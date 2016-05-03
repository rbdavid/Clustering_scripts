#!/mnt/lustre_fs/users/mjmcc/apps/python2.7/bin/python
# ----------------------------------------
# USAGE:

# ----------------------------------------
# PREAMBLE:

import numpy as np
from numpy.linalg import *
import MDAnalysis
from MDAnalysis.analysis.align import *
from MDAnalysis.analysis.rms import *
import sys

# ----------------------------------------
# VARIABLE DECLARATION

pdb_file = sys.argv[1]
traj_loc = sys.argv[2]
start = int(sys.argv[3])
end = int(sys.argv[4])

alignment = 'backbone and (resid 19:25 or resid 50:55 or resid 90:94 or resid 112:117 or resid 142:149 or resid 165:169 or resid 190:194 or resid 214:218 or resid 236:240 or resid 253:258 or resid 303:307)'

zeros = np.zeros
dot_prod = np.dot
rmsd = MDAnalysis.analysis.rms.rmsd
sqrt = np.sqrt
flush = sys.stdout.flush

thresh = 1E-5
maxIter = 100

# ----------------------------------------
# SUBROUTINES:

def ffprint(string):
	print '%s' %(string)
	flush()

# ----------------------------------------
# MAIN PROGRAM:
# ATOM SELECTION - load the pdb and traj into the desired MDAnalysis universe objects
u = MDAnalysis.Universe(pdb_file)
avg_pdb = MDAnalysis.Universe(pdb_file)

u_all = u.select_atoms('all')
backbone = u.select_atoms('backbone')
u_important = u.select_atoms('protein or nucleic or resname A5 or resname A3 or resname U5 or resname atp or resname adp or resname PHX or resname MG and not name H*') 
u_align = u.select_atoms(alignment)
u_substrate = u.select_atoms('nucleic or resname A5 or resname A3 or resname U5 or resname atp or resname adp or resname PHX or resname MG and not name H*')

avg_important = avg_pdb.select_atoms('protein or nucleic or resname A5 or resname A3 or resname U5 or resname atp or resname adp or resname PHX or resname MG and not name H*')

u_important_atoms = len(u_important.atoms)
align_atoms = len(u_align.atoms)
substrate_res = len(u_substrate.residues)

ffprint('%d   %d' %(u_important_atoms, align_atoms))

# Determining number of steps
temp = start
nSteps = 0
while temp <= end:
	u.load_new('%s/Truncated/production.%s/production.%s.dcd' %(traj_loc,start,start))
	nSteps += len(u.trajectory)
	temp += 1

ffprint(nSteps)

# Array Declaration
all_coord = zeros((nSteps,u_important_atoms,3),dtype=np.float32)
avgCoord = zeros((u_important_atoms,3),dtype=np.float32)
all_align = zeros((nSteps,align_atoms,3),dtype=np.float32)
avgAlign = zeros((align_atoms,3),dtype=np.float32)

# Trajectory Analysis: 
ffprint('Beginning trajectory analysis')
temp = 0 
while start <= end:
	ffprint('Loading trajectory %s' %(start))
	u.load_new('%s/Truncated/production.%s/production.%s.dcd' %(traj_loc,start,start))

	for ts in u.trajectory:
		dimensions = u.dimensions[:3]
		dimensions2 = dimensions/2.0

		u_all.translate(-backbone.center_of_geometry())

		for i in range(substrate_res):
			COM = np.zeros(3)
			t = zeros(3)
			COM = u_substrate.residues[i].center_of_mass()

			for j in range(3):
				if COM[j] < -dimensions2[j]:
					t[j] = dimensions[j]
				elif COM[j] > dimensions2[j]:
					t[j] = -dimensions[j]
			u_substrate.residues[i].atoms.translate(t)
		
		avgCoord += u_important.positions
		avgAlign += u_align.positions
		all_coord[temp] = u_important.positions
		all_align[temp] = u_align.positions
		temp += 1
	start += 1

ffprint(nSteps)
if temp != nSteps:
	ffprint('Failed to analyze all timesteps; fucked shit up')

avgCoord /= float(nSteps)
avgAlign /= float(nSteps)
ffprint(len(all_coord))
ffprint(len(all_coord[0]))
ffprint('Finished with the trajectory analysis')

# Calculating and Aligning to the average positions
iteration = 0
residual = thresh + 10.0 					# arbitrary assignment greater than thresh
ffprint('Beginning iterative process of calculating average positions and aligning to the average')
while residual > thresh and iteration < maxIter:		
	tempAvgCoord = zeros((u_important_atoms,3),dtype=np.float32)		# zeroing out the tempAvgCoord array every iteration
	tempAvgAlign = zeros((align_atoms,3),dtype=np.float32)
	for i in range(nSteps):
		R, d = rotation_matrix(all_align[i,:,:],avgAlign)
		all_align[i,:,:] = dot_prod(all_align[i,:,:],R.T)
		all_coord[i,:,:] = dot_prod(all_coord[i,:,:],R.T)
		tempAvgAlign += all_align[i,:,:]
		tempAvgCoord += all_coord[i,:,:]			# recalculate the average coordinates to optimize the average position
	tempAvgCoord /= float(nSteps)				# finishing the average
	tempAvgAlign /= float(nSteps)
	residual = rmsd(avgAlign, tempAvgAlign)			# calculating the rmsd between avg and tempAvg to quantify our iterative optimization of the average positions	
	iteration += 1
	avgCoord = tempAvgCoord
	avgAlign = tempAvgAlign
	print 'Step:', iteration, 'Residual=', residual
ffprint('Average structure has converged')				# Now have the iteratively aligned avgCoord array, as well as the iteratively aligned (COG-corrected and rotated) allCoord array

# Print out pdb of average structure
ffprint('Writing a pdb of the average structure.')
#avg_important.residues.set_positions(avgCoord)
avg_important.set_positions(avgCoord)
avg_important.write('avg_structure.pdb')
ffprint('Finished writing pdb of the average structure')

