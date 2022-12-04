import sys
sys.path.append('../source')

import rhf
import uhf
from integral import iEri
import numpy as np
import os
'''
This illustrates the difference in RHF and UHF when dealing
with the dissociation of the hydrogen molecule as the bond
length is increased from 0.5-12 bohr. Compare with Szabo &
Ostlund pg 166 fig.3.5

Molecule is aligned along x-axis with H1 at [0,0,0].
'''
mol = """name=h2
matrix=c
diis=off
basis=sto-3g
uhfmix=0.02
post={}

H1 1 0.00000000000000  0.00000000 0.00000000000
H2 1 1.39838996182773  0.00000000 0.00000000000

end
"""
fileName = 'h2.hpf'
f = open(fileName, 'w')
f.write(mol)
f.close()

#create molecular atom and basis objects
molAtom, molBasis, molData = rhf.mol([], file = fileName)

#clean up file
if os.path.exists(fileName):
    os.remove(fileName)

#solve base geometry for <00|00>
rhf.scf(molAtom, molBasis, molData, [])
Heri = rhf.ERI[iEri(0,0,0,0)] * 0.5

#initial position of H2
a = molAtom[1].center[:] 

#definition of range 
start = 0.5
points = 100

nAtoms = len(molAtom)

erhf = []
euhf = []
separation = []

#set geo to original geometry
geo = np.zeros((nAtoms,3))
for atom in range(nAtoms):
	geo[atom,:] = molAtom[atom].center

for i in range(points):

	#update geometry and rebuild centers with it
	geo[1,0] = start + i * 0.125
	molAtom, molBasis = rhf.rebuildCenters(molAtom, molBasis, geo)

	#calculate energies
	separation.append(geo[1,0])
	erhf.append(rhf.scf(molAtom, molBasis, molData, []))
	euhf.append(uhf.scf(molAtom, molBasis, molData, []))
	
#get single hydrogen atom energy
molAtom = molAtom[:1]
molBasis = molBasis[:1]
molAtom, molBasis = rhf.rebuildCenters(molAtom, molBasis, geo)
molData['multiplicity'] = 2
H = uhf.scf(molAtom, molBasis, molData, [])

#plot results
import matplotlib.pyplot as pl

pl.plot(separation, erhf, '.r')
pl.plot(separation, euhf, '.m')

pl.plot([0,13], [2*H, 2*H], 'k.--')
pl.plot([0,13], [2*H + Heri,2*H + Heri], 'k.--')

pl.text(12, -0.65, 'rhf')
pl.text(12, -1.00, 'uhf')
pl.text(1.8,-0.52,'2H + <11|11>/2', size='x-small')
pl.text(1.8,-0.91,'2H', size='x-small')

pl.xlabel('bond length (bohr)')
pl.ylabel('energy (Hartree)')

pl.title('H$_2$' + ' dissociation rhf v uhf')

pl.show()

