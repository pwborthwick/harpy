#main program
'''
if you change geometry you must rebuild molAtom and molBasis
with molAtom, molBasis = rebuildCenters(molAtom, molBasis, geo)
where geo is a (nAtoms, 3) array of the new geometry.

if you don't want output then either replace lists with [] or 
change the name of the item(s) to say 'mintsx'.
'''

import rhf
import uhf
import rohf
import time
import sys
from numpy import load

t = time.time()

#concatenate arguments harpy.py-rhf.-v.-m.-LiH.
hfType = 'rhf'
args = ''
for arg in sys.argv:
    args += arg + '.'

#get hf type, remove fom args list once processed
if '-uhf.' in args: hfType = 'uhf'
if '-rohf.' in args: hfType = 'rohf'
args = args.replace('-uhf.','').replace('-rhf.','').replace('-rohf.','').replace('harpy.py.','')

#verbose or minimal
molList, scfList = [[],['SCF', 'postSCF']]
if '-v.' in args:
    molList = ['geometry', 'orbitals']
    scfList = [ 'preSCF', 'SCF', 'postSCF']

if '-m.' in args: scfList.append('mints')
args = args.replace('-v.','').replace('-m.','')

#get project file
molFile = 'project.hpf'
if '.hpf' in args:
    sgra = args[::-1]
    i = sgra.find('fph.')
    j = sgra.find('-', i)
    molFile = sgra[i:j][::-1]
    args = args.replace('-'+molFile+'.','')

#only thing left in args is molecule
molecule = ''
if '-' in args:
    molecule = args[1:-1]

#set up geometry and basis
if hfType == 'rhf':
    molAtom, molBasis, molData = rhf.mol(molList, file=molFile, molecule=molecule)
    #check if there is a density matrix for this molecule in the basis
    name, basisName = [molData['name'], molData['basis']]
    mints = '../mints/' + name + '-' + basisName + '-mints.npz'
    try:
        data = load(mints)
        density = data['d']
    except FileNotFoundError:
        density = None

elif hfType in ['uhf', 'rohf']:
    molAtom, molBasis, molData = rhf.mol(molList, file=molFile, molecule=molecule, method=hfType)

#do scf calculation
if hfType == 'rhf':
    print(rhf.scf(molAtom, molBasis, molData , scfList, density), ' in ',round(time.time()-t,3),'s')
elif hfType == 'uhf':
    print(uhf.scf(molAtom, molBasis, molData , scfList), ' in ',round(time.time()-t,3),'s')
elif hfType == 'rohf':
    print(rohf.scf(molAtom, molBasis, molData , scfList), ' in ',round(time.time()-t,3),'s')