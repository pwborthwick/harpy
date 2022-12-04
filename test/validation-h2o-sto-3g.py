from __future__ import division
import numpy as np
import time
import os
from numpy.linalg import eigh
from scipy.linalg import fractional_matrix_power as fractPow
import sys
sys.path.append('../source')
import rhf
from atom import getConstant

'''
Test of water in STO-3G basis for all testable codes
'''

#read command line for 'F'
args = sys.argv[1:]
if args == []: 
	mode = 'Q'
else:
	mode = args[0]

sysStart = time.time()
processTime = sysStart

def stepTime(step):
	#elapsed time for current step

	if step == 1:
		t = time.time() - sysStart
	else:
		t = time.time() - processTime

	h = 0
	if t >3600:
		h = int(t//3600)
		t -= h*3600
	m = 0
	if t > 60:
		m = int(t//60)
		t -= m*60
	s = round(t,1)

	return ('00' + str(h))[-2:] + ':' + ('00' + str(m))[-2:] + ':' + ('00' + str(s))[-4:]

def passing(actual, reference, tol):

	if abs(actual-reference) < 10 **(-tol-1):
		return 'True'
	else:
		return 'False'

def validityCheck(actual, reference, tolerance = 1e-6):

	if np.allclose(actual, reference, tolerance):
		return 'True'
	else:
		return 'False'

outputScheme = "{:<12} {:<12} {:<48} {:>13.8f} {:<6} {:<2} {:<3}"

#define molecule and run parameters
molecule = """name=h2o
matrix=c
basis=sto-3g
post={}
diis=on

O1 8 0.000000000000 -0.143225816552 0.000000000000
H1 1 1.638036840407  1.136548822547 0.000000000000
H2 1 -1.638036840407 1.136548822547 0.000000000000

end
"""
fileName = '../test/h2o.hpf'
f = open(fileName, 'w')
f.write(molecule)
f.close()

molAtom, molBasis, molData = rhf.mol([], file = fileName)
#clean up file
if os.path.exists(fileName):
    os.remove(fileName)

#do basis EHF SCF calculation
eSCF = rhf.scf(molAtom, molBasis, molData, [])

#==================Header
print('='*101)
print(' '*40, 'Water Test in STO-3G                        mode[', mode, ']')
print('-'*101)
print('  Elapsed       Step                            Process                         Value    Pass   Check')
print('-'*101)
#==================SCF energy
value = eSCF
print(outputScheme.format(stepTime(1), stepTime(0), 'SCF energy',\
      value, validityCheck(value, -74.942079928192), 6, 'CW'))

#==================Mulliken charges
processTime = time.time()
from post import mulliken
value = 8 - mulliken(rhf.density, rhf.S, molAtom, molBasis)[0]
print(outputScheme.format(stepTime(1), stepTime(0), 'Mulliken oxygen charge ', \
	  value, validityCheck(value, -0.253146052405), 6, 'CW'))

#==================dipole moment
processTime = time.time()
from post import buildDipole
value = np.sqrt(buildDipole(molAtom, molBasis, rhf.density, 'origin')[3])
print(outputScheme.format(stepTime(1), stepTime(0), 'Dipole Moment resultant', \
	  value, validityCheck(value, 0.603521296525), 6, 'CW'))

#==================quadrupole moment
processTime = time.time()
from post import buildQuadrupole
value = buildQuadrupole(molAtom, molBasis, rhf.density, 'origin')[12]
print(outputScheme.format(stepTime(1), stepTime(0), 'Quadrupole Moment amplitude', \
	  value, validityCheck(value, 1.3376918702067326), 6, 'HA'))

#==================Polarizabilities
processTime = time.time()
from post import polarizabilities
value, isotropic, _, _ = polarizabilities(molAtom, molBasis, rhf.C, rhf.fock, rhf.ERI, 5, 'nuclear charge')
reference = [7.93556215, 3.06821073, 0.05038621, 3.68471969539233]
comparison = [value[0], value[1], value[2], isotropic]
print(outputScheme.format(stepTime(1), stepTime(0), 'Polarizabilities                    [isotropic]', \
	  isotropic, validityCheck(reference, comparison), 6, 'HA'))

#==================Hyperpolarizabilities
processTime = time.time()
from post import hyperPolarizabilities
X = fractPow(rhf.S, -0.5)
orthogonalFock = np.dot(X.T , np.dot(rhf.fock , X))
#diagonalise Fock
e, orthogonalC = eigh(orthogonalFock)
_, value = hyperPolarizabilities(molAtom, molBasis, rhf.C, rhf.ERI, e, rhf.fock, 5, 'nuclear charge')
reference = [-9.34242926,-5.20670354, 1.38579626e-01, -9.34242926e+00, 1.38579626e-01 ]
comparison = [value[0,1], value[1,1], value[2,1], value[3,0], value[5,2]]
print(outputScheme.format(stepTime(1), stepTime(0), 'Hyperpolarizabilities                     [0,1]', \
	  value[0,1], validityCheck(reference, comparison), 6, 'HA'))

print('='*101)
#==================Moller-Plesset (2)
processTime = time.time()
from mp import mp2
p, ap = mp2(molAtom, 0, molBasis, rhf.C, e, rhf.ERI, 'p-ap')
print(outputScheme.format(stepTime(1), stepTime(0), 'Moller-Plesset (2)', \
	  p+ap, validityCheck(p+ap, -0.049149636120), 6, 'CW'))

#==================Moller-Plesset spin-component (2)
processTime = time.time()
from mp import mp2
value = mp2(molAtom, 0, molBasis, rhf.C, e, rhf.ERI, 'scs')
print(outputScheme.format(stepTime(1), stepTime(0), 'Spin-Component Scaled MP (2)', \
	  value, validityCheck(value, -0.0562875057281451), 6, 'PN'))

#==================Orbital-Optimised MP2
processTime = time.time()
from mp import orbitalOptimisedMP2
from atom import nuclearRepulsion
value = orbitalOptimisedMP2(rhf.C, rhf.coreH, e, molBasis, nuclearRepulsion(molAtom), rhf.ERI, 10) -eSCF
print(outputScheme.format(stepTime(1), stepTime(0), 'Orbital-Optimised MP (2)', \
	  value, validityCheck(value, -0.04939136780163267), 6, 'HA'))

# #==================Laplace Transform MP2
processTime = time.time()
from mp import mp2LaplaceTransform
value = mp2LaplaceTransform(molBasis, rhf.C, rhf.ERI, e, 5, eSCF, meshSize=40)
value = sum(value)
print(outputScheme.format(stepTime(1), stepTime(0), 'Laplace Transform MP (2)', \
	  value, validityCheck(value, -0.0491458273 ), 6, 'PN'))

print('='*101)
#==================CIS
processTime = time.time()
from ci import cis, ciDegeneracy
value, _ = cis(molAtom, 0, molBasis, rhf.C, rhf.fock, rhf.ERI)
degen = ciDegeneracy(value)
reference =  [0.2872554996 , 0.3444249963, 0.3564617587, 0.3659889948, 0.3945137992]
comparison = [degen[0][0], degen[1][0], degen[2][0], degen[3][0], degen[4][0]]
print(outputScheme.format(stepTime(1), stepTime(0), 'CIS                                         [0]', \
      degen[0][0], validityCheck(reference, comparison), 6, 'CW'))

#==================RPA
processTime = time.time()
from ci import randomPhaseApproximation
value = randomPhaseApproximation(molAtom, 0, molBasis, rhf.C, rhf.fock, rhf.ERI, 'linear')
degen = ciDegeneracy(value)
reference =  [0.2851637170 ,0.2997434467 , 0.3526266606, 0.3547782530, 0.3651313107]
comparison = [degen[0][0], degen[1][0], degen[2][0], degen[3][0], degen[4][0]]
print(outputScheme.format(stepTime(1), stepTime(0), 'Random Phase Approximation                  [0]', \
      degen[0][0], validityCheck(reference, comparison), 6, 'CW'))

#==================sa-CIS-singles
processTime = time.time()
from ci import ciSpinAdaptedSingles
value = ciSpinAdaptedSingles(molAtom, 0, molBasis, rhf.C, rhf.fock, rhf.ERI)
reference =  [0.3564617587, 0.4160717386, 0.5056282877 , 0.5551918860 , 0.6553184485]
comparison = [value[0], value[1], value[2], value[3], value[4]]
print(outputScheme.format(stepTime(1), stepTime(0), 'Spin-Adapted CI singles                     [0]', \
      value[0], validityCheck(reference, comparison), 6, 'CW'))

#==================sa-CIS-triples
processTime = time.time()
from ci import ciSpinAdaptedTriples
value = ciSpinAdaptedTriples(molAtom, 0, molBasis, rhf.C, rhf.fock, rhf.ERI)
reference =  [0.2872554996, 0.3444249963, 0.3659889948, 0.3945137992, 0.5142899971]
comparison = [value[0], value[1], value[2], value[3], value[4]]
print(outputScheme.format(stepTime(1), stepTime(0), 'Spin-Adapted CI triples                     [0]', \
      value[0], validityCheck(reference, comparison), 6, 'CW'))

print('='*101)
#==================CCSD
processTime = time.time()
from cc.scc import ccsd
value, ts, td = ccsd(molAtom, rhf.C, 0, rhf.fock, rhf.ERI, 50, 1e-8, eSCF, 'on')
print(outputScheme.format(stepTime(1), stepTime(0), 'Coupled-Cluster Singles and Doubles', \
	  value, validityCheck(value, -0.070680088376), 6, 'CW'))

#==================(T)
processTime = time.time()
from cc.scc import amplitudesT3
from integral import buildFockMOspin, buildEriMO, buildEriDoubleBar
#get fock in MO spin basis
fockMOspin = buildFockMOspin(14, rhf.C, rhf.fock)
#get two-electron repulsion integrals in MO basis
eriMO = buildEriMO(rhf.C, rhf.ERI)
#transform eri from MO to spin basis
eriMOspin = buildEriDoubleBar(14, eriMO)
value = amplitudesT3(fockMOspin, ts, td, eriMOspin, 10)
print(outputScheme.format(stepTime(1), stepTime(0), 'CCSD Perturbative Triples', \
	  value, validityCheck(value, -0.000099877272), 6, 'CW'))

if mode == 'F':
	#==================CCD
	processTime = time.time()
	from cc.scc import ccd
	value = ccd(molAtom, rhf.C, 0, rhf.fock, rhf.ERI, 50, 1e-8, eSCF)
	print(outputScheme.format(stepTime(1), stepTime(0), 'Coupled-Cluster Doubles', \
	      value, validityCheck(value, -0.07015051), 6, 'PN'))

#==================LCCD
processTime = time.time()
from cc.scc import lccd
value = lccd(molAtom, rhf.C, 0, rhf.fock, rhf.ERI, 50, 1e-8, eSCF)
print(outputScheme.format(stepTime(1), stepTime(0), 'Linear Coupled-Cluster Doubles', \
	  value, validityCheck(value, -0.07192915), 6, 'P4'))

#==================LCCSD
processTime = time.time()
from cc.scc import lccsd
value, _, _ = lccsd(molAtom, rhf.C, 0, rhf.fock, rhf.ERI, 50, 1e-8, 'on', eSCF)
print(outputScheme.format(stepTime(1), stepTime(0), 'Linear Coupled-Cluster Singles and Doubles', \
	  value, validityCheck(value, -0.07257659), 6, 'HA'))

#==================CC2
processTime = time.time()
from cc.scc import cc2
value, _, _ = cc2(molAtom, rhf.C, 0, rhf.fock, rhf.ERI, 50, 1e-8, eSCF, 'on')
print(outputScheme.format(stepTime(1), stepTime(0), 'Coupled-Cluster (2)', \
	  value, validityCheck(value, -74.9914791 + 74.94207992818629), 6, 'P4'))

#==================CCSD-Lambda
processTime = time.time()
from cc.scc import ccsd_lambda
value, l1, l2, _ = ccsd_lambda(fockMOspin, eriMOspin, ts, td, 10, 50, 1e-8)
print(outputScheme.format(stepTime(1), stepTime(0), 'Coupled-Cluster Singles and Doubles-\u039B',  \
	  value, validityCheck(value, -0.068888201463), 6, 'P4'))

print('='*101)
#===================Electron Propagator 2
processTime = time.time()
from ep import electronPropagator2
value = electronPropagator2(molBasis, rhf.C, rhf.ERI, e, 5, startOrbital = 2, nOrbitals = 4)
reference = [-10.3403, -7.4044, 12.8950, 15.6734]
comparison = [value[0], value[1], value[2], value[3]]
print(outputScheme.format(stepTime(1), stepTime(0), 'Electron Propagator (2)                     [0]', \
      value[0], validityCheck(reference, comparison, 1e-4), 4, 'HA'))

#===================spin-adapted Electron Propagator 2
processTime = time.time()
from ep import electronPropagator2spin
value = electronPropagator2spin(molBasis, rhf.C, rhf.ERI, e, 5, nOrbitals = 5)
reference = [-541.7558, -29.8607, -14.5980, -10.3404, -7.4044]
comparison = [value[0], value[1], value[2], value[3], value[4]]
print(outputScheme.format(stepTime(1), stepTime(0), 'Spin-Adapted Electron Propagator (2)        [0]', \
      value[0], validityCheck(reference, comparison, 1e-4), 4, 'HA'))

#===================spin-adapted Electron Propagator 3
processTime = time.time()
from ep import electronPropagator3spin
value = electronPropagator3spin(molBasis, rhf.C, rhf.ERI, e, 5, nOrbitals = 5)
reference = [-543.6307, -29.2271, -14.8363, -10.7673, -8.1825]
comparison = [value[0], value[1], value[2], value[3], value[4]]
print(outputScheme.format(stepTime(1), stepTime(0), 'Spin-Adapted Electron Propagator (3)        [0]', \
      value[0], validityCheck(reference, comparison, 1e-4), 4, 'PN'))

#==================Associated Green's Function Correction to Koopman
processTime = time.time()
from ep import koopmanAGFcorrection
value = koopmanAGFcorrection(molBasis ,rhf.C ,rhf.ERI, e, 5, nOrbitals = 5)
reference = [-12.6644, -5.3342, -1.6200, -2.5340, -3.5334]
comparison = [value[0][1], value[1][1], value[2][1], value[3][1], value[4][1]]
print(outputScheme.format(stepTime(1), stepTime(0), 'Koopman Green Function Correction           [0]', \
      value[0][1], validityCheck(reference, comparison, 1e-4), 4, 'HA'))

print('='*101)
#==================FCI
processTime = time.time()
from fci import fci
value = fci(molAtom, molBasis, 0, rhf.C, rhf.ERI, rhf.coreH)[0] - eSCF + nuclearRepulsion(molAtom)
print(outputScheme.format(stepTime(1), stepTime(0), 'Full Configuration Interaction              [0]', \
	  value, validityCheck(value, -0.07090027025095935), 6, 'P4'))

#==================CIS slater
processTime = time.time()
from fci import ciss
value, _ = ciss(molAtom, molBasis, 0, rhf.C, rhf.ERI, rhf.coreH)
degen = ciDegeneracy(value)
reference =  [0.28725544 , 0.34442492, 0.35646170, 0.36598893, 0.39451371]
comparison = [degen[0][0], degen[1][0], degen[2][0], degen[3][0], degen[4][0]]
print(outputScheme.format(stepTime(1), stepTime(0), 'Determinant CIS                             [0]', \
	  degen[0][0], validityCheck(reference, comparison), 6, 'MM'))

#===================CISD slater
processTime = time.time()
from fci import cisd
from atom import nuclearRepulsion
value = cisd(molAtom, molBasis, 0, rhf.C, rhf.ERI, rhf.coreH)[0] - eSCF + nuclearRepulsion(molAtom)
print(outputScheme.format(stepTime(1), stepTime(0), 'Determinant CISD                            [0]', \
	  value,  validityCheck(value, -0.06914309), 6, 'MM'))

print('='*101)
#==================Forces - Ocypete
processTime = time.time()
from force import forces
value = forces(molAtom, molBasis, rhf.density, rhf.fock, engine = 'aello', type = 'analytic')
reference = [9.74414370e-02, -8.63000979e-02, -4.87207185e-02, 8.63000979e-02, -4.87207185e-02]
comparison = [value[0,1], value[1,0], value[1,1], value[2,0], value[2,1]]
print(outputScheme.format(stepTime(1), stepTime(0), 'Forces (ocypete - cython)                 [0,1]', \
	  value[0,1], validityCheck(reference, comparison), 6, 'MM'))

if mode == 'F':
	#===================Forces - native
	processTime = time.time()
	from force import forces
	value = forces(molAtom, molBasis, rhf.density, rhf.fock, engine = 'native', type = 'analytic')
	reference = [9.74414370e-02, -8.63000979e-02, -4.87207185e-02, 8.63000979e-02, -4.87207185e-02]
	comparison = [value[0,1], value[1,0], value[1,1], value[2,0], value[2,1]]
	print(outputScheme.format(stepTime(1), stepTime(0), 'Forces (native  - python)                 [0,1]', \
	      value[0,1], validityCheck(reference, comparison), 6, 'MM'))

print('='*101)
#===================rESP
processTime = time.time()
from rESP import restrainedESP
con, res = restrainedESP(molAtom, molBasis, molData, { 'sphere' : 'con', 'points' : ['density', 1], 'shell' : [4, 0.2, 1.4], 
                        'file' : ['w', 'esp.npz', 'clear'], 'view' : False, 'constrain' : [[0,[-2,3]]], 'restrain' : {} } )
reference = [-0.47703059, 0.23851529 ,0.23851529, -0.47551183, 0.23775591, 0.23775591]
comparison = [con[0], con[1], con[2], res[0], res[1], res[2]]
print(outputScheme.format(stepTime(1), stepTime(0), 'Restrained Electrostatic Potential          [0]', \
	  con[0], validityCheck(reference, comparison), 6, 'PN'))

print('='*101)
#=======================EOM-CCSD
processTime = time.time()
data = {'method':'ccsd','electrons':10, 'cycle_limit': 50, 'precision':1e-10, 'verbose':False}
from cc.fcc import coupledCluster
cc = coupledCluster(fockMOspin, eriMOspin, rhf.e, data)

from cc.fcc import eom_ccsd
eom = eom_ccsd(cc, roots=[7,11], partitioned = False)
reference = [7.4901, 8.7959, 9.8321, 10.0122, 10.7445]
comparison = [i[0] for i in eom.excitations]
print(outputScheme.format(stepTime(1), stepTime(0), 'EOM-CCSD  (fast)                            [0]', \
      comparison[0], validityCheck(reference, comparison, 1e-4), 4, 'GA'))

if mode == 'F':
	print('='*101)
	#===================EOM-CCSD
	processTime = time.time()
	from eom import eomccsd
	value, _ = eomccsd(fockMOspin, eriMOspin, ts, td, 10, 4, 14, partitioned = False, dialog = False)
	reference = [7.4901, 8.7959, 9.8321, 10.0122, 10.7445]
	value = np.sort(value * getConstant('hartree->eV'))
	degen = ciDegeneracy(value)
	comparison = []
	for excitation in degen:
		if excitation[0] > 5 and excitation[0] < 11:
			comparison.append(excitation[0])
	print(outputScheme.format(stepTime(1), stepTime(0), 'EOM-CCSD                                    [0]', \
	      comparison[0], validityCheck(reference, comparison, 1e-4), 4, 'GA'))

	#====================EOM-MBPT(2)
	processTime = time.time()
	from eom import eommbpt2
	value, _ = eommbpt2(fockMOspin, eriMOspin, 10, 4, 14, partitioned = False, dialog = False)
	reference = [7.1222, 8.4365, 9.4985, 9.6904, 10.4213]
	value = np.sort(value * getConstant('hartree->eV'))
	degen = ciDegeneracy(value)
	comparison = []
	for excitation in degen:
		if excitation[0] > 5 and excitation[0] < 11:
			comparison.append(excitation[0])
	print(outputScheme.format(stepTime(1), stepTime(0), 'EOM-MBPT (2)                                [0]', \
	      comparison[0], validityCheck(reference, comparison, 1e-4), 4, 'GA'))

print('='*101)
#========================TDHF-transition
processTime = time.time()
from tdhf import TDHFproperties
residues = TDHFproperties(molAtom, molBasis, 0, rhf.C, rhf.fock, rhf.ERI, 'tamm-dancoff', [5, 2, 5])
reference = [0.35646177543487845, 0.09925794226244171, 0.0023412739974616056, -0.30230280991042413, \
            -5.925026183856348e-16, -1.0994804162166085e-15, 0.1305217446887937, 0.03186115503468884]
r = residues[0]
comparison = [r['excitation energy'], r['electric transition dipole (length gauge)'][2], r['oscillator strength (length gauge)'],\
              r['magnetic transition dipole (length gauge)'][0], r['rotatory strength (length gauge)'],r['rotatory strength (velocity gauge)'],\
              r['electric transition dipole (velocity gauge)'][2], r[ 'oscillator strength (velocity gauge)']]
print(outputScheme.format(stepTime(1), stepTime(0), 'TDHF transition properties         [excitation]', \
	  comparison[0], validityCheck(reference, comparison), 6, 'P4'))

#========================TDHF-spectrum
processTime = time.time()
from tdhf import TDHFspectrum
dict = {'data' : residues, 'units' : 'nm', 'type' : 'opa', 'points' : 5000, 'shape' : 'gaussian', 'gamma' : 0.02}
spectrum = TDHFspectrum(dict)
reference = [ 57.87004982, 7.58675674e-22, 127.82114569, 9.70790360e-01]
comparison = [spectrum['x'][0], spectrum['y'][0], spectrum['poles'][0], spectrum['bars'][0]]
print(outputScheme.format(stepTime(1), stepTime(0), 'TDHF OPA spectrum                           [x]', \
	  comparison[0], validityCheck(reference, comparison), 6, 'P4'))

print('='*101)
#========================cogus
processTime = time.time()
from cogus import symbolicGeneratedCoupledCluster
_, value, _ = symbolicGeneratedCoupledCluster('ccd', rhf.fock, rhf.ERI, rhf.C, e, [0, nuclearRepulsion(molAtom), 10])
print(outputScheme.format(stepTime(1), stepTime(0), 'cogus - ccd', \
	    value, validityCheck(value, -0.07015048), 8, 'PN'))

processTime = time.time()
_, value, _ = symbolicGeneratedCoupledCluster('ccsd', rhf.fock, rhf.ERI, rhf.C, e, [0, nuclearRepulsion(molAtom), 10])
print(outputScheme.format(stepTime(1), stepTime(0), 'cogus - ccsd', \
	    value, validityCheck(value, -0.07068008709615015), 8, 'P4'))

processTime = time.time()
_, value, _ = symbolicGeneratedCoupledCluster('ccsdt', rhf.fock, rhf.ERI, rhf.C, e, [0, nuclearRepulsion(molAtom), 10])
print(outputScheme.format(stepTime(1), stepTime(0), 'cogus - ccsdt', \
	    value, validityCheck(value, -0.07081280801921253), 8, 'P4'))

processTime = time.time()
_, value, cp = symbolicGeneratedCoupledCluster('ccsd_t', rhf.fock, rhf.ERI, rhf.C, e, [0, nuclearRepulsion(molAtom), 10])
print(outputScheme.format(stepTime(1), stepTime(0), 'cogus - ccsd(t)', \
	    cp, validityCheck(cp, -9.987726961762642e-05), 8, 'P4'))

processTime = time.time()
_, value, _ = symbolicGeneratedCoupledCluster('cc2', rhf.fock, rhf.ERI, rhf.C, e, [0, nuclearRepulsion(molAtom), 10])
print(outputScheme.format(stepTime(1), stepTime(0), 'cogus - cc2', \
	    value, validityCheck(value, -0.0493991397445086), 8, 'P4'))

processTime = time.time()
_, value, _ = symbolicGeneratedCoupledCluster('cc3', rhf.fock, rhf.ERI, rhf.C, e, [0, nuclearRepulsion(molAtom), 10])
print(outputScheme.format(stepTime(1), stepTime(0), 'cogus - cc3', \
	    value, validityCheck(value, -0.07077803146036388), 8, 'P4'))

processTime = time.time()
_, value, _ = symbolicGeneratedCoupledCluster('lccd', rhf.fock, rhf.ERI, rhf.C, e, [0, nuclearRepulsion(molAtom), 10])
print(outputScheme.format(stepTime(1), stepTime(0), 'cogus - lccd', \
	    value, validityCheck(value, -0.07192916394222108), 8, 'HA'))

processTime = time.time()
_, value, _ = symbolicGeneratedCoupledCluster('lccsd', rhf.fock, rhf.ERI, rhf.C, e, [0, nuclearRepulsion(molAtom), 10])
print(outputScheme.format(stepTime(1), stepTime(0), 'cogus - lccsd', \
	    value, validityCheck(value, -0.07257658934412553), 8, 'HA'))

processTime = time.time()
from cogus import symbolicGeneratedLambda
_, value, _, _ = symbolicGeneratedLambda('ccsd', rhf.fock, rhf.ERI, rhf.C, e, rhf.coreH, [0, nuclearRepulsion(molAtom), 10])
print(outputScheme.format(stepTime(1), stepTime(0), 'cogus - \u039B', \
	    value, validityCheck(value, -0.07068008881678622), 8, 'HA'))

print('='*101)
#==================MBPT
processTime = time.time()
from mbpt import mbptEvaluateMPn
value = mbptEvaluateMPn(molAtom, molBasis, molData, eSCF)
print(outputScheme.format(stepTime(1), stepTime(0), 'Many-Bodied Perturbation Theory             [2]', \
	  value[0], validityCheck(value[0], -0.0491496272), 6, 'P4'))
print(outputScheme.format(stepTime(1), stepTime(0), 'Many-Bodied Perturbation Theory             [3]', \
	  value[1], validityCheck(value[1], -0.0141878317), 6, 'P4'))
print(outputScheme.format(stepTime(1), stepTime(0), 'Many-Bodied Perturbation Theory             [4]', \
	  value[2], validityCheck(value[2], -0.0046902727), 6, 'HA'))

print('='*101)
#==================UHF
processTime = time.time()
import uhf
value = uhf.scf(molAtom, molBasis, molData, [])
print(outputScheme.format(stepTime(1), stepTime(0), 'uhf energy', \
	    value, validityCheck(value,  -74.942079928192356), 6, 'PS'))
molData['charge'] = -1
molData['multiplicity'] = 2
value = uhf.scf(molAtom, molBasis, molData, [])
print(outputScheme.format(stepTime(1), stepTime(0), 'uhf energy cation', \
	    value, validityCheck(value, -74.487850244828891 ), 6, 'PS'))
