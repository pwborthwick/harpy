from __future__ import division
#import system modules
import sys
sys.path.append('../source')

import os
import numpy as np
import rhf
from basis import electronCount

def getMolecule(basis):
	#get the molecular definition objects

	mol = """name=h2o
matrix=c
diis=on
basis=sto-3g
charge=0
multiplicity=1

O1 8 0.000000000000 -0.143225816552 0.000000000000
H1 1 1.638036840407  1.136548822547 0.000000000000
H2 1 -1.638036840407 1.136548822547 0.000000000000

end
	"""
	#replace basis if necessary
	if basis != None:
		mol = mol.replace('sto-3g', basis)

	fileName = 'h2o.hpf'
	f = open(fileName, 'w')
	f.write(mol)
	f.close()

	molAtom, molBasis, molData = rhf.mol([], file = fileName)

	#clean up file
	if os.path.exists(fileName):
	    os.remove(fileName)

	molData['basis'] = basis
	eSCF = rhf.scf(molAtom, molBasis, molData, [])

	return eSCF, molAtom, molBasis, molData

def getOrbitalEnergies(fock, S):
	#compute the orbital energies

    from scipy.linalg import fractional_matrix_power as fractPow
    X = fractPow(S, -0.5)

    orthogonal_fock = np.dot(X.T, np.dot(fock, X))
    e, _ = np.linalg.eigh(orthogonal_fock)

    return e
 
def getOrbitalCharacter(orbital, basis, atom):
    #get the AO characteristic of MO
 
    aoMax = np.where(orbital == np.amax(orbital))

    return (basis[aoMax[0][0]].symbol, atom[basis[aoMax[0][0]].atom].id)

def getMOwithCharacter(character, c, basis, atom):
    #find the MO with the given AO character

    for i in range(c.shape[1]):
        ao = getOrbitalCharacter(c[:,i], basis, atom)
        if ao == character: 
            return i

def getKedgeExcitations(k, n, c, basis, atom, eMO):
    #get the n excitations

    lumo = electronCount(atom, data['charge'])//2

    #get MO corresponding to 1s O1 and energy
    mo = getMOwithCharacter(k, c, basis, atom)

    k = np.zeros((n))
    for i in range(n):
        k[i] = abs(eMO[lumo + i]- eMO[mo])

    return k

def getDipoles(atom, basis):
    #compute the dipole moments

    from post import dipoleComponent

    cartesians = ('x', 'y', 'z')
    nbf = len(basis)
    mu = np.zeros((3, nbf, nbf))

    for i in range(mu.shape[0]):
        mu[i] = dipoleComponent(atom, basis, cartesians[i], 'origin')

    return mu

def getTransitionProperties(mu, c, i, a):
    #compute the transition dipoles

    tmu = np.zeros(3)
    for x in range(3):
        tmu[x] = np.einsum('j,k,jk', c[:, i], c[:, a], mu[x])

    oscillatorStrength = (2/3) * abs(eMO[a] - eMO[i])
    oscillatorStrength *= np.sum(abs(tmu)**2)

    return tmu, oscillatorStrength

def lorentzian(e0, e, tau):
#Lorentzian broadening

   gamma = 1.0/tau
   g = (gamma/2.0)**2.0/((e0-e)**2.0 + (gamma/2.0)**2.0)

   return g

e, atom, basis, data = getMolecule('6-31g')
eMO = getOrbitalEnergies(rhf.fock, rhf.S)

#get n k-edge excitations
n = 5
kedge = getKedgeExcitations(('1s', 'O1'), n, rhf.C, basis, atom, eMO)

#lumo
lumo = electronCount(atom, data['charge'])//2

#1s of oxygen O1
mo = getMOwithCharacter(('1s','O1'), rhf.C, basis, atom)

#get transition dipoles and oscillator strengths
transitionDipoles = []
oscillatorStrength = []
for i in range(n):
    a = i + lumo
    mu = getDipoles(atom, basis)
    tmu, osc = getTransitionProperties(mu, rhf.C, mo , a)
    transitionDipoles.append(tmu)
    oscillatorStrength.append(osc)

#normalize oscillator strength
oscillatorStrength /= max(oscillatorStrength)

#plot broadened spectra
import matplotlib.pyplot as py
from atom import getConstant

py.title('$H_2O$ k-edge spectrum')
py.grid()   
points = 0
margin, tau, npoints = [0.5, 40, 100]

for i in range(n):
    start = (kedge[i]) - margin
    finish = (kedge[i]) + margin
    x = np.linspace(start, finish, npoints)

    points = lorentzian(kedge[i], x, tau) * oscillatorStrength[i]
    py.plot(x * getConstant('hartree->eV'), points,'k')
    py.bar(kedge[i] * getConstant('hartree->eV'), oscillatorStrength[i], color='orange')

    py.text(kedge[i] * getConstant('hartree->eV') - margin, 1.0, str(i + lumo), fontsize='x-small')

xl, xr = py.xlim()
py.text(xl + 2, 1,'0->', fontsize='x-small')
py.xlabel('excitation energy (eV)')
py.ylabel('oscillator strength (norm)')  

py.show()

print('Largest oscillator strength from transition oxygen 1s ->', getOrbitalCharacter(rhf.C[:,9], basis, atom)[0])