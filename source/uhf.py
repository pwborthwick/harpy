from __future__ import division
#import system modules
import numpy as np
from scipy.linalg import fractional_matrix_power as fractPow
from numpy.linalg import eigh, solve, LinAlgError
import sys
#import class modules
from atom import atom, nuclearRepulsion, getConstant
from basis import checkBasis, buildBasis, electronCount
#import user modules
from post import buildDipole
from view import geometry, orbitals, pre, preSCF, SCF, postSCF, post, uhfOrbitals
from integral import buildOverlap, buildKinetic, buildCoulomb, buildEri, \
                     buildHamiltonian, iEri, expandEri
from diis import diis_f

def getOccupancy(e, orbitals, nBasis):
    #make occupancy lists

    occupancy = np.zeros((2,nBasis))

    for spin in range(2):
        idx = np.argsort(e[spin])

        occupancy[spin][idx[:orbitals[spin]]] = 1

    return occupancy

def buildUnrestrictedFock(H, eri, cycle, density):
    #the unrestricted spin Fock matrices

    nBasis = H.shape[0]
    n = nBasis
    fock = np.array([H, H])

    #initial guess for each spin is core Hamiltonian
    if cycle == 0:
        return fock

    fock[0] += np.einsum('rs,pqrs->pq', density[0], eri, optimize=True) - np.einsum('rs,psrq->pq', density[0], eri, optimize=True)
    fock[0] += np.einsum('rs,pqrs->pq', density[1], eri, optimize=True)
    fock[1] += np.einsum('rs,pqrs->pq', density[1], eri, optimize=True) - np.einsum('rs,psrq->pq', density[1], eri, optimize=True)
    fock[1] += np.einsum('rs,pqrs->pq', density[0], eri, optimize=True)

    return fock

def buildUnrestrictedDensity(c, orbitals):
    #alpha and beta spin density matrices

    nBasis = c[0].shape[0]
    density = np.zeros((2, nBasis, nBasis)) 

    for spin in range(2):
        density[spin] = np.einsum('pr,qr->pq', c[spin][:, :orbitals[spin]], c[spin][:, :orbitals[spin]], optimize=True)
    return  density

def rmsDensity(density, preDensity):
    #compute RMS of density matrix

    return  np.sqrt(np.mean(np.square(density - preDensity)))


def rebuildCenters(molAtom, molBasis, geo):
    #a change of geometry means atom.center and  basis.center need updating
    for atom in range(0, len(molAtom)):
        molAtom[atom].center = geo[atom, :]

        for basis in range(0, len(molBasis)):
            if molBasis[basis].atom == atom:
                molBasis[basis].center = geo[atom, :]

    return molAtom, molBasis

def scf(molAtom, molBasis, run, show):

    global density_total, c, occupancy

    #options
    name, basisName, diisStatus, integralEngine, gaugeOrigin, hamiltonianGuess, charge, multiplicity, \
          iterations, diisCapacity, convergence, uhfmix, p, _, _, _, _  = run.values()

    #pre-SCF matrices - S{overlap}  T{1e kinetic/exchange/resonance}  V{1e Coulomb} ERI(2e electron repulsion)
    if integralEngine == 'native':
        S = buildOverlap(molBasis)
        T = buildKinetic(molBasis)
        V = buildCoulomb(molAtom, molBasis)
        ERI = buildEri(molBasis)
    elif integralEngine == 'aello':
        from aello import aello
        S, T, V, ERI = aello(molAtom, molBasis)

    coreH, initialFock = buildHamiltonian(hamiltonianGuess,S,T,V)

    nBasis = S.shape[0]
    ERI = expandEri(ERI, nBasis)

    density = np.zeros((2, nBasis, nBasis))

    #orthogonalise matrix X
    X = np.zeros((nBasis, nBasis))
    X = fractPow(S, -0.5)

    #check multiplicity valid for electron count
    electrons = electronCount(molAtom, charge)

    unpaired = (multiplicity - 1)
    paired = electrons - unpaired

    if ((paired + unpaired) != electrons):
        print('multiplicity not compatible')
        post(False)
        sys.exit('multiplicity not compatible')

    #alpha and beta spin orbital counts
    orbitals = [(electrons - unpaired)//2 + unpaired, (electrons - unpaired)//2]
    if 'orbitals' in show:
        uhfOrbitals(orbitals[0], orbitals[1], multiplicity)

    #define storage for diis
    if diisStatus == 'on':
        diis = diis_f(diisCapacity)

    #SCF loop
    for cycle in range(iterations):

        #build fock matrices and orthogonalise
        if cycle == 0:
            fock = buildUnrestrictedFock(initialFock, ERI, cycle , density)
        else:
            fock = buildUnrestrictedFock(coreH, ERI, cycle , density)

        #do diis if selected
        if (cycle != 0) and (diisStatus == 'on'):
            fock = diis.build(fock, density, S, X)

        #compute SCF energy
        energy = 0.0
        for i in range(0, len(molBasis)):
            for j in range(0, len(molBasis)):
                energy += 0.5 * density[0][i,j] * (fock[0][i,j] + coreH[i,j])
        for i in range(0, len(molBasis)):
            for j in range(0, len(molBasis)):
                energy += 0.5 * density[1][i,j] * (fock[1][i,j] + coreH[i,j])

        #orthogonalise fock matrix
        e = np.zeros((2, nBasis))
        c = np.zeros_like(density)
        for spin in range(2):
            orthogonal_fock = np.dot(X.T, np.dot(fock[spin], X))
            e[spin] , orthogonal_c  = eigh(orthogonal_fock)
            c[spin] = np.dot(X, orthogonal_c)

        #break symmetry
        if (orbitals[0] == orbitals[1]) and (cycle == 0):
            #mix the AOs (rows)
            cHOMO = c[0][orbitals[0]-1,:]
            cLUMO = c[0][orbitals[0],:]
            c[0][orbitals[0]-1,:] = 1.0/((1 + uhfmix**2))**0.5*(cHOMO + uhfmix*cLUMO)
            c[0][orbitals[0],:]   = 1.0/((1 + uhfmix**2))**0.5*(-uhfmix*cHOMO + cLUMO)

        #calculate densities over occupied MO
        occupancy = getOccupancy(e, orbitals, nBasis)
        density = buildUnrestrictedDensity(c, orbitals)

        #convergence control
        if cycle != 0:
            deltaEnergy = abs(preEnergy - energy)
            if 'SCF' in show:
                SCF(energy, deltaEnergy, (rmsDensity(density[0], preDensity[0]) + rmsDensity(density[1], preDensity[1]))*0.5 ,\
                    cycle, diisStatus, iterations, convergence)

            if (deltaEnergy < convergence) and (rmsDensity(density[0], preDensity[0]) < convergence) \
                                           and (rmsDensity(density[1], preDensity[1]) < convergence) :
                break

        preEnergy = energy
        preDensity = density

    #if failed to converge exit with messages
    if cycle == iterations:
        print('SCF failed to converge in ' + str(cycle) + ' iterations')
        if diisStatus == 'off': print('Try diis = \'on\'')

        post(False)
        sys.exit('convergence failure')

    #final eigensolution of final Fock matrix are orbital energies (e) and MO coefficients(C)
    totalEnergy = energy + nuclearRepulsion(molAtom)

    #spin analysis
    spinMatrix = np.dot(c[1][:, :orbitals[1]].T, np.dot(S, c[0][:, :orbitals[0]]))
    spinContamination = orbitals[1] - np.vdot(spinMatrix, spinMatrix)
    density_total = density[0] + density[1]
    spin_density = density[0] - density[1]

    def spinSquare(c, occupancy, S):
        #compute <S^2>

        #get occupied molecular eigenvectors
        cOccupied = [c[0]*occupancy[0], c[1]*occupancy[1]]
        moOccupied = [cOccupied[0][:,~np.all(cOccupied[0]==0, axis=0)],  cOccupied[1][:,~np.all(cOccupied[1]==0, axis=0)]]
        nOccupied = [moOccupied[0].shape[1], moOccupied[1].shape[1]]

        #components of total spin
        spin = np.dot(moOccupied[0].T, np.dot(S, moOccupied[1]))
        spin_xy = (nOccupied[0] + nOccupied[1]) * 0.5 - np.einsum('ij,ij->', spin, spin)
        spin_z = (nOccupied[0] - nOccupied[1])**2 * 0.25
        spin_total = (spin_xy + spin_z)

        multiplicity = np.sqrt(spin_total + 0.25) - 0.5

        return spin_total, multiplicity*2 + 1

    total_spin, multiplicity = spinSquare(c, occupancy, S)

    if 'SCF' in show:
        postSCF([totalEnergy, cycle, energy],'uhf')

    if 'postSCF' in show:
        postSCF([e[0], e[1], density_total, spin_density, total_spin, spinContamination, multiplicity], 'uhf-post')

        if 'ch' in p:
            population = []
            population.append(np.einsum('ij,ji->i', density[0], S, optimize=True))
            population.append(np.einsum('ij,ji->i', density[1], S, optimize=True))

            mulliken = []
            atomCharge = np.zeros((2,len(molAtom)))
            atomSpinCharge = np.zeros(atomCharge.shape[1])
            totalCharge = np.zeros(2)

            for spin in range(2):
                for i in range(population[spin].shape[0]):
                    mulliken.append([molBasis[i].symbol, molAtom[molBasis[i].atom].id, round(population[spin][i],6)])
                    atomCharge[spin][molBasis[i].atom] += round(population[spin][i],6)
                    totalCharge[spin] += population[spin][i]

            spinPopulation = np.einsum('ij,ji->i', spin_density, S)
            for i in range(spinPopulation.shape[0]):
                atomSpinCharge[molBasis[i].atom] += spinPopulation[i]

            atoms = []
            for atom in molAtom:
                atoms.append([atom.id, atom.number])

            postSCF([nBasis, mulliken, totalCharge, atomCharge, atoms, spinPopulation, atomSpinCharge], 'uhf-mull')

        if 'di' in p: 
            import aello
            dipoles = buildDipole(molAtom, molBasis, density_total*0.5, gaugeOrigin)

    #clean up outfile
    post()

    return totalEnergy