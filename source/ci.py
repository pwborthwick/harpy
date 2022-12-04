from __future__ import division
from integral import buildFockMOspin, buildEriMO, buildEriDoubleBar
from numpy import zeros, sort, real, dot, sqrt, eye, append, block
from basis import electronCount
from numpy.linalg import eigh, eigvals, eig, qr, norm
from scipy.linalg import sqrtm
from view import postSCF
from integral import iEri

def cis(atoms, charge, bases, eigenVectors, fock, ERI):
    #compute Configuration Interaction Singles

    spinOrbitals = len(bases) * 2
    #get fock matrix in MO spin basis
    fockMOspin = buildFockMOspin(spinOrbitals, eigenVectors, fock)
    
    #get two-electron repulsion integrals in MO basis
    eriMO = buildEriMO(eigenVectors, ERI)

    #transform eri from MO to spin basis
    eriMOspin = buildEriDoubleBar(spinOrbitals, eriMO)

    #orbital occupation
    nElectrons = electronCount(atoms, charge)
    nOccupied  = int(nElectrons/2)
    nVirtual   = len(bases) - nOccupied

    #build singly excited determinant basis Hamiltonian
    ciHamiltonian = zeros( (nOccupied*nVirtual*4,nOccupied*nVirtual*4) )

    ia = 0
    for i in range(0, nElectrons):
        for a in range(nElectrons, spinOrbitals):

            jb = 0
            for j in range(0, nElectrons):
                for b in range(nElectrons, spinOrbitals):

                    if (i == j):
                        ciHamiltonian[ia,jb] += fockMOspin[a,b]
                    if (a == b):
                        ciHamiltonian[ia,jb] -= fockMOspin[i,j]

                    ciHamiltonian[ia,jb] += eriMOspin[a,j,i,b]

                    jb += 1
            ia += 1

    #diagonalise Hamiltonian
    e, ccis = eigh(ciHamiltonian)

    #excitations
    jumps = excitations(e, ccis, nElectrons, nOccupied, nVirtual)

    #detect degeneracy
    postSCF(ciDegeneracy(e), 'ci')

    #Davidson
    n = min(5, len(ciHamiltonian)//2)
    postSCF(blockDavidson(n, ciHamiltonian),'bd')

    #output jumps
    postSCF(jumps, 'ju')

    return e, ccis

def ciDegeneracy(e):
    #from sorted [low->high] produce [value, degeneracy] tuples

    count = 0
    ciEnergy = []
    energy = e[count]
    degeneracy = 0

    #search forward for equal energies
    while True:

        #loop until new energy value
        while True:
            degeneracy += 1
            count += 1
            if count == len(e):                 #got to end of list
                break
            if abs(energy - e[count]) > 1e-12:  #same to within tolerence
                break

        if degeneracy == 3:
            ciEnergy.append([energy, 't'])
        elif degeneracy == 2:
            ciEnergy.append([energy, 'd'])
        elif degeneracy == 1:
            ciEnergy.append([energy, 's'])
        else:
            ciEnergy.append([energy, '(' + str(degeneracy) + ')'])

        if count >= len(e):
            break

        energy = e[count]
        degeneracy = 0

    return ciEnergy

def ciSpinAdaptedSingles(atoms, charge, bases, eigenVectors, fock, ERI):
    #configuration interaction spin-adapted singles

    spinOrbitals = len(bases) * 2
    #get fock matrix in MO basis
    fockMO = dot(eigenVectors.T , dot(fock, eigenVectors))
    
    #get two-electron repulsion integrals in MO basis
    eriMO = buildEriMO(eigenVectors, ERI)

    #orbital occupation
    nElectrons = electronCount(atoms, charge)
    nOccupied  = int(nElectrons/2)
    nVirtual   = len(bases) - nOccupied
    n   = len(bases)

    #build singly excited determinant basis Hamiltonian
    ciHamiltonian = zeros( (nOccupied*nVirtual, nOccupied*nVirtual) )

    ia = 0
    for i in range(0, nOccupied):
        for a in range(nOccupied, n):

            jb = 0
            for j in range(0, nOccupied):
                for b in range(nOccupied, n):

                    sum = 0.0
                    if (i == j) and (a == b):
                        sum  += fockMO[a,a] - fockMO[i,i]

                    sum += 2.0 * eriMO[iEri(a,i,j,b)] - eriMO[iEri(a,b,j,i)]
                    ciHamiltonian[ia,jb] = sum

                    jb += 1
            ia += 1

    #diagonalise Hamiltonian
    e = eigvals(ciHamiltonian)

    #order eigenvalues
    e.sort()

    postSCF(e, 'cisas')

    return e

def ciSpinAdaptedTriples(atoms, charge, bases, eigenVectors, fock, ERI):
    #configuration interaction spin-adapted singles

    spinOrbitals = len(bases) * 2
    #get fock matrix in MO basis
    fockMO = dot(eigenVectors.T , dot(fock, eigenVectors))
    
    #get two-electron repulsion integrals in MO basis
    eriMO = buildEriMO(eigenVectors, ERI)

    #orbital occupation
    nElectrons = electronCount(atoms, charge)
    nOccupied  = int(nElectrons/2)
    nVirtual   = len(bases) - nOccupied
    n   = len(bases)

    #build singly excited determinant basis Hamiltonian
    ciHamiltonian = zeros( (nOccupied*nVirtual, nOccupied*nVirtual) )

    ia = 0
    for i in range(0, nOccupied):
        for a in range(nOccupied, n):

            jb = 0
            for j in range(0, nOccupied):
                for b in range(nOccupied, n):

                    sum = 0.0
                    if (i == j) and (a == b):
                        sum  += fockMO[a,a] - fockMO[i,i]

                    sum -= eriMO[iEri(a,b,j,i)]
                    ciHamiltonian[ia,jb] = sum

                    jb += 1
            ia += 1
    
    #diagonalise Hamiltonian
    e = eigvals(ciHamiltonian)

    #order eigenvalues
    e.sort()

    postSCF(e, 'cisat')

    return e

def randomPhaseApproximation(atoms, charge, bases, eigenVectors, fock, ERI, type = 'tamm-dancoff'):
    #time-dependent Hartree-Fock (TDHF) random phase approximation (RPA)

    spinOrbitals = len(bases) * 2
    
    #get fock matrix in MO spin basis
    fockMOspin = buildFockMOspin(spinOrbitals, eigenVectors, fock)
    
    #get two-electron repulsion integrals in MO basis
    eriMO = buildEriMO(eigenVectors, ERI)

    #transform eri from MO to spin basis
    eriMOspin = buildEriDoubleBar(spinOrbitals, eriMO)

    #orbital occupation
    nElectrons = electronCount(atoms, charge)
    nOccupied  = int(nElectrons/2)
    nVirtual   = len(bases) - nOccupied

    A = zeros((nOccupied*nVirtual*4, nOccupied*nVirtual*4))
    B = zeros((nOccupied*nVirtual*4, nOccupied*nVirtual*4)) 

    ia = 0
    for i in range(0, nElectrons):
        for a in range(nElectrons, spinOrbitals):

            jb = 0
            for j in range(0, nElectrons):
                for b in range(nElectrons, spinOrbitals):

                    if (i == j):
                        A[ia,jb] += fockMOspin[a,b]
                    if (a == b):
                        A[ia,jb] -= fockMOspin[i,j]

                    A[ia,jb] += eriMOspin[b,i,j,a]
                    B[ia,jb] += eriMOspin[b,a,j,i]

                    jb += 1
            ia += 1

    if type == 'linear':
        HA = A + B
        HB = A - B
        ciHamiltonian = dot(HA, HB)
    elif type == 'block':
        ciHamiltonian = block([ [A,   B],
                                [-B, -A]  ])
    elif type == 'hermitian':
        sqm = sqrtm(A-B)
        ciHamiltonian = dot(sqm, dot(A+B, sqm))
        
    elif type == 'tamm-dancoff':
        ciHamiltonian = A
        e, v = eigh(ciHamiltonian)

        #sort eigen solutions
        idx = e.argsort()
        e = e[idx].real
        v = v[:,idx].real

        return e, v

    elif type == 'raw':
        return A, B

    #diagonalise Hamiltonian
    e = eigvals(ciHamiltonian)

    #order eigenvalues
    e.sort()

    #'hermitian' and 'linear' gives square of eigenvalues
    if type in 'hermitian, linear' : 
        e = sqrt(real(e))
    else:
        e = e[nOccupied*nVirtual*4:].real

    #detect degeneracy
    postSCF(ciDegeneracy(e), 'rpa')
    
    return e


def excitations(ecis, ccis, nElectrons, nOccupied, nVirtual):
    #calculate the jumps and contributions

    levels = []
    for i in range(0, nOccupied*2):
        for j in range(nOccupied*2, (nOccupied+nVirtual)*2):
            levels.append(str(i) + ' -> ' + str(j))

    contributions = []
    #loop over energies and find contributions > 10%
    for i in range(0, ccis.shape[1]):
        #loop down levels
        for j in range(0, ccis.shape[0]):
            percent = round(ccis[j,i] * ccis[j,i] * 100)
            if  percent > 10:
                contributions.append([i , round(ecis[i],6), percent, levels[j]])

    return contributions

def blockDavidson(nLowestEigen, h):
    #simplistic implementation of block Davidson algorithm for finding n lowest eigenvalues, real symmetric

    tolerence =  1e-8
    n = h.shape[0]
    iterations = n

    #sub-space can't be greater than dimension of h
    if nLowestEigen > n//2: nLowestEigen = n//2

    #initial sub-space guess
    nInitialGuessVectors = 2 * nLowestEigen
    t = eye(n, nInitialGuessVectors)

    V = zeros((n, 0))
    V = append(V, t, axis=1)

    I = eye(n)

    #begin algorithm
    m = nInitialGuessVectors
    cycle = 0
    tau = 0
    preTheta = 1

    while True:
        if m > nInitialGuessVectors:
            preTheta = theta[:nLowestEigen]

        T = dot(V[:,:m].T,dot(h,V[:,:m]))
        rho,S = eig(T)
        idx = rho.argsort()
        theta = rho[idx]
        s = S[:,idx]
        for j in range(0,nInitialGuessVectors):
            w = dot((h - theta[j]*I),dot(V[:,:m],s[:,j])) 
            #refresh the sub-space block 
            t[:, j] = real(w/(theta[j]-h[j,j] + 1e-14))    
        #append new block
        V = append(V , t, axis=1)
        #orthogonalise
        V, a = qr(V)

        #check convergence
        delta = norm(theta[:nLowestEigen] - preTheta)
        if delta < tolerence:
            break

        m += nInitialGuessVectors
        cycle += 1
        if cycle == iterations:
            break

    if (cycle < iterations):
        return real(theta[:nLowestEigen])
    else:
        return []

import numpy as np
class cis_d(object):
    #class for CIS(D)

    def __init__(self, hf, roots=5, solver='eigh', method='cis(d)'):

        self.hf = hf
        self.roots = roots
        self.solver = solver
        self.method = method

        self.fs, self.gs, self.eps = self.get_spin_quantities()
        self.get_spin_metrics()

        self.cache = {}
        if self.method == 'cis-mp2':
            self.cache['cis-mp2'] = self.get_perturbative_doubles_mp2()
        else:
            self.cache['cisd']    = self.get_perturbative_doubles_cis()

    def get_spin_quantities(self):
        #return the eri, Fock and orbital energies in spin MO basis

        from cc.fcc import spinMO

        spin = spinMO(self.hf.rhf.e, self.hf.rhf.ERI, self.hf.rhf.C, self.hf.rhf.fock)

        return spin.fs, spin.gs, np.kron(self.hf.rhf.e, np.ones(2))

    def get_spin_metrics(self):
        #get the dimensions of spin quantities

        from basis import electronCount
        charge = self.hf.data['charge']
        nocc = electronCount(self.hf.atoms, charge)
        nvir = 2*len(self.hf.basis) - nocc

        self.occupations = (nocc, nvir, nocc*nvir)

        #get the orbital slices
        n = np.newaxis
        o = slice(None, nocc)
        v = slice(nocc, None)

        self.slice = (n, o, v)

    def check_scf_energy(self, e_scf):
        #check that Harpy energy equals spin quantities version

        from atom import nuclearRepulsion
        from math import isclose

        n, o, v = self.slice
        hf_energy = np.einsum('ii', self.fs[o, o]) - 0.5 * np.einsum('ijij', self.gs[o, o, o, o]) + nuclearRepulsion(self.hf.atoms)

        return isclose(hf_energy, e_scf, rel_tol=1e-8)

    def get_mp2_energy(self):
        #compute the MP2 energy correction

        n, o, v = self.slice 

        #orbital energy denominator
        dd = 1.0 / (self.eps[v, n, n, n] + self.eps[n, v, n, n] 
                  - self.eps[n, n, o, n] - self.eps[n, n, n, o] )

        #mp2 energy
        e_mp2 = -0.25 * np.einsum('ijab,abij->', self.gs[o,o,v,v]**2, dd, optimize=True)

        self.cache['mp2'] = e_mp2

        return e_mp2

    def cis_solve_direct(self):
        #solve the CIS eigenvalue problem with eigh

        nocc, nvir, nrot = self.occupations
        n, o, v = self.slice

        a  = np.einsum('ab,ij->iajb',np.diag(np.diag(self.fs)[v]),np.diag(np.ones(nocc)), optimize=True) 
        a -= np.einsum('ij,ab->iajb',np.diag(np.diag(self.fs)[o]),np.diag(np.ones(nvir)), optimize=True) 
        a += np.einsum('ajib->iajb', self.gs[v,o,o,v], optimize=True) 

        #reshape for solving
        a = a.reshape(nrot, nrot)

        #direct solve
        try:
            e_cis, u = np.linalg.eigh(a)
            converged = True
        except np.linalg.LinAlgError as e:
            print('matrix solve error ', e)
            converged = False

        self.cache['cis'] = e_cis
        self.cache['u']   = u

        return e_cis, u, converged

    def cis_solve_davidson(self):
        #davidson solve for cis

        def cis_diagonal(self):
            #diagonal pre-conditioner for cis Davidson

            n, o, v = self.slice ; nocc, nvir, nrot = self.occupations

            ds = self.eps[o, n] - self.eps[n, v]

            #initialize to fock diagonal
            diagonal = -ds.ravel()

            cis_diagonal = diagonal.reshape(nocc, nvir)
            cis_diagonal -= np.einsum('aiai->ia', self.gs[v, o, v, o], optimize=True)

            return diagonal

        def cis_initial_guess(self, diagonal, f=1):
            #guess vector for cis Davidson

            #get largest absolute values on diagonal matrix as best guess
            args = np.argsort(np.absolute(diagonal))

            #we only have nocc*nvir roots available
            if self.roots > len(args):
                print('reducing requested roots - exceeded ', len(args))
                self.roots = len(args)

            guess_vectors = np.zeros((diagonal.size, self.roots * f))
            for root in range(self.roots * f):
                guess_vectors[args[root], root] = 1.0

            return guess_vectors

        def cis_matvec(cis):

            #construct the cis matrix dot product with arbitary vector (r)

            n, o, v = self.slice ; nocc, nvir, nrot = self.occupations

            ds = self.eps[o, n] - self.eps[n, v]

            cis = np.array(cis)
            r   = np.zeros_like(cis)

            cis_s = cis.reshape(nocc, nvir)
            r_s   = r.reshape(nocc, nvir)

            #sconfiguration interaction singles
            r_s -= np.einsum('ia,ia->ia', ds, cis_s, optimize=True)
            r_s -= np.einsum('ajbi,jb->ia', self.gs[v, o, v, o], cis_s, optimize=True)

            return r

        guess_vector_factor, tol, vectors_per_root = 1, 1e-8, 20

        #get diagonal preconditioner
        diagonal = cis_diagonal(self)

        #generate initial guess from diagonal
        guess_vectors = cis_initial_guess(self, diagonal, guess_vector_factor)


        from adc.adc import davidson
        e_cis, u, converged = davidson(cis_matvec, guess_vectors, diagonal, tol=tol,
                                            vectors_per_root = vectors_per_root )

        self.cache['cis'] = e_cis
        self.cache['u']   = u

        return e_cis, u, converged

    def get_perturbative_doubles_cis(self):
        #get the cis(d) corrected roots

        nocc, nvir, _ = self.occupations
        n, o, v = self.slice

        #orbital energy denominator
        dd = 1.0 / (self.eps[v, n, n, n] + self.eps[n, v, n, n] 
                  - self.eps[n, n, o, n] - self.eps[n, n, n, o] )

        #get CIS solution
        if self.solver == 'eigh':
            e_cis, u_cis, converged = self.cis_solve_direct()
        if self.solver == 'davidson':
            e_cis, u_cis, converged = self.cis_solve_davidson()

        #not converged exit
        if not converged: exit('cis failed to converge')

        cisd_correction = []

        print('***CIS(D)***\n root      CIS          CIS(D)         \u0394 \n---------------------------------------------')
        for root in range(self.roots):

            #get amplitude vector and clean 
            b = u_cis[:, root]
            
            b[abs(b)<1e-14] = 0

            #re-shape amplitude vector
            b = b.reshape(nocc, nvir).T

            #a tensor
            a = -self.gs[v,v,o,o]*dd

            #u tensor
            u =  np.einsum('icab,cj->abij', self.gs[o,v,v,v], b, optimize=True)
            u -= np.einsum('jcab,ci->abij', self.gs[o,v,v,v], b, optimize=True)
            u += np.einsum('ijka,bk->abij', self.gs[o,o,o,v], b, optimize=True)
            u -= np.einsum('ijkb,ak->abij', self.gs[o,o,o,v], b, optimize=True)

            #v tensor
            V =  0.5 * np.einsum('jkbc,bi,cajk->ai', self.gs[o,o,v,v], b, a, optimize=True)
            V += 0.5 * np.einsum('jkbc,aj,cbik->ai', self.gs[o,o,v,v], b, a, optimize=True)
            V += np.einsum('jkbc,bj,acik->ai', self.gs[o,o,v,v], b, a, optimize=True)

            #double excitations - correlation term electrons not involved in excitation
            e_cisd  = np.einsum('ai,ai->', b, V, optimize=True)

            #shifted denominator
            d_abij_omega = 1.0 / (self.eps[v, n, n, n] + self.eps[n, v, n, n] 
                                - self.eps[n, n, o, n] - self.eps[n, n, n, o] - e_cis[root])

            e_cisd_direct = -0.25 * np.einsum('abij,abij->', u*u, d_abij_omega, optimize=True)

            e_mp2 = self.get_mp2_energy()
            print('  {:>2d}   {:>10.6f}    {:>10.6f} ({:>10.6f} )'.
                            format(root+1, e_cis[root], e_cisd_direct + e_cisd + e_mp2 + e_cis[root],
                                e_cisd_direct + e_cisd + e_mp2 ))

            cisd_correction.append(e_cisd_direct + e_cisd + e_mp2)

        return cisd_correction

    def get_perturbative_doubles_mp2(self):
        #get the cis-mp2 corrected roots

        nocc, nvir, _ = self.occupations
        n, o, v = self.slice

        #orbital energy denominator
        dd = 1.0 / (self.eps[v, n, n, n] + self.eps[n, v, n, n] 
                  - self.eps[n, n, o, n] - self.eps[n, n, n, o] )
        dt = 1.0 / (self.eps[v, n, n, n, n, n] + self.eps[n, v, n, n, n, n]
                  + self.eps[n, n, v, n, n, n] - self.eps[n, n, n, o, n, n]
                  - self.eps[n, n, n, n, o, n] - self.eps[n, n, n, n, n, o])

        #get CIS solution
        if self.solver == 'eigh':
            e_cis, u_cis, converged = self.cis_solve_direct()
        if self.solver == 'davidson':
            e_cis, u_cis, converged = self.cis_solve_davidson()

        #not converged exit
        if not converged: exit('cis failed to converge')

        cis_mp2_correction = []

        print('***CIS-MP2***\n root      CIS          CIS-MP2        \u0394 \n---------------------------------------------')
        for root in range(self.roots):

            #get amplitude vector and clean 
            b = u_cis[:, root]
            
            b[abs(b)<1e-14] = 0

            #re-shape amplitude vector
            b = b.reshape(nocc, nvir).T

            #shifted denominators
            d_abij_omega =    1.0 / (self.eps[v, n, n, n] + self.eps[n, v, n, n] 
                                   - self.eps[n, n, o, n] - self.eps[n, n, n, o] - e_cis[root])

            d_abcijk_omega =  1.0 / (self.eps[v, n, n, n, n, n] + self.eps[n, v, n, n, n, n]
                                   + self.eps[n, n, v, n, n, n] - self.eps[n, n, n, o, n, n]
                                   - self.eps[n, n, n, n, o, n] - self.eps[n, n, n, n, n, o] - e_cis[root])

            #u tensor
            u =  np.einsum('icab,cj->abij', self.gs[o,v,v,v], b, optimize=True)
            u -= np.einsum('jcab,ci->abij', self.gs[o,v,v,v], b, optimize=True)
            u += np.einsum('ijka,bk->abij', self.gs[o,o,o,v], b, optimize=True)
            u -= np.einsum('ijkb,ak->abij', self.gs[o,o,o,v], b, optimize=True)
            e_cis_mp2 = -0.25 * np.einsum('abij,abij->', u*u, d_abij_omega, optimize=True)

            #u tensor
            u =  np.einsum('jkbc,ai->abcijk', self.gs[o,o,v,v], b, optimize=True)
            u += np.einsum('jkca,bi->abcijk', self.gs[o,o,v,v], b, optimize=True)
            u += np.einsum('jkab,ci->abcijk', self.gs[o,o,v,v], b, optimize=True)
            u += np.einsum('kibc,aj->abcijk', self.gs[o,o,v,v], b, optimize=True)
            u += np.einsum('kica,bj->abcijk', self.gs[o,o,v,v], b, optimize=True)
            u += np.einsum('kiab,cj->abcijk', self.gs[o,o,v,v], b, optimize=True)
            u += np.einsum('ijbc,ak->abcijk', self.gs[o,o,v,v], b, optimize=True)
            u += np.einsum('ijca,bk->abcijk', self.gs[o,o,v,v], b, optimize=True)
            u += np.einsum('ijab,ck->abcijk', self.gs[o,o,v,v], b, optimize=True)
            e_cis_mp2 += -(1/36) * np.einsum('abcijk,abcijk->', u*u, d_abcijk_omega, optimize=True)

            print('  {:>2d}   {:>10.6f}    {:>10.6f} ({:>10.6f} )'.
                            format(root+1, e_cis[root], e_cis_mp2 + e_cis[root],
                                e_cis_mp2 ))

            cis_mp2_correction.append(e_cis_mp2)

        return cis_mp2_correction

