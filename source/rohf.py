from __future__ import division
import numpy as np
import scipy

from integral import buildOverlap, buildKinetic, buildCoulomb, buildEri, \
                     buildHamiltonian, iEri, expandEri
from basis import electronCount
from atom import nuclearRepulsion
from view import geometry, orbitals, pre, preSCF, SCF, postSCF, post, uhfOrbitals
from post import buildDipole

from diis import diis_f

def rmsDensity(density, preDensity):
    #compute RMS of density matrix

    return  np.sqrt(np.mean(np.square(density - preDensity)))

def get_occupations(eps_list, atoms, charge, multiplicity):
    #compute the orbital occupancies from orbital energies

    if len(eps_list) == 1:
        eps = eps_list[0]
        eps_alpha = eps_beta = eps
    else:
        eps, eps_alpha, eps_beta = eps_list

    nmo = len(eps)

    mo_occupied = np.zeros(nmo)
    ne = electronCount(atoms, charge)

    #spin (unpaired electrons)
    spin = multiplicity - 1
    electrons = ((ne-spin)//2 + spin, (ne-spin)//2)

    #fill occupancies
    mo_occupancy = np.zeros((nmo))
    eps_sorted = np.argsort(eps)
    core_indices = eps_sorted[:electrons[1]]

    if spin != 0:
        open_indices = eps_sorted[electrons[1]:]
        open_indices = open_indices[np.argsort(eps_alpha[open_indices])[:spin]]
        mo_occupancy[open_indices] = 1

    mo_occupancy[core_indices] = 2 

    return mo_occupancy

def get_densities(c, occupations):
    #get the one-particle density matrices for closed and open

    alpha = occupations > 0
    beta  = occupations == 2

    dm_alpha = np.einsum('pr,r,qr', c, alpha, c, optimize=True)
    dm_beta  = np.einsum('pr,r,qr', c, beta, c, optimize=True)

    return np.array([dm_alpha, dm_beta])


def get_coulomb_exchange(density, eri):
    #form the coulomb and exchange HF integrals

    j = np.zeros_like(density) ; k = np.zeros_like(density)
    for spin in range(density.shape[0]):
           j[spin] = np.einsum('ijkl,ji->kl', eri, density[spin], optimize=True)
           k[spin] = np.einsum('ikjl,ji->kl', eri, density[spin], optimize=True)

    return j, k

def make_fock(h, v, dm, s):
    #construct the rohf Roothaan effective fock matrix components

    #uhf fock operators
    f_alpha = h + v[0] ; f_beta = h + v[1]

    #Roothaan effective Fock

    nbf = s.shape[0]

    f_core = (f_alpha + f_beta) * 0.5

    #project uhf focks - alpha all non-zero , beta - all doubly occupied
    core     = np.einsum('pr,rq->pq', dm[1], s, optimize=True)
    open     = np.einsum('pr,rq->pq', dm[0] - dm[1], s, optimize=True)
    virtual  = np.eye(nbf) - np.einsum('pr,rq->pq', dm[0], s, optimize=True)

    #make fock 
    fock  = 0.5 * np.einsum('sp,sr,rq->pq', core, f_core, core, optimize=True)
    fock += 0.5 * np.einsum('sp,sr,rq->pq', open, f_core, open, optimize=True)
    fock += 0.5 * np.einsum('sp,sr,rq->pq', virtual, f_core, virtual, optimize=True)

    fock += np.einsum('sp,sr,rq->pq', open, f_beta, core, optimize=True)
    fock += np.einsum('sp,sr,rq->pq', open, f_alpha, virtual, optimize=True)
    fock += np.einsum('sp,sr,rq->pq', virtual, f_core, core, optimize=True)

    roothaan_fock = fock + fock.transpose(1,0)

    return f_alpha, f_beta, roothaan_fock

def eigensolution(fock, x):
    #solve the fock, S system

    fock_prime = np.einsum('rp,rs,sq->pq', x, fock[2], x, optimize=True)
    e , c  =np.linalg.eigh(fock_prime)
    c =  np.einsum('pr,rq->pq',x, c, optimize=True)

    e_alpha = np.einsum('pi,ps,si->i', c, fock[0], c, optimize=True)
    e_beta =  np.einsum('pi,ps,si->i', c, fock[1], c, optimize=True)

    return e, e_alpha, e_beta, c

def get_rohf_energy(density, h1e, v):
    #compute the rohf energy

    e =  np.einsum('ij,ji->', h1e, density[0], optimize=True)
    e += np.einsum('ij,ji->', h1e, density[1], optimize=True)

    e_coulomb = (np.einsum('ij,ji->', v[0], density[0]) + np.einsum('ij,ji->', v[1], density[1])) * 0.5
  
    return (e, e_coulomb)

def scf(molAtom, molBasis, run, show):

    global density_total

    #options
    name, basisName, diisStatus, integralEngine, gaugeOrigin, hamiltonianGuess, charge, multiplicity, \
          iterations, diisCapacity, convergence, uhfmix, p, _, _, _, _  = run.values()

    #pre-SCF matrices - s{overlap}  t{1e kinetic/exchange/resonance}  v{1e Coulomb} ERI(2e electron repulsion)
    if integralEngine == 'native':
        s = buildOverlap(molBasis)
        t = buildKinetic(molBasis)
        v = buildCoulomb(molAtom, molBasis)
        eri = buildEri(molBasis)
    elif integralEngine == 'aello':
        from aello import aello
        s, t, v, eri = aello(molAtom, molBasis)

    #get nuclear repulsion energy
    nuclear_repulsion = nuclearRepulsion(molAtom)

    #get eri to tensor from linear form
    nbf = s.shape[0]
    eri = expandEri(eri, nbf)

    #get one-electron hamiltonian
    one_electron_h = t + v
    eps, c = scipy.linalg.eigh(one_electron_h, s)

    #determine the alpha and beta occupancies
    mo_occupancy = get_occupations([eps], molAtom, charge, multiplicity)

    #check we have right number of electrons
    if electronCount(molAtom, charge) != np.sum(mo_occupancy):
        exit('charge and multiplicity incompatible')

    #report if requested
    if 'orbitals' in show:
        uhfOrbitals(sum(i > 0 for i in mo_occupancy), sum(i == 2 for i in mo_occupancy), multiplicity)

    #get alpha and beta density matrices
    dm = get_densities(c, mo_occupancy)

    #coulomb and exchange integrals
    j, k = get_coulomb_exchange(dm, eri)
    jk = j - k

    #rohf energy components
    rohf_energy = get_rohf_energy(dm, one_electron_h, jk)

    #symmetric orthogonalization
    x =  scipy.linalg.fractional_matrix_power(s, -0.5)

    #convergence buffers
    scf_energy =  [np.sum(rohf_energy)]
    scf_density = [dm]

    #define storage for diis
    if diisStatus == 'on':
        diis = diis_f(diisCapacity)

    for cycle in range(iterations):

        #get cycle fock's and solve for orbital energies and coefficients
        f_alpha, f_beta, fock = make_fock(one_electron_h, jk, dm, s)

        #do diis if selected
        if (cycle != 0) and (diisStatus == 'on'):
            fock = diis.build(fock, dm[0]+dm[1], s, x)

        e, e_alpha, e_beta, c = eigensolution((f_alpha, f_beta, fock), x)

        mo_occupancy = get_occupations([e, e_alpha, e_beta], molAtom, charge, multiplicity)

        dm = get_densities(c, mo_occupancy)

        #recalculate jk and get new energy for this cycle
        j, k = get_coulomb_exchange(dm, eri)
        jk = j[0] + j[1] - k

        rohf_energy = get_rohf_energy(dm, one_electron_h, jk)

        #update buffers
        scf_energy.append(np.sum(rohf_energy))
        scf_density.append(dm)

        #convergence control
        if cycle != 0:
            rms_alpha = rmsDensity(scf_density[-1][0], scf_density[-2][0]) ; rms_beta = rmsDensity(scf_density[-1][1], scf_density[-2][1])
            if (abs(scf_energy[1] - scf_energy[0]) < convergence) and (rms_alpha < convergence) and (rms_beta < convergence):
                break
            else:
                SCF(scf_energy[-1], abs(scf_energy[1] - scf_energy[0]), (rms_alpha + rms_beta)*0.5, cycle, 'off', iterations, convergence)

        del scf_energy[0]
        del scf_density[0]

    if cycle == iterations:
        print('SCF failed to converge in ' + str(cycle) + ' iterations')
        if diisStatus == 'off': print('Try diis = \'on\'')
        post(False)
        sys.exit('convergence failure')

    if 'SCF' in show:
        postSCF([scf_energy[-1 ] + nuclear_repulsion, cycle, scf_energy[-1 ]],'uhf')

    if 'postSCF' in show:
        def spin_analysis(c, occupancy, s):
            #spin contamination and spin statistics

            alpha = (occupancy > 0).astype(np.int)  ; beta = (occupancy == 2).astype(np.int)

            #get occupied molecular eigenvectors
            c_occ = [c[0]*alpha, c[1]*beta]
            mo_occ = [c_occ[0][:,~np.all(c_occ[0]==0, axis=0)],  c_occ[1][:,~np.all(c_occ[1]==0, axis=0)]]
            nalpha, nbeta = [mo_occ[0].shape[1], mo_occ[1].shape[1]]

            #components of total spin
            spin = np.einsum('rp,rs,sq->pq',mo_occ[0], s, mo_occ[1], optimize=True)
            spin_xy = (nalpha + nbeta) * 0.5 - np.einsum('ij,ij->', spin, spin, optimize=True)
            spin_z  = (nalpha - nbeta)**2 * 0.25
            spin_total = (spin_xy + spin_z)

            multiplicity = np.sqrt(spin_total + 0.25) - 0.5

            return spin_total, multiplicity*2 + 1, nalpha, nbeta

        spin_total, spin_multiplicity, nalpha, nbeta = spin_analysis([c,c], mo_occupancy, s)

        spin_matrix = np.einsum('rp,rs,sq->pq', c[:, :nbeta], s, c[:, :nalpha], optimize=True)
        spin_contamination = nbeta - np.einsum('pq,pq->',spin_matrix, spin_matrix, optimize=True)
        density_total = dm[0] + dm[1] ;  spin_density  = dm[0] - dm[1]
        postSCF([e_alpha, e_beta, density_total, spin_density, spin_total, spin_contamination, spin_multiplicity], 'uhf-post')

        #if requested Mulliken populations
        if 'ch' in p:
            population = []
            population.append(np.einsum('ij,ji->i', dm[0], s, optimize=True))
            population.append(np.einsum('ij,ji->i', dm[1], s, optimize=True))

            mulliken = []
            atom_charge = np.zeros((2,len(molAtom)))
            atom_spin_charge = np.zeros(atom_charge.shape[1])
            total_charge = np.zeros(2)

            for spin in range(2):
                for i in range(population[spin].shape[0]):
                    mulliken.append([molBasis[i].symbol, molAtom[molBasis[i].atom].id, round(population[spin][i],6)])
                    atom_charge[spin][molBasis[i].atom] += round(population[spin][i],6)
                    total_charge[spin] += population[spin][i]

            spin_population = np.einsum('ij,ji->i', spin_density, s)
            for i in range(spin_population.shape[0]):
                atom_spin_charge[molBasis[i].atom] += spin_population[i]

            atoms = []
            for atom in molAtom:
                atoms.append([atom.id, atom.number])

            postSCF([nbf, mulliken, total_charge, atom_charge, atoms, spin_population, atom_spin_charge], 'uhf-mull')

        #if requested dipole moment
        if 'di' in p: 
            import aello
            dipoles = buildDipole(molAtom, molBasis, density_total*0.5, gaugeOrigin)

    #clean up outfile
    post()

    return scf_energy[-1 ] + nuclear_repulsion