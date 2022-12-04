from __future__ import division
import numpy as np
import sys
sys.path.append('../../harpy/codes')
from cc.fcc import spinMO
from diis import diis_c

'''
extra diagram to add to CCSD to implement CCSDT-n methods
'''
def cluster_triples_n_diagrams(name, d_tensor, f, g, o, v, t1=None, t2=None, t3=None):
    #evaluate extra diagrams for ccsdt-n

    if name in ['ccsdt-1a', 'ccsdt-1b', 'ccsdt-2', 'ccsdt-3', 'ccsdt-4']:
        #diagram S_{7}
        singles = 0.25 * np.einsum('mnef,aefimn->ai', g[o,o,v,v], t3, optimize=True)

        #diagrams D_{10a-19c}
        doubles =  np.einsum('em,abeijm->abij', f[v,o], t3, optimize=True)
        t =  0.5 * np.einsum('aefijm,bmef->abij', t3, g[v,o,v,v], optimize=True)
        doubles += t - t.transpose(1,0,2,3)
        t = -0.5 * np.einsum('abeimn,mnje->abij', t3, g[o,o,o,v], optimize=True)
        doubles += t - t.transpose(0,1,3,2)

        #diagrams T_{1a-1b}, T_{2a-2b}
        t = np.einsum('aeij,bcek->abcijk', t2, g[v,v,v,o], optimize=True)
        t = t - t.transpose(0,1,2,5,4,3) - t.transpose(0,1,2,3,5,4)
        triples = t - t.transpose(1,0,2,3,4,5) - t.transpose(2,1,0,3,4,5)
        t = -np.einsum('abim,mcjk->abcijk', t2, g[o,v,o,o], optimize=True)
        t =  t - t.transpose(0,1,2,4,3,5) - t.transpose(0,1,2,5,4,3)
        triples += t - t.transpose(2,1,0,3,4,5) - t.transpose(0,2,1,3,4,5)
        t = np.einsum('ce,abeijk->abcijk', f[v,v], t3, optimize=True)
        triples += t - t.transpose(2,1,0,3,4,5) - t.transpose(0,2,1,3,4,5)
        t = -np.einsum('km,abcijm->abcijk', f[o,o], t3, optimize=True)
        triples += t - t.transpose(0,1,2,5,4,3) - t.transpose(0,1,2,3,5,4)

    if name in ['ccsdt-1b', 'ccsdt-2', 'ccsdt-3', 'ccsdt-4']:
        #diagrams D_{11a-11c}
        doubles += np.einsum('mnef,em,fabnij->abij', g[o,o,v,v], t1, t3, optimize=True)
        t = -0.5 * np.einsum('mnef,am,efbinj->abij', g[o,o,v,v], t1, t3, optimize=True)
        doubles += t - t.transpose(1,0,2,3)
        t = -0.5 * np.einsum('mnef,ei,afbmnj->abij', g[o,o,v,v], t1, t3, optimize=True)
        doubles += t - t.transpose(0,1,3,2)

    if name in ['ccsdt-2', 'ccsdt-3', 'ccsdt-4']:
        #diagrams T_{3a-3e}
        t = -np.einsum('em,aeij,bcmk->abcijk', f[v,o], t2, t2, optimize=True)
        t = t - t.transpose(0,1,2,5,4,3) - t.transpose(0,1,2,3,5,4)
        triples += t - t.transpose(1,0,2,3,4,5) - t.transpose(2,1,0,3,4,5)
        t = np.einsum('mbef,aeim,fcjk->abcijk',g[o,v,v,v], t2, t2, optimize=True)
        t = t - t.transpose(0,1,2,4,3,5) - t.transpose(0,1,2,5,4,3)
        triples += t - t.transpose(1,0,2,3,4,5) - t.transpose(0,2,1,3,4,5) + t.transpose(1,2,0,3,4,5) + t.transpose(2,0,1,3,4,5) - t.transpose(2,1,0,3,4,5)
        t = -np.einsum('mnej,aeim,bcnk->abcijk', g[o,o,v,o], t2, t2, optimize=True)
        t = t - t.transpose(0,1,2,4,3,5) - t.transpose(0,1,2,3,5,4) + t.transpose(0,1,2,4,5,3) + t.transpose(0,1,2,5,3,4) - t.transpose(0,1,2,5,4,3)
        triples += t - t.transpose(1,0,2,3,4,5) - t.transpose(2,1,0,3,4,5)
        t = -0.5 * np.einsum('mcef,abim,efjk->abcijk', g[o,v,v,v], t2, t2, optimize=True)
        t = t - t.transpose(0,1,2,4,3,5) - t.transpose(0,1,2,5,4,3)
        triples += t - t.transpose(2,1,0,3,4,5) - t.transpose(0,2,1,3,4,5)   
        t = 0.5 * np.einsum('mnek,aeij,bcmn->abcijk', g[o,o,v,o], t2, t2, optimize=True)
        t = t - t.transpose(0,1,2,5,4,3) - t.transpose(0,1,2,3,5,4)
        triples += t - t.transpose(1,0,2,3,4,5) - t.transpose(2,1,0,3,4,5)

    if name in ['ccsdt-3', 'ccsdt-4']:
        #diagrams T_{4a-4d}
        t = np.einsum('abef,ei,fcjk->abcijk', g[v,v,v,v], t1, t2, optimize=True)
        t = t - t.transpose(0,1,2,4,3,5) - t.transpose(0,1,2,5,4,3)
        triples += t - t.transpose(2,1,0,3,4,5) - t.transpose(0,2,1,3,4,5)
        t = np.einsum('mnij,am,bcnk->abcijk', g[o,o,o,o], t1, t2, optimize=True)
        t = t - t.transpose(0,1,2,5,4,3) - t.transpose(0,1,2,3,5,4)
        triples += t - t.transpose(1,0,2,3,4,5) - t.transpose(2,1,0,3,4,5)
        t = -np.einsum('amie,ej,bcmk->abcijk', g[v,o,o,v], t1, t2, optimize=True)
        t = t - t.transpose(0,1,2,4,3,5) - t.transpose(0,1,2,3,5,4) + t.transpose(0,1,2,4,5,3) + t.transpose(0,1,2,5,3,4) - t.transpose(0,1,2,5,4,3)
        triples += t - t.transpose(1,0,2,3,4,5) - t.transpose(2,1,0,3,4,5)
        t = -np.einsum('amie,bm,ecjk->abcijk', g[v,o,o,v], t1, t2, optimize=True)
        t = t - t.transpose(0,1,2,4,3,5) - t.transpose(0,1,2,5,4,3)
        triples += t - t.transpose(1,0,2,3,4,5) - t.transpose(0,2,1,3,4,5) + t.transpose(1,2,0,3,4,5) + t.transpose(2,0,1,3,4,5) - t.transpose(2,1,0,3,4,5)
        #diagram T_{7a-7d}
        t = -np.einsum('mbef,ei,am,fcjk->abcijk', g[o,v,v,v], t1, t1, t2, optimize=True)
        t = t - t.transpose(0,1,2,4,3,5) - t.transpose(0,1,2,5,4,3)
        triples += t - t.transpose(1,0,2,3,4,5) - t.transpose(0,2,1,3,4,5) + t.transpose(1,2,0,3,4,5) + t.transpose(2,0,1,3,4,5) - t.transpose(2,1,0,3,4,5)
        t = np.einsum('mnej,ei,am,bcnk->abcijk', g[o,o,v,o], t1, t1, t2, optimize=True) 
        t = t - t.transpose(0,1,2,4,3,5) - t.transpose(0,1,2,3,5,4) + t.transpose(0,1,2,4,5,3) + t.transpose(0,1,2,5,3,4) - t.transpose(0,1,2,5,4,3)
        triples += t - t.transpose(1,0,2,3,4,5) - t.transpose(2,1,0,3,4,5)
        t = -np.einsum('amef,ei,fj,bcmk->abcijk', g[v,o,v,v], t1, t1, t2, optimize=True) 
        t = t - t.transpose(0,1,2,5,4,3) - t.transpose(0,1,2,3,5,4)
        triples += t - t.transpose(1,0,2,3,4,5) - t.transpose(2,1,0,3,4,5)
        t = np.einsum('mnie,am,bn,ecjk->abcijk', g[o,o,o,v], t1, t1, t2, optimize=True)                      
        t = t - t.transpose(0,1,2,4,3,5) - t.transpose(0,1,2,5,4,3)
        triples += t - t.transpose(2,1,0,3,4,5) - t.transpose(0,2,1,3,4,5)
        #diagrams T_{8a-8e}
        t = -np.einsum('mnef,em,abin,fcjk->abcijk', g[o,o,v,v], t1, t2, t2, optimize=True)
        t = t - t.transpose(0,1,2,4,3,5) - t.transpose(0,1,2,5,4,3)
        triples += t - t.transpose(2,1,0,3,4,5) - t.transpose(0,2,1,3,4,5)
        t = -np.einsum('mnef,fj,aeim,bcnk->abcijk', g[o,o,v,v], t1, t2, t2, optimize=True)
        t = t - t.transpose(0,1,2,4,3,5) - t.transpose(0,1,2,3,5,4) + t.transpose(0,1,2,4,5,3) + t.transpose(0,1,2,5,3,4) - t.transpose(0,1,2,5,4,3)
        triples += t - t.transpose(1,0,2,3,4,5) - t.transpose(2,1,0,3,4,5)      
        t = -np.einsum('mnef,bn,aeim,fcjk->abcijk', g[o,o,v,v], t1, t2, t2, optimize=True)
        t = t - t.transpose(0,1,2,4,3,5) - t.transpose(0,1,2,5,4,3)
        triples += t - t.transpose(1,0,2,3,4,5) - t.transpose(0,2,1,3,4,5) + t.transpose(1,2,0,3,4,5) + t.transpose(2,0,1,3,4,5) - t.transpose(2,1,0,3,4,5)
        t = 0.5 * np.einsum('mnef,ei,abmn,fcjk->abcijk', g[o,o,v,v], t1, t2, t2, optimize=True)      
        t = t - t.transpose(0,1,2,4,3,5) - t.transpose(0,1,2,5,4,3)
        triples += t - t.transpose(2,1,0,3,4,5) - t.transpose(0,2,1,3,4,5)
        t = 0.5 * np.einsum('mnef,am,efij,bcnk->abcijk', g[o,o,v,v], t1, t2, t2, optimize=True)  
        t = t - t.transpose(0,1,2,5,4,3) - t.transpose(0,1,2,3,5,4)       
        triples += t - t.transpose(1,0,2,3,4,5) - t.transpose(2,1,0,3,4,5)   

    if name in ['ccsdt-4']:
        #diagrams T_{2c - 2e}
        t = 0.5 * np.einsum('abef,efcijk->abcijk', g[v,v,v,v], t3, optimize=True)
        triples += t - t.transpose(2,1,0,3,4,5) - t.transpose(0,2,1,3,4,5)
        t = 0.5 * np.einsum('mnij,abcmnk->abcijk', g[o,o,o,o], t3, optimize=True)
        triples += t - t.transpose(0,1,2,5,4,3) - t.transpose(0,1,2,3,5,4)
        t = np.einsum('amie,ebcmjk->abcijk', g[v,o,o,v], t3, optimize=True)
        t = t - t.transpose(0,1,2,4,3,5) - t.transpose(0,1,2,5,4,3)
        triples += t - t.transpose(1,0,2,3,4,5) - t.transpose(2,1,0,3,4,5)

    return [singles * d_tensor[0], doubles * d_tensor[1], triples * d_tensor[2] + t3]

'''
use the symbolically generated COGUS CCSD, CCSDT and CCSD(T) to get basic amplitudes
'''
def coupledClusterTriplesVariations(name, fock, eri, c, e, scfData, runData):
    #execute symbolically generated cluster codes

    #check we can handle method
    if not name in ['ccsd', 'ccsdt','ccsd(t)', 'ccsdt-1a', 'ccsdt-1b', 'ccsdt-2', 'ccsdt-3', 'ccsdt-4'] : return 0.0

    #get code for correct method
    if name == 'ccsdt':
        from ccsdt  import cc_energy, cc_singles, cc_doubles, cc_triples
    else:
        from ccsd_t import cc_energy, cc_singles, cc_doubles, cc_triples, cc_perturbation_energy

    #run and scf values
    use_td_guess, iterations, tolerance, verbose = runData
    charge, nuclearRepulsion, electrons = scfData

    #orbital occupations
    spinOrbitals = (fock.shape[0]) * 2
    nocc = electrons
    nvir = spinOrbitals - nocc

    #get two-electron repulsion integrals in MO basis
    mo = spinMO(e, eri, c, fock)
    gs = mo.gs
    fs = mo.fs

    #slices
    n = np.newaxis
    o = slice(None,nocc)
    v = slice(nocc, None)

    #D tensors
    eps = np.kron(e, np.ones(2))

    d = [np.reciprocal(-eps[v, n] + eps[n, o]),
         np.reciprocal(-eps[v, n, n, n] - eps[n, v, n, n] + eps[n, n, o, n] + eps[n, n, n, o]),
         np.reciprocal(- eps[ v, n, n, n, n, n] - eps[n, v, n, n, n, n] - eps[n, n, v, n, n, n]
                       + eps[n, n, n, o, n, n] + eps[n, n, n, n, o, n] + eps[n, n, n, n, n, o])]

    #HF energy3
    HFenergy = 1.0 * np.einsum('ii', fs[o, o]) -0.5 * np.einsum('ijij', gs[o, o, o, o])
    print('Hartree-Fock SCF electronic energy           {:<14.10f}  au\nTotal energy                                 {:<14.10f}  au\n'.
                                                                                format(HFenergy, HFenergy + nuclearRepulsion))
    if verbose: 
        print('cycle           energy                \u0394E\n---------------------------------------------')
    #amplitude initialisation
    ts = np.zeros((nvir, nocc))
    td = np.zeros((nvir, nvir, nocc, nocc))
    tt = np.zeros((nvir, nvir, nvir, nocc, nocc, nocc))

    #use initial guess
    if use_td_guess: td = gs[v,v,o,o].copy()*d[1]

    #get initial cluster energy
    cycleEnergy =  [cc_energy(fs, gs, o, v, t1=ts, t2=td, t3=tt ) - HFenergy]

    #initiate DIIS
    diis = diis_c(6, [ts, td, tt])

    triples = np.zeros_like(tt)

    #iterations
    for cycle in range(iterations):

        #update DIIS cache
        diis.refresh_store([ts, td, tt])
        #update amplitudes
        singles = cc_singles(fs, gs, o, v, t1=ts, t2=td, t3=tt) * d[0] + ts 
        doubles = cc_doubles(fs, gs, o, v, t1=ts, t2=td, t3=tt) * d[1] + td
        if name in ['ccsdt', 'ccsd(t)']:
            triples = cc_triples(fs, gs, o, v, t1=ts, t2=td, t3=tt) * d[2] + tt

        #additional diagrams
        if name in ['ccsdt-1a', 'ccsdt-1b', 'ccsdt-2', 'ccsdt-3', 'ccsdt-4']:
            triples_corrections = cluster_triples_n_diagrams(name, d, fs, gs, o, v, t1=ts, t2=td, t3=tt)

            singles += triples_corrections[0]
            doubles += triples_corrections[1]
            triples =  triples_corrections[2]

        #recalculate energy
        cycleEnergy.append(cc_energy(fs, gs, o, v, t1=singles, t2=doubles, t3=triples) - HFenergy)
        deltaEnergy = np.abs(cycleEnergy[-2] - cycleEnergy[-1])

        #convergence test
        if deltaEnergy < tolerance:

            print('Final {:8} correction is                  {:<14.10f} au'.format(name, cycleEnergy[-1]))
            print('Final energy with {:8} correction        {:<14.10f}  au'.format(name, cycleEnergy[-1] + HFenergy))
            break
        else:
            ts = singles
            td = doubles
            tt = triples
            lastCycleEnergy = cycleEnergy
            if verbose: print('{:>3d}       {:>15.10f}        {:>12.10f} '.format(cycle, cycleEnergy[-1], deltaEnergy))

            #interpolated DIIS values
            ts, td, tt = diis.build([ts, td, tt])
            del cycleEnergy[0]
    else:
        print("Did not converge")
        exit('cc failed')

    perturbationEnergy = 0.0
    if name == 'ccsd(t)': 
        perturbativeTriples = cc_triples(fs, gs, o, v, t1=singles, t2=doubles, t3=triples)
        triples = perturbativeTriples + np.reciprocal(d[2]) * triples
        triples = triples * d[2]
        l1, l2 = [singles.transpose(1,0) ,doubles.transpose(2,3,0,1)]

        perturbationEnergy = cc_perturbation_energy(fs, gs, o, v, l1, l2, triples)
        print('\nccsd(t) (t) perturbative correction           {:<14.10f} au'.format(perturbationEnergy))
        print('Final ccsd(t) energy with all corrections    {:<14.10f}  au'.format(cycleEnergy[-1] + HFenergy+ perturbationEnergy))

if __name__ == '__main__':

    import rhf
    molAtom, molBasis, molData = rhf.mol([])
    eSCF = rhf.scf(molAtom, molBasis,molData, [])

    #get data for coupled-cluster
    from atom import nuclearRepulsion ; from basis import electronCount
    charge, nuclearRepulsion, electrons = [molData['charge'], nuclearRepulsion(molAtom), electronCount(molAtom, molData['charge'])]
    f, c, e, eri = [rhf.fock, rhf.C, rhf.e, rhf.ERI]   

    coupledClusterTriplesVariations('ccsdt-1a', f, eri, c, e,
                                   [charge, nuclearRepulsion, electrons],
                                   [True, 50, 1e-10, True])