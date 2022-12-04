from __future__ import division
import numpy as np
from diis import diis_c
from view import postSCF
from atom import getConstant
from adc.adc import davidson
from cc.fcc import spinMO

class unitaryCoupledCluster(object):
    #Unitary Coupled Cluster CC2

    o = None ; v = None
    d_tensors = []
    HFenergy = 0.0

    def __init__(self, fs, gs, e, data):

        self.fs = fs
        self.gs = gs
        self.e = e
        self.ss = None
        self.sd = None
        self.method, self.electrons, self.cycle_limit, self.convergence, self.verbose, self.roots = data.values()
        self.converged = False

        self.guess_vector_factor = 1

        #check method implemented
        if not self.method in ['ucc2', 'ucc2-ee', 'ucc2-s-ee', 'ucc3', 'ucc(4)'] : return None

        self.mp2 = 0.0  

        self.energy = self.iterator(self.update_amplitudes) 

        if self.method in ['ucc2-ee', 'ucc2-s-ee']: self.excitations()

    def initialise_amplitudes(self):
        #set initial amplitudes

        nocc = self.electrons
        n = np.newaxis
        o = slice(None, nocc)
        v = slice(nocc, None)

        eps = np.kron(self.e, np.ones(2))

        #d tensors
        ds = 1.0 / (eps[o, n] - eps[n, v] )
        dd = 1.0 / (eps[o, n, n, n] + eps[n, o, n, n] - eps[n, n, v, n] - eps[n, n, n, v] )
        dt = 1.0 / (eps[o, n, n, n, n, n] + eps[n, o, n, n, n, n] + eps[n, n, o, n, n, n]- \
                    eps[n, n, n, v, n, n] - eps[n, n, n, n, v, n] - eps[n, n, n, n, n, v] )
 
        self.d_tensors = [ds, dd, dt]

        #initial amplitudes and mp2 energy
        self.ss = np.zeros_like(self.fs[o,v]) ; self.sd = self.gs[o, o, v, v]*dd ; self.st = np.zeros_like(dt)
        self.mp2 = 0.25 * np.einsum('ijab,ijab->', self.gs[o, o, v, v], self.sd, optimize=True)
        self.td = self.sd.copy()

        #Hartree-Fock energy
        self.HFenergy = np.einsum('ii', self.fs[o, o]) - 0.5 * np.einsum('ijij', self.gs[o, o, o, o])

        #class variables
        self.o = o ; self.v = v

    def update_amplitudes(self, iterative=True):
        #compute the next cycle amplitudes

        o = self.o ; v = self.v

        if self.method in ['ucc2', 'ucc2-s-ee', 'ucc2-ee']:
            if not iterative:
                s1 =  0.5 * np.einsum('ajbc,ijbc->ia', self.gs[v,o,v,v], self.sd, optimize=True)
                s1 -= 0.5 * np.einsum('jkib,jkab->ia', self.gs[o,o,o,v], self.sd, optimize=True)

                self.ss = s1 * self.d_tensors[0]

            s2 = self.gs[o,o,v,v].copy()
            s2 += 0.5 * np.einsum('abcd,ijcd->ijab', self.gs[v,v,v,v], self.sd, optimize=True)
            s2 += 0.5 * np.einsum('klij,klab->ijab', self.gs[o,o,o,o], self.sd, optimize=True)

            t = np.einsum('kbcj,ikac->ijab', self.gs[o,v,v,o], self.sd, optimize=True)
            s2 += t - t.transpose(1,0,2,3) - t.transpose(0,1,3,2) + t.transpose(1,0,3,2)

            self.sd = s2 * self.d_tensors[1]

        if self.method == 'ucc3':
            s1 =  0.5 * np.einsum('ajbc,ijbc->ia', self.gs[v,o,v,v], self.sd, optimize=True)
            s1 -= 0.5 * np.einsum('jkib,jkab->ia', self.gs[o,o,o,v], self.sd, optimize=True)
            s1 += np.einsum('ajib,jb->ia', self.gs[v,o,o,v], self.ss, optimize=True)
            s1 += 0.5 * np.einsum('abij,jb->ia', self.gs[v,v,o,o], self.ss, optimize=True)
            s1 -= 0.5 * np.einsum('alik,ljcb,kjcb->ia', self.gs[v,o,o,o], self.sd, self.sd, optimize=True)
            s1 += 0.5 * np.einsum('adic,kjcb,kjdb->ia', self.gs[v,v,o,v], self.sd, self.sd, optimize=True)
            s1 -= 0.25* np.einsum('bcid,jkad,jkbc->ia', self.gs[v,v,o,v], self.sd, self.sd, optimize=True)
            s1 += 0.25* np.einsum('aljk,ilbc,jkbc->ia', self.gs[v,o,o,o], self.sd, self.sd, optimize=True)
            s1 -= np.einsum('acjd,ikbd,jkbc->ia', self.gs[v,v,o,v], self.sd, self.sd, optimize=True)
            s1 += np.einsum('blik,jlac,jkbc->ia', self.gs[v,o,o,o], self.sd, self.sd, optimize=True)
            s1 -= 0.25*np.einsum('bljk,ilac,jkbc->ia', self.gs[v,o,o,o], self.sd, self.sd, optimize=True)
            s1 += 0.25*np.einsum('bdjc,ikac,jkbd->ia', self.gs[v,v,o,v], self.sd, self.sd, optimize=True)

            self.ss = s1 * self.d_tensors[0]

            s2 = self.gs[o,o,v,v].copy()
            s2 += 0.5 * np.einsum('abcd,ijcd->ijab', self.gs[v,v,v,v], self.sd, optimize=True)
            s2 += 0.5 * np.einsum('klij,klab->ijab', self.gs[o,o,o,o], self.sd, optimize=True)

            t = np.einsum('kbcj,ikac->ijab', self.gs[o,v,v,o], self.sd, optimize=True)
            s2 += t - t.transpose(1,0,2,3) - t.transpose(0,1,3,2) + t.transpose(1,0,3,2)

            t = np.einsum('abcj,ic->ijab', self.gs[v,v,v,o], self.ss, optimize=True)
            s2 += t - t.transpose(1,0,2,3)

            t = -np.einsum('kbij,ka->ijab', self.gs[o,v,o,o], self.ss, optimize=True)
            s2 += t - t.transpose(0,1,3,2)

            t = (1/3) * np.einsum('klcd,ikac,ljdb->ijab', self.gs[o,o,v,v], self.sd, self.sd, optimize=True)
            s2 += t - t.transpose(1,0,2,3) - t.transpose(0,1,3,2) + t.transpose(1,0,3,2)

            s2 += (1/6) * np.einsum('klcd,ijcd,klab->ijab', self.gs[o,o,v,v], self.sd, self.sd, optimize=True)

            t = -(1/3) * np.einsum('klcd,ijac,klbd->ijab', self.gs[o,o,v,v], self.sd, self.sd, optimize=True)
            s2 += t - t.transpose(0,1,3,2)

            t = -(1/3) * np.einsum('klcd,ikab,jlcd->ijab', self.gs[o,o,v,v], self.sd, self.sd, optimize=True)
            s2 += t - t.transpose(1,0,2,3)

            t = (1/3) * np.einsum('acik,ljdb,klcd->ijab', self.gs[v,v,o,o], self.sd, self.sd, optimize=True)
            s2 += t - t.transpose(1,0,2,3) - t.transpose(0,1,3,2) + t.transpose(1,0,3,2)

            s2 += (1/12) * np.einsum('cdij,klab,klcd->ijab', self.gs[v,v,o,o], self.sd, self.sd, optimize=True)
            s2 += (1/12) * np.einsum('abkl,ijcd,klcd->ijab', self.gs[v,v,o,o], self.sd, self.sd, optimize=True)

            t = -(1/6) * np.einsum('cdki,ljab,klcd->ijab', self.gs[v,v,o,o], self.sd, self.sd, optimize=True)
            s2 += t - t.transpose(1,0,2,3)

            t = -(1/6) * np.einsum('cakl,ijdb,klcd->ijab', self.gs[v,v,o,o], self.sd, self.sd, optimize=True)
            s2 += t - t.transpose(0,1,3,2)

            t = -(1/6) * np.einsum('acij,klbd,klcd->ijab', self.gs[v,v,o,o], self.sd, self.sd, optimize=True)
            s2 += t - t.transpose(0,1,3,2)

            t = -(1/6) * np.einsum('abik,jlcd,klcd->ijab', self.gs[v,v,o,o], self.sd, self.sd, optimize=True)
            s2 += t - t.transpose(1,0,2,3)

            self.sd = s2 * self.d_tensors[1]

        if self.method == 'ucc(4)':
            s1 =  0.5 * np.einsum('ajbc,ijbc->ia', self.gs[v,o,v,v], self.sd, optimize=True)
            s1 -= 0.5 * np.einsum('jkib,jkab->ia', self.gs[o,o,o,v], self.sd, optimize=True)

            self.ss = s1 * self.d_tensors[0]

            s2 = self.gs[o,o,v,v].copy()
            s2 += 0.5 * np.einsum('abcd,ijcd->ijab', self.gs[v,v,v,v], self.sd, optimize=True)
            s2 += 0.5 * np.einsum('klij,klab->ijab', self.gs[o,o,o,o], self.sd, optimize=True)

            t = np.einsum('kbcj,ikac->ijab', self.gs[o,v,v,o], self.sd, optimize=True)
            s2 += t - t.transpose(1,0,2,3) - t.transpose(0,1,3,2) + t.transpose(1,0,3,2)
            t = np.einsum('abcj,ic->ijab', self.gs[v,v,v,o], self.ss, optimize=True)
            s2 += t - t.transpose(1,0,2,3)
            t = -np.einsum('kbij,ka->ijab', self.gs[o,v,o,o], self.ss, optimize=True)
            s2 += t - t.transpose(0,1,3,2)

            t = 0.25 * np.einsum('klcd,ikac,jlbd->ijab', self.gs[o,o,v,v], self.sd, self.sd, optimize=True)
            s2 += t - t.transpose(1,0,2,3) - t.transpose(0,1,3,2) + t.transpose(1,0,3,2)
            t = -0.25 * np.einsum('klcd,ikab,jlcd->ijab', self.gs[o,o,v,v], self.sd, self.sd, optimize=True) 
            s2 += t - t.transpose(1,0,2,3) 
            t = -0.25 * np.einsum('klcd,ijac,klbd->ijab', self.gs[o,o,v,v], self.sd, self.sd, optimize=True)
            s2 += t - t.transpose(0,1,3,2)     
            s2 += 0.125 * np.einsum('klcd,ijcd,klab->ijab', self.gs[o,o,v,v], self.sd, self.sd, optimize=True)    

            t = 0.5 * np.einsum('dblj,ikac,klcd->ijab', self.gs[v,v,o,o], self.sd, self.sd, optimize=True)
            s2 += t - t.transpose(1,0,2,3) - t.transpose(0,1,3,2) + t.transpose(1,0,3,2)
            s2 += 0.125 * np.einsum('abkl,klcd,ijcd->ijab', self.gs[v,v,o,o], self.sd, self.sd, optimize=True)
            s2 += 0.125 * np.einsum('cdij,klcd,klab->ijab', self.gs[v,v,o,o], self.sd, self.sd, optimize=True)
            t = -0.25 * np.einsum('bdkl,klcd,ijac->ijab', self.gs[v,v,o,o], self.sd, self.sd, optimize=True)
            s2 += t - t.transpose(0,1,3,2)
            t = -0.25 * np.einsum('cdjl,klcd,ikab->ijab', self.gs[v,v,o,o], self.sd, self.sd, optimize=True)
            s2 += t - t.transpose(1,0,2,3)
            t = -0.25 * np.einsum('dbij,klca,klcd->ijab', self.gs[v,v,o,o], self.sd, self.sd, optimize=True)
            s2 += t - t.transpose(0,1,3,2)
            t = -0.25 * np.einsum('ablj,ikdc,klcd->ijab', self.gs[v,v,o,o], self.sd, self.sd, optimize=True)
            s2 += t - t.transpose(1,0,2,3)

            t = 0.5 * np.einsum('bkcd,ijkacd->ijab', self.gs[v,o,v,v], self.st, optimize=True)
            s2 += t - t.transpose(0,1,3,2)
            t = -0.5 * np.einsum('kljc,iklabc->ijab', self.gs[o,o,o,v], self.st, optimize=True)
            s2 += t - t.transpose(1,0,2,3)

            self.sd = s2 * self.d_tensors[1]

            t = np.einsum('bcdk,ijad->ijkabc', self.gs[v,v,v,o], self.sd, optimize=True)
            t = t - t.transpose(0,1,2,4,3,5) - t.transpose(0,1,2,5,4,3)
            s3 = t - t.transpose(2,1,0,3,4,5) - t.transpose(0,2,1,3,4,5)
            t = -np.einsum('lcjk,ilab->ijkabc', self.gs[o,v,o,o], self.sd, optimize=True)
            t = t - t.transpose(0,1,2,5,4,3) - t.transpose(0,1,2,3,5,4)
            s3 += t - t.transpose(1,0,2,3,4,5) - t.transpose(2,1,0,3,4,5)

            self.dt = s3 * self.d_tensors[2]

    def cluster_energy(self):
        #compute the coupled-cluster energy correction

        o = self.o ; v = self.v

        e = 1.0 * np.einsum('ia,ia->', self.fs[o,v], self.ss, optimize=True)

        e += 0.25 * np.einsum('ijab,ijab->', self.gs[o, o, v, v], self.sd, optimize=True)

        if self.method == 'ucc3':
            e += (1/6) * np.einsum('ijab,ia,jb->', self.gs[o,o,v,v], self.ss, self.ss, optimize=True)
        if self.method == 'ucc(4)':
            t =  0.5 * np.einsum('dblj,ikac,klcd,ijab->', self.gs[v,v,o,o], self.sd, self.sd, self.sd, optimize=True)
            t += 0.0625 * np.einsum('abkl,ijab,klcd,ijcd->', self.gs[v,v,o,o], self.sd, self.sd, self.sd, optimize=True)
            t -= 0.25 * np.einsum('ablj,klcd,ijab,ikdc->', self.gs[v,v,o,o], self.sd, self.sd, self.sd, optimize=True)
            t -= 0.25 * np.einsum('dbij,klcd,ijab,klca->', self.gs[v,v,o,o], self.sd, self.sd, self.sd, optimize=True)
            e -= 0.5 * t

        return e

    def iterator(self, func):
        #consistent field iterations

        #initiate the amplitudes
        self.initialise_amplitudes()

        #initialise diis buffers
        amplitudes = [self.ss, self.sd] if self.method != 'ucc(4)' else [self.ss, self.sd, self.st]
        diis = diis_c(6, amplitudes)

        cycle_energy = [self.cluster_energy()]

        for cycle in range(self.cycle_limit):   

            #store pre-update amplitudes
            amplitudes = [self.ss, self.sd] if self.method != 'ucc(4)' else [self.ss, self.sd, self.st]
            diis.refresh_store(amplitudes)

            func()

            #calculate current cycle energy
            cycle_energy.append(self.cluster_energy()) 

            #test convergence
            delta_energy = np.abs(cycle_energy[-2] - cycle_energy[-1])
            if delta_energy < self.convergence:
                self.converged = True
                break
            else:
                if self.verbose: print('cycle = {:>3d}  energy = {:>15.10f}   \u0394E = {:>12.10f} '.format(cycle, cycle_energy[-1], delta_energy))
                del cycle_energy[0]

            #diis build extrapolated amplitudes
            amplitudes = [self.ss, self.sd] if self.method != 'ucc(4)' else [self.ss, self.sd, self.st]
            amplitudes = diis.build(amplitudes)
            if self.method != 'ucc(4)':
                self.ss, self.sd = amplitudes 
            else:
                self.ss, self.sd, self.st = amplitudes                 

        if self.converged:
            self.update_amplitudes(iterative=False)
            return {self.method: cycle_energy[-1], 'mp2': self.mp2, 'eHF':self.HFenergy}


    def excitations(self):
        #UCC2 secular matrix

        o, v, n = self.o, self.v, np.newaxis
        nocc, nvir = self.ss.shape
        nrot = nocc * nvir

        ds, dd = np.reciprocal(self.d_tensors[0]), np.reciprocal(self.d_tensors[1])
        sd =  self.sd if not self.method == 'ucc2-s-ee' else self.td

        def ucc_diagonal(self):
            #compute the diagonal of the ADC matrix as a pre-conditioner for the davidson iterations

            #initialize to fock diagonal
            diagonal = -np.concatenate([ds.ravel(), dd.swapaxes(1, 2).ravel()])

            ucc_diagonal = diagonal[:nrot].reshape(nocc, nvir)

            ucc_diagonal -= np.einsum('aiai->ia', self.gs[v, o, v, o], optimize=True)

            ucc_diagonal += 0.5  * np.einsum('acik,ikac->ia', self.gs[v, v, o, o], sd, optimize=True)
            ucc_diagonal += 0.5  * np.einsum('ikac,ikac->ia', self.gs[o, o, v, v], sd, optimize=True) 
            ucc_diagonal -= 0.25 * np.einsum('cdik,ikcd->i',  self.gs[v, v, o, o], sd, optimize=True)[:, n]
            ucc_diagonal -= 0.25 * np.einsum('ikcd,ikcd->i',  self.gs[o, o, v, v], sd, optimize=True)[:, n]
            ucc_diagonal -= 0.25 * np.einsum('ackl,klac->a',  self.gs[v, v, o, o], sd, optimize=True)[n, :]
            ucc_diagonal -= 0.25 * np.einsum('klac,klac->a',  self.gs[o, o, v, v], sd, optimize=True)[n, :]

            return diagonal

        def ucc_initial_guess(self, diagonal, f=1):
            #initial vector to start Davidson

            #get largest absolute values on diagonal matrix as best guess
            args = np.argsort(np.absolute(diagonal))
            guess_vectors = np.zeros((diagonal.size, self.roots * f))
            for root in range(self.roots * f):
                guess_vectors[args[root], root] = 1.0

            return guess_vectors

        def matvec(ucc):
        #construct the UCC2 blocks of second order matrix dot product with arbitary vector (r)

            ucc = np.array(ucc)
            r   = np.zeros_like(ucc)

            ucc_s = ucc[:nrot].reshape(nocc, nvir)
            r_s  = r[:nrot].reshape(nocc, nvir)

            #singles - singles block
            r_s -= np.einsum('ia,ia->ia', ds, ucc_s, optimize=True)

            r_s -= np.einsum('ajbi,jb->ia', gs[v, o, v, o], ucc_s, optimize=True)

            r_s += 0.5 * np.einsum('acik,jkbc,jb->ia', gs[v, v, o, o], sd, ucc_s, optimize=True)
            r_s += 0.5 * np.einsum('jkbc,ikac,jb->ia', gs[o, o, v, v], sd, ucc_s, optimize=True)

            t    = -np.einsum('cdik,jkcd->ij', gs[v, v, o, o], sd, optimize=True)
            t   += -np.einsum('jkcd,ikcd->ij', gs[o, o, v, v], sd, optimize=True)
            r_s += 0.25 * np.einsum('ij,ja->ia', t, ucc_s, optimize=True)

            t    = -np.einsum('ackl,klbc->ab', gs[v, v, o, o], sd, optimize=True)
            t   += -np.einsum('klbc,klac->ab', gs[o, o, v, v], sd, optimize=True)
            r_s += 0.25 * np.einsum('ab,ib->ia', t, ucc_s, optimize=True)

            ucc_d = ucc[nrot:].reshape(nocc, nvir, nocc, nvir)
            r_d  = r[nrot:].reshape(nocc, nvir, nocc, nvir)

            #singles - doubles block
            r_s += 0.5 * np.einsum('klid,kald->ia', gs[o, o, o, v], ucc_d, optimize=True)
            r_s -= 0.5 * np.einsum('klic,kcla->ia', gs[o, o, o, v], ucc_d, optimize=True)
            r_s -= 0.5 * np.einsum('alcd,icld->ia', gs[v, o, v, v], ucc_d, optimize=True)
            r_s += 0.5 * np.einsum('akcd,kcid->ia', gs[v, o, v, v], ucc_d, optimize=True)

            #doubles - singles block
            r_d += 0.5 * np.einsum('kbij,ka->iajb', gs[o, v, o, o], ucc_s, optimize=True)
            r_d -= 0.5 * np.einsum('kaij,kb->iajb', gs[o, v, o, o], ucc_s, optimize=True)
            r_d -= 0.5 * np.einsum('abcj,ic->iajb', gs[v, v, v, o], ucc_s, optimize=True)
            r_d += 0.5 * np.einsum('abci,jc->iajb', gs[v, v, v, o], ucc_s, optimize=True)

            #doubles - doubles block
            r_d -= np.einsum('ijab,iajb->iajb', dd, ucc_d, optimize=True)

            return r

        #get diagonal preconditioner
        diagonal = ucc_diagonal(self)

        #generate initial guess from diagonal
        guess_vectors = ucc_initial_guess(self, diagonal, self.guess_vector_factor)

        self.tol = 1e-8 ; sort_on_absolute=True ; self.vectors_per_root=30
        e, v, self.converged = davidson(matvec, guess_vectors, diagonal, tol=self.tol,
                                        vectors_per_root = self.vectors_per_root )

        print(e)

if __name__ == '__main__':

    #do an SCF computation
    import rhf
    molAtom, molBasis, molData = rhf.mol([])
    eSCF = rhf.scf(molAtom, molBasis,molData, [])

    #get data for coupled-cluster
    from atom import nuclearRepulsion ; from basis import electronCount
    charge, nuclearRepulsion, electrons = [molData['charge'], nuclearRepulsion(molAtom), electronCount(molAtom, molData['charge'])]
    f, c, e, eri = [rhf.fock, rhf.C, rhf.e, rhf.ERI]   

    #get fock and eri in molecular spin basis from spinMO class
    mo = spinMO(e, eri, c, f)
    gs = mo.gs
    fs = mo.fs

    data = {'method':'ucc(4)', 'electrons':electrons, 'cycle_limit': 50, 'precision':1e-10, 'verbose':True, 'roots':5}
    cc = unitaryCoupledCluster(fs, gs, e, data)
    if cc.converged:
        cc.energy['nuclear'] = nuclearRepulsion

        print(cc.energy)
