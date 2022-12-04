from __future__ import division
import numpy as np
from diis import diis_c
from view import postSCF
from atom import getConstant

class coupledCluster(object):
    #class to provide coupled cluster facilities
    #ccsd reference  compute the F and W intermediates from J. Gauss and J. F. Stanton: Coupled-cluster calculations of 
    #Nuclear magnetic resonance chemical shifts - J. Chem. Phys., Vol. 103, No. 9, 1 September 1995

    o = None ; v = None
    d_tensors = []
    HFenergy = 0.0

    def __init__(self, fs, gs, e, data):
        #initialise with spin fock eris and amplitudes

        self.fs = fs
        self.gs = gs
        self.e = e
        self.ts = None
        self.td = None
        self.method, self.electrons, self.cycle_limit, self.convergence, self.verbose = data.values()
        self.converged = False

        #check method implemented
        if not self.method in ['ccsd', 'ccsd(t)', 'ccd', 'cc2', 'lccd', 'lccsd', 'qcisd'] : return None

        func_amplitude = self.update_amplitudes if not self.method in ['lccd', 'lccsd'] else self.update_linear_amplitudes

        self.mp2 = 0.0  

        self.energy = self.iterator(func_amplitude)
        if self.method == 'ccsd(t)': 
            self.perturbative_triples()

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
        self.ts = np.zeros_like(self.fs[o,v]) ; self.td = self.gs[o, o, v, v]*dd
        self.mp2 = 0.25 * np.einsum('ijab,ijab->', self.gs[o, o, v, v], self.td, optimize=True)

        #Hartree-Fock energy
        self.HFenergy = np.einsum('ii', self.fs[o, o]) - 0.5 * np.einsum('ijij', self.gs[o, o, o, o])

        #class variables
        self.o = o ; self.v = v

    def tau(self, tilde=True):
        #build the tau and tau-tilde Table III (d)

        tau = self.td.copy()
        f = 0.5 if tilde else 1.0
        t = f * np.einsum('ia,jb->ijab', self.ts, self.ts, optimize=True)

        tau += t - t.transpose(0,1,3,2)

        return tau

    def intermediates(self, _slice, tilde=True):
        #intermedates from Table III(a)

        if not _slice in ['oo','vv','ov','oooo','vvvv','ovvo']:
            print('no pre-evaluated slice [', _slice, '] - transpose axes')
            exit()

        o = self.o ; v = self.v

        if tilde:
            if _slice == 'oo':
                im = self.fs[o,o].copy()
                np.fill_diagonal(im, 0.0)
                im += 0.5 * np.einsum('ie,me->mi', self.ts, self.fs[o, v],optimize=True)
                im += np.einsum('ne,mnie->mi', self.ts, self.gs[o, o, o, v], optimize=True)
                im += 0.5 * np.einsum('inef,mnef->mi', self.tau(), self.gs[o, o, v, v], optimize=True)

            if _slice == 'vv':
                im = self.fs[v,v].copy()
                np.fill_diagonal(im, 0.0)
                im -= 0.5 * np.einsum('ma,me->ae', self.ts, self.fs[o, v],optimize=True)
                im += np.einsum('mf,mafe->ae', self.ts, self.gs[o, v, v, v], optimize=True)
                im -= 0.5 * np.einsum('mnaf,mnef->ae', self.tau(), self.gs[o, o, v, v], optimize=True)

            if _slice == 'ov':
                im = self.fs[o,v].copy()
                im += np.einsum('nf,mnef->me', self.ts, self.gs[o, o, v, v], optimize=True)

            if _slice == 'oooo':
                im = self.gs[o, o, o, o].copy()
                t = np.einsum('je,mnie->mnij', self.ts, self.gs[o, o, o, v], optimize=True)
                im += t - t.transpose(0,1,3,2)
                im += 0.25 * np.einsum('ijef,mnef->mnij', self.tau(tilde=False), self.gs[o, o, v, v], optimize=True)

            if _slice == 'vvvv':
                im = self.gs[v, v, v, v].copy()
                t = -np.einsum('mb,amef->abef', self.ts, self.gs[v, o, v, v], optimize=True)
                im += (t - t.transpose(1,0,2,3))
                im += 0.25 * np.einsum('mnab,mnef->abef', self.tau(tilde=False), self.gs[o, o, v, v], optimize=True)

            if _slice == 'ovvo':
                im = self.gs[o, v, v, o].copy()
                im += np.einsum('jf,mbef->mbej', self.ts, self.gs[o, v, v, v], optimize=True)
                im -= np.einsum('nb,mnej->mbej', self.ts, self.gs[o, o, v, o], optimize=True)
                im -= 0.5 * np.einsum('jnfb,mnef->mbej', self.td, self.gs[o, o, v, v], optimize=True)
                im -= np.einsum('jf,nb,mnef->mbej', self.ts, self.ts, self.gs[o, o, v, v], optimize=True)

        return im

    def update_amplitudes(self):
        #compute the next cycle amplitudes

        o = self.o ; v = self.v

        foo = self.intermediates('oo') ; fvv = self.intermediates('vv') ; fov = self.intermediates('ov')

        #singles amplitudes
        if self.method in ['ccsd', 'ccsd(t)', 'cc2']:

            #single amplitudes  Table I (a)
            t1 = self.fs[o,v].copy()
            t1 += np.einsum('ie,ae->ia', self.ts, fvv, optimize=True)            
            t1 -= np.einsum('ma,mi->ia', self.ts, foo, optimize=True)
            t1 += np.einsum('imae,me->ia', self.td, fov, optimize=True)
            t1 -= np.einsum('nf,naif->ia', self.ts, self.gs[o, v, o, v], optimize=True)
            t1 -= 0.5 * np.einsum('imef,maef->ia', self.td, self.gs[o, v, v, v], optimize=True)
            t1 -= 0.5 * np.einsum('mnae,nmei->ia', self.td, self.gs[o, o, v, o], optimize=True)

        elif self.method in ['ccd']:
            t1 = self.ts.copy()

        elif self.method in ['qcisd']:
            foo = self.fs[o,o] ; fvv = self.fs[v,v]
            np.fill_diagonal(foo, 0.0) ; np.fill_diagonal(fvv, 0.0)

            t1 = np.einsum('ic,ac->ia', self.ts, fvv, optimize=True)
            t1 -= np.einsum('ka,ki->ia', self.ts, foo, optimize=True)
            t1 += np.einsum('kc,akic->ia', self.ts, self.gs[v,o,o,v], optimize=True)

            t1 += np.einsum('ikac,kc->ia', self.td, self.fs[o,v], optimize=True)
            t1 += 0.5 * np.einsum('kicd,kacd->ia', self.td, self.gs[o,v,v,v], optimize=True)
            t1 -= 0.5 * np.einsum('klca,klci->ia', self.td, self.gs[o,o,v,o], optimize=True)

            t1 -= 0.5 * np.einsum('ic,klda,lkcd->ia', self.ts, self.td, self.gs[o,o,v,v], optimize=True)
            t1 -= 0.5 * np.einsum('ka,licd,lkcd->ia', self.ts, self.td, self.gs[o,o,v,v], optimize=True)
            t1 += np.einsum('kc,lida,klcd->ia', self.ts, self.td, self.gs[o,o,v,v], optimize=True)

        #doubles amplitudes
        t2 = self.gs[o, o, v, v].copy()

        if self.method in ['ccsd', 'ccsd(t)', 'ccd']:

            #doubles amplitudes Table I (b)
            woooo = self.intermediates('oooo') ; wvvvv = self.intermediates('vvvv') ; wovvo = self.intermediates('ovvo')

            t2 += 0.5 * np.einsum('mnab,mnij->ijab', self.tau(tilde=False), woooo, optimize=True)

            t2 += 0.5 * np.einsum('ijef,abef->ijab', self.tau(tilde=False), wvvvv, optimize=True)

            t = np.einsum('abej,ie->ijab', self.gs[v, v, v, o], self.ts, optimize=True)
            t2 += t - t.transpose(1,0,2,3)

            t = np.einsum('ma,mbij->ijab', self.ts, self.gs[o, v, o, o], optimize=True)
            t2 += -(t - t.transpose(0,1,3,2))

            t = np.einsum('imae,mbej->ijab', self.td, wovvo, optimize=True) 
            t -= np.einsum('ie,ma,mbej->ijab', self.ts, self.ts, self.gs[o, v, v, o], optimize=True)
            t2 += t - t.transpose(0,1,3,2) - t.transpose(1,0,2,3) + t.transpose(1,0,2,3).transpose(0,1,3,2)

            t = np.einsum('ijae,be->ijab', self.td, fvv, optimize=True)
            t -= 0.5 * np.einsum('ijae,mb,me->ijab', self.td, self.ts, fov, optimize=True)
            t2 += t - t.transpose(0,1,3,2)

            t = np.einsum('imab,mj->ijab', self.td, foo, optimize=True)
            t += 0.5 * np.einsum('imab,je,me->ijab', self.td, self.ts, fov, optimize=True)
            t2 += -(t - t.transpose(1,0,2,3))

        elif self.method in ['cc2']:

            foo = self.fs[o,o] ; fvv =self.fs[v,v]
            np.fill_diagonal(foo, 0.0) ; np.fill_diagonal(fvv, 0.0)

            t = -np.einsum('mj,imab->ijab', foo, self.td, optimize=True)
            t2 += t - t.transpose(1,0,2,3)

            t = -np.einsum('ijmb,ma->ijab', self.gs[o, o, o, v], self.ts, optimize=True)
            t2 += t - t.transpose(0,1,3,2)

            t2 += np.einsum('efab,ie,jf->ijab', self.gs[v, v, v, v], self.ts, self.ts, optimize=True)

            t2 += np.einsum('ijmn,ma,nb->ijab', self.gs[o, o, o, o], self.ts, self.ts, optimize=True)

            t = np.einsum('be,ijae->ijab', fvv, self.td, optimize=True)

            t = np.einsum('ejab,ie->ijab', self.gs[v, o, v, v], self.ts, optimize=True)
            t2 += t - t.transpose(1,0,2,3)

            t = np.einsum('ejmn,ma,nb,ie->ijab', self.gs[v, o, o, o], self.ts, self.ts, self.ts, optimize=True)
            t2 += t - t.transpose(1,0,2,3)

            t2 += np.einsum('efmn,ma,nb,ie,jf->ijab', self.gs[v, v, o, o], self.ts, self.ts, self.ts, self.ts, optimize=True)

            t = -np.einsum('ejmb,ie,ma->ijab', self.gs[v, o, o, v], self.ts, self.ts, optimize=True)
            t2 += t - t.transpose(1,0,2,3) - t.transpose(0,1,3,2) + t.transpose(1,0,2,3).transpose(0,1,3,2)

            t = -np.einsum('efmb,ma,ie,jf->ijab', self.gs[v, v, o, v], self.ts, self.ts, self.ts, optimize=True)
            t2 += t - t.transpose(0,1,3,2)

        elif self.method in ['qcisd']:

            t = np.einsum('ic,abcj->ijab', self.ts, self.gs[v,v,v,o], optimize=True)
            t2 += t - t.transpose(1,0,2,3)
            t = -np.einsum('ka,kbij->ijab', self.ts, self.gs[o,v,o,o], optimize=True)
            t2 += t - t.transpose(0,1,3,2)

            t = np.einsum('ijac,bc->ijab', self.td, fvv, optimize=True)
            t2 += t - t.transpose(0,1,3,2)
            t = -np.einsum('ikab,kj->ijab', self.td, foo, optimize=True)
            t2 += t - t.transpose(1,0,2,3)
            t2 += 0.5 * np.einsum('ijcd,abcd->ijab', self.td, self.gs[v,v,v,v], optimize=True)
            t2 += 0.5 * np.einsum('klab,klij->ijab', self.td, self.gs[o,o,o,o], optimize=True)
            t = np.einsum('ikac,kbcj->ijab', self.td, self.gs[o,v,v,o])
            t2 += t - t.transpose(0,1,3,2)  - t.transpose(1,0,2,3) + t.transpose(1,0,3,2)

            t2 += 0.25 * np.einsum('ijcd,klab,klcd->ijab', self.td, self.td, self.gs[o,o,v,v], optimize=True)
            t = np.einsum('ikac,jlbd,klcd->ijab', self.td, self.td, self.gs[o,o,v,v], optimize=True)
            t2 += t - t.transpose(1,0,2,3)
            t = -0.5 * np.einsum('ikdc,ljab,klcd->ijab', self.td, self.td, self.gs[o,o,v,v], optimize=True)
            t2 += t - t.transpose(1,0,2,3)
            t = -0.5 * np.einsum('lkac,ijdb,klcd->ijab', self.td, self.td, self.gs[o,o,v,v], optimize=True)
            t2 += t - t.transpose(0,1,3,2)

        #update amplitudes
        self.ts = t1*self.d_tensors[0]
        self.td = t2*self.d_tensors[1]

    def update_linear_amplitudes(self):
        #linear coupled-cluster - only td and ts.

        o = self.o ; v = self.v

        #doubles amplitdes - no singles for linear     
        foo = self.fs[o,o] ; fvv = self.fs[v,v]
        np.fill_diagonal(foo, 0.0) ; np.fill_diagonal(fvv, 0.0)

        t2 = self.gs[o, o, v, v].copy()

        t = np.einsum('be,ijae->ijab', fvv, self.td, optimize=True)
        t2 += t - t.transpose(0,1,3,2)

        t = -np.einsum('mj,imab->ijab', foo, self.td, optimize=True)
        t2 += t - t.transpose(1,0,2,3)

        t2 += 0.5 * np.einsum('abef,ijef->ijab', self.gs[v, v, v, v], self.td, optimize=True)

        t2 += 0.5 * np.einsum('mnij,mnab->ijab', self.gs[o, o, o, o], self.td, optimize=True)

        t = np.einsum('mbej,imae->ijab', self.gs[o, v, v, o], self.td, optimize=True)
        t2 += t - t.transpose(1,0,2,3) - t.transpose(0,1,3,2) + t.transpose(1,0,2,3).transpose(0,1,3,2)

        #block for adding 's' to lccd
        if self.method == 'lccsd':

            t1 = self.fs[o,v].copy()

            t1 += np.einsum('me,imae->ia', self.fs[o,v], self.td, optimize=True)

            t1 += 0.5 * np.einsum('efam,imef->ia', self.gs[v, v, v, o], self.td, optimize=True)

            t1 -= 0.5 * np.einsum('iemn,mnae->ia', self.gs[o, v, o, o], self.td, optimize=True)

            t1 += np.einsum('ae,ie->ia', fvv, self.ts, optimize=True)

            t1 -= np.einsum('mi,ma->ia', foo, self.ts, optimize=True)

            t1 += np.einsum('ieam,me->ia', self.gs[o, v, v, o], self.ts, optimize=True)

            t = np.einsum('ejab,ie->ijab', self.gs[v, o, v, v], self.ts, optimize=True)
            t2 += t - t.transpose(1,0,2,3)

            t = -np.einsum('ijmb,ma->ijab', self.gs[o, o, o, v], self.ts, optimize=True)
            t2 += t - t.transpose(0,1,3,2)

            self.ts = t1*self.d_tensors[0]

        self.td = t2*self.d_tensors[1]


    def cluster_energy(self):
        #compute the coupled-cluster energy correction

        o = self.o ; v = self.v

        e = 1.0 * np.einsum('ia,ia->', self.fs[o,v], self.ts, optimize=True)

        e += 0.25 * np.einsum('ijab,ijab->', self.gs[o, o, v, v], self.td, optimize=True)

        #non-linear term
        if not self.method in ['lccd', 'lccsd', 'qcisd']:
            e += 0.5 * np.einsum('ijab,ia,jb->', self.gs[o, o, v, v], self.ts, self.ts, optimize=True)

        return e

    def perturbative_triples(self):
        #compute the perturbative triples contribution

        o = self.o ; v = self.v    

        #cyclic permutation list
        cyclic = {'ijkbac':-1,'ijkcba':-1,'jikabc':-1,'jikbac':+1,\
                  'jikcba':+1,'kjiabc':-1,'kjibac':+1,'kjicba':+1}

        #disconnected triples amplitudes
        t = np.einsum('ia,jkbc->ijkabc', self.ts, self.gs[o, o, v, v], optimize=True)
        disconnected_tt = t.copy()
        for i in cyclic:
            disconnected_tt += cyclic[i] * np.einsum('ijkabc->' + i, t, optimize=True)

        #connected triples amplitudes
        t =  np.einsum('jkae,eibc->ijkabc', self.td, self.gs[v, o, v, v], optimize=True)
        t -= np.einsum('imbc,majk->ijkabc', self.td, self.gs[o, v, o, o], optimize=True)
        connected_tt = t.copy()
        for i in cyclic:
            connected_tt += cyclic[i] * np.einsum('ijkabc->' + i, t, optimize=True)

        tt = (disconnected_tt + connected_tt) * self.d_tensors[2]

        perturbation_energy = np.einsum('ijkabc,ijkabc->', connected_tt, tt) / 36.0

        #add triples perturbation correction to energy dictionary
        self.energy['pt'] = perturbation_energy


    def iterator(self, func):
        #consistent field iterations

        #initiate the amplitudes
        self.initialise_amplitudes()

        #initialise diis buffers
        diis = diis_c(6, [self.ts, self.td])

        cycle_energy = [self.cluster_energy()]

        for cycle in range(self.cycle_limit):   

            #store pre-update amplitudes
            diis.refresh_store([self.ts, self.td])

            func()

            #calculate current cycle energy
            cycle_energy.append(self.cluster_energy()) 

            #test convergence
            delta_energy = np.abs(cycle_energy[-2] - cycle_energy[-1])
            if delta_energy < self.convergence:
                self.converged = True
                return {'cc': cycle_energy[-1], 'mp2': self.mp2, 'eHF':self.HFenergy}
                break
            else:
                if self.verbose: print('cycle = {:>3d}  energy = {:>15.10f}   \u0394E = {:>12.10f} '.format(cycle, cycle_energy[-1], delta_energy))
                del cycle_energy[0]

            #diis build extrapolated amplitudes
            self.ts, self.td = diis.build([self.ts, self.td])

class spinMO(object):
    #class to provide integrals in the spin molecular basis

    def __init__(self, e, eri, c, f):
        #instance variables
        self.e = e
        self.eri = eri
        self.c = c
        self.f = f

        self.gs = self.gMOspin(self.e, self.c, self.eri, self.c.shape[0])
        self.fs = self.fMOspin(self.f, self.c)

    def gMOspin(self, e, c, eri, nbf):
        #construct MO spin eri - eri is linear array

        def iEri(i,j,k,l):
            #index into the four-index eri integrals
            p = max(i*(i+1)/2 + j, j*(j+1)/2 + i)
            q = max(k*(k+1)/2 + l, l*(l+1)/2 + k)
            return  int(max(p*(p+1)/2 + q, q*(q+1)/2 + p))

        #get 4 index eri and spinblock to spin basis
        g = np.zeros((nbf,nbf,nbf,nbf))
        for i in range(nbf):
            for j in range(nbf):
                for k in range(nbf):
                    for l in range(nbf):
                        g[i,j,k,l] = eri[iEri(i,j,k,l)]
     
        spinBlock = np.kron(np.eye(2), np.kron(np.eye(2), g).T)
        g = spinBlock.transpose(0,2,1,3) - spinBlock.transpose(0,2,3,1)

        #prepare orbital energies
        eps = np.concatenate((e,e), axis=0)
        cs = np.block([
                     [c, np.zeros_like(c)],
                     [np.zeros_like(c), c]])
        cs = cs[:, eps.argsort()]
        eps = np.sort(eps)

        #eri to MO
        g = np.einsum('pQRS,pP->PQRS', np.einsum('pqRS,qQ->pQRS', np.einsum('pqrS,rR->pqRS', np.einsum('pqrs,sS->pqrS', \
        g, cs, optimize=True), cs, optimize=True), cs, optimize=True), cs, optimize=True)

        return g

    def fMOspin(self, f, c):
        #construct MO spin fock

        cs = np.kron(c, np.eye(2))
        fs = np.dot(cs.T, np.dot(np.kron(f, np.eye(2)), cs))

        return fs

class ccsdLambda(object):
    #class to provide coupled cluster facilities - lambda
    #ccsd reference  compute the F and W intermediates from J. Gauss and J. F. Stanton: Coupled-cluster calculations of 
    #Nuclear magnetic resonance chemical shifts - J. Chem. Phys., Vol. 103, No. 9, 1 September 1995

    o = None ; v = None
    d_tensors = []
    HFenergy = 0.0

    def __init__(self, fs, gs, e, data):
        #initialise with spin fock eris and amplitudes

        print('..running CCSD')
        self.cc = coupledCluster(fs, gs, e, data)
        print('--running Lambda')
        self.method, self.electrons, self.cycle_limit, self.convergence, self.verbose = data.values()
        self.converged = False
        self.e = e
        self.fs = fs
        self.gs = gs

        #get the lambda initial amplitudes
        self.energy = self.iterator()

    def initialise_amplitudes(self):
        #initialisation 

        nocc = self.electrons
        n = np.newaxis
        o = slice(None, nocc)
        v = slice(nocc, None)

        eps = np.kron(self.e, np.ones(2))

        #d tensors
        ds = 1.0 / (eps[o, n] - eps[n, v] )
        dd = 1.0 / (eps[o, n, n, n] + eps[n, o, n, n] - eps[n, n, v, n] - eps[n, n, n, v] )
 
        self.d_tensors = [ds.transpose(1,0), dd.transpose(2,3,0,1)]

        #initial amplitudes
        self.ts = self.cc.ts ; self.td = self.cc.td
        self.ls = np.zeros_like(self.ts).transpose(1,0) ; self.ld = self.td.transpose(2,3,0,1)

        #Hartree-Fock energy
        self.HFenergy = np.einsum('ii', self.fs[o, o]) - 0.5 * np.einsum('ijij', self.gs[o, o, o, o])

        #class variables
        self.o = o ; self.v = v

    def intermediates(self, _slice, tilde=False):
        #lambda intermedates from Table III(b)

        if not _slice in ['oo','vv','ov','oooo','vvvv','ovvo','ooov','vovv','ovoo','vvvo','OO','VV']:
            print('no pre-evaluated slice [', _slice, '] - transpose axes')
            exit()

        o = self.o ; v = self.v

        fov = self.cc.intermediates('ov', tilde=True).copy()

        if not tilde:
            if _slice == 'vv':
                im = self.cc.intermediates('vv', tilde=True)
                im -= 0.5 * np.einsum('me,ma->ae', fov, self.ts, optimize=True)

            if _slice == 'oo':
                im = self.cc.intermediates('oo', tilde=True)
                im += 0.5 * np.einsum('me,ie->mi', fov, self.ts, optimize=True)

            if _slice == 'ov':
                im = fov

            if _slice == 'oooo':
                im  = self.cc.intermediates('oooo', tilde=True)
                im += 0.25 * np.einsum('ijef,mnef->mnij', self.cc.tau(tilde=False), self.gs[o, o, v, v], optimize=True)

            if _slice == 'vvvv':
                im = self.cc.intermediates('vvvv', tilde=True)
                im += 0.25 * np.einsum('mnab,mnef->abef', self.cc.tau(tilde=False), self.gs[o, o, v, v], optimize=True)

            if _slice == 'ovvo':
                im = self.cc.intermediates('ovvo', tilde=True)
                im -= 0.5 * np.einsum('jnfb,mnef->mbej', self.td, self.gs[o, o, v, v], optimize=True)

            if _slice == 'ooov':
                im =  self.gs[o, o, o, v].copy()
                im += np.einsum('if,mnfe->mnie', self.ts, self.gs[o, o, v, v], optimize=True)

            if _slice == 'vovv':
                im = self.gs[v, o, v, v].copy()
                im += np.einsum('na,mnef->amef', self.ts, self.gs[o, o, v, v], optimize=True)

            if _slice == 'ovoo':
                im = self.gs[o, v, o, o].copy()
                im -= np.einsum('me,ijbe->mbij', fov, self.td, optimize=True)
                im -= np.einsum('nb,mnij->mbij', self.ts, self.intermediates('oooo', tilde=False), optimize=True)
                im += 0.5 * np.einsum('ijef,mbef->mbij', self.cc.tau(tilde=False), self.gs[o, v, v, v], optimize=True)
                t = np.einsum('jnbe,mnie->mbij', self.td, self.gs[o, o, o, v], optimize=True)
                im += t - t.transpose(0,1,3,2)
                t = np.einsum('ie,mbej->mbij', self.ts, self.gs[o, v, v, o], optimize=True)
                t -= np.einsum('ie,njbf,mnef->mbij', self.ts, self.td, self.gs[o, o, v, v], optimize=True)
                im += t - t.transpose(0,1,3,2)

            if _slice == 'vvvo':
                im = self.gs[v, v, v, o].copy()
                im -= np.einsum('me,miab->abei', fov, self.td, optimize=True)
                im += np.einsum('if,abef->abei', self.ts, self.intermediates('vvvv', tilde=False), optimize=True)
                im += 0.5 * np.einsum('mnab,mnei->abei', self.cc.tau(tilde=False), self.gs[o, o, v, o], optimize=True)
                t = -np.einsum('miaf,mbef->abei', self.td, self.gs[o, v, v, v], optimize=True)
                im += t - t.transpose(1,0,2,3)
                t = -np.einsum('ma,mbei->abei', self.ts, self.gs[o, v, v, o], optimize=True)
                t += np.einsum('ma,nibf,mnef->abei', self.ts, self.td, self.gs[o, o, v, v], optimize=True)
                im += t - t.transpose(1,0,2,3)

            if _slice == 'VV':
                im = -0.5 * np.einsum('afmn,mnef->ae', self.ld, self.td, optimize=True)

            if _slice == 'OO':
                im = 0.5 * np.einsum('efin,mnef->mi', self.ld, self.td, optimize=True)

            return im

    def update_amplitudes(self):
        #compute the next cycle amplitudes

        o = self.o ; v = self.v

        #singles lambda Table II (a)
        fov = self.intermediates('ov', tilde=False) ; foo = self.intermediates('oo', tilde=False)
        fvv = self.intermediates('vv', tilde=False) ; wovvo = self.intermediates('ovvo', tilde=False)
        wvvvo = self.intermediates('vvvo', tilde=False) ; wovoo = self.intermediates('ovoo', tilde=False)
        wvovv = self.intermediates('vovv', tilde=False) ; wooov = self.intermediates('ooov', tilde=False)
        woooo = self.intermediates('oooo', tilde=False) ; wvvvv = self.intermediates('vvvv', tilde=False)
        goo = self.intermediates('OO', tilde=False) ; gvv = self.intermediates('VV', tilde=False)

        l1 = fov.copy().transpose(1,0)
        l1 += np.einsum('ei,ae->ai', self.ls, fvv, optimize=True)
        l1 -= np.einsum('am,im->ai', self.ls, foo, optimize=True)
        l1 += np.einsum('em,ieam->ai', self.ls, wovvo, optimize=True)
        l1 += 0.5 * np.einsum('efim,efam->ai', self.ld, wvvvo, optimize=True)
        l1 -= 0.5 * np.einsum('aemn,iemn->ai', self.ld, wovoo, optimize=True)
        l1 -= np.einsum('ef,eifa->ai', gvv, wvovv, optimize=True)
        l1 -= np.einsum('mn,mina->ai', goo, wooov, optimize=True)

        #doubles lambda Table II (b)

        l2 = self.gs[v, v, o, o].copy()

        t = np.einsum('aeij,eb->abij', self.ld, fvv, optimize=True)
        l2 += t - t.transpose(1,0,2,3)

        t = -np.einsum('abim,jm->abij', self.ld, foo, optimize=True)
        l2 += t - t.transpose(0,1,3,2)

        l2 += 0.5 * np.einsum('abmn,ijmn->abij', self.ld, woooo, optimize=True)

        l2 += 0.5 * np.einsum('efab,efij->abij', wvvvv, self.ld, optimize=True)

        t = np.einsum('ei,ejab->abij', self.ls, wvovv, optimize=True)
        l2 += t - t.transpose(0,1,3,2)

        t = -np.einsum('am,ijmb->abij', self.ls, wooov, optimize=True)
        l2 += t - t.transpose(1,0,2,3)

        t = np.einsum('aeim,jebm->abij', self.ld, wovvo, optimize=True)
        l2 += t - t.transpose(1,0,2,3) - t.transpose(0,1,3,2) + t.transpose(1,0,2,3).transpose(0,1,3,2)

        t = np.einsum('ai,jb->abij', self.ls, fov)
        l2 += t - t.transpose(1,0,2,3) - t.transpose(0,1,3,2) + t.transpose(1,0,2,3).transpose(0,1,3,2)

        t = np.einsum('ijae,be->abij', self.gs[o, o, v, v], gvv, optimize=True)
        l2 += t - t.transpose(1,0,2,3)

        t = -np.einsum('imab,mj->abij', self.gs[o, o, v, v], goo, optimize=True)
        l2 += t - t.transpose(0,1,3,2)

        self.ls = l1*self.d_tensors[0] ; self.ld = l2*self.d_tensors[1]

    def oprdm(self):
        #one-particle reduced density matrix

        n = sum(self.ls.shape)
        gamma = np.zeros((n, n))
        o = self.o ; v = self.v

        gamma[v, o] = np.einsum('ai->ai', self.ls, optimize=True)

        gamma[o, v]  =  np.einsum('ia->ia', self.cc.ts, optimize=True)
        gamma[o, v] += np.einsum('bj,ijab->ia', self.ls, self.cc.td, optimize=True)
        gamma[o, v] -= np.einsum('bj,ja,ib->ia', self.ls, self.cc.ts, self.cc.ts, optimize=True)
        gamma[o, v] -= 0.5 * np.einsum('bcjk,ikbc,ja->ia', self.ld, self.cc.td, self.cc.ts, optimize=True)    
        gamma[o, v] -= 0.5 * np.einsum('bcjk,jkac,ib->ia', self.ld, self.cc.td, self.cc.ts, optimize=True)

        gamma[v, v]  =  np.einsum('ai,ib->ab', self.ls, self.cc.ts, optimize=True)
        gamma[v, v] += 0.5 * np.einsum('acij,ijbc->ab', self.ld, self.cc.td, optimize=True)

        gamma[o, o]  = -np.einsum('aj,ia->ij', self.ls, self.cc.ts, optimize=True)
        gamma[o, o] -= 0.5 * np.einsum('abjk,ikab->ij', self.ld, self.cc.td, optimize=True)
        gamma[o, o] += np.einsum('ij->ij', np.eye(self.ls.shape[1]), optimize=True)

        return gamma

    def lambda_energy(self):

        o = self.o ; v = self.v

        e = np.einsum('ia,ai->', self.fs[o,v], self.ls, optimize=True)

        e += 0.25 * np.einsum('abij,abij->', self.ld, self.gs[v, v, o, o], optimize=True)

        return e

    def lagrangian_energy(self):

        o = self.o ; v = self.v

        #diagonals of Fock matrix
        foo = np.diag(np.diag(self.cc.fs[o,o])) 
        fvv = np.diag(np.diag(self.cc.fs[v,v])) 

        #singles amplitudes recalculated
        t1  = self.cc.ts/self.cc.d_tensors[0]
        t1 -= np.einsum('ma,mi->ia', self.ts, foo, optimize=True)
        t1 += np.einsum('ie,ae->ia', self.ts, fvv, optimize=True)

        #doubles amplitudes recalculated
        t2 = self.td/self.cc.d_tensors[1]
        t  = np.einsum('imab,mj->ijab', self.td, foo, optimize=True)
        t2 += -(t - t.transpose(1,0,2,3))
        t  = np.einsum('ijae,be->ijab', self.td, fvv, optimize=True)
        t2 += t - t.transpose(0,1,3,2)

        lagrange = self.cc.energy['cc']
        lagrange += np.einsum('ai,ia->', self.ls, t1, optimize=True)
        lagrange += np.einsum('abij,ijab->', self.ld, t2, optimize=True)

        return lagrange

    def lambda_perturbative_triples(self):
        #perturbative triples correction to lambda CCSD

        o = self.o ; v = self.v

        #permutations are: i/jk a/bc, i/jk c/ab    and   k/ij a/bc
        permutation_set = [{'ijkabc':+1,'ijkbac':-1,'ijkcba':-1,'jikabc':-1,'jikbac':+1,'jikcba':+1,'kjiabc':-1,'kjibac':+1,'kjicba':+1},
                           {'ijkcab':+1,'ijkacb':-1,'ijkbac':-1,'jikcab':-1,'jikacb':+1,'jikbac':+1,'kjicab':-1,'kjiacb':+1,'kjibac':+1},
                           {'kijabc':+1,'ikjabc':-1,'jikabc':-1,'kijbac':-1,'ikjbac':+1,'jikbac':+1,'kijcba':-1,'ikjcba':+1,'jikcba':+1}]

        #triples orbital energy divisor
        dt = self.cc.d_tensors[2]

        #triples correction
        lll = np.zeros_like(dt)

        #lambda triples
        t = np.einsum('dkbc,adij->ijkabc', self.gs[v, o, v, v], self.ld, optimize=True)
        for i in permutation_set[2]:
            lll += permutation_set[2][i] * np.einsum('ijkabc->' + i, t, optimize=True)

        t = np.einsum('jklc,abil->ijkabc', self.gs[o, o, o, v], self.ld, optimize=True)
        for i in permutation_set[1]:
            lll -= permutation_set[1][i] * np.einsum('ijkabc->' + i, t, optimize=True)

        t = np.einsum('ai,bcjk->ijkabc', self.ls, self.gs[v, v, o, o], optimize=True)
        for i in permutation_set[0]:
            lll += permutation_set[0][i] * np.einsum('ijkabc->' + i, t, optimize=True)

        t = np.einsum('ia,bcjk->ijkabc', self.fs[o,v], self.ld, optimize=True)
        for i in permutation_set[0]:
            lll += permutation_set[0][i] * np.einsum('ijkabc->' + i, t, optimize=True)


        #t triples
        ttt = np.zeros_like(dt)

        t = np.einsum('bcdk,ijad->ijkabc', self.gs[v, v, v, o], self.cc.td, optimize=True)
        for i in permutation_set[2]:
            ttt += permutation_set[2][i] * np.einsum('ijkabc->' + i, t, optimize=True)

        t = np.einsum('lcjk,ilab->ijkabc', self.gs[o, v, o, o], self.cc.td, optimize=True)
        for i in permutation_set[1]:
            ttt -= permutation_set[1][i] * np.einsum('ijkabc->' + i, t, optimize=True)

        ttt *= dt

        lambda_correction = np.einsum('ijkabc,ijkabc->', lll, ttt, optimize=True)/36.0
        
        return lambda_correction

    def iterator(self):
        #iterate lambda amplitudes to convergence

        self.initialise_amplitudes()

        #initialise diis buffers
        diis = diis_c(6, [self.ls, self.ld])

        cycle_energy = [self.lambda_energy()]

        for cycle in range(self.cycle_limit):

            #store pre-update amplitudes
            diis.refresh_store([self.ls, self.ld])

            self.update_amplitudes()
            cycle_energy.append(self.lambda_energy())

            #test convergence
            delta_energy = np.abs(cycle_energy[-2] - cycle_energy[-1])
            if delta_energy < self.convergence:
                self.converged = True
                return_energies = {'cc': cycle_energy[-1], 'eHF':self.HFenergy, 'lagrange':self.lagrangian_energy()}
                if '(t)' in self.method:
                    return_energies['pt'] = self.cc.energy['pt']
                    return_energies['pl'] = self.lambda_perturbative_triples()
                return return_energies
                break
            else:
                if self.verbose: print('cycle = {:>3d}  energy = {:>15.10f}   \u0394E = {:>12.10f} '.format(cycle, cycle_energy[-1], delta_energy))
                del cycle_energy[0]

            #diis build extrapolated amplitudes
            self.ls, self.ld = diis.build([self.ls, self.ld])

def fastCoupledCluster(type, fock, eps, c, eri, nuclearRepulsion, data):
    #einsum coupled-cluster

    #get fock and eri in molecular spin basis from spinMO class
    mo = spinMO(eps, eri, c, fock)
    gs = mo.gs
    fs = mo.fs

    cc_data = {'method':type,'electrons':data['electrons'], 'cycle_limit': data['cycle_limit'], \
               'precision':data['precision'], 'verbose':False}

    #call coupled-cluster class
    if type in ['ccd', 'ccsd', 'ccsd(t)', 'cc2', 'lccd', 'lccsd', 'qcisd']:
        cc = coupledCluster(fs, gs, eps, cc_data)

        if cc.converged:
            cc.energy['nuclear'] = nuclearRepulsion
            energy = cc.energy
            postSCF([type, cc.energy], '+c')

    elif type in ['lambda']:
        cc_data['method'] = 'ccsd'
        l = ccsdLambda(fs, gs, eps, cc_data)
        if l.converged:
            l.energy['nuclear'] = nuclearRepulsion
            energy = l.energy
            postSCF([type, energy], '+c')   
             
    return energy

class eom_ccsd(object):
    #class for equation of motion coupled-cluster singles and doubles

    def __init__(self, cc, roots=[1, 50], partitioned=False):
        self.cc = cc
        self.roots = roots
        self.partitioned = partitioned

        self.e, self.v = self.do_eom_ccsd(cc)

        from ci import ciDegeneracy
        e = np.sort( self.e * getConstant('hartree->eV'))
        excitation_tuples = ciDegeneracy(e)

        self.excitations = []
        for excitation in excitation_tuples:
            if excitation[0] > self.roots[0] and excitation[0] < self.roots[1]: 
                self.excitations.append(excitation)


    def intermediates(self, cc, _slice, tilde=False):
        #intermediates for EOM

        o = cc.o ; v = cc.v
        fov = cc.intermediates('ov', tilde=True).copy()

        if not tilde:

            if _slice == 'vv':
                im = cc.intermediates('vv', tilde=True) + np.diag(np.diag(cc.fs[v,v]))
                im -= 0.5 * np.einsum('me,ma->ae', fov, cc.ts, optimize=True)

            if _slice == 'oo':
                im = cc.intermediates('oo', tilde=True) + np.diag(np.diag(cc.fs[o,o]))
                im += 0.5 * np.einsum('me,ie->mi', fov, cc.ts, optimize=True)

            if _slice == 'ov':
                im = fov

            if _slice == 'oovv':
                im = cc.gs[o,o,v,v]

            if _slice == 'oooo':
                im  = cc.intermediates('oooo', tilde=True)
                im += 0.25 * np.einsum('ijef,mnef->mnij', cc.tau(tilde=False), cc.gs[o, o, v, v], optimize=True)

            if _slice == 'vvvv':
                im = cc.intermediates('vvvv', tilde=True)
                im += 0.25 * np.einsum('mnab,mnef->abef', cc.tau(tilde=False), cc.gs[o, o, v, v], optimize=True)

            if _slice == 'ovvo':
                im = cc.intermediates('ovvo', tilde=True)
                im -= 0.5 * np.einsum('jnfb,mnef->mbej', cc.td, cc.gs[o, o, v, v], optimize=True)

            if _slice == 'ooov':
                im =  cc.gs[o, o, o, v].copy()
                im += np.einsum('if,mnfe->mnie', cc.ts, cc.gs[o, o, v, v], optimize=True)

            if _slice == 'vovv':
                im = cc.gs[v, o, v, v].copy()
                im += np.einsum('na,mnef->amef', cc.ts, cc.gs[o, o, v, v], optimize=True)

            if _slice == 'ovoo':
                im = cc.gs[o, v, o, o].copy()
                im -= np.einsum('me,ijbe->mbij', fov, cc.td, optimize=True)
                im -= np.einsum('nb,mnij->mbij', cc.ts, self.intermediates(cc, 'oooo', tilde=False), optimize=True)
                im += 0.5 * np.einsum('ijef,mbef->mbij', cc.tau(tilde=False), cc.gs[o, v, v, v], optimize=True)
                t = np.einsum('jnbe,mnie->mbij', cc.td, cc.gs[o, o, o, v], optimize=True)
                im += t - t.transpose(0,1,3,2)
                t = np.einsum('ie,mbej->mbij', cc.ts, cc.gs[o, v, v, o], optimize=True)
                t -= np.einsum('ie,njbf,mnef->mbij', cc.ts, cc.td, cc.gs[o, o, v, v], optimize=True)
                im += t - t.transpose(0,1,3,2)

            if _slice == 'vvvo':
                im = cc.gs[v, v, v, o].copy()
                im -= np.einsum('me,miab->abei', fov, cc.td, optimize=True)
                im += np.einsum('if,abef->abei', cc.ts, self.intermediates(cc, 'vvvv', tilde=False), optimize=True)
                im += 0.5 * np.einsum('mnab,mnei->abei', cc.tau(tilde=False), cc.gs[o, o, v, o], optimize=True)
                t = -np.einsum('miaf,mbef->abei', cc.td, cc.gs[o, v, v, v], optimize=True)
                im += t - t.transpose(1,0,2,3)
                t = -np.einsum('ma,mbei->abei', cc.ts, cc.gs[o, v, v, o], optimize=True)
                t += np.einsum('ma,nibf,mnef->abei', cc.ts, cc.td, cc.gs[o, o, v, v], optimize=True)
                im += t - t.transpose(1,0,2,3)

        return im

    def do_eom_ccsd(self, cc):
        #construct the EOM Hamiltonian

        nocc, nvir = self.cc.ts.shape 
        nrot = nocc * nvir

        #kronecker delta
        kd = np.eye(nocc + nvir)

        #short version variables
        o = self.cc.o ; v = self.cc.v
        fs, gs, ts, td = self.cc.fs, self.cc.gs, self.cc.ts, self.cc.td

        #get singles-singles block    

        hss  = np.einsum('ac,ik->iakc', self.intermediates(cc, 'vv'), kd[o,o], optimize=True)
        hss -= np.einsum('ki,ac->iakc', self.intermediates(cc, 'oo'), kd[v,v], optimize=True)
        hss += np.einsum('akic->iakc', self.intermediates(cc, 'ovvo').transpose(1,0,3,2))
        hss = hss.reshape(nrot, nrot).copy()

        #get singles-doubles 
        hsd  = np.einsum('ld,ac,ik->iakcld', self.intermediates(cc, 'ov'), kd[v,v], kd[o,o], optimize=True)
        hsd += 0.5 * np.einsum('alcd,ik->iakcld', self.intermediates(cc, 'vovv'), kd[o,o], optimize=True)
        hsd -= 0.5 * np.einsum('klid,ac->iakcld', self.intermediates(cc, 'ooov'), kd[v,v], optimize=True)
        hsd = hsd.reshape(nrot, nrot*nrot).copy()

        #get doubles-singles 
        t = np.einsum('kaij,bc->iajbkc', self.intermediates(cc, 'ovoo'), kd[v,v], optimize=True)
        hds = t - t.transpose(0,3,2,1,4,5)
        t = np.einsum('abcj,ik->iajbkc', self.intermediates(cc, 'vvvo'), kd[o,o], optimize=True)
        hds += t - t.transpose(2,1,0,3,4,5)
        t = np.einsum('bkec,ijae->iajbkc', self.intermediates(cc, 'vovv'), td, optimize=True)
        hds += t - t.transpose(0,3,2,1,4,5)
        t = -np.einsum('mkjc,imab->iajbkc', self.intermediates(cc, 'ooov'), td, optimize=True)
        hds += t - t.transpose(2,1,0,3,4,5)
        hds = hds.reshape(nrot*nrot, nrot).copy()

        if not self.partitioned:

            t = np.einsum('bc,ad,il,jk->iajbkcld', self.intermediates(cc, 'vv'), kd[v,v], kd[o,o], kd[o,o], optimize=True)
            hdd = t - t.transpose(0,3,2,1,4,5,6,7)
            t = -np.einsum('kj,ad,il,bc->iajbkcld', self.intermediates(cc, 'oo'),  kd[v,v], kd[o,o], kd[v,v], optimize=True)
            hdd += t - t.transpose(2,1,0,3,4,5,6,7)
            hdd += 0.5 * np.einsum('abcd,ik,jl->iajbkcld', self.intermediates(cc, 'vvvv'), kd[o,o], kd[o,o], optimize=True)
            hdd += 0.5 * np.einsum('klij,ac,bd->iajbkcld', self.intermediates(cc, 'oooo'), kd[v,v], kd[v,v], optimize=True)
            t = np.einsum('akic,jl,db->iajbkcld', self.intermediates(cc, 'ovvo').transpose(1,0,3,2), kd[o,o], kd[v,v], optimize=True)
            hdd += t - t.transpose(2,1,0,3,4,5,6,7) - t.transpose(0,3,2,1,4,5,6,7) + t.transpose(2,3,0,1,4,5,6,7) 
            t = -0.5 * np.einsum('lkec,ijeb,ad->iajbkcld', self.intermediates(cc, 'oovv'), td, kd[v,v], optimize=True)
            hdd += t - t.transpose(0,3,2,1,4,5,6,7)
            t = 0.5 * np.einsum('kmcd,miab,jl->iajbkcld', self.intermediates(cc, 'oovv'), td, kd[o,o], optimize=True)
            hdd += t - t.transpose(2,1,0,3,4,5,6,7)

        else:
            t = np.einsum('bc,jk,il,ad->iajbkcld', fs[v,v], kd[o,o], kd[o,o], kd[v,v], optimize=True)
            hdd = t - t.transpose(0,3,2,1,4,5,6,7) 
            t = np.einsum('ki,jl,ad,bc->iajbkcld', fs[o,o], kd[o,o], kd[v,v], kd[v,v], optimize=True) 
            hdd += t - t.transpose(2,1,0,3,4,5,6,7)
        
        hdd = hdd.reshape(nrot*nrot, nrot*nrot).copy()

        #construct Hamiltonian from blocks
        eom_matrix = np.bmat([[hss, hsd],
                              [hds, hdd]])

        #solve dense Hamiltonian
        from scipy.linalg import eig
        e, v = eig(eom_matrix)

        return e.real, v.real

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

    #call coupled-cluster class
    data = {'method':'ccsd','electrons':electrons, 'cycle_limit': 50, 'precision':1e-10, 'verbose':True}
    cc = coupledCluster(fs, gs, e, data)

    if cc.converged:
        cc.energy['nuclear'] = nuclearRepulsion

        print(cc.energy)

    data['method'] = 'ccsd(t)'
    l = ccsdLambda(fs, gs, e, data)
    if l.converged:
        l.energy['nuclear'] = nuclearRepulsion

        print(l.energy)

    #EOM-CCSD
    eom = eom_ccsd(cc, roots=[4, 41], partitioned=False)

    print('root   energy (eV)   multiplicity\n---------------------------------')
    for i, root in enumerate(eom.excitations):
        print('{:<2d}      {:<10.5f}        {:<s}'.format(i, root[0], root[1]))
