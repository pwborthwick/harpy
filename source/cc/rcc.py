from __future__ import division
import numpy as np
from diis import diis_c
from atom import getConstant


'''
Restricted (spin-summed) Coupled-Cluster for CCD and CCSD
'''

class restrictedCoupledCluster(object):
    #restricted (spin-summed) Coupled-Cluster

    def __init__(self, f, g, e, data):
        #initialise with spin fock eris and amplitudes

        self.f = f
        self.g = g
        self.e = e
        self.ts = None
        self.td = None
        self.method, self.electrons, self.cycle_limit, self.convergence, self.verbose, self.HFenergy = data.values()
        self.converged = False

        #check method implemented
        if not self.method in ['ccd', 'ccsd'] : return None

        self.initialise_amplitudes()
        func_amplitude = self.update_amplitudes

        self.mp2 = 0.0  

        self.energy = self.iterator(func_amplitude)

    def initialise_amplitudes(self):
        #set initial amplitudes

        nocc = self.electrons//2
        n = np.newaxis
        o = slice(None, nocc)
        v = slice(nocc, None)

        eps = self.e
        #d tensors
        ds = 1.0 / (eps[o, n] - eps[n, v] )
        dd = 1.0 / (eps[o, n, n, n] + eps[n, o, n, n] - eps[n, n, v, n] - eps[n, n, n, v] )
        dt = 1.0 / (eps[o, n, n, n, n, n] + eps[n, o, n, n, n, n] + eps[n, n, o, n, n, n]- \
                    eps[n, n, n, v, n, n] - eps[n, n, n, n, v, n] - eps[n, n, n, n, n, v] )
  
        #class variables 
        self.d_tensors = [ds, dd, dt]
        self.o = o ; self.v = v

        #initial amplitudes and mp2 energy
        self.ts = np.zeros_like(ds) ; self.td = self.g[o, o, v, v] * dd
        self.mp2 = self.cluster_energy()

    def update_amplitudes(self):
        #compute the next cycle amplitudes

        o = self.o ; v = self.v

        np.fill_diagonal(self.f, 0.0)

        #singles amplitudes
        if self.method in ['ccd', 'ccsd']:

            #no singles 
            t1 = np.zeros_like(self.ts)

            #doubles amplitudes
            t2 = self.g[o,o,v,v].copy()
            t2 += np.einsum('ijef,efab->ijab', self.td, self.g[v,v,v,v], optimize=True)
            t2 += np.einsum('mnab,ijmn->ijab', self.td, self.g[o,o,o,o], optimize=True)
            t2 += np.einsum('mjae,infb,nmef->ijab', self.td, self.td, self.g[o,o,v,v],optimize=True)
            t2 -= 2.0*np.einsum('ijae,mnbf,mnef->ijab', self.td, self.td, self.g[o,o,v,v], optimize=True)
            t2 += np.einsum('ijae,nmbf,mnef->ijab', self.td, self.td, self.g[o,o,v,v], optimize=True)
            t2 -= 2.0*np.einsum('miae,jnbf,mnef->ijab', self.td, self.td, self.g[o,o,v,v], optimize=True)
            t2 += np.einsum('miae,njbf,mnef->ijab', self.td, self.td, self.g[o,o,v,v], optimize=True)
            t2 += np.einsum('imae,njbf,nmef->ijab', self.td, self.td, self.g[o,o,v,v],optimize=True)
            t2 += 4.0*np.einsum('imae,jnbf,mnef->ijab', self.td, self.td, self.g[o,o,v,v], optimize=True)
            t2 -= 2.0*np.einsum('imae,njbf,mnef->ijab', self.td, self.td, self.g[o,o,v,v], optimize=True)
            t2 += np.einsum('mnae,ijfb,mnef->ijab', self.td, self.td, self.g[o,o,v,v], optimize=True)
            t2 -= 2.0*np.einsum('nmae,ijfb,mnef-> ijab', self.td, self.td, self.g[o,o,v,v], optimize=True)
            t2 += np.einsum('ijfe,nmab,mnef->ijab', self.td, self.td, self.g[o,o,v,v], optimize=True)
            t2 += np.einsum('miae,njfb,nmef->ijab', self.td, self.td, self.g[o,o,v,v], optimize=True)
            t2 -= 2.0*np.einsum('imae,jnbf,nmef->ijab', self.td, self.td, self.g[o,o,v,v], optimize=True)

            t =  np.einsum('eb,ijae->ijab', self.f[v,v], self.td, optimize=True)
            t -= np.einsum('im,mjab->ijab', self.f[o,o], self.td, optimize=True)
            t += 2.0*np.einsum('jmbe,miea->ijab', self.td, self.g[o,o,v,v], optimize=True)
            t -= np.einsum('mjbe,miea->ijab', self.td, self.g[o,o,v,v], optimize=True)
            t -= np.einsum('imeb,jema->ijab', self.td, self.g[o,v,o,v], optimize=True)
            t -= np.einsum('imae,jemb->ijab', self.td, self.g[o,v,o,v], optimize=True)
            t += np.einsum('jmef,inab,mnef->ijab', self.td, self.td, self.g[o,o,v,v], optimize=True)
            t -= 2.0*np.einsum('mjef,inab,mnef->ijab', self.td, self.td, self.g[o,o,v,v], optimize=True)
            
            if self.method == 'ccd': t2 += t + t.transpose(1,0,3,2)

        if self.method in ['ccsd']:

            #singles CCSD
            t1 = self.f[o,v].copy()
            t1 -= np.einsum('im,ma->ia', self.f[o,o], self.ts, optimize=True)
            t1 += np.einsum('ea,ie->ia', self.f[v,v], self.ts, optimize=True)
            t1 -= np.einsum('me,ie,ma->ia', self.f[o,v], self.ts, self.ts, optimize=True)
            t1 += 2.0*np.einsum('me,imae->ia', self.f[o,v], self.td, optimize=True)
            t1 -= np.einsum('me,miae->ia', self.f[o,v], self.td, optimize=True)
            t1 -= np.einsum('me,iema->ia', self.ts, self.g[o,v,o,v], optimize=True)
            t1 += 2.0*np.einsum('me,miea->ia', self.ts, self.g[o,o,v,v], optimize=True)
            t1 -= 2.0*np.einsum('me,na,nmie->ia', self.ts, self.ts, self.g[o,o,o,v], optimize=True)
            t1 -= np.einsum('me,if,mafe->ia', self.ts, self.ts, self.g[o,v,v,v], optimize=True)
            t1 += 2.0*np.einsum('me,if,maef->ia', self.ts, self.ts, self.g[o,v,v,v], optimize=True)
            t1 += np.einsum('me,na,mnie->ia', self.ts, self.ts, self.g[o,o,o,v], optimize=True)
            t1 += np.einsum('me,if,na,nmef->ia', self.ts, self.ts, self.ts, self.g[o,o,v,v], optimize=True)
            t1 -= 2.0*np.einsum('me,if,na,mnef->ia', self.ts, self.ts, self.ts, self.g[o,o,v,v], optimize=True)
            t1 -= np.einsum('mief,mafe->ia', self.td, self.g[o,v,v,v], optimize=True)
            t1 += 2.0*np.einsum('imef,mafe->ia', self.td, self.g[o,v,v,v], optimize=True)
            t1 -= 2.0*np.einsum('mnae,mnie->ia', self.td, self.g[o,o,o,v], optimize=True)
            t1 += np.einsum('nmae,mnie->ia', self.td, self.g[o,o,o,v], optimize=True)
            t1 -= 2.0*np.einsum('me,inaf,nmef->ia', self.ts, self.td, self.g[o,o,v,v], optimize=True)
            t1 -= 2.0*np.einsum('me,niaf,mnef->ia', self.ts, self.td, self.g[o,o,v,v], optimize=True)
            t1 += np.einsum('me,niaf,nmef->ia', self.ts, self.td, self.g[o,o,v,v], optimize=True)
            t1 -= 2.0*np.einsum('ie,nmaf,nmef->ia', self.ts, self.td, self.g[o,o,v,v], optimize=True)
            t1 += np.einsum('ie,nmaf,mnef->ia', self.ts, self.td, self.g[o,o,v,v], optimize=True)
            t1 -= 2.0*np.einsum('na,imfe,mnef->ia', self.ts, self.td, self.g[o,o,v,v], optimize=True)
            t1 += np.einsum('na,imef,mnef->ia', self.ts, self.td, self.g[o,o,v,v], optimize=True)
            t1 += 4.0*np.einsum('me,inaf,mnef->ia', self.ts, self.td, self.g[o,o,v,v], optimize=True)

            #doubles CCSD
            t2 += np.einsum('ie,jf,efab->ijab', self.ts, self.ts, self.g[v,v,v,v], optimize=True)
            t2 += np.einsum('ma,nb,ijmn->ijab', self.ts, self.ts, self.g[o,o,o,o], optimize=True)
            t2 -= np.einsum('ie,jf,ma,mbef->ijab', self.ts, self.ts, self.ts, self.g[o,v,v,v], optimize=True)
            t2 -= np.einsum('ie,jf,mb,mafe->ijab', self.ts, self.ts, self.ts, self.g[o,v,v,v], optimize=True)
            t2 += np.einsum('ie,ma,nb,nmje->ijab', self.ts, self.ts, self.ts, self.g[o,o,o,v], optimize=True)
            t2 += np.einsum('je,ma,nb,mnie->ijab', self.ts, self.ts, self.ts, self.g[o,o,o,v], optimize=True)
            t2 += np.einsum('ie,jf,ma,nb,mnef->ijab', self.ts, self.ts, self.ts, self.ts, self.g[o,o,v,v], optimize=True)
            t2 += np.einsum('ie,jf,nmab,nmef->ijab', self.ts, self.ts, self.td, self.g[o,o,v,v], optimize=True)
            t2 += np.einsum('ma,nb,ijfe,nmef->ijab', self.ts, self.ts, self.td, self.g[o,o,v,v], optimize=True)
            
            t -= np.einsum('mb,jima->ijab', self.ts, self.g[o,o,o,v], optimize=True)
            t += np.einsum('je,ieab->ijab', self.ts, self.g[o,v,v,v], optimize=True)
            t += np.einsum('me,ie,mjab->ijab', self.f[o,v], self.ts, self.td, optimize=True)
            t -= np.einsum('me,ma,ijeb->ijab', self.f[o,v], self.ts, self.td, optimize=True)
            t -= np.einsum('ie,ma,mjeb->ijab', self.ts, self.ts, self.g[o,o,v,v], optimize=True)
            t -= np.einsum('ie,mb,jema->ijab', self.ts, self.ts, self.g[o,v,o,v], optimize=True)
            t -= 2.0*np.einsum('nb,imae,nmje->ijab', self.ts, self.td, self.g[o,o,o,v], optimize=True)
            t += np.einsum('nb,miae,nmje->ijab', self.ts, self.td, self.g[o,o,o,v], optimize=True)
            t -= np.einsum('je,imfb,maef->ijab', self.ts, self.td, self.g[o,v,v,v], optimize=True)
            t -= np.einsum('je,miaf,mbfe->ijab', self.ts, self.td, self.g[o,v,v,v], optimize=True)
            t -= np.einsum('je,imaf,mbef->ijab', self.ts, self.td, self.g[o,v,v,v], optimize=True)
            t += np.einsum('je,nmab,nmie->ijab', self.ts, self.td, self.g[o,o,o,v], optimize=True)
            t += np.einsum('nb,imae,mnje->ijab', self.ts, self.td, self.g[o,o,o,v], optimize=True)
            t -= np.einsum('ma,ijfe,mbfe->ijab', self.ts, self.td, self.g[o,v,v,v], optimize=True)
            t += np.einsum('ma,ineb,nmje->ijab', self.ts, self.td, self.g[o,o,o,v], optimize=True)
            t += 2.0*np.einsum('je,imaf,mbfe->ijab', self.ts, self.td, self.g[o,v,v,v], optimize=True)
            t -= np.einsum('me,ijaf,mbfe->ijab', self.ts, self.td, self.g[o,v,v,v], optimize=True)
            t += 2.0*np.einsum('me,ijaf,mbef->ijab', self.ts, self.td, self.g[o,v,v,v], optimize=True)
            t += np.einsum('me,inab,mnje->ijab', self.ts, self.td, self.g[o,o,o,v], optimize=True)
            t -= 2.0*np.einsum('me,inab,nmje->ijab', self.ts, self.td, self.g[o,o,o,v], optimize=True)
            t -= 2.0*np.einsum('me,jf,inab,mnef->ijab', self.ts, self.ts, self.td, self.g[o,o,v,v], optimize=True)
            t += np.einsum('me,jf,inab,nmef->ijab', self.ts, self.ts, self.td, self.g[o,o,v,v], optimize=True)
            t -= 2.0*np.einsum('me,na,ijfb,mnef->ijab', self.ts, self.ts, self.td, self.g[o,o,v,v], optimize=True)
            t += np.einsum('me,na,ijfb,nmef->ijab', self.ts, self.ts, self.td, self.g[o,o,v,v], optimize=True)
            t += np.einsum('ie,ma,njbf,mnef->ijab', self.ts, self.ts, self.td, self.g[o,o,v,v], optimize=True)
            t -= 2.0*np.einsum('ie,ma,jnbf,mnef->ijab', self.ts, self.ts, self.td, self.g[o,o,v,v], optimize=True)
            t += np.einsum('ie,ma,njfb,nmef->ijab', self.ts, self.ts, self.td, self.g[o,o,v,v], optimize=True)
            t += np.einsum('ie,nb,mjaf,mnef->ijab', self.ts, self.ts, self.td, self.g[o,o,v,v], optimize=True)
            
            t2 += t + t.transpose(1,0,3,2)

        self.ts = t1 * self.d_tensors[0]
        self.td = t2 * self.d_tensors[1]

    def cluster_energy(self):
        #compute the coupled-cluster energy correction

        o = self.o ; v = self.v

        e = np.einsum('nemf, mnef -> ', (2.0*self.td.transpose(1,2,0,3) - self.td.transpose(0,2,1,3)), self.g[o,o,v,v], optimize =True)

        if self.method in ['ccsd']:
            e -= np.einsum('ne,mf,mnef->', self.ts, self.ts, self.g[o,o,v,v], optimize=True)
            e += 2.0 * np.einsum('me,me->', self.f[o,v], self.ts, optimize=True)
            e += 2.0 * np.einsum('ne,mf,nmef->', self.ts, self.ts, self.g[o,o,v,v], optimize=True)

        return e


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

if __name__ == '__main__':

    #do an SCF computation
    import rhf
    molAtom, molBasis, molData = rhf.mol([])
    eSCF = rhf.scf(molAtom, molBasis,molData, [])

    #get data for coupled-cluster
    from atom import nuclearRepulsion ; from basis import electronCount
    charge, nuclearRepulsion, electrons = [molData['charge'], nuclearRepulsion(molAtom), electronCount(molAtom, molData['charge'])]
    f, c, e, eri, d, ch = [rhf.fock, rhf.C, rhf.e, rhf.ERI, rhf.density, rhf.coreH]   

    #Hartree-Fock electronic energy
    eHF = float(np.einsum('pq,pq->',d , (f + ch), optimize=True))

    def gMO(c, f, eri, nbf):
        #construct MO eri and Fock - eri is linear array in Chemists notation

        from integral import buildEriMO, expandEri

        g = expandEri(buildEriMO(c, eri), nbf).reshape(nbf, nbf, nbf, nbf).swapaxes(1, 2)
        f = np.einsum('rp,rs,sq->pq', c, f, c, optimize=True)

        return f, g

    #transform to molecular basis
    f, g = gMO(c, f, eri, c.shape[0])

    #call restricted coupled-cluster class
    data = {'method':'ccd','electrons':electrons, 'cycle_limit': 50, 'precision':1e-10, 'verbose':True, 'scf energy':eHF}
    cc = restrictedCoupledCluster(f, g, e, data)

    if cc.converged:
        cc.energy['nuclear'] = nuclearRepulsion

        print(cc.energy)

