from __future__ import division
import numpy as np
from atom import getConstant
from cc.fcc import spinMO

def davidson(matrix, guess_subspace, diagonal, cycles=None, sort_on_absolute=True, tol=1e-8, vectors_per_root=30):
    #asymmetric Davidson iterator
    #this code is based on O Backhouse's solver in psi4numpy (ADC_helper)

    converged = False

    if callable(matrix):
        matvec = matrix
    else:
        matvec = lambda x: np.dot(matrix, x)

    if sort_on_absolute:
        selector = lambda x: np.argsort(np.absolute(x))[:k]
    else:
        selector = lambda x: np.argsort(x)[:k]

    k = guess_subspace.shape[1]
    b = guess_subspace.copy()
    theta = np.zeros((k,))

    if cycles is None:
        cycles = k * 15
    
    for cycle in range(cycles):

        #orthogonalize sub-space set
        b, r = np.linalg.qr(b)
        ex_theta = theta[:k]

        #solve sub-space Hamiltonian
        s = np.zeros_like(b)
        for i in range(b.shape[1]):
            s[:,i] = matvec(b[:,i])

        g = np.dot(b.T, s)
        theta, S = np.linalg.eigh(g)

        #selected biggest eigenvalues (theta) and corresponding vectors (S)
        idx = selector(theta)
        theta = theta[idx]
        S = S[:,idx]

        #augment sub-space
        b_augment = []
        for i in range(k):
            w  = np.dot(s, S[:,i])
            w -= np.dot(b, S[:,i]) * theta[i]
            q = w / (theta[i] - diagonal[i] + 1e-30)
            b_augment.append(q)

        #check for converged roots
        if np.linalg.norm(theta[:k] - ex_theta) < tol:
            converged = True
            b = np.dot(b, S)
            break

        else:
            #collapse sub-space if too large or augment sub-space
            if b.shape[1] >= (k * vectors_per_root):
                b = np.dot(b, S)
                theta = ex_theta
            else:
                b = np.hstack([b, np.column_stack(b_augment)])

    b = b[:, :k]

    return theta, b, converged

class ADC(object):
    #class for Algebraic Diagramatic Construction method for excited states
    from cc.fcc import spinMO

    def __init__(self, mode, rhf, electrons, roots=3, tolerance=1e-8, solve=[1,20]):

        self.mode = mode
        self.roots = roots
        self.tol = tolerance
        self.electrons = electrons

        self.guess_vector_factor, self.vectors_per_root = solve

        #check SCF has been computed
        try: 
            rhf.e
        except AttributeError:
            print('SCF not performed - no integrals')

        self.rhf = rhf

        #set converged flag and do computation
        self.converged = False
        if self.mode == 'ee':
            self.eig_energy, self.eig_vector = self.do_adc_ee()
        elif self.mode in ['ea', 'ip']:
            self.eig_energy, self.eig_vector = self.do_adc_ea_ip()

    def get(self, type):        
        #get quantities 

        if type == 'e': return self.eig_energy
        if type == 'v': return self.eig_vector

    def get_spin_quantities(self):
        #from harpy globals get the spin eri and no energies

        spin = spinMO(self.rhf.e, self.rhf.ERI, self.rhf.C, self.rhf.fock)

        return spin.gs, np.kron(self.rhf.e, np.ones(2)), spin.fs

    def get_spin_metrics(self):
        #compute the ocupations of spin orbitals

        nocc = self.electrons 
        nvir = self.rhf.S.shape[0] * 2 - nocc

        #occupation slices
        n = np.newaxis
        o = slice(None, nocc)
        v = slice(nocc, None)

        return (nocc, nvir, nocc*nvir), (n, o, v)

    def moller_plesset(self, silent=False):
        #do moller-plesset second order

        n, o, v = self.slice

        #d tensors - d singles and d doubles
        ds = self.eps[o, n] - self.eps[n, v]
        dd = self.eps[o, n, n, n] + self.eps[n, o, n, n] - self.eps[n, n, v, n] - self.eps[n, n, n, v]

        #doubles cluster amplitudes
        td = self.gs[o, o, v, v] / dd

        #output the MP2 energy summary
        mp2 = 0.25 * np.einsum('ijab,ijab->', td, self.gs[o, o, v, v]) 

        if not silent:
            print('\nmoller-plesset(2) energy summary')
            print('rhf total energy       {:>16.10f}'.format(self.rhf.SCFenergy))
            print('mp2 correlation energy {:>16.10f}'.format(mp2))
            print('mp2 corrected energy   {:>16.10f}\n'.format(self.rhf.SCFenergy+mp2))

        return ds, dd, td

    def do_adc_ee(self):
        #shell for ADC computation

        def adc_diagonal(self, ds, dd, td):
            #compute the diagonal of the ADC matrix as a pre-conditioner for the davidson iterations

            n, o, v = self.slice ; nocc, nvir, nrot = self.occupations

            #initialize to fock diagonal
            diagonal = -np.concatenate([ds.ravel(), dd.swapaxes(1, 2).ravel()])

            adc_diagonal = diagonal[:nrot].reshape(nocc, nvir)

            adc_diagonal -= np.einsum('aiai->ia', self.gs[v, o, v, o], optimize=True)

            adc_diagonal += 0.5  * np.einsum('acik,ikac->ia', self.gs[v, v, o, o], td, optimize=True)
            adc_diagonal += 0.5  * np.einsum('ikac,ikac->ia', self.gs[o, o, v, v], td, optimize=True) 
            adc_diagonal -= 0.25 * np.einsum('cdik,ikcd->i',  self.gs[v, v, o, o], td, optimize=True)[:, n]
            adc_diagonal -= 0.25 * np.einsum('ikcd,ikcd->i',  self.gs[o, o, v, v], td, optimize=True)[:, n]
            adc_diagonal -= 0.25 * np.einsum('ackl,klac->a',  self.gs[v, v, o, o], td, optimize=True)[n, :]
            adc_diagonal -= 0.25 * np.einsum('klac,klac->a',  self.gs[o, o, v, v], td, optimize=True)[n, :]

            return diagonal

        def adc_initial_guess(self, diagonal, f=1):
            #initial vector to start Davidson

            #get largest absolute values on diagonal matrix as best guess
            args = np.argsort(np.absolute(diagonal))
            guess_vectors = np.zeros((diagonal.size, self.roots * f))
            for root in range(self.roots * f):
                guess_vectors[args[root], root] = 1.0

            return guess_vectors

        def matvec(adc):
        #construct the ADC blocks of EE-ADC second order matrix dot product with arbitary vector (r)

            adc = np.array(adc)
            r   = np.zeros_like(adc)

            adc_s = adc[:nrot].reshape(nocc, nvir)
            r_s  = r[:nrot].reshape(nocc, nvir)

            #singles - singles block
            r_s -= np.einsum('ia,ia->ia', ds, adc_s, optimize=True)

            r_s -= np.einsum('ajbi,jb->ia', gs[v, o, v, o], adc_s, optimize=True)

            r_s += 0.5 * np.einsum('acik,jkbc,jb->ia', gs[v, v, o, o], td, adc_s, optimize=True)
            r_s += 0.5 * np.einsum('jkbc,ikac,jb->ia', gs[o, o, v, v], td, adc_s, optimize=True)

            t    = -np.einsum('cdik,jkcd->ij', gs[v, v, o, o], td, optimize=True)
            t   += -np.einsum('jkcd,ikcd->ij', gs[o, o, v, v], td, optimize=True)
            r_s += 0.25 * np.einsum('ij,ja->ia', t, adc_s, optimize=True)

            t    = -np.einsum('ackl,klbc->ab', gs[v, v, o, o], td, optimize=True)
            t   += -np.einsum('klbc,klac->ab', gs[o, o, v, v], td, optimize=True)
            r_s += 0.25 * np.einsum('ab,ib->ia', t, adc_s, optimize=True)

            adc_d = adc[nrot:].reshape(nocc, nvir, nocc, nvir)
            r_d  = r[nrot:].reshape(nocc, nvir, nocc, nvir)

            #singles - doubles block
            r_s += 0.5 * np.einsum('klid,kald->ia', gs[o, o, o, v], adc_d, optimize=True)
            r_s -= 0.5 * np.einsum('klic,kcla->ia', gs[o, o, o, v], adc_d, optimize=True)
            r_s -= 0.5 * np.einsum('alcd,icld->ia', gs[v, o, v, v], adc_d, optimize=True)
            r_s += 0.5 * np.einsum('akcd,kcid->ia', gs[v, o, v, v], adc_d, optimize=True)

            #doubles - singles block
            r_d += 0.5 * np.einsum('kbij,ka->iajb', gs[o, v, o, o], adc_s, optimize=True)
            r_d -= 0.5 * np.einsum('kaij,kb->iajb', gs[o, v, o, o], adc_s, optimize=True)
            r_d -= 0.5 * np.einsum('abcj,ic->iajb', gs[v, v, v, o], adc_s, optimize=True)
            r_d += 0.5 * np.einsum('abci,jc->iajb', gs[v, v, v, o], adc_s, optimize=True)

            #doubles - doubles block
            r_d -= np.einsum('ijab,iajb->iajb', dd, adc_d, optimize=True)

            return r

        #get spin versions of integrals
        self.gs, self.eps, self.fs = self.get_spin_quantities()

        #get spin metrics
        self.occupations, self.slice = self.get_spin_metrics()

        #do Moller-Plesset (2)
        ds, dd, td = self.moller_plesset()

        #get diagonal preconditioner
        diagonal = adc_diagonal(self, ds, dd, td)

        #generate initial guess from diagonal
        guess_vectors = adc_initial_guess(self, diagonal, self.guess_vector_factor)

        #global for this routine
        n, o, v = self.slice  ; nocc, nvir, nrot = self.occupations
        gs = self.gs ; tol = self.tol

        e, v, self.converged = davidson(matvec, guess_vectors, diagonal, tol=self.tol,
                                        vectors_per_root = self.vectors_per_root )

        return e, v    

    def do_adc_ea_ip(self):
        #shell for ADC computation

        def adc_diagonal(self, ds, dd, td):
            #compute the diagonal of the ADC matrix as a pre-conditioner for the davidson iterations

            n, o, v = self.slice ; nocc, nvir, nrot = self.occupations

            #initialize to fock diagonal
            if self.mode == 'ip':

                dh = self.eps[o, n, n] + ds[n]

                # Construct the single-singles (1h-1h) contribution
                hh = np.diag(self.eps[o])
                hh += 0.25 * np.einsum('ikab,jkab->ij', td, self.gs[o, o, v, v], optimize=True)
                hh += 0.25 * np.einsum('jkab,ikab->ij', td, self.gs[o, o, v, v], optimize=True)

                diagonal = np.concatenate([np.diag(hh), dh.ravel()])
                return diagonal, hh, dh

            elif self.mode == 'ea':

                dp = ds[:, :, n] - self.eps[n, n, v]

                # Construct the single-singles (1p-1p) contribution
                pp = np.diag(self.eps[v])
                pp -= 0.25 * np.einsum('ijac,ijbc->ab', td, self.gs[o, o, v, v], optimize=True) 
                pp -= 0.25 * np.einsum('ijbc,ijac->ab', td, self.gs[o, o, v, v], optimize=True)

                diagonal = np.concatenate([np.diag(pp), -dp.ravel()])
                return diagonal, pp, dp

        def adc_initial_guess(self, diagonal, f=1):
            #initial vector to start Davidson

            #get largest absolute values on diagonal matrix as best guess
            args = np.argsort(np.absolute(diagonal))
            guess_vectors = np.zeros((diagonal.size, self.roots * f))
            for root in range(self.roots * f):
                guess_vectors[args[root], root] = 1.0

            return guess_vectors

        def matvec_ip(adc):
        #construct the ADC blocks of IP-ADC second order matrix dot product with arbitary vector (r)

            r    = np.zeros_like(adc)   ;      adc     = np.array(adc)
            r_i  = r[:nocc]             ;      adc_i   = adc[:nocc]    

            adc_ija = adc[nocc:].reshape(nocc, nocc, nvir)  ; r_ija = r[nocc:].reshape(nocc, nocc, nvir)

            r_i += np.dot(shp, adc_i)
            r_i += np.sqrt(0.5) * np.einsum('ijak,ija->k', gs[o, o, v, o], adc_ija, optimize=True) 

            r_ija += np.sqrt(0.5) * np.einsum('ijak,k->ija', gs[o, o, v, o], adc_i, optimize=True) 
            r_ija += np.einsum('ija,ija->ija', dhp, adc_ija, optimize=True)
                     
            return r

        def matvec_ea(adc):
        #construct the ADC blocks of EA-ADC second order matrix dot product with arbitary vector (r)

            r    = np.zeros_like(adc)   ;      adc     = np.array(adc)
            r_a  = r[:nvir]             ;      adc_a   = adc[:nvir]    

            adc_iab = adc[nvir:].reshape(nocc, nvir, nvir)  ; r_iab = r[nvir:].reshape(nocc, nvir, nvir)

            r_a += np.dot(shp, adc_a)
            r_a += np.sqrt(0.5) * np.einsum('abic,iab->c', gs[v, v, o, v], adc_iab, optimize=True) 

            r_iab += np.sqrt(0.5) * np.einsum('abic,c->iab', gs[v, v, o, v], adc_a, optimize=True) 
            r_iab -= np.einsum('iab,iab->iab', dhp, adc_iab, optimize=True)
                     
            return r

        #get spin versions of integrals
        self.gs, self.eps, self.fs = self.get_spin_quantities()

        #get spin metrics
        self.occupations, self.slice = self.get_spin_metrics()

        #do Moller-Plesset (2)
        ds, dd, td = self.moller_plesset()

        #get diagonal preconditioner
        diagonal, shp, dhp = adc_diagonal(self, ds, dd, td)

        #generate initial guess from diagonal
        guess_vectors = adc_initial_guess(self, diagonal, self.guess_vector_factor)

        #global for this routine
        n, o, v = self.slice  ; nocc, nvir, nrot = self.occupations
        gs = self.gs ; tol = self.tol

        matvec = matvec_ip if self.mode == 'ip' else matvec_ea
        e, v, self.converged = davidson(matvec, guess_vectors, diagonal, tol=self.tol,
                                        vectors_per_root = self.vectors_per_root)
        
        if self.mode == 'ip': 
            e = -e  ; v = -v

        return e, v

class adc_analyse(object):
    #class to process results from adc calculation

    THRESHOLD = 0.1
    ORBITAL_ROOT = 1

    def __init__(self, adc):

        self.adc = adc
        self.root = -1

    def adc_energy(self):
        #ccompute the energies key is 'energy' [multiplicity, e(Hr), e(eV)]

        m = lambda x: len([n for n, i in enumerate(eigenvalue) if abs(i-x)<1e-6])
        eigenvalue = self.adc.eig_energy

        adc_energy = []
        for i in range(self.adc.roots):
            multiplicity = m(eigenvalue[i])
            adc_energy.append([multiplicity, eigenvalue[i], eigenvalue[i]*getConstant('hartree->eV')])

        return adc_energy[self.root]

    def adc_norm(self):
        #compute the norms

        nocc, nvir, _ = self.adc.occupations

        blocks = {'ip':['1h', '2h-1p'] , 'ea':['1p', '2p-1h'], 'ee': ['1h-1p', '2h-2p']}
        f = {'ip': nocc, 'ea': nvir, 'ee': nocc*nvir}

        mode = self.adc.mode

        #singles
        v = self.adc.eig_vector[:f[self.adc.mode], self.root]
        norm_single = np.einsum('i,i->', v, v, optimize=True)

        #doubles
        v = self.adc.eig_vector[f[self.adc.mode]:, self.root]
        norm_double = np.einsum('i,i->', v, v, optimize=True)

        adc_norm = [blocks[self.adc.mode], float(norm_single), float(norm_double)]
        
        return adc_norm

    def adc_state_ip_ea(self):
        #compute the excitations

        has_block = [False, False, False]
        dominant_state = [None, -1, None, -1]
        state_cache = []

        nocc, nvir, nrot = self.adc.occupations
        s_block = {'ip': nocc, 'ea': nvir}
        d_block = {'ip':(nocc, nocc, nvir), 'ea':(nocc, nvir, nvir)}

        spatial = [i//2 for i in d_block[self.adc.mode]]

        #singles excitation
        u = (self.adc.eig_vector[:s_block[self.adc.mode], self.root]**2.0).reshape(s_block[self.adc.mode])

        #aa and bb
        blocks = [u[::2].ravel() , u[1::2].ravel()]

        x = np.sqrt(blocks[0] + blocks[1])
        idx = np.argsort(x)[::-1]
        x = x[idx]

        dominant = np.count_nonzero(x > adc_analyse.THRESHOLD)

        if idx != [] and (self.adc.mode == 'ea'):
            for n, i in enumerate(idx):
                idx[n] += spatial[0]

        if idx != []:
            dominant_state = [idx[0], x[0], None, -1] ; has_block[0] = True
            state_cache.append([[i for i in idx[:dominant]], [i for i in x[:dominant]]])

        #doubles excitation
        u = (self.adc.eig_vector[s_block[self.adc.mode]:, self.root]**2.0).reshape(d_block[self.adc.mode])

        #aaa, bbb, aba and baba
        blocks = [u[::2,::2,::2].ravel(), u[1::2,1::2,1::2].ravel(), u[::2,1::2,::2].ravel(), u[1::2,::2,1::2].ravel()]

        for n, block in enumerate([[0,1],[2,3]]):

            x = np.sqrt(blocks[block[0]] + blocks[block[1]])*np.sqrt(2)
            idx = np.argsort(x)[::-1]
            x = x[idx]

            dominant = np.count_nonzero(x > adc_analyse.THRESHOLD)
            ix = np.vstack(np.unravel_index(idx[:dominant], tuple(spatial))).transpose().tolist()

            #get proper sequence for virtual orbitals
            if ix != []: 
                mask = [2] if self.adc.mode == 'ip' else [1,2]
                for i in ix:
                    for j in mask:
                        i[j] += spatial[0]

                if x[0] > dominant_state[3]: dominant_state[2:] = [ix[0], x[0]]
                has_block[n+1] = True
            state_cache.append([ix, [i for i in x[:dominant]]])

        return has_block, dominant_state, state_cache

    def adc_state_ee(self):
        #compute the excitations

        has_block = [False, False]
        dominant_state = [None, None]
        state_cache = []

        nocc, nvir, nrot = self.adc.occupations
        s_block = {'ee':(nocc,nvir)}
        d_block = {'ee':(nocc, nvir, nocc, nvir)}

        spatial = [i//2 for i in s_block[self.adc.mode]]

        #singles excitation
        u = (self.adc.eig_vector[:nocc*nvir, self.root]**2.0).reshape(s_block[self.adc.mode])

        blocks = [u[::2,::2].ravel(), u[1::2,1::2].ravel(), u[::2,1::2].ravel(), u[1::2,::2].ravel()]

        x = np.sqrt(blocks[0] + blocks[1] + blocks[2] + blocks[3])/np.sqrt(2)
        idx = np.argsort(x)[::-1]
        x = x[idx]

        dominant = np.count_nonzero(x > adc_analyse.THRESHOLD)
        ix = np.vstack(np.unravel_index(idx[:dominant], tuple(spatial))).transpose().tolist()

        #get proper sequence for virtual orbitals
        if ix != []: 
            mask = [1]
            for i in ix:
                for j in mask:
                    i[j] += spatial[0]

            has_block[0] = True
            dominant_state = [ix[0], x[0]]
        state_cache.append([ix, [i for i in x[:dominant]]])

        return has_block, dominant_state, state_cache

    def summary(self):
        #output to console

        if self.adc.mode == 'ip': head = ['1h','2h-1p','i','i  j  a']
        if self.adc.mode == 'ea': head = ['1p','2p-1h','a','i  a  b']
        if self.adc.mode == 'ee': head = ['1h-1p','2h-2p','i  a','']

        if self.adc.mode == 'ee':
            print('  n   m         energy          {:2s}:-----------------------> {:5s}:---->'.
                                                   format(head[0], head[1]))
            print('              Hr       eV       norm       {:1s}                 norm '.
                                                   format(head[2]))
            print('--------------------------------------------------------------------------')
        else:
            print('  n   m         energy          {:2s}:-----------------------> {:5s}:------------------------>'.
                                                   format(head[0], head[1]))
            print('              Hr       eV       norm       {:1s}                   norm       {:7s}         '.
                                                   format(head[2],    head[3]))
            print('-------------------------------------------------------------------------------------------')

        root = 1 ; cycle_energy = 0.0

        #from cache and dominant root orbitals are numbered from base 0

        for i in range(self.adc.roots):

            self.root = i
            e = self.adc_energy()
            n = self.adc_norm()

            #ignore 0.0000 single normed roots and only print 1 of each multiplicity
            if np.isclose(n[1], 0.0000) : continue
            if np.isclose(e[1], cycle_energy): 
                continue
            else: cycle_energy = e[1]

            state = self.adc_state_ee if self.adc.mode == 'ee' else self.adc_state_ip_ea

            has_block, dominant_state, cache = state()

            print(' {:>2d}  {:>2d} {:>10.6f}{:>10.6f}'.
                                                    format(root, e[0], e[1], e[2]), end='')
            if self.adc.mode == 'ee' and has_block[0]:
                print('  {:>8.4f}   [{:>2d},{:>2d} ]{:>7.4f}'.
                                                    format(n[1], dominant_state[0][0] + adc_analyse.ORBITAL_ROOT, 
                                                                 dominant_state[0][1] + adc_analyse.ORBITAL_ROOT, 
                                                                 dominant_state[1]), end='')
            elif has_block[0]:
                print('  {:>8.4f}   [{:>2d} ]{:>7.4f}'.
                                                    format(n[1], dominant_state[0] + adc_analyse.ORBITAL_ROOT, 
                                                                 dominant_state[1]), end='')
            else:
                print('                           ', end='')

            if self.adc.mode != 'ee' and (has_block[1] or has_block[2]):
                print('      {:>8.4f}   [{:>2d},{:>2d},{:>2d} ]{:>7.4f}'.
                                                    format(n[2], dominant_state[2][0] + adc_analyse.ORBITAL_ROOT, 
                                                                 dominant_state[2][1] + adc_analyse.ORBITAL_ROOT,
                                                                 dominant_state[2][2] + adc_analyse.ORBITAL_ROOT,
                                                                 dominant_state[3]))
            else:
                print('      {:>8.4f}'.format(n[2]))

            root += 1

    def detail(self, root):
        #detailed list of excitations

        self.root = root
        e = self.adc_energy()
        n = self.adc_norm()

        print('\n-->state (root) {:>2d}'.
                                                    format(root))
        print('   polarization type = {:>2s}        energy = {:<9.6f} eV'.
                                                    format(self.adc.mode.upper(), e[1]))

        HOMO = self.adc.occupations[0]//2  ; LUMO = HOMO + 1
        print('   multiplicity = {:>2d}             homo->lumo  {:>2d}->{:>2d}'.format(e[0], HOMO, LUMO))
 
        state = self.adc_state_ee if self.adc.mode == 'ee' else self.adc_state_ip_ea

        block_label   = {'ip':['1h','2h-1p'],'ea':['1p','2p-2h'],'ee':['1h-1p','2h-2p']}
        idx_label     = {'ip':['i','i  j  a'], 'ea':['a','i  a  b'], 'ee':['i  a', '']}
        sym_label     = ['\u03B1->\u03B1','\u03B1\u03B1->\u03B2\u03B2', '\u03B1\u03B2->\u03B1\u03B2']

        has_block, dominant_state, cache = state()
        idx, x = zip(*cache)

        print('block  type           excitation\n--------------------------------------'.format())
        for n, block in enumerate(has_block):

            if block:
                for m, i in enumerate(idx[n]):
                    if self.adc.mode in ['ip','ea']:
                        if n == 0:
                            print('{:<5s}  {:<5s}         [{:>2d} ]     {:>7.4f}'.
                            format(block_label[self.adc.mode][min(n,1)], sym_label[n], i + adc_analyse.ORBITAL_ROOT, x[n][m]))
                        elif n == 1:
                            if (m//2)*2 == m:continue
                            print('{:<5s}  {:<5s}     [{:>2d},{:>2d},{:>2d} ]  {:>7.4f}'.
                            format(block_label[self.adc.mode][min(n,1)], sym_label[n], i[0] + adc_analyse.ORBITAL_ROOT, 
                                   i[1] + adc_analyse.ORBITAL_ROOT, i[2] + adc_analyse.ORBITAL_ROOT, x[n][m]))
                        else:
                            print('{:<5s}  {:<5s}     [{:>2d},{:>2d},{:>2d} ]  {:>7.4f}'.
                            format(block_label[self.adc.mode][min(n,1)], sym_label[n], i[0] + adc_analyse.ORBITAL_ROOT, 
                                   i[1] + adc_analyse.ORBITAL_ROOT, i[2] + adc_analyse.ORBITAL_ROOT, x[n][m]))
                    else:
                        print('{:<5s}  {:<5s}     [{:>2d},{:>2d} ]     {:>7.4f}'.
                        format(block_label[self.adc.mode][min(n,1)], sym_label[n], i[0] + adc_analyse.ORBITAL_ROOT, 
                               i[1] + adc_analyse.ORBITAL_ROOT, x[n][m]))

    def transition_density(self, root, mpdm, type='ee'):
        #return the transition density for ee type

        if type != 'ee':
            print('transition density only available for electron excitation')
            return None

        #collect the properties needed for computation
        nocc, nvir, nrot = self.adc.occupations  ; n, o, v = self.adc.slice
        ds, dd, td = self.adc.moller_plesset(silent=True)

        #get amplitude vectors
        u1 = self.adc.eig_vector[:nocc*nvir, root].reshape(nocc, nvir)
        u2 = self.adc.eig_vector[nocc*nvir:, root].reshape(nocc, nvir, nocc, nvir)
        
        gs, _, _ = self.adc.get_spin_quantities()

        t = np.einsum('ikac,kbjc->ijab', td, gs[o,v,o,v], optimize=True)
        tD = t - t.transpose(1,0,2,3) - t.transpose(0,1,3,2) + t.transpose(1,0,3,2)
        tD -= 0.5 * np.einsum('ijcd,abcd->ijab', td, gs[v,v,v,v]) + 0.5 * np.einsum('klab,klij->ijab', td, gs[o,o,o,o], optimize=True)
        tD /= dd

        #the transition density matrix
        dm = np.zeros((nocc+nvir, nocc+nvir))

        #0th order
        dm[v, o] += u1.transpose(1, 0) 
        #1st order
        dm[o, v] += np.einsum('ijab,jb->ia', td, u1, optimize=True) 

        #2nd order
        dm[o, o] -= np.einsum('ia,ja->ij', mpdm[o,v], u1, optimize=True)
        dm[o, o] += np.einsum('iakb,jkab->ij', u2, td, optimize=True)

        dm[v, v] += np.einsum('ia,ib->ab', u1, mpdm[o,v], optimize=True)
        dm[v, v] -= np.einsum('iajc,ijbc->ab', u2, td, optimize=True)

        dm[o, v] -= np.einsum('ijab,jb->ia', tD, u1, optimize=True)

        dm[v, o] += 0.5 * (np.einsum('ijab,jkbc,kc->ai', td, td, u1, optimize=True)
                          -np.einsum('ab,ib->ai', mpdm[v,v], u1, optimize=True)
                          +np.einsum('ja,ij->ai', u1, mpdm[o,o], optimize=True))

        return dm

class hf_reference(object):
    #holder class for rhf, molAtom and molBasis structures

    def __init__(self, rhf, atoms, basis, data):

        self.rhf = rhf
        self.atoms = atoms
        self.basis = basis
        self.data  = data

class mp2_properties(object):
    #class to provide mp2 density and dipoles

    def __init__(self, hf, adc):

        self.hf = hf
        self.adc = adc

    def mp2_density(self, type='unrelaxed'):
        #compute the mp2 level density matrix in spin basis

        ds, dd, td = self.adc.moller_plesset(silent=True)

        #particle block and enforce symmetry
        oo = -0.5* np.einsum('ikab,jkab->ij', td, td, optimize=True)
        oo = 0.5 * (oo + np.transpose(oo))

        #hole block and enforce symmetry
        vv = 0.5 * np.einsum('ijac,ijbc->ab', td, td, optimize=True) 
        vv = 0.5 * (vv + np.transpose(vv))

        #paticle-hole block needed for relaxed density
        gs, _ , _ = self.adc.get_spin_quantities()
        n, o, v = self.adc.slice
        ov = -0.5 * (np.einsum('ijbc,jabc->ia', td, gs[o,v,v,v], optimize=True) +
                     np.einsum('jkib,jkab->ia', gs[o,o,o,v], td, optimize=True) 
                    )/ds

        if type == 'unrelaxed': ov = np.zeros_like(ov)

        dm = np.block([[oo, ov] , [np.transpose(ov), vv]])

        return dm

    def dipoles(self):
        #compute dipoles at various levels

        spin_to_spatial = lambda x: (x[::2,::2] + x[1::2,1::2] + x[::2,1::2] + x[1::2,::2]) 

        #get components 
        from post import dipoleComponent
        mu_component = [dipoleComponent(self.hf.atoms, self.hf.basis, x, 'origin') for x in ['x','y','z']]
        charges      = [a.number for a in self.hf.atoms]
        centers      = [a.center for a in self.hf.atoms]

        #ao->mo
        mu_mo = np.einsum('rp,xrs,sq->xpq', self.hf.rhf.C, mu_component, self.hf.rhf.C, optimize=True)

        #compute nuclear dipole contribution
        nuclear_dipole = np.einsum('i,ix->x', charges, centers)

        #HF reference dipole
        nocc, nvir, _ = self.adc.occupations

        dm = np.zeros((nocc+nvir, nocc+nvir))
        np.fill_diagonal(dm[:nocc, :nocc], 1.0)

        #contract density to spatial
        dm = spin_to_spatial(dm)
        mu_rhf = np.asarray([np.einsum("ij,ij->", mu_mo[k], dm) for k in range(3)])
        hf_reference_dipole = nuclear_dipole - mu_rhf

        #compute unrelaxed dipole at mp2 level
        udm = self.mp2_density(type = 'unrelaxed')
        udm = spin_to_spatial(udm) + dm
        mu_mp2 = np.asarray([np.einsum("ij,ij->", mu_mo[k], udm) for k in range(3)])
        mu_mp2_unrelaxed = nuclear_dipole - mu_mp2

        #compute relaxed dipole at mp2 level
        rdm = self.mp2_density(type = 'relaxed')
        rdm = spin_to_spatial(rdm) + dm
        mu_mp2 = np.asarray([np.einsum("ij,ij->", mu_mo[k], rdm) for k in range(3)])
        mu_mp2_relaxed = nuclear_dipole - mu_mp2

        return {'hf': hf_reference_dipole, 'mpu': mu_mp2_unrelaxed, 'mpr': mu_mp2_relaxed}
       
if __name__ == '__main__':

    import rhf
    molAtom, molBasis, molData = rhf.mol([])
    e_scf = rhf.scf(molAtom, molBasis, molData, [])

    from basis import electronCount

    charge, electrons = [molData['charge'], electronCount(molAtom, molData['charge'])]

    adc = ADC('ee', rhf, electrons, roots=6, solve=[2, 10])

    adc_a = adc_analyse(adc)
    adc_a.summary()
    adc_a.detail(0)
    adc_a.detail(3)
    adc_a.detail(4)

    mp_prop = mp2_properties(hf_reference(rhf, molAtom, molBasis, molData), adc)
    dipoles = mp_prop.dipoles()

    print()
    caption = ['hf  reference dipole', 'mp2 unrelaxed dipole', 'mp2 relaxed dipole']
    for i, mu in enumerate(['hf', 'mpu', 'mpr']):
        x , y, z = dipoles[mu] * getConstant('au->debye')
        print('{:<20s}    x= {:<8.4f}   y= {:<8.4f}   z= {:<8.4f}  D'.format(caption[i], x, y, z))


    adc = ADC('ee', rhf, electrons, roots=20, solve=[2, 10])
    adc_a = adc_analyse(adc)
    root = 14
    dm = adc_a.transition_density(root, mp_prop.mp2_density(type='relaxed'))

    from post import dipoleComponent
    mu_component = [dipoleComponent(mp_prop.hf.atoms, mp_prop.hf.basis, x, 'origin') for x in ['x','y','z']]
    charges      = [a.number for a in mp_prop.hf.atoms]
    centers      = [a.center for a in mp_prop.hf.atoms]

    #ao->mo
    mu_mo = np.kron(np.einsum('rp,xrs,sq->xpq', mp_prop.hf.rhf.C, mu_component, mp_prop.hf.rhf.C, optimize=True), np.eye(2))

    #get transition dipole moment
    tdm = np.einsum('ia,xia->x', dm, mu_mo, optimize=True)

    os = (2/3) * adc.eig_energy[root] * np.einsum('x,x->', tdm, tdm, optimize=True)
    print('electric length gauge oscillator strength = {:<8.6f}  for excitation {:<8.6f} eV'.
                             format(os , adc.eig_energy[root]*getConstant('hartree->eV')))
