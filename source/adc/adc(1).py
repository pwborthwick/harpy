from __future__ import division
import numpy as np
from atom import getConstant
from cc.fcc import spinMO
from adc import *

class first_order_adc(object):

    def __init__(self, ADC, HF, solver='eigh'):

        self.ADC    = ADC
        self.HF     = HF
        self.solver = solver

        #get mp2 quantities
        ds, dd, td = self.ADC.moller_plesset(silent=True)

        #get diagonal preconditioner
        diagonal = self.adc_diagonal(ds, dd, td)

        #generate initial guess from diagonal
        guess_vectors = self.adc_initial_guess(diagonal, self.ADC.guess_vector_factor)

        #global for this routine
        n, o, v = self.ADC.slice  ; nocc, nvir, nrot = self.ADC.occupations
        gs = adc.gs ; tol = adc.tol

        if solver == 'davidson':
            e, u, converged = davidson(self.matvec, guess_vectors, diagonal, tol=self.ADC.tol,
                                             vectors_per_root = self.ADC.vectors_per_root )
        elif solver == 'eigh':
            e, u, converged = self.direct_solve()

        self.cache = {'e':e[:self.ADC.roots], 'u':u[:, :self.ADC.roots], 'c':converged}

    def get(self):
        #return results

            return self.cache['e'], self.cache['u'], self.cache['c']

    def get_transition_properties(self):
        #collect the different transition property and write to cache

        function_calls = [self.get_cis_transition_moments, self.get_adc_transition_moments] * 3
        cache_labels   = [['dipole:electric:length:CIS',   'oscillator:electric:length:CIS'],
                          ['dipole:electric:length:ADC',   'oscillator:electric:length:ADC'],
                          ['dipole:electric:velocity:CIS', 'oscillator:electric:velocity:CIS'],
                          ['dipole:electric:velocity:ADC', 'oscillator:electric:velocity:ADC'],
                          ['dipole:magnetic:length:CIS',   ''],
                          ['dipole:magnetic:length:ADC',   '']]
        operator_id = ['d', 'd', 'n', 'n', 'a', 'a']

        #transition dipoles and oscillator strengths
        for i, f in enumerate(function_calls):

            properties = []
            operator = f(operator_id[i])

            for root, tdm in enumerate(operator):
                if 'electric' in cache_labels[i][0]:
                    properties.append((2/3) * self.cache['e'][root] * np.einsum('x,x->', tdm, tdm, optimize=True))
                    self.cache[cache_labels[i][1]] = properties

            self.cache[cache_labels[i][0]] = operator

        #cross-section
        properties = []
        for root in range(self.ADC.roots):
            properties.append(self.cache['oscillator:electric:length:CIS'][root] * 2 * getConstant('alpha') * (np.pi)**2)
        self.cache['cross-section:CIS'] = properties
        properties = []
        for root in range(self.ADC.roots):
            properties.append(self.cache['oscillator:electric:length:ADC'][root] * 2 * getConstant('alpha') * (np.pi)**2)
        self.cache['cross-section:ADC'] = properties

    def get_adc_transition_moments(self, type='d'):
        #compute the transition dipole and oscillator strength at ADC(1) level

        transition_properties = []


        mu_mo = self.get_reference_dipole_components(type)
        n, o, v = self.ADC.slice ; nocc, nvir, nrot = self.ADC.occupations

        #get MP2 properties
        ds, dd ,td = adc.moller_plesset(silent=True)

        for root in range(self.ADC.roots):

            #ADC ph amplitude
            u = self.cache['u'][:, root].reshape(nocc, nvir)

            #get ADC transition density
            dm = np.zeros((nocc+nvir, nocc+nvir))       
            dm[v, o] += u.transpose(1, 0) 
            dm[o, v] += np.einsum('ijab,jb->ia', td, u, optimize=True)  

            #get transition dipole moment
            tdm = np.einsum('ia,xia->x', dm, mu_mo, optimize=True)

            transition_properties.append(tdm)

        return transition_properties


    def get_cis_transition_moments(self, type='d'):
        #compute the transition dipole momemnts and oscilator strength at CIS level

        transition_properties = []

        mu_mo = self.get_reference_dipole_components(type)
        n, o, v = self.ADC.slice ; nocc, nvir, nrot = self.ADC.occupations

        for root in range(self.ADC.roots):

            #get CIS transition density
            dm = self.cache['u'][:, root].reshape(nocc, nvir)

            #get transition dipole moment
            tdm = np.einsum('ia,xia->x', dm, mu_mo[:, o, v], optimize=True)

            transition_properties.append(tdm)

        return transition_properties


    def get_reference_dipole_components(self, type='d'):
        #return the ground state dipole components

        #get components 
        if   type == 'd':
            from post import dipoleComponent
            mu_component = [dipoleComponent(self.HF.atoms, self.HF.basis, x, 'nuclear charge') for x in ['x','y','z']]

        elif type == 'n':
            from integral import buildNabla
            mu_component = [buildNabla(self.HF.atoms, self.HF.basis, x) for x in ['x','y','z']]

        elif type == 'a':
            from integral import buildAngular
            mu_component = [0.5 * buildAngular(self.HF.atoms, self.HF.basis, x, 'origin') for x in ['x','y','z']]

        else:
            print('property operator type[', type, ']not supported')

        #ao->mo
        mu_mo = np.kron(np.einsum('rp,xrs,sq->xpq', self.HF.rhf.C, mu_component, self.HF.rhf.C, optimize=True), np.eye(2))

        return mu_mo

    def adc_diagonal(self, ds, dd, td):
        #compute the diagonal of the ADC matrix as a pre-conditioner for the davidson iterations

        n, o, v = self.ADC.slice ; nocc, nvir, nrot = self.ADC.occupations

        #initialize to fock diagonal
        diagonal = -ds.ravel()

        adc_diagonal = diagonal[:nrot].reshape(nocc, nvir)
        adc_diagonal -= np.einsum('aiai->ia', self.ADC.gs[v, o, v, o], optimize=True)

        return diagonal

    def adc_initial_guess(self, diagonal, f=1):
        #initial vector to start Davidson

        #get largest absolute values on diagonal matrix as best guess
        args = np.argsort(np.absolute(diagonal))

        #we only have nocc*nvir roots available
        if self.ADC.roots > len(args):
            print('reducing requested roots - exceeded ', len(args))
            self.ADC.roots = len(args)

        guess_vectors = np.zeros((diagonal.size, self.ADC.roots * f))
        for root in range(self.ADC.roots * f):
            guess_vectors[args[root], root] = 1.0

        return guess_vectors

    def matvec(self, adc):
        #construct the self blocks of EE-self first order matrix dot product with arbitary vector (r)

        n, o, v = self.ADC.slice ; nocc, nvir, nrot = self.ADC.occupations
        ds, dd, td = self.ADC.moller_plesset(silent=True)

        adc = np.array(adc)
        r   = np.zeros_like(adc)

        adc_s = adc[:nrot].reshape(nocc, nvir)
        r_s  = r[:nrot].reshape(nocc, nvir)

        #singles - singles block
        r_s -= np.einsum('ia,ia->ia', ds, adc_s, optimize=True)
        r_s -= np.einsum('ajbi,jb->ia', self.ADC.gs[v, o, v, o], adc_s, optimize=True)

        return r

    def direct_solve(self):
        #self(1) = CIS

        n, o, v = self.ADC.slice ; nocc, nvir, nrot = self.ADC.occupations

        #excitations
        a  = np.einsum('ab,ij->iajb',np.diag(np.diag(self.ADC.fs)[v]),np.diag(np.ones(nocc))) 
        a -= np.einsum('ij,ab->iajb',np.diag(np.diag(self.ADC.fs)[o]),np.diag(np.ones(nvir))) 
        a += np.einsum('ajib->iajb',self.ADC.gs[v,o,o,v], optimize=True) 

        #reshape for solving
        a = a.reshape(nrot, nrot)

        try:
            e, u = np.linalg.eigh(a)
            converged = True
        except np.linalg.LinAlgError as e:
            print('matrix solve error ', e)
            converged = False

        return e, u, converged


if __name__ == '__main__':

    def get_excitation_labels(n, type, nocc):
        #put HOMO/LUMO labels on excitation

        if type == 'o':
            frontier = 'HOMO' if n == (nocc-1) else 'HOMO-' + str(nocc-1-n)
        else:
            frontier = 'LUMO' if n == 0 else 'LUMO+' + str(n)

        return frontier

    import rhf
    molAtom, molBasis, molData = rhf.mol([])
    e_scf = rhf.scf(molAtom, molBasis, molData, [])

    from basis import electronCount

    charge, electrons = [molData['charge'], electronCount(molAtom, molData['charge'])]

    #call adc class with zero roots to initialise object then reset roots
    adc = ADC('ee', rhf, electrons, roots=0, solve=[1, 10])
    adc.roots = 20

    #create ground state class for acd
    hf = hf_reference(rhf, molAtom, molBasis, molData)

    #create instance of adc(1) class
    adc_1 = first_order_adc(adc, hf, solver='eigh')
    energy, v, converged = adc_1.get()

    #instance of ground state RHF object       
    adc_1.get_transition_properties()


    if converged:

        nocc, nvir , _ = adc.occupations
        nocc_spatial, nvir_spatial = nocc//2, nvir//2

        print('ADC(1) Excited States')      
        print('-----------------------------------------------------------------------------------------------')
        print('root                energy                      excitation             osc.  (CIS)       (ADC)')
        print('-----------------------------------------------------------------------------------------------')
        current_energy = energy[0] ; current_root = 1 ; multiple = 0

        state = []
        for i, e in enumerate(energy):
    
            if not np.isclose(current_energy, e):
                multiplicity = '[' + str(multiple) + ']'

                eigenvector = (v[:, i-1]**2).reshape(nocc, nvir)

                compacted_to_spatial = (eigenvector[::2,::2].ravel()  + eigenvector[1::2,1::2].ravel() + 
                                        eigenvector[::2,1::2].ravel() + eigenvector[1::2,::2].ravel())/np.sqrt(2)

                idx = np.argsort(compacted_to_spatial)[::-1]
                maximum_excitations = compacted_to_spatial[idx]
                ix = np.unravel_index(idx, (nocc_spatial, nvir_spatial))

                oscillator_strength_cis = adc_1.cache['oscillator:electric:length:CIS'][i-1] 
                oscillator_strength_adc = adc_1.cache['oscillator:electric:length:ADC'][i-1]


                print('  {:<2d}  {:4s}  {:>9.6f}   {:>9.6f}       {:>8.4f}  {:>8s} -> {:8s}'.
                     format(current_root, multiplicity, current_energy, current_energy*getConstant('hartree->eV'), 
                            maximum_excitations[0], get_excitation_labels(ix[0][0],'o', nocc_spatial),
                            get_excitation_labels(ix[1][0],'v', nocc_spatial)), end='')

                if not np.isclose(oscillator_strength_cis, 0.0) :
                    print('       {:<9.4f}  {:<9.4f}'.format(oscillator_strength_cis, oscillator_strength_adc))
                else:
                    print()
                current_energy = e ; current_root += 1 ; multiple = 1
            else:
                multiple +=1

            state.append(current_root)

        #get full properties for singlet state
        root = 14
        print('\nExcited state ', state[root], '     root number ', root, '      energy ', round(adc_1.cache['e'][root], 6))
        print('------------------------------------ ------------------------------------------------------------------')
        print('   type       gauge                    CIS                                       ADC             ')
        print('                              dipole           oscillator             dipole           oscillator')
        print('------------------------------------------------------------------------------------------------------')
        cis_edl = adc_1.cache['dipole:electric:length:CIS'][root]
        adc_edl = adc_1.cache['dipole:electric:length:ADC'][root]
        cis_osc = adc_1.cache['oscillator:electric:length:CIS'][root] ; adc_osc = adc_1.cache['oscillator:electric:length:ADC'][root] 
        print(' electric    length  [{:>7.4f} {:>7.4f} {:>7.4f}]  {:>7.4f}      [{:>7.4f} {:>7.4f} {:>7.4f}]  {:>7.4f}'.
               format(cis_edl[0], cis_edl[1], cis_edl[2], cis_osc, adc_edl[0], adc_edl[1], adc_edl[2], adc_osc))

        cis_edv = adc_1.cache['dipole:electric:velocity:CIS'][root]
        adc_edv = adc_1.cache['dipole:electric:velocity:ADC'][root]
        cis_osc = adc_1.cache['oscillator:electric:velocity:CIS'][root] ; adc_osc = adc_1.cache['oscillator:electric:velocity:ADC'][root] 
        print(' electric   velocity [{:>7.4f} {:>7.4f} {:>7.4f}]  {:>7.4f}      [{:>7.4f} {:>7.4f} {:>7.4f}]  {:>7.4f}'.
               format(cis_edv[0], cis_edv[1], cis_edv[2], cis_osc, adc_edv[0], adc_edv[1], adc_edv[2], adc_osc))

        cis_mdl = adc_1.cache['dipole:magnetic:length:CIS'][root]
        adc_mdl = adc_1.cache['dipole:magnetic:length:ADC'][root]
        print(' magnetic    length  [{:>7.4f} {:>7.4f} {:>7.4f}]               [{:>7.4f} {:>7.4f} {:>7.4f}] '.
               format(cis_mdl[0], cis_mdl[1], cis_mdl[2],  adc_mdl[0], adc_mdl[1], adc_mdl[2]))

        print('\nCross-sections        CIS       {:>9.6f}   ADC   {:>9.6f}'.
               format(adc_1.cache['cross-section:CIS'][root], adc_1.cache['cross-section:ADC'][root])  )

    else:
        print('ADC(1) failed to converge')

    def lorentzian(e0, e, tau):
        #Lorentzian broadening

       gamma = 1.0/tau
       g = (gamma/2.0)**2.0/((e0-e)**2.0 + (gamma/2.0)**2.0)

       return g


    bars  = ['oscillator:electric:length:CIS', 'oscillator:electric:velocity:CIS']
    broad = ['oscillator:electric:length:ADC', 'oscillator:electric:velocity:ADC']

    import matplotlib.pyplot as py

    fig, ax = py.subplots(2, 1, sharex=True)
    fig.subplots_adjust(hspace=0.04)

    fig.suptitle('Oscillator Strengths')

    #bar plots
    ax[0].bar(adc_1.cache['e']*getConstant('hartree->eV'), adc_1.cache[bars[0]], 
        width=0.05, color='orange', align='edge', label='length')
    ax[0].bar(adc_1.cache['e']*getConstant('hartree->eV'), adc_1.cache[bars[1]], 
        width=-0.05, color='black', align='edge', label='velocity')
    ax[0].get_xaxis().set_visible(False)
    ax[0].set_ylabel('CIS')

    ax[0].legend(loc="upper left")

    tau, margin, n = [40, 0.5, 50 ]

    #broadening plots
    for i, e in enumerate(adc_1.cache['e']):

        eV = e * getConstant('hartree->eV')
        x = np.linspace(eV - margin, eV + margin, n) 
        if not np.isclose(adc_1.cache[broad[0]][i], 0):
            lorentz = lorentzian(eV, x, tau) * adc_1.cache[broad[0]][i]
            ax[1].plot(x, lorentz, color='orange')
        if not np.isclose(adc_1.cache[broad[1]][i], 0):
            lorentz = lorentzian(eV, x, tau) * adc_1.cache[broad[1]][i]
            ax[1].plot(x, lorentz, color='black')

    ax[1].set_ylabel('ADC(1)')
    ax[1].set_xlabel('Energy (eV)')

    py.show()