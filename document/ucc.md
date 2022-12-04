### Unitary Coupled-Cluster

This module implements ucc2, ucc3 and excited state ucc2. These are unitary coupled-cluster methods.

1.  **class unitaryCoupledCluster(object)**

   The initiator takes the arguments (fs, gs, e, data) where *fs* is the MO spin Fock matrix, *gs* are the MO spin two electron repusion integrals, *e* are the AO basis orbital energies and *data* is a dictionary {'method':.., 'electrons':.., 'cycle_limit'.., 'precision':.., 'verbose':.., 'roots:...'} supplied to the class by the caller. *method* is the unitary coupled-cluster method, *electrons* are the number of electrons, *cycle_limit* is the maximum number of iterations allowed, *precision* is the number of decimal places to converge to, *verbose* is a boolean flag - if True information on each cycle will be printed and *roots* are the number of roots to be returned if excited-state is selected. \
The available *method*s are 'ucc2', 'ucc3', 'ucc2-s-ee', 'ucc2-ee' and 'ucc(4).
On exit the class has instance variables *ss* the singles amplitudes, *sd* the doubles amplitudes, *converged* a boolean flag (which should be checked on exit)
indicating a successful convergence of the iterations and 'energy'. *ss* is of dimension [o,v], *sd* is of dimension [o,o,v,v]. *energy* is a dictionary with keys
   + 'cc'      - unitary coupled-cluster correction
   + 'eHF'     - Hartree-Fock electronic energy
   + 'mp2'     - Moller-Plesset 2 energy
   + 'nuclear' - nuclear repulsion energy (user added see example code)

 In the ucc2 method the singles and doubles de-coupled with the doubles being formally equivalent to linear ccd. ucc2-ee is the excited state version of ucc2 using converged ss and sd amplitudes. ucc2-s-ee is the 'strict' version of ucc2-ee using td amplitudes and is equivalent to adc(2).


   There are eight routines \
   *initialise_amplitudes(self)* - sets td,ss and sd initial values and computes d-tensors (inverses). Computes HF energy. \
   *update_amplitudes(self, iterative=True)* - generate next guess of the amplitudes. \
   *cluster_energy(self)* - compute the cluster correction energy. \
   *iterator(self, func)* - the main iteration loop, amplitude update function passed as arguement.
   *excitations(self)* - secular equations for ucc2

  This is an example of the use of the classes.
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

    data = {'method':'ucc2-s-ee', 'electrons':electrons, 'cycle_limit': 50, 'precision':1e-10, 'verbose':True, 'roots':5}
    cc = unitaryCoupledCluster(fs, gs, e, data)
    if cc.converged:
        cc.energy['nuclear'] = nuclearRepulsion

        print(cc.energy)

The results for the methods are (Crawford geometry STO-3G)
{'ucc2': -0.07192916403651912, 'mp2': -0.049149636689279796, 'eHF': -82.94444701585147, 'nuclear': 8.00236706181077}
{'ucc3': -0.07011956942398256, 'mp2': -0.049149636689279796, 'eHF': -82.94444701585147, 'nuclear': 8.00236706181077}

[0.28550929 0.28550929 0.28550929 0.34391619 0.37229159]
{'ucc2-s-ee': -0.07192916403651912, 'mp2': -0.049149636689279796, 'eHF': -82.94444701585147, 'nuclear': 8.00236706181077}
{'ucc(4)': -0.07054260489227335, 'mp2': -0.049149636689279796, 'eHF': -82.94444701585147, 'nuclear': 8.00236706181077}

The ucc2 and ucc3 methods are taken from [Hodecker](https://core.ac.uk/download/pdf/322693292.pdf) and ucc(4) from [Rodney J.Bartlett, Stanislaw A.Kucharski & Jozef Noga](https://www.sciencedirect.com/science/article/abs/pii/S0009261489873725).
]