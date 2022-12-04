### Fast Coupled-Cluster

The module scc.py has coupled-cluster routines written for clarity as explicit for-loops. This module contains a different (and faster) approach. In order to do coupled-cluster you need to first convert from atomic basis to a molecular spin basis (at least that's the way we'll do it). This is a class to do just that 

1.  **class spinMO(object)**

The initiator takes arguements (e, eri, c, f) where *e* are the orbital energies of a converged HF calculation, *eri* are the two electron repulsion integrals (in  our case given as the linear array form), *c* are the orbital coefficients and *f* is the final Fock matrix. All quantities have dimension number of basis functions. To get a instance of the class use

        mo = spinMO(eps, eri, C, fock)
        spinFock = mo.fs  ; spinEri = np.gs

  The class has instance variables *gs* the MO spin two electron replusion integrals and *fs* the MO spin Fock integrals. These arrays are of length 2\*number of basis functions in all dimensions ie [n,n,n,n] and [n,n]

  There are two routines\
    *gMOspin(self, e, c, eri, nbf)* - for the two electron repulsion integrals and\
    *fMOspin(self, f, c)* - for the Fock matrix.

The main coupled-cluster class is

2.  **class coupledCluster(object)**

   The initiator takes the arguements (fs, gs, e, data) where *fs* is the MO spin Fock matrix, *gs* are the MO spin two electron repusion integrals, *e* are the AO basis orbital energies and *data* is a dictionary {'method':.., 'electrons':.., 'cycle_limit'.., 'precision':.., 'verbose':..} supplied to the class by the caller. *method* is the coupled-cluster method, *electrons* are the number of electrons, *cycle_limit* is the maximum number of iterations allowed, *precision* is the number of decimal places to converge to and *verbose* is a boolean flag - if True information on each cycle will be printed. \
The available *method*s are 'ccd', 'ccsd', 'ccsd(t)', 'cc2', 'qcisd' and linear ccd 'lccd' and linear ccsd 'lccsd'.
On exit the class has instance variables *ts* the singles amplitudes, *td* the doubles amplitudes, *converged* a boolean flag (which should be checked on exit)
indicating a successful convergence of the iterations and 'energy'. *ts* is of dimension [o,v], *td* is of dimension [o,o,v,v]. *energy* is a dictionary with keys
   + 'cc'      - coupled-cluster correction
   + 'pt'      - perturbative triples correction (optional)
   + 'eHF'     - Hartree-Fock electronic energy
   + 'mp2'     - Moller-Plesset 2 energy
   + 'nuclear' - nuclear repulsion energy (user added see example code)

   The class has a method *intermediates(slice, tau)* where *slice* can be one of 'oo', 'vv', 'ov', 'oooo', 'vvvv' or 'ovvo' and tau is a boolean (default=True) which for coupled-cluster will be True (False for Lambda calculations). To get the F<sub>ae</sub> intermediate use

        mo = cc.itermediates('vv')

   There are eight routines \
   *initialise_amplitudes(self)* - sets ts and td initial values and computes d-tensors (inverses). Computes HF energy. \
   *tau(self, tilde=True)* - the &tau; functions. \
   *intermediates(self, \_slice, tilde=True)* - F and W coupled-cluster intermediates. \
   *update_amplitudes(self)* - generate next guess of the amplitudes for ccd, ccsd, ccsd(t) and cc2. \
   *update_linear_amplitudes(self)* - generate next guess of the amplitudes for lccd, lccsd. \
   *cluster_energy(self)* - compute the cluster correction energy. \
   *perturbative_triples(self)* - perturbative triples correction. \
   *iterator(self, func)* - the main iteration loop, amplitude update function passed as arguement.

   For the intermediates (tilde type) F-type intermediates are 'oo', 'vv' and 'ov', and for W-type 'oooo', 'vvvv' and 'ovvo'.

There is a DIIS class specifically for the coupled-cluster class 

3. **diis(object)**
   
The initiator takes the arguements *ts* initial singles amplitudes, *td* initial doubles amplitudes and *capacity* the size of the diis buffers. See the iterator routine in coupledCluster class for details on it's implementation.

   There are two routines \
   *refresh_store(self, ts, td)* - adds new ts and td to a buffer. \
   *build(self, ts, td)* - builds the B-matrix and weights and generates extrapolated ts and td. \


4. **fastCoupledCluster(type, fock, eps, c, eri, nuclearRepulsion, data)**

parameters - *type* is the couple-cluster method (see class for options), *fock* is the Fock matrix in the AO basis, *eps* the orbital energies corresponding to *c* the orbital coefficients in the AO basis, *eri* the two electron repulsion integrals in AO basis (linear form), *nuclearRepulsion* is the energy of the nuclear-nuclear interaction and *data* is a dictionary corresponding to that to be passed to the coupledCluster class initiator. Routine calls spinMO class then the coupledCluster class. The output is written to the output file 'harpy.html' and the energy dictionary from coupledCluster class is returned.
   The main rhf module calls this routine in response to the **post={}** keys '+s', '+d', 't', '+2', '+q', +l' and '+L' for ccd, ccsd, ccsd(t), cc2, qcisd, lccd and lccsd respectively. The key for lambda is '+^' which returns the lambda pseudo-energy.
   
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

        #call coupled-cluster class
        data = {'method':'ccsd(t)','electrons':electrons, 'cycle_limit': 50, 'precision':1e-10, 'verbose':False}
        cc = coupledCluster(fs, gs, e, data)

        if cc.converged:
            cc.energy['nuclear'] = nuclearRepulsion

    >>>{'cc': -0.07068008870929447, 'mp2': -0.04914963668931829, 'eHF': -82.94444701585344, 'pt': -9.987727029497426e-05, 'nuclear': 8.00236706181077}

The are the results for the methods \
ccsd(t) {'cc': -0.07068008870929447, 'mp2': -0.04914963668931829, 'eHF': -82.94444701585344, 'pt': -0.00009987727029497426, 'nuclear': 8.00236706181077} \
ccsd    {'cc': -0.07068008870929447, 'mp2': -0.04914963668931829, 'eHF': -82.94444701585344, 'nuclear': 8.00236706181077} \
ccd     {'cc': -0.07015048756595696, 'mp2': -0.04914963668931829, 'eHF': -82.94444701585344, 'nuclear': 8.00236706181077} \
cc2     {'cc': -0.049399113048481456, 'mp2': -0.04914963668931829, 'eHF': -82.94444701585344, 'nuclear': 8.00236706181077} \
qcisd   {'cc': -0.07071279699443576, 'mp2': -0.049149636689279796, 'eHF': -82.94444701585147, 'nuclear': 8.00236706181077} \
lccd    {'cc': -0.0719291640366139, 'mp2': -0.04914963668931829, 'eHF': -82.94444701585344, 'nuclear': 8.00236706181077} \
lccsd   {'cc': -0.07257658932696193, 'mp2': -0.04914963668931829, 'eHF': -82.94444701585344, 'nuclear': 8.00236706181077}

As a note, the ccsd(t) with explicit loop does a computation in about 50s (old Celeron laptop) for H<sub>2</sub>O in STO-3G basis, these routines do all the above coupled-cluster methods in about 1s.

5. **class ccsdLambda(object)**

The initiator takes the arguements (fs, gs, e, data) where *fs* is the MO spin Fock matrix, *gs* are the MO spin two electron repusion integrals, *e* are the AO basis orbital energies and *data* is a dictionary {'method': 'ccsd', 'electrons':.., 'cycle_limit'.., 'precision':.., 'verbose':..} supplied to the class by the caller. *method* is the coupled-cluster method, *electrons* are the number of electrons, *cycle_limit* is the maximum number of iterations allowed, *precision* is the number of decimal places to converge to and *verbose* is a boolean flag - if True information on each cycle will be printed.\The available *method*s are 'ccd', 'ccsd', 'ccsd(t)', 'cc2' and linear ccd 'lccd' and linear ccsd 'lccsd'.\
   On exit the class has instance variables *ts* the singles amplitudes, *td* the doubles amplitudes, *converged* a boolean flag (which should be checked on exit)
   indicating a successful convergence of the iterations and 'energy'. *ts* is of dimension [o,v], *td* is of dimension [o,o,v,v]. *energy* is a dictionary with keys
   + 'cc'      - coupled-cluster correction
   + 'eHF'     - Hartree-Fock electronic energy
   + 'lagrange'- Lagrangian energy
   + 'nuclear' - nuclear repulsion energy (user added)

The class has a method *intermediates(slice, tau)* where *slice* can be one of 'oo', 'vv', 'ov', 'oooo', 'vvvv' or 'ovvo' and tau is a boolean (default=False) which for lambda coupled-cluster will be False (True for coupled-cluster calculations). There are five routines\
   *initialise_amplitudes(self)* - sets ts and td initial values and computes d-tensors (inverses). Computes HF energy.
   *intermediates(self, \_slice, tilde=False)* - F, W and G coupled-cluster intermediates.
   *update_amplitudes(self)* - generate next guess of the amplitudes for ccd, ccsd, ccsd(t) and cc2.
   *lambda_energy(self)* - compute the cluster correction energy.
   *lagrangian_energy* - compute the Lagrangian energy.
   *iterator(self)* - the main iteration loop, amplitude update function passed as arguement.

The intermediates (non-tilde) are for F-type 'oo', 'vv' and 'ov' and for W-type 'oooo', 'vvvv', 'ovvo', 'ooov', 'vovv', 'ovoo' and 'vvvo'. In addition there are two G-type 3-body intermediates 'OO' and 'VV'.

The output is
{'cc': -0.06888821143156079, 'eHF': -82.94444701585394, 'lagrange': -0.07068008870920173, 'nuclear': 8.00236706181077}
   
Add the following to the previous code to run lambda

    data['method'] = 'ccsd'
    l = ccsdLambda(fs, gs, e, data)
    if l.converged:
        l.energy['nuclear'] = nuclearRepulsion

    >>>{'cc': -0.06888821143156079, 'eHF': -82.94444701585394, 'lagrange': -0.07068008870920173, 'nuclear': 8.00236706181077}

The code include here can be run as 'python3 cc/fcc/py' from the harpy/source directory.

The method 'ccsd(t)' has been added for lambda ie *data[method] = 'ccsd(t)'*. This will do a &Lambda;CCSD(T) computation, that is adding a pertutbative triples to the &Lambda;CCSD calculation. In this case the *energy* dictionary has added keys
  + 'pt' - perturbative triples correction at CCSD level
  + 'pl' - perturbative triples correction at &Lambda;CCSD level.

The &Lambda; perturbative triples has not been fully tested against other sources.

The lambda class has a _oprdm_ method which returns the one-particle response density matrix.

6. **class eom_ccsd(object)**

The initiator takes the arguements (cc, roots, partitioned) where *cc* is an instance of the coupled-cluster class (converged), *roots* is a range of enrgy values in eV between which the excitations are to be reported (in *excitations* class variable). The range is given as a list ie [15, 30]. *partitioned* is a boolean flag which if set to True will use an approximate DD-block, that is P-EOM_CCSD. The raw eigenvalues and eigenvectors are given in class variables .e and .v (in atomic units) and the class variable .excitations contains elements of the type [energy, multiplicity] eg [3.56109, 't'] for energies in eV is range *roots*. The class is run as 
```
    #EOM-CCSD
    eom = eom_ccsd(cc, roots=[4, 19], partitioned=False)
    print('root   energy (eV)   multiplicity\n---------------------------------')
    for i, root in enumerate(eom.excitations):
        print('{:<2d}      {:<10.5f}        {:<s}'.format(i, root[0], root[1]))
```
with output for H<sub>2</sub> (0.74 interbond in angstroms) in 3-21g
```
---------------------------------
root   energy (eV)   multiplicity
---------------------------------
0       10.85265          t *
1       15.89841          s
2       26.47121          t
3       30.52162          s
4       31.88140          s
5       40.40196          t
```
* values added in parentheses are from Gaussian (supplied by Josh Goings)