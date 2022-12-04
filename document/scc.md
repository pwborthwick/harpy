# Coupled Cluster Module - Single, Doubles and Perturbative Triples

This module uses code based on the paper **J.F. Stanton, J. Gauss, J.D. Watts, and R.J. Bartlett, J. Chem. Phys. 
volume 94, pp. 4334-4345 (1991)**. There is a Crawford project on the Singles and Doubles [see](https://github.com/CrawfordGroup/ProgrammingProjects/tree/master/Project%2305) and on the Perturbative Triples [see](https://github.com/CrawfordGroup/ProgrammingProjects/tree/master/Project%2306). Joshua Goings has another good discussion with code [here](https://joshuagoings.com/2013/07/17/coupled-cluster-with-singles-and-doubles-ccsd-in-python/). These routines use full spin-spin arrays, a more efficient storage regime would use occupied-virtual slicing. The routines here are coded with explicit loops rather than einsum so it's slow - but explicit. For faster versions see either cogus.py or fcc.py.

1.  **tauSpin(i, j, a, b, ts, td)**

    parameters - *i* and *j* refer to occupied spin basis functions, *a*, *b* are virtual spin basis functions.
					 *ts* are the singles amplitudes an array \[spinOrbitals, spinOrbitals]. Spin orbital count being twice the number
					 of basis functions. *td* are doubles amplitudes an array or order \[spinOrbitals, spinOrbitals, spinOrbitals, spinOrbitals].
					 Returns float. Equation (9) of Stanton et al.

2.  **tautauSpin(i, j, a, b, ts, td)**

    parameters - *i* and *j* refer to occupied spin basis functions, *a*, *b* are virtual spin basis functions.
					 *ts* are the singles amplitudes and *td* the doubles amplitude. Returns float. Equation (10).

3.  **updateIntermediates(fs, ts, td, eriMOspin, nElectrons)**

    parameters - *fs* is the Fock matrix in the MO spin basis, *ts* and *td* are the singles and doubles amplitudes, *eriMOspin* are the 2-electron
		             repulsion integrals in the MO spin basis as a \[spinOrbitals, spinOrbitals, spinOrbitals, spinOrbitals] array. *nElectrons*
		             is the electron count. Evaluates equations (3) - (8) and returns the updated ts and td arrays. Calls functions tau, tauSpin, 
		             amplitudesT1 and amplitudesT2. Builds the f and w occupied (\_o), unoccupied (\_u) and mixed (\_m) arrays used by amplitudesT1 and 
		             amplitudeT2. The f and w arrays are global variables to this and the amplitude T1 and T2 routines.
                 
4.  **amplitudesT1()**

    parameters - none. Returns updated ts. Equation (1).

5.  **amplitudesT2()**

    parameters - none. Returns updated td. Equation (2).

6.  **amplitudesT3(fs, ts, td, eriMOspin, nElectrons)**

    parameters - *fs* is the Fock matrix in the MO spin basis, *ts* and *td* are the singles and doubles amplitudes, *eriMOspin* are the 2-electron
		             repulsion integrals in the MO spin basis as a \[spinOrbitals, spinOrbitals, spinOrbitals, spinOrbitals] array. *nElectrons*
		             is the electron count. Returns the perturbative triples correction as a float.

7.  **ccsd(atoms, eigenVectors, charge, fock, ERI, iterations, convergence, totalEnergy, diisStatus)**

    parameters - *atoms* is an array of atom objects (the molecular atoms), *eigenVectors* are the eigenfunctions for the final Fock matrix,
		*charge* the molecular charge, *fock* the final fock matrix, *ERI* the 2-electron repulsion integrals, *iterations* the maximum number
		of iterations allowed, *convergence* is the convergence tolerence, *totalEnergy* the final converged SCF energy
		(not needed for calculation) and *diisStatus* indicates whether DIIS is to be used (='on') or not (='off').\
    &nbsp; &nbsp; This routine calls buildFockMOspin to get the Fock matrix in the MO spin basis, then buildEriMO to get 2-electron 
		repulsion integrals in the MO basis and subsequently buildEriDoubleBar to transform to a double-bar spin basis integral in physicist notation. The mp2 
		energy is found using the doubles amplitudes. The calculation of the amplitudes is then calculated in a loop until the 
		amplitude difference (of both amplitudes) is below the convergence threshold. *diis* can be employed which works on the 
		concatenation of the two (linear) amplitude arrays into a single object and using the difference of two successive cycles 
		(which should tend to 0 at convergence) as the *diis* error object. Returns CCSD correction and single and double amplitudes.

8.  **ccsdEnergy(fs, ts, td, eriMOspin, nElectrons)**

    parameters - *fs* is the Fock matrix in the MO spin basis, *ts* and *td* are the singles and doubles amplitudes, *eriMOspin* are the 2-electron
		             repulsion integrals in the MO spin basis as a (spinOrbitals, spinOrbitals, spinOrbitals, spinOrbitals) array. *nElectrons*
		             is the electron count. Returns the total ccsd energy as float.

- - -
## LCCD
The linear coupled-cluster doubles are a subset of the CCD equations. Hence there are no T<sub>1</sub> amplitudes and the T<sub>2</sub> amplitudes are (diagram labelling from Shavitt & Bartlett - Many-Body Methods in Chemistry and Physics Figure 9.2)
+ g<sub>abij</sub> **\[D1]**
+ P(ab){*f*<sub>bc</sub> t<sup>ae</sup><sub>ij</sub>} **\[D2a]**
+ -P(ij){*f*<sub>mj</sub >t<sup>ab</sup><sub>im</sub>} **\[D2b]**
+ 0.5 g<sub>abef</sub> t<sup>ef</sup><sub>ij</sub> **\[D2c]**
+ 0.5 g<sub>mnij</sub >t<sup>ab</sup><sub>mn</sub> **\[D2d]**
+ P(ab)P(ij){g<sub>mbej</sub> t<sup>ae</sup><sub>im</sub>} **\[D2e]**
+ 0.25 g<sub>mnef</sub> t<sup>ef</sup><sub>ij</sub> t<sup>ab</sup><sub>mn</sub>  **\[D3a]**
+ P(ij){g<sub>mnef</sub> t<sup>ae</sup><sub>im</sub> t<sup>bf</sup><sub>jn</sub>} **\[D3b]**
+ -0.5P(ij){g<sub>mnef</sub> t<sup>fe</sup><sub>im</sub> t<sup>ab</sup><sub>nj</sub>} **\[D3c]**
+ -0.5P(ab){g<sub>mnef</sub> t<sup>ae</sup><sub>nm</sub> t<sup>fb</sup><sub>ij</sub>} **\[D3d]**

For the Linear CCD we use **\[D1], \[D2a], \[D2b], \[D2c], \[D2d] and \[D2e]**

9.  **lccd(atoms, eigenVectors, charge, fock, ERI, iterations, convergence):**

    parameters - *atoms* is an array of atom objects (the molecular atoms), *eigenVectors* are the eigenfunctions for the final Fock matrix,
			*charge* the molecular charge, *fock* the final fock matrix, *ERI* the 2-electron repulsion integrals, *iterations* the number 
			of iterations and *convergence* are for control of the iteration process. Returns the linear coupled cluster doubles energy 
			(LCCD) also known as coupled electron pair approximation zero (CEPA0). Formula used is [here](https://github.com/bge6/psi4numpy/blob/master/Tutorials/08_CEPA0_and_CCD/8b_CEPA0_and_CCD.ipynb).

10. **lccdAmplitudes(td, eriMOspin, fockMOspin, nElectrons, spinOrbitals)**

    parameters - *td* the doubles amplitudes, *eriMOspin* are the 2-electron repulsion integrals in the MO spin basis,  *fockMOspin* is the fock 				 	matrix in the MO spin basis, *nElectrons* is the electron count and *spinOrbitals* the total orbitals in the spin basis.

11. **ccdEnergy( td, eriMOspin, nElectrons, spinOrbitals)**

    parameters - *td* the doubles amplitudes, *eriMOspin* are the 2-electron repulsion integrals in the MO spin basis, *nElectrons* is the 	 				 	electron count and *spinOrbitals* the total orbitals in the spin basis.
- - -
## CCD
The equations are given above in the LCCD section.

12. **ccd(atoms, eigenVectors, charge, fock, ERI, iterations, convergence):**

    parameters - *atoms* is an array of atom objects (the molecular atoms), *eigenVectors* are the eigenfunctions for the final Fock matrix,
		 *charge* the molecular charge, *fock* the final fock matrix, *ERI* the 2-electron repulsion integrals, *iterations* the number of 
		iterations and *convergence* are for control of the iteration process. Returns the coupled cluster doubles energy (CCD). 
		Formula used is [here](https://github.com/bge6/psi4numpy/blob/master/Tutorials/08_CEPA0_and_CCD/8b_CEPA0_and_CCD.ipynb).

13. **ccdAmplitudes(td, eriMOspin, fockMOspin, nElectrons, spinOrbitals)**

    parameters - *td* the doubles amplitudes, *eriMOspin* are the 2-electron repulsion integrals in the MO spin basis,  *fockMOspin* is the fock 				 		matrix in the MO spin basis, *nElectrons* is the electron count and *spinOrbitals* the total orbitals in the spin basis.

- - -
## &Lambda;-CCSD
These equations are taken from [Gauss and Stanton J. Chem. Phys., Vol. 103, No. 9, 1 September 1995](http://www2.chemia.uj.edu.pl/~migda/Literatura/pdf/JCP03561.pdf)\
**Lambda intermediates**
+ F<sup>&lambda;</sup><sub>ae</sub> = F<sub>ae</sub> - 0.5t<sup>a</sup><sub>m</sub>F<sub>me</sub>
+ F<sup>&lambda;</sup><sub>mi</sub> = F<sub>mi</sub> + 0.5t<sup>e</sup><sub>i</sub>F<sub>me</sub>
+ F<sup>&lambda;</sup><sub>me</sub> = F<sub>mi</sub>

+ + F<sub>ae</sub> = *f*<sub>ae</sub> - t<sup>a</sup><sub>m</sub>*f*<sub>me</sub> + t<sup>f</sup><sub>m</sub> g<sub>amef</sub> - 0.5 &tau;<sup>af</sup><sub>mn</sub> g<sub>mnef</sub> 
+ + F<sub>mi</sub> = *f*<sub>mi</sub> + t<sup>e</sup><sub>i</sub>*f*<sub>me</sub> + t<sup>e</sup><sub>n</sub> g<sub>mnie</sub> + 0.5 &tau;<sup>ef</sup><sub>in</sub> g<sub>mnef</sub> 
+ + F<sub>me</sub> = *f*<sub>me</sub> + t<sup>f</sup><sub>n</sub> g<sub>mnef</sub>

+ W<sup>&lambda;</sup><sub>mnij</sub> = W<sub>mnij</sub> + 0.25&tau;<sup>ef</sup><sub>ij</sub>g<sub>mnef</sub>
+ W<sup>&lambda;</sup><sub>abef</sub> = W<sub>abef</sub> + 0.25&tau;<sup>ab</sup><sub>mn</sub>g<sub>mnef</sub>
+ W<sup>&lambda;</sup><sub>mbej</sub> = W<sub>mbej</sub> - 0.5&tau;<sup>fb</sup><sub>jn</sub>g<sub>mnef</sub>

+ + W<sub>mnij</sub> = g<sub>mnij</sub> + P(ij){t<sup>e</sup><sub>j</sub> g<sub>mnie</sub>} + 0.5&tau;<sup>ef</sup><sub>ij</sub> g<sub>mnef</sub> 
+ + W<sub>abef</sub> = g<sub>abef</sub> - P(ab){t<sup>b</sup><sub>m</sub> g<sub>amef</sub>} + 0.5&tau;<sup>ab</sup><sub>mn</sub> g<sub>mnef</sub> 
+ + W<sub>mbej</sub> = W<sub>(ovvo)</sub> = g<sub>mbej</sub> + t<sup>f</sup><sub>j</sub> g<sub>mbef</sub> - t<sup>b</sup><sub>n</sub> g<sub>mnej</sub> - (t<sup>fb</sup><sub>jn</sub> + t<sup>f</sup><sub>j</sub>t<sup>b</sup><sub>n</sub>) g<sub>nmfe</sub>

+ W<sup>&lambda;</sup><sub>mnie</sub> = W<sub>mnie</sub>
+ W<sup>&lambda;</sup><sub>amef</sub> = W<sub>amef</sub>
+ W<sup>&lambda;</sup><sub>mbij</sub> = W<sub>mbij</sub>
+ W<sup>&lambda;</sup><sub>abei</sub> = W<sub>abei</sub>

+ + W<sub>mnie</sub> = g<sub>mnie</sub> + t<sup>f</sup><sub>i</sub> g<sub>mnfe</sub> 
+ + W<sub>amef</sub> = g<sub>amef</sub> - t<sup>a</sup><sub>n</sub> g<sub>nmef</sub>
+ + W<sub>mbij</sub> = g<sub>mbij</sub> - F<sub>me</sub>t<sup>be</sup><sub>ij</sub> - t<sup>b</sup><sub>n</sub>W<sub>mnij</sub> + 0.5&tau;<sup>ef</sup><sub>ij</sub> g<sub>mbef</sub> + P(ij){t<sup>be</sup><sub>jn</sub> g<sub>mnie</sub>} + P(ij){t<sup>e</sup><sub>i</sub> (g<sub>mbej</sub> - t<sup>bf</sup><sub>nj</sub> g<sub>mnef</sub>)} 
+ + W<sub>abei</sub> = g<sub>abei</sub> - F<sub>me</sub>t<sup>ab</sup><sub>mi</sub> + t<sup>f</sup><sub>i</sub>W<sub>abef</sub> + 0.5&tau;<sup>ab</sup><sub>mn</sub> g<sub>mnei</sub> - P(ab){t<sup>af</sup><sub>mi</sub> g<sub>mbef</sub>} - P(ab{t<sup>a</sup><sub>m</sub> (g<sub>mbei</sub> - t<sup>bf</sup><sub>ni</sub> g<sub>mnef</sub>)} 

**Three-body Terms**
+ G<sub>ae</sub> = -0.5t<sup>ef</sup><sub>mn</sub>&lambda;<sup>mn</sup><sub>af</sub>
+ G<sub>mi</sub> = 0.5t<sup>ef</sup><sub>mn</sub>&lambda;<sup>in</sup><sub>ef</sub>

**Lambda Equations**\
**&Lambda;<sub>1</sub>** (&lambda;<sup>i</sup><sub>a</sub>) = F<sup>&lambda;</sup><sub>ia</sub> + &lambda;<sup>i</sup><sub>e</sub>F<sup>&lambda;</sup><sub>ea</sub> - &lambda;<sup>m</sup><sub>a</sub>F<sup>&lambda;</sup><sub>im</sub> + &lambda;<sup>m</sup><sub>e</sub>W<sup>&lambda;</sup><sub>ieam</sub> + 0.5&lambda;<sup>im</sup><sub>ef</sub>W<sup>&lambda;</sup><sub>efam</sub> - 0.5&lambda;<sup>mn</sup><sub>ae</sub>W<sup>&lambda;</sup><sub>iemn</sub> - G<sub>ef</sub>W<sup>&lambda;</sup><sub>eifa</sub> - G<sub>mn</sub>W<sub>mina</sub>\
**&Lambda;<sub>2</sub>** (&lambda;<sup>ij</sup><sub>ab</sub>) = g<sub>ijab</sub>  - P(ij){&lambda;<sup>im</sup><sub>ab</sub>F<sup>%lambda;</sup><sub>jm</sub>} + 0.5&lambda;<sup>mn</sup><sub>ab</sub>W<sup>&lambda;</sup><sub>ijmn</sub> + 0.5&lambda;<sup>ij</sup><sub>ef</sub>W<sup>&lambda;</sup><sub>efab</sub> + P(ij){&lambda;<sup>i</sup><sub>e</sub>W<sup>&lambda;</sup><sub>ejab</sub>} - P(ab){&lambda;<sup>m</sup><sub>a</sub>W<sup>&lambda;</sup><sub>ijmb</sub>} + P(ij)P(ab){&lambda;<sup>im</sup><sub>ae</sub>W<sup>&lambda;</sup><sub>jebm</sub>} + P(ij)P(ab){&lambda;<sup>i</sup><sub>a</sub>F<sup>&lambda;</sup><sub>jb</sub>} + P(ab){g<sub>ijae</sub>G<sub>be</sub>} - P(ij){g<sub>imab</sub>G<sub>mj</sub> }

**&Lambda;-CCSD Pseudoenergy**\
E<sub>&lambda;</sub> = &lambda;<sup>i</sup><sub>a</sub>f<sub>ai</sub> + 0.25&lambda;<sup>ij</sup><sub>ab</sub>g<sub>abij</sub>

14.  **ccsd_lambda(fs, eriMOspin, ts, td, nElectrons, iterations, tolerance)**

     parameters - *fs* is the converged Fock matrix in the molecular spin basis, *eriMOspin* are the electron repulsion integrals in the molecular spin basis, *ts* are the converged single amplitudes from a CCSD computation, *td* are the converged doubles amplitudes from a CCSD computation, *nElectrons* are the number of electrons, *iterations* are the maximum number of cycles and *tolerance* is the criterion for convergence. Returns the &Lambda;-CCSD pseudo-energy, the single and doubles &Lambda;-amplitudes and the intermediates.

- - -
## CC2
CC2 is an approximate scheme to CCSD. The T<sub>1</sub> amplitudes are the same as CCSD, however the T<sub>2</sub> amplitudes are given by <&psi;<sup>ab</sup><sub>ij</sub>| **H<sub>N</sub>** e<sup>T<sub>1</sub></sup> + **F<sub>N</sub>** T<sub>2</sub>|0> = 0\
where **H<sub>N</sub>** and **F<sub>N</sub>** are the normal-ordered Hamiltonian and Fock matrices. **\[&nbsp;]** are the coupled-cluster diagram designations. Equations for T-amplitudes taken from reference above (Gauss and Stanton) Table I(a) & (b) and (tilde) intermediates taken from Table III.

T<sub>1</sub> = f<sub>ai</sub> + *F*<sub>ae</sub>t<sup>e</sup><sub>i</sub> - *F*<sub>mi</sub>t<sup>a</sup><sub>m</sub> + *F*<sub>me</sub>t<sup>ae</sup><sub>im</sub> +
t<sup>e</sup><sub>m</sub>g<sub>amie</sub> - 0.5t<sup>ae</sup><sub>mn</sub>g<sub>mnie</sub> + 0.5t<sup>ef</sup><sub>im</sub>g<sub>amef</sub>
+ f<sub>ai</sub> **\[S<sub>1</sub>]**
+ *F*<sub>ae</sub>t<sup>e</sup><sub>i</sub>
+ + t<sup>e</sup><sub>i</sub>f<sub>ae</sub> **\[S<sub>3<sub>a</sub></sub>]**
+ + -0.5t<sup>e</sup><sub>i</sub>t<sup>a</sup><sub>m</sub>f<sub>me</sub> **\[S<sub>5<sub>a</sub></sub>]**
+ + t<sup>e</sup><sub>i</sub>t<sup>f</sup><sub>m</sub>g<sub>amef</sub> **\[S<sub>5<sub>b</sub></sub>]**
+ + -t<sup>e</sup><sub>i</sub> &tau;<sup>af</sup><sub>mn</sub>g<sub>mnef</sub>
+ + + -0.5t<sup>e</sup><sub>i</sub> t<sup>af</sup><sub>mn</sub>g<sub>mnef</sub> **\[S<sub>4<sub>a</sub></sub>]**
+ + + -0.5t<sup>e</sup><sub>i</sub> t<sup>a</sup><sub>m</sub>t<sup>f</sup><sub>n</sub>g<sub>mnef</sub> **\[S<sub>6</sub>]**
+ *F*<sub>mi</sub>t<sup>a</sup><sub>m</sub>
+ + -t<sup>a</sup><sub>m</sub>f<sub>mi</sub> **\[S<sub>3<sub>b</sub></sub>]**
+ + -0.5t<sup>a</sup><sub>m</sub>t<sup>e</sup><sub>i</sub>f<sub>me</sub> **\[S<sub>5<sub>a</sub></sub>]**
+ + -t<sup>a</sup><sub>m</sub>t<sup>e</sup><sub>n</sub>g<sub>mnie</sub> **\[S<sub>5<sub>c</sub></sub>]**
+ + -t<sup>a</sup><sub>m</sub> &tau;<sup>ef</sup><sub>in</sub>g<sub>mnef</sub>
+ + + -0.5t<sup>a</sup><sub>m</sub> t<sup>ef</sup><sub>in</sub>g<sub>mnef</sub> **\[S<sub>4<sub>b</sub></sub>]**
+ + + -0.5t<sup>a</sup><sub>m</sub> t<sup>e</sup><sub>i</sub>t<sup>f</sup><sub>n</sub>g<sub>mnef</sub> **\[S<sub>6</sub>]**
+ *F*<sub>me</sub>t<sup>ae</sup><sub>im</sub>
+ + t<sup>ae</sup><sub>im</sub>f<sub>me</sub> **\[S<sub>2<sub>a</sub></sub>]**
+ + t<sup>ae</sup><sub>im</sub>t<sup>f</sup><sub>n</sub>g<sub>mnef</sub> **\[S<sub>4<sub>c</sub></sub>]**
+ t<sup>e</sup><sub>m</sub>g<sub>amie</sub> **\[S<sub>3<sub>c</sub></sub>]**
+ -0.5t<sup>ae</sup><sub>mn</sub>g<sub>mnie</sub> **\[S<sub>2<sub>c</sub></sub>]**
+ 0.5t<sup>ef</sup><sub>im</sub>g<sub>amef</sub> **\[S<sub>2<sub>b</sub></sub>]**

T<sub>2</sub> = g<sub>abij</sub> + P(ij){t<sup>ae</sup><sub>ij</sub>(*F*<sub>be</sub> - 0.5t<sup>b</sup><sub>n</sub>*F*<sub>me</sub>)} - P(ab){t<sup>ab</sup><sub>im</sub>(*F*<sub>mj</sub> + 0.5t<sup>e</sup><sub>j</sub>*F*<sub>me</sub>)} + 0.5&tau;<sup>ab</sup><sub>mn</sub>*W*<sub>mnij</sub> + 0.5&tau;<sup>ef</sup><sub>ij</sub>*W*<sub>abef</sub> + P(ab)P(ij){t<sup>ae</sup><sub>im</sub>*W*<sub>mbej</sub> - t<sup>e</sup><sub>i</sub>t<sup>a</sup><sub>m</sub>g<sub>mbej</sub>} + P(ij){t<sup>e</sup><sub>i</sub>g<sub>abej</sub>} - P(ab){t<sup>a</sup><sub>m</sub>g<sub>mbij</sub>}

Since only T<sub>1</sub> operates on **H<sub>N</sub>** we can reduce the above equation to\
T<sub>2</sub> = g<sub>abij</sub> + 0.5&tau;<sup>ab</sup><sub>mn</sub>*W*<sub>mnij</sub> + 0.5&tau;<sup>ef</sup><sub>ij</sub>*W*<sub>abef</sub> - P(ab)P(ij)t<sup>e</sup><sub>i</sub>t<sup>a</sup><sub>m</sub>g<sub>mbej</sub> + P(ij)t<sup>e</sup><sub>i</sub>g<sub>abej</sub> - P(ab)t<sup>a</sup><sub>m</sub>g<sub>mbij</sub>
+ g<sub>abij</sub> **\[D<sub>1</sub>]**
+ 0.5&tau;<sup>ab</sup><sub>mn</sub>*W*<sub>mnij</sub> = t<sup>a</sup><sub>m</sub>t<sup>b</sup><sub>n</sub>*W*<sub>mnij</sub> 
+ + t<sup>a</sup><sub>m</sub>t<sup>b</sup><sub>n</sub>g<sub>mnij</sub> **\[D<sub>6<sub>b</sub></sub>]**
+ + t<sup>a</sup><sub>m</sub>t<sup>b</sup><sub>n</sub>P(ij){t<sup>e</sup><sub>j</sub>g<sub>mnie</sub>} **\[D<sub>8<sub>b</sub></sub>]**
+ + 0.5t<sup>a</sup><sub>m</sub>t<sup>b</sup><sub>n</sub>t<sup>e</sup><sub>i</sub>t<sup>f</sup><sub>j</sub>g<sub>mnef</sub> **\[D<sub>9</sub>]**
+ 0.5&tau;<sup>ef</sup><sub>ij</sub>*W*<sub>abef</sub>
+ + t<sup>e</sup><sub>i</sub>t<sup>f</sup><sub>j</sub>g<sub>abef</sub> **\[D<sub>6<sub>a</sub></sub>]**
+ + -t<sup>e</sup><sub>i</sub>t<sup>f</sup><sub>j</sub>P(ab){t<sup>b</sup><sub>m</sub>g<sub>amef</sub>} **\[D<sub>8<sub>a</sub></sub>]**
+ + 0.5t<sup>e</sup><sub>i</sub>t<sup>f</sup><sub>j</sub>t<sup>a</sup><sub>m</sub>t<sup>b</sup><sub>n</sub>g<sub>mnef</sub> **\[D<sub>9</sub>]**
+ P(ab)P(ij)t<sup>e</sup><sub>i</sub>t<sup>a</sup><sub>m</sub>g<sub>mbej</sub> **\[D<sub>6<sub>c</sub></sub>]**
+ P(ij)t<sup>e</sup><sub>i</sub>g<sub>abej</sub> **\[D<sub>4<sub>a</sub></sub>]**
+ -P(ab)t<sup>a</sup><sub>m</sub>g<sub>mbij</sub> **\[D<sub>4<sub>b</sub></sub>]**

Need to add in the T<sub>2</sub> operating on **F<sub>N</sub>**
+ t<sup>ae</sup><sub>ij</sub>f<sub>be</sub> **\[D<sub>2<sub>a</sub></sub>]**
+ -t<sup>ab</sup><sub>im</sub>f<sub>mj</sub> **\[D<sub>2<sub>b</sub></sub>]**

15. **cc2(atoms, eigenVectors, charge, fock, ERI, iterations, convergence, totalEnergy, diisStatus)**
  
    parameters - *atoms* is an array of atom objects (the molecular atoms), *eigenVectors* are the eigenfunctions for the final Fock matrix,
	charge the molecular charge, *fock* the final fock matrix, *ERI* the 2-electron repulsion integrals, *iterations* the number
	of iterations allowed from iteration, *convergence* is the convergence tolerence, *totalEnergy* the final converged energy
	(not needed for calculation) and *diisStatus* indicates whether DIIS is to be used (='on') or not (='off'). Returns the CC2 level energy correction and the T<sub>1</sub> and T<sub>2</sub> amplitudes.

- - -
## LCCSD
For LCCSD we add in **\[S1], \[S2a], \[S2b], \[S2c], \[S3a], \[S3b] and \[S3c]** single and **\[D4a]** and **\[D4b]** from the LCCD equations.

16. **lccsd(atoms, eigenVectors, charge, fock, ERI, iterations, convergence, diisStatus, SCFenergy)**
 
    parameters - *atoms* is an array of atom objects (the molecular atoms), *eigenVectors* are the eigenfunctions for the final Fock matrix,
	charge the molecular charge, *fock* the final fock matrix, *ERI* the 2-electron repulsion integrals, *iterations* the number
	of iterations allowed from iteration, *convergence* is the convergence tolerence, *totalEnergy* the final converged energy
	(not needed for calculation) and *diisStatus* indicates whether DIIS is to be used (='on') or not (='off'). Returns the LCCSD level energy correction and the T<sub>1</sub> and T<sub>2</sub> amplitudes.
