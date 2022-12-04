# Integral Module - integrals and integral matrices

This module uses code derived from the excellent article by Joshua Goings entitled 'A (hopefully)'
gentle guide to the computer implementation of molecular integrals over Gaussian basis functions' [see]
(https://joshuagoings.com/2017/04/28/integrals/)

1.  **e(ia, ja, type, r, ie, je)**

	parameters - *i* and *j* refer to basis functions, so ia is angular momentum on orbital i ie is exponent on orbital i. Type is an integer depends on integral being evaluated. *r* is tuple of vector between atom centers. Uses exp, returns float.

2.  **overlap(ia, ja, ie, je, ir, jr)**

	parameters - *i* and *j* refer to basis functions, so ir is atom center coordinates of i. Uses pow, returns float.

3.  **s(iBasis, jBasis)**

	parameters - *iBasis* , *jBasis* are basis objects. Returns overlap between iBasis and jBasis. Returns float.

4.  **buildOverlap(bases)**

	parameters - *bases* is an array of basis objects (the molecular basis). Computes the overlap matrix. Returns \[nbf, nbf] array where nbf are the number of basis functions.

5.  **kinetic(ia, ja, ie, je, ir, jr)**

	parameters - *i* and *j* refer to basis functions, so ir is atom center coordinates of i. Calls overlap returns float. Kinetic energy integral between two Gaussians.

6.  **k(iBasis, jBasis)**

	parameters - *iBasis* , *jBasis* are basis objects. Returns kinetic between contracted Gaussians and returns a float.

7.  **buildKinetic(bases)**

	parameters - *bases* is an array of basis objects (the molecular basis). Computes the kinetic matrix. Returns \[nbf, nbf] array. One electron exchange.

8.  **j(v, n, p, r, rnorm)**

	parameters - *v* is vector of order of Coulomb Hermite derivatives in x,y,z. *n* is order of Boys function. *p* is sum of exponents of functions comprising composite center and *r* is vector of distance between composite center *p* and a nuclear center *c*. *rnorm* is distance between *p* and *c*. Returns float, recursive function. Calls Boys.

9.  **coulomb(ia, ja, ie, je, ir, jr, kr)**

	parameters - *i* and *j* refer to basis functions, so kr is atom center coordinates of k. Computes the 1e-coulomb integral between two Gaussians. Calls e and j, returns float.

10. **v(iBasis, jBasis, r)**

    parameters - *iBasis* , *jBasis* are basis objects, r is center of nucleus. Returns coulomb between contracted Gaussians for nuclear center c. Calls coulomb and returns float.

11. **buildCoulomb(atoms, bases)**

	parameters - *bases* is an array of basis objects (the molecular basis) and *atoms* is an array of atom objects. Computes the nuclear repulsion integral matrix. Returns \[nbf,nbf] array.

12. **er(ie, je, ke, le, ia, ja, ka, la, ir, jr, kr, lr)**

	parameters - *i*, *j*, *k*, *l* refer to basis functions, so *kr* is atom center coordinates of *k*. Computes the electron-electron repulsion between Gaussians. Calls e and j and uses math.pow.

13. **eri(iBasis, jBasis, kBasis, lBasis)**

	parameters - *iBasis* , *jBasis*, *kBasis* and *lBasis* are basis objects. Returns 2-electron repulsion between contracted Gaussians and returns a float. Calls er.

14. **buildEri(bases)**

	parameters - *bases* is an array of basis objects (the molecular basis). Computes the 2-electron repulsion integrals as a linear array. Symmetry is taken into account so only upper triangle of ij, kl matrix is stored. More information of handling the indexes of 2-electron integrals can be found in the Crawford project #3 [see](https://github.com/CrawfordGroup/ProgrammingProjects/tree/master/Project%2303). Integrals are in Mulliken (chemists) order.

15. **iEri(i, j, k, l)**

  	parameters - *i*, *j*, *k*, *l* are basis indexes. Returns a pointer into array formed in buildEri.

16. **buildHamiltonian(type, S, K, J)**

    parameters - *type* method of initial guess for fock matrix, currently either 'core' or 'gwh'. *S* is overlap matrix, *K* is 1e-exchange matrix and *J* the 1e-coulomb matrix. If type is 'core' returns (J+K), (J+K). If 'gwh' returns (J+K), 1.75.S\[i,j].(H\[i,i] + H\[j,j])/2 - the generalised Wolfberg-Helmholtz approximation. Return values are core Hamiltonian, initial Fock guess.

17. **buildDensity(n, occupiedOrbitals, C)**

	parameters - *n* is number of basis functions, *occupiedOrbitals* are the number of occupied orbitals (twice the number of electrons) and *C* are the eigenvectors of the final Fock matrix. This is C\[i,k].C\[j,k] with i,j over n and k over occupiedOrbitals. Returns \[nbf,nbf] array.

18. **buildFock(H, eri, cycle, D, engine = 'aello')**

	parameters - *H* is the core Hamiltonian matrix, *eri* the linear array of 2-electron repulsion integrals, *cycle* the current SCF iteration (not needed for calculation) and *D* the density matrix. The last parameter is the integral engine selected either 'native' or 'aello'. the default is the cython **aello** engine (doesn't operate on first cycle). If density is a zero matrix then it's first cycle and Fock is equal to core Hamiltonian, otherwise equal to H plus G matrices. The G matrix is G\[i,m] = D\[k,l].(2<im|kl> - <ik|ml>). Returns \[nbf,nbf] array.

19. **boys(n, T)**

	parameters - *n* order of Boys function, *T* is point of evaluation. Calls the confluent hypergeometric function 1F1 from scipy.special. Returns float. Used in j.

20. **mu(iBasis, jBasis, kr, direction)**

	parameters - *iBasis* and *jBasis* are basis objects. *kr* is atom center of atom *k* and *direction* is a cartesian axis \['x'|'y'|'z'].

21. **dipole(ia, ja, ie, je, ir, jr, kr, direction)**

	parameters - *i*, *j*, *k* refer to basis functions, so *kr* is atom center coordinates of *k* and *direction* is a cartesian axis \['x'|'y'|'z']. Computes the dipole component in the 'direction' specified. Calls e, uses math.pow and returns float. Calls dipole, returns float. Note dipoleComponent and buildDipole are in post module.

22. **buildEriMO(eigenvectors, ERI)**

	parameters - *eigenvectors* are eigenfunctions of final Fock matrix. *ERI* array of the 2-electron repulsion integrals. Transforms eri to the MO basis, following this scheme [see](https://github.com/CrawfordGroup/ProgrammingProjects/blob/master/Project%2304/hints/hint2.md). Returns linear array.

23. **buildFockMOspin(spinOrbitals, eigenvectors, fock)**

	parameters - *spinOrbitals* are the number of spin orbitals (twice the number of basis functions), *eigenvectors* are eigenfunctions of final Fock matrix and *fock* the final Fock matrix. Transform Fock matrix from AO basis -> MO basis -> MO spin basis. Returns \[2nbf,2nbf] array.

24. **buildEriSingleBar(spinOrbitals, eriMO)**

	parameters - *spinOrbitals* are the number of spin orbitals and *eriMO* are the 2-electron repulsion integrals in the MO basis. Returns <ij|kl> as a \[2nbf,2nbf,2nbf,2nbf]) array.

25. **buildEriSingleBar(spinOrbitals, eriMO)**

	parameters - *spinOrbitals* are the number of spin orbitals and *eriMO* are the 2-electron repulsion integrals in the MO basis. Returns <ij||kl> as a \[2nbf,2nbf,2nbf,2nbf] array. <ij||kl> <ik|jl> - <il|jk>.

26.	**eriTransform(eri)**

	parameters - *eri* is a 2-electron repulsion integrals. ab|cd -> ac|bd.
	
27. **expandEri(eriMO, nBasis)**

	parameters - *eriMO* the linear eri in the molecular basis and *nBasis* is the number of basis functions. Using this is a lot quicker than recalculating the tensor from scratch. returns eriMO as a 4-index tensor.

28. **d(iBasis, jBasis, direction):**

    parameters - *iBasis* and *jBasis* are basis objects. *kr* is atom center of atom *k* and *direction* is a cartesian axis \[*x*|*y*|*z*]. Calls nabla to calculate matrix for vector differential operator.

29. **nabla(ia, ja, ie, je, ir, jr, direction):**

    parameters - *i*, *j*, *k* refer to basis functions, so *kr* is atom center coordinates of k and direction is a cartesian axis \[*x*|*y*|*z*]. Computes the nabla component in the *direction* specified. Calls e, uses math.pow and returns float. Calls nabla, returns float. 

30. **buildNabla(atoms, bases, direction):**

	parameters - *bases* is an array of basis objects (the molecular basis). *atoms* is an array of atom objects. Computes the matrix for the vector differential operator (del or nabla).

31. **ang(ia, ja, ie, je, ir, jr, kr, direction):**

	parameters - *i*, *j*, *k* refer to basis functions, so *kr* is atom center coordinates of k and direction is a cartesian axis \[*x*|*y*|*z*]. Computes the angular momentum component in the *direction* specified. Calls e, uses math.pow and returns float. Returns float. 

32. **a(iBasis, jBasis, kr, direction):**

	parameters - *iBasis* and *jBasis* are basis objects. *kr* is atom center of atom *k* and *direction* is a cartesian axis \[*x*|*y*|*z*]. Calls ang to compute the angular momentum operator.

33. **buildAngular(atoms, bases, direction, gaugeOrigin):**

	parameters -  *atoms* is an array of atom objects and *bases* is an array of basis objects (the molecular basis). Computes the angular momentum integral matrix in the *direction* specified using *gaugeOrigin*.
	
34. **quadrupole(ia, ja, ie, je, ir, jr, kr, direction):**

	parameters - *i*, *j*, *k* refer to basis functions, so *kr* is atom center coordinates of k and direction is a cartesian axis \[*x*|*y*|*z*]. Computes the quadrupole momentum component in the *direction* specified. Calls e, uses math.pow and returns float. Returns float. 

35. **q(iBasis, jBasis, kr, direction):**

	parameters - *iBasis* and *jBasis* are basis objects. *kr* is atom center of atom *k* and *direction* is a cartesian axis \[*x*|*y*|*z*]. Calls quadrupole to compute the quadrupole moment operator.

36. **electricField(atoms, bases, direction, gaugeOrigin):**

	parameters -  *atoms* is an array of atom objects and *bases* is an array of basis objects (the molecular basis). Computes the electric field integral matrix in the *direction* specified using *gaugeOrigin*.

37. **ef(iBasis, jBasis, kr, direction):**

	parameters - *iBasis* and *jBasis* are basis objects. *kr* is atom center of atom *k* and *direction* is a cartesian axis \[*x*|*y*|*z*]. Calls quadrupole to compute the electric field operator.

38. **electric(ia, ja, ie, je, ir, jr, kr, direction):**

	parameters - *i*, *j*, *k* refer to basis functions, so *kr* is atom center coordinates of k and direction is a cartesian axis \[*x*|*y*|*z*]. Computes the electric field component in the *direction* specified. Calls e, uses math.pow and returns float. Returns float. 
