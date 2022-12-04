# Configuration Interaction Module 


There is a Crawford project on the CI [see](https://github.com/CrawfordGroup/ProgrammingProjects/tree/master/Project%2312). See also fci module.

1.	**cis(atoms, charge, bases, eigenVectors, fock, ERI)**

 	parameters - *atoms* is an array of atom objects (the molecular atom set), *charge* is the molecular charge, *bases* is an array of basis objects (the molecular basis), *eigenVectors* are the final eigenfunctions of the fock matrix, *fock* is the final Fock matrix and *ERI* the 2-electron repulsion integrals. This routine calls buildFockMOspin to get the Fock matrix in the MO spin basis, then buildEriMO to get 2-electron repulsion integrals in the MO basis and subsequently buildEriDoubleBar to transform to a spin basis integral. A CI Hamiltonian is computed and the eigen solution found. This gives a mixture of 
 	tuples states which a returned as a unique value and a multiplicity by ciDegeneracy function. Returns the arrays of eigenvalues and eigenvectors. A call to blockDavidson returns the (5) lowest eigenvalues from a block Davidson algorithm. A call to excitations produces a list of all contributions to the orbital excitations above 10% and the levels involved in the jump. Returns the eigenvalues and eigenvectors.

2. **ciDegeneracy(e)**

   parameters - *e* is an array of (possibly degenerate) values pre-sorted in ascending order. Returns an array of 2-pules [energy, degeneracy],
 					 where is is the degeneracy (*s* - singlet, *d* - doublet, *t* - triplet or \[*n*] - n'let).

3. **spinAdaptedSingles(atoms, charge, bases, eigenVectors, fock, ERI)**

   parameters - *atoms* is an array of atom objects (the molecular atom set), *charge* is the molecular charge, bases is an array of basis objects (the molecular basis), *eigenVectors* are the final eigenfunctions of the fock matrix, *fock* is the final Fock matrix and *ERI* the 2-electron repulsion integrals. This routine constructs the Fock matrix in the MO basis and calls buildEriMO to get the 2-electron repulsion integrals in the mO basis as well. A spin-adapted CI hamiltonian for singlets is then constructed and the eigen solution found. Returns eigenvalues.

4. **spinAdaptedTriples(atoms, charge, bases, eigenVectors, fock, ERI)**

   parameters - *atoms* is an array of atom objects (the molecular atom set), *charge* is the molecular charge, bases is an array of basis objects (the molecular basis), *eigenVectors* are the final eigenfunctions of the fock matrix, *fock* is the final Fock matrix and *ERI* the 2-electron repulsion integrals. This routine constructs the Fock matrix in the MO basis and calls buildEriMO to get the 2-electron repulsion integrals in the mO basis as well. A spin-adapted CI hamiltonian for triplets is then constructed and the eigen solution found. Returns eigenvalues.

5. **randomPhaseApproximation(atoms, charge, bases, eigenVectors, fock, ERI)**

    parameters - *atoms* is an array of atom objects (the molecular atom set), *charge* is the molecular charge, bases is an array of basis objects (the molecular basis), *eigenVectors* are the final eigenfunctions of the fock matrix, *fock* is the final Fock matrix, *ERI* the 2-electron repulsion integrals and *type* is the method of constructing Hamiltonian. *type* can be 'block' this is just the CIS method, 'linear' is using (A-B)(A+B) to find square of eigenvalues, 'hermitian' is using (A-B)<sup>&#189;</sup>(A+B)(A-B)<sup>&#189;</sup> to find the square of the eigenvectors and 'raw' which just returns the A and B matrices. There is also a 'tamm-dancoff' option which just uses the A matrix. This option will also solve for eigenvectors and returns sorted eigenvalues and eigenvectors. This routine calls buildFockMOspin to get the Fock matrix in the MO spin basis, then buildEriMO to get 2-electron repulsion integrals in the MO basis and subsequently buildEriDoubleBar to transform to a spin basis integral. A and B matrices are constructed and (A+B)(A-B) solved as an eigenvalue problem to get the CI energies squared. ciDegeneracy is then used to filter out the degeneracies.
					 
6. **excitations(ecis, ccis, nElectrons, nOccupied, nVirtual)**

   parameters - *ecis* and *ccis* are the eigenvalues and vectors of the CI Hamiltonian, n\_ are the various orbital occupation counts. Returns an array of arrays each containing \[energy level number, energy value, % contribution, jump given as eg *3 -> 5*].

7.	**blockDavidson(nLowestEigen, h)**

	parameters - *nLowestEigen*	 is the number of the lowest eigenvalues the routine is to return, *h* is a CI hamiltonian to be diagonalised. Returns the eigenvalues as an array or *None* if algorithm failed. Use with sparse matrices.


Also included is a class to handle CIS(D) computations. The class can do calculations using both a direct solver (numpy.linalg.eigh) or the Davidson iterative solver from adc moldule. 
To do a CIS(D) computation use code as follows which also shows getting the mp2 energy. The class has a *cache* property which is a dictionary with keys 'mp2', 'cis', 'cisd' and 'u'. These are the respective MP2 and CIS energies, the CIS(D) **correction** and the CIS eigenvectors from the computation. These are simply retrieved as eg ```mp2_energy = cisd.cache['mp2```']. The *method* can be either CIS(D) or CIS_MP2, in the latter case the *cache* key 'cisd' will not exist and is replaced by the keyword 'cis-mp2'
```
from adc.adc import hf_reference

#run RHF scf computation
import rhf
molAtom, molBasis, molData = rhf.mol([])
e_scf = rhf.scf(molAtom, molBasis, molData, [])

#ground state class
hf = hf_reference(rhf, molAtom, molBasis, molData)

from ci import cis_d
roots = 10
cisd = cis_d(hf, roots, solver='davidson', method='cis(d)')
```
which will produce an output like
```
***CIS(D)***
 root      CIS          CIS(D)         Δ 
---------------------------------------------
   1     0.378234      0.339494 ( -0.038740 )
   2     0.378234      0.339494 ( -0.038740 )
   3     0.378234      0.339494 ( -0.038740 )
   4     0.450863      0.423928 ( -0.026935 )
   5     0.450863      0.423928 ( -0.026935 )
   6     0.450863      0.423928 ( -0.026935 )
   7     0.454664      0.405082 ( -0.049582 )
   8     0.497466      0.465175 ( -0.032292 )
   9     0.497466      0.465175 ( -0.032292 )
  10     0.497466      0.465175 ( -0.032292 )
```
This is for the geometry prescribed in [this reference](https://hirata-lab.chemistry.illinois.edu//cis_data.out). The value given by this reference for CIS(D) is -0.03874026915629841. The output for CIS-MP2 (first root) is ```1     0.378234      0.342717 ( -0.035517 )``` compared to the reference value of -0.03551743011037764.

