# Moller-Plesset Module - post SCF calculations

A discussion of MP2 can be found in [Crawford projects]  (https://github.com/CrawfordGroup/ProgrammingProjects/tree/master/Project%2304).

1.	**mollerPlesset(atoms, charge, bases, eigenVectors, eigenValues, ERI, e)**

	parameters - *atoms* is an array of atom objects (the molecular atom set), *charge* is the molecular charge, *bases* is an array of basis objects (the molecular basis), *eigenVectors* are the eigenfunctions of the final Fock matrix and *eigenValues* the	 corresponding eigenvalues, *ERI* the 2-electron repulsion integrals and *e* the final SCF total energy. Calls mp2 and mp3 and returns mp2 (parallel and anti-parallel components) and mp3 energy corrections to the output module.

2.	**mp2(atoms, charge, bases, eigenVectors, eigenValues, ERI)**

	parameters - *atoms* is an array of atom objects (the molecular atom set), *charge* is the molecular charge, *bases* is an array of basis objects (the molecular basis), *eigenVectors* are the eigenfunctions of the final Fock matrix and *eigenValues* the corresponding eigenvalues and *ERI* are the 2-electron repulsion integrals. *type* can be either 'p-ap' which returns the parallel and anti-parallel spin components of the mp2 correction (default), 'scs' which will return the spin-conponent scaled correction or 'no' which returns the natural orbitals and their occupations (Trunks, Salter, Sosa, Bartlett - Theory and implementation of MBPT density matrix. An application to one-electron properties [here](https://www.sciencedirect.com/science/article/abs/pii/0009261488802495)). Calls buildEriMO to transform ERI to the molecular basis, and returns are dependent on *type*.
 
3.	**mp3(atoms, charge, bases, eigenVectors, fock, ERI)**

	parameters - *atoms* is an array of atom objects (the molecular atom set), *charge* is the molecular charge, *bases* is an array of basis objects (the molecular basis), *eigenVectors* are the eigenfunctions of the final Fock matrix, *fock* is the final Fock matrix and *ERI* are the 2-electron repulsion integrals. Calls buildEriMO to transform ERI to the molecular basis and returns the mp3 energy correction	

4.	**orbitalOptimisedMP2(eigenVectors, h, e, molBasis, eNuclear, ERI, nElectrons)**

	parameters - *eigenVectors* are the final orbital coefficients, *h* is the core hamiltonian, *e* are the final orbital energies, *molBasis* is an array of basis objects( the molecular basis), *eNuclear* is the nuclear repulsion energy, *ERI* are the 2-electron repulsion integrals and *nElectrons* are the number of electrons. Calls integral.iEri. This is a version of the algorithm found in psi4numpy tutorials [here](https://github.com/psi4/psi4numpy/blob/master/Tutorials/10_Orbital_Optimized_Methods/10a_orbital-optimized-mp2.ipynb). Outputs results to view.postSCF (*omp*). Contains functions hSpinBlock(h, spinOrbitals, nBasis), eriSpinBlock(eri, spinOrbitals, nBasis) and eriSpinMO(gao, eigenVectors). Returns omp energy.

5.	**eriMOpartition(eri, co, cv, nBasis, no, nv)**

	parameters - *eri* are the 2-electron repulsion integrals in tensor form, *co* and *cv* are partitioned orbital coefficients (co = c[:,o] cv = c[:,v]),*nBasis* is the number of basis functions, *no* and *nv* are the number of occupied and virtual orbitals. Returns eri as tensor [o,v,o,v].

6.	**mp2LaplaceTransform(molBasis, c, ERI, eps, nOccupied, E, meshSize=40)**

	parameters - *molBasis* an array of basis objects (the molecular basis), *c* the orbital eigenfunctions, *ERI* the 2-electron  repulsion integrals (linear), *eps* the orbital energies, *nOccupied* the number of occupied orbitals, *E* the converged scf energy and *meshSize* the size of the quadrature grid. Uses Laplace transforms based on psi4numpty [here](https://github.com/psi4/psi4numpy/blob/master/Moller-Plesset/LT-MP2.py). Sends results to postSCF (*mplp*) for output. Returns mp2Laplace energy correction.
	
7.	**mp2UnrelaxedDensity(c, e, eri, nbf, nOccupied)**

	parameters - *c* are the orbital eigenfunctions, *e* the orbital energies, *eri* the 2-electron  repulsion integrals (linear), *nbf* are the number of basis functions and *nOccupied* are the number of doubly occupied orbitals. Returns the unrelaxed mp2 level density matrix.
