# Electron Propagator

This module follows the psi4numpy implementation [here](https://github.com/psi4/psi4numpy/tree/master/Electron-Propagator). 

1.	**electronPropagator2( molBasis, c, ERI, scfEnergy, eps, nOccupied, startOrbital = 2, nOrbitals = 4)**

	parameters - *molBasis* is an array of basis objects, *c* are the final orbital eigenvectors, *ERI* the (linear) 2-electron repulsion integrals, *scfEnergy* is the final converged total energy, *eps* the final orbital energies, *nOccupied* the number of occupied orbitals, *startOrbital* is orbital from which to start analysis and *nOrbitals* the number of orbitals to analyse. Returns to postSCF a list of the orbital energies (Koopman) and ep2 energies representing the ionising potentials. Returns ep2 energies.

2.	**electronPropagator2spin(molBasis, c, ERI, eigenValues, nOccupied, nOrbitals = 5)**	

	parameters - *molBasis* is an array of basis objects, *c* are the final orbital eigenvectors, *ERI* the (linear) 2-electron repulsion integrals, *eigenValues* are the final orbital energies, *nOccupied* the number of occupied orbitals, and *nOrbitals* the number of 	orbitals to analyse. Returns to postSCF a list of the orbital energies (Koopman) and ep2 energies representing the ionising potentials. Returns ep3 energies.

3.  **koopmanAGFcorrection(molBasis ,c ,ERI, eigenValues, nOccupied, nOrbitals = 5):**

    parameters -  *molBasis* is an array of basis objects, *c* are the final orbital eigenvectors, *ERI* the (linear) 2-electron repulsion integrals, *eigenValues* are the final orbital energies, *nOccupied* the number of occupied orbitals, and *nOrbitals* the number of orbitals to analyse (n below HOMO). This computes the approximate Green function correction to the IP. See Szabo and Ostlund pg 403. Returns a list of [[Koopman energy, AGF correction], ...]