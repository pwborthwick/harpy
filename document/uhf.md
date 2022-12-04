# Unrestricted Hartree-Fock Module - SCF calculations

A discussion of the algorithm can be found in eg [here](https://www2.chemistry.msu.edu/faculty/harrison/web_docs_Hartree_Fock/HF3.pdf). You will need from numpy - sqrt, zeros, dot and vdot from numpy.linalg - fractional_matrix_power, eigh, from math - pow. All functions are imported from their modules explicitly so you can see where they come from. This uses **mol** from rhf to read input file and process defaults.

1.	**buildUnrestrictedFock(H, eri, cycle, density)**

	parameters - *H* the core Hamiltonian, *eri* the 2-electron repulsion integrals, *cycle* the current SCF iteration, *density* are the alpha and beta spin density matrices in format (2, nBasis, nBasis). Returns an alpha and beta spin Fock matrices in same format as density. If cycle is 0 then both matrices are returned as the core Hamiltonian.

2. 	**buildUnrestrictedDensity(c, orbitals)**

  	parameters - *c* are the alpha and beta spin Fock eigenvectors in format (2, nBasis, nBasis), *orbitals* are the &alpha; and &beta; occupancy numbers. Returns the spin density matrices.

3. 	**rmsDensity(density, preDensity)**

	parameters - *density*, *preDensity* two consecutive cycles of a density matrix. Returns square root of the square of the difference of the two matrices.


4.	**rebuildCenters(molAtom, molBasis, geo)**

	parameters - *molAtom* is an array of atom objects, *molBasis* is an array of basis objects and *geo* is an array of cartesian coordinates. This routine is used to copy a new coordinate set represented by geo into molAtom and molBasis .center attributes. Returns the rebuilt object arrays which can then be used as arguements for scf to calculate an energy for a changed geometry.
			
5.	**scf(molAtom, molBasis, run, show)**

	parameters - *molAtom* is an array of atom objects, *molBasis* is an array of basis objects, *run* are the parameters that define the scf, *show* are the output sections to be displayed. If show selects *postSCF* the following are displayed
	spin matrix, spin contamination, total density P<sup>a</sup> + P<sup>b</sup>, spin density  P<sup>a</sup> - P<sup>b</sup>, and if post={ch,di} is in the project file then a Mulliken population analysis and dipole calculation are performed. There is a 'uhfmix' parameter that can be included in the .hpf molecule definition, by default this is set to 0.0 so closed shell uhf computations default to rhf. In, for example, the H<sub>2</sub> dissociation example the mixing parameter is set to 0.02 which results in a mixing of the humo and lumo on the alpha orbitals.

	Returns total energy and total density is global.

6.  **getOccupancy(e, orbitals, nBasis)**

    parameters - *e* are the orbital energies, *orbitals* are the number of &alpha; and &beta; orbitals and *nBasis* is the number of basis functions. Returns occupancy vectors.

7.  **spinSquare(c, occupancy, S)**
    
    parameters - *c* are the molecular eigenvectors shape is (2, nBasis, nBasis), *occupancy* are the alpha and beta occupancies and *S* is the overlap integrals. Returns <S<sup>2</sup>> and the multiplicity 2\*S + 1.

H<sub>2</sub>O in 6-31G basis - charge -1, multiplicity 2. Comparison with pyscf.

| energy     | source  |
|---------|-------|
|-75.801683 | harpy |
| -75.801683 | pyscf |

mulliken population

|     O1     |   H1  | H2  | description  |
|-----------|----| ------|---------------|
|-0.714275 | -0.142862 | -0.142862 | harpy charge |
| -0.12926 | 0.56463 | 0.56463 | harpy spin |
| -0.71428 | -0.14286 | -0.14286 | pyscf charge |
| -0.12926 | 0.56463 | 0.56463 | pyscf spin |

total spin is 0.771032
multiplicity is 2.021