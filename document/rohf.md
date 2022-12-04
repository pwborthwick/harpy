## Restricted Open-Shell Hartree-Fock
 
For a discussion of ROHF see [this](http://vergil.chemistry.gatech.edu/notes/cis/node5.html). Another useful article is [ROHF theory made simple](https://aip.scitation.org/doi/10.1063/1.3503173). The treatment here follows the pyscf approach.

1. **rmsDensity(density, preDensity)**

   parameters - *density* is the density matrix in the current cycle and *preDensity* is the density matrix from the previous cycle. Return the root-mean square difference of the matrices.

2. **get_occupations(eps_list, atoms, charge, multiplicity)**
   
   parameters - *eps_list* orbital energies either \[e] or \[e, e<sup>&alpha;</sup>, e<sup>&beta;</sup>], *atoms* the array of atom objects, *charge* the net charge off the molecule and *multiplicity* is the multiplicity (unpaired electrons -1). This routine returns the occupancy of the atomic orbitals eg water in sto-3g (7 orbitals) with charge +1 (9 electrons) and multiplicity 2 (1 unpaired electron) we have [2, 2, 2, 2, 1, 0, 0] returned. To do this the routine first determines the number of &alpha; and &beta; electrons - the number of electrons we can assign to doubly occupied orbitals (n<sub>c</sub>) is the total electrons - the number which are to be assigned to single orbitals alone ie n<sub>e</sub>-(multiplicity - 1) for our example 9-1=8. So 8 electrons are assigned to 4 doubly occupied orbitals 4 for &alpha; and 4 for &beta; with the remaining unpaired electrons being assigned to &alpha;. This gives an occupancy of \[2, 2, 2, 2, 1, 0, 0] = \[1, 1, 1, 1, 1, 0, 0]<sup>&alpha;</sup> + \[1, 1, 1, 1, 0, 0, 0]<sup>&beta;</sup>. In practice we sort the combined orbital energies by assigning an occupation of 2 to the lowest (core) n<sub>c</sub> orbitals, then we take the indices of the sorted highest orbitals not already assigned and take the corresponding energies from the &alpha; orbitals. These (non-core) orbital energies are then sorted and the lowest used to take (multiplicity -1) orbitals to assign an occupancy of 1 to - all remaining orbitals are assigned an occupancy of 0.

3. **get_densities(c, occupations)**

   parameters - *c* orbital coefficients (see above)and *occupations* are atomic orbital occupancies calculated from *get_occupations*. Returns the alpha and beta densitys as a (2, nbf, nbf) array.

4. **get_coulomb_exchange(density, eri)**

   parameters - *density* are the alpha and beta densities as returned by *get_densities* and *eri* are the two-electron repulsion integrals. Returns the alpha and beta j and k integrals.

5. **make_fock(h, v, dm, s, roothaan=False)**

   parameters - *h* is the one-electron hamiltonian, *v* are alpha and beta effective potential (j+k), *dm* are the alpha and beta densities, *s* is the overlap matrix and *roothaan* is a boolean flag determining if the Roothaan effective Fock matrix is returned. The routine returns the &alpha; and &beta; Fock matrices and if *roothaan* is True also the Roothaan effective Fock matrix constructed according to 
  ![image](https://user-images.githubusercontent.com/73105740/144617311-e3909a47-a4a7-41a0-86be-0c58cd7f6544.png)

6. **eigensolution(fock, x)**

   parameters - *fock* is tuple of &alpha; and &beta; Focks and *x* is the inverse square root of the overlap matrix. Solves for orbital energies for (F<sup>&alpha;</sup> + F<sup>&beta;</sup>), F<sup>&alpha;</sup> and F<sup>&beta;</sup> and returns them.
   
7. **get_rohf_energy(density, h1e, v)**  

   parameters - *density* are the &alpha; and &beta; densities, *h1e* is the one-electron hamiltonian and *v* are the exchange and coulomb integrals. Returns the one-electron energy and the two-electron component of the energy separately.

8. **spin_analysis(c, occupancy, s)**

   parameters - *c* the orbital coefficients (2, nbf, nbf), *occupancy* an array of occupation numbers and *s* the overlpa matrix. Computes the actual total spin (<S<sup>2</sup>>) and multiplicity and returns them with the number of occupied &alpha; and &beta; electrons. 

9. **scf(molAtom, molBasis, run, show)**

   parameters - *molAtom* an array of atom objects, *molBasis* an array of orbital objects (the molecular basis), *run* is a distionary of the parameters determining the characteristics of the computation and *show* is a list of properties to be output. This routine gets the molecular integrals needed and performs the scf iteration. If a converged solution is obtained an analysis of the spin properties is carried out and if requested the Mulliken population analysis and dipole computation are performed.

Note that unlike uhf the spin contamination is zero and the multiplicity is exact. We can run our (modified) H<sub>2</sub> dissociation program from harpy/test to see if ROHF handles that problem and we get

![image](https://user-images.githubusercontent.com/73105740/145675142-efc2e4dd-7646-4787-b26e-0655242b1580.png)

