# Post Module - post SCF calculations

Dipole and Mulliken are covered in [Crawford projects]  (https://github.com/CrawfordGroup/ProgrammingProjects/tree/master/Project%2303). A useful discussion of charges can be found [here](https://mattermodeling.stackexchange.com/questions/1439/what-are-the-types-of-charge-analysis)

1.	**charges(D, S, atoms, bases)**

	parameters - *D* is the final density matrix, *S* the overlap matrix, *atoms* an array of atom objects (the molecular atom
	set) and *bases* an array of basis objects (the molecular basis). The routine calls the different methods of charge and population analysis. Calls Mulliken and Lowdin. No return value.

2.	**mulliken(D, S, atoms, bases))**

	parameters - *D* is the final density matrix, *S* the overlap matrix, *atoms* is an array of atom objects (the molecular atom
	set) and *bases* an array of basis objects (the molecular basis). Performs a Mulliken population analysis, passing
	gross atomic populations and orbital populations to the output module and returning gross atomic populations.

3.	**lowdin(D, S, atoms, bases))**

	parameters - *D* is the final density matrix, *S* the overlap matrix, *atoms* an array of atom objects (the molecular atom
		             set) and *bases* an array of basis objects (the molecular basis). Performs a Lowdin population analysis, passing
		             gross atomic populations to the output module and returning gross atomic populations.

4.	**energyPartition(E, N, D, K, V, G, e, en)**

	parameters - *E* is final electronic energy, *N* the nuclear repulsion energy, *D* the final density matrix, *T* the exchange (kinetic) energy matrix, *V* the Coulomb (potential) energy matrix, *G* the matrix in Fock = CoreHamiltonian + G, *e* are eigenvalues of final fock (orbital energies) and *en* is the number of electrons. Performs a analysis of the components of the energies. Returns the energy partitioning to the output module.

5. **dipoleComponent(atoms, bases, type, gauge)**

	parameters - *atoms* is an array of atom objects (the molecular atom set) and *bases* an array of basis objects (the molecular basis), *type* is a cartesian axis *x*|*y*|*z* and *gauge* is the type of gauge origin. Calls gaugeCenter (atom module) and mu (integral module) returns float.

6. **buildDipole(atoms, bases, density, gaugeOrigin, engine = 'aello'):**

	parameters - *atoms* is an array of atom objects (the molecular atom set) and *bases* an array of basis objects (the molecular basis), *density* is the final density matrix. *gaugeOrigin* is the gauge type string and *engine* is the integral engine to use. Calls dipoleComponent returns an array cells \[0,1,2] are electron dipole components, \[3] is square of resultant. Array is returned and written to output module.

7. **buildMp2Dipole(atoms, bases, gauge, c, e, eri, nOccupied)**
    parameters - *atoms* is an array of atom objects (the molecular atom set) and *bases* an array of basis objects (the molecular basis), *gauge* is the gauge type string, *c* are the converged eigenvectors, *e* the final orbital energies, *eri* are the 2-electron repulsion integrals (as linear list) and *nOccupied* is the number of doubly occupied orbitals. Returns the mp2 level unrelaxed dipole moment. List is returned and output written.
    
8. **quadrupoleComponent(atoms, bases, type, gauge)**

	parameters - *atoms* is an array of atom objects (the molecular atom set) and *bases* an array of basis objects (the molecular basis), *type* is a cartesian axis *xx*|*yy*|*zz*|*xy*|*yz*|*zx*| and *gauge* is the type of gauge origin. Calls gaugeCenter (atom module) and q (integral module) returns float.
	
9. **buildQuadrupole(atoms, bases, density, gaugeOrigin)**

	parameters - *atoms* is an array of atom objects (the molecular atom set) and *bases* an array of basis objects (the molecular basis), *density* is the final density matrix. *gaugeOrigin* is the gauge type string. Calls quadrupoleComponent returns an array of electron quadrupole components, [xx,yy,zz,xy,yz,zx]. Array is returned writes to output module.

10. **polarizabilities(atoms, bases, c, f, ERI, nOccupied, gauge)**

	parameters - *atoms* is an array of atom objects (the molecular atom set) and *bases* an array of basis objects (the molecular basis),
	*c* are the final eigenvectors, *f* the final fock matrix, *ERI* the 2-electron repulsion integrals, *nOccupied* the number of occupied orbitals and *gauge* is the type of gauge origin. Returns the principal polarizations, the isotropic polarization, the responses and the dipole tensors. Based on the psi4numpy tutorial [here](https://github.com/psi4/psi4numpy/blob/master/Tutorials/06_Molecular_Properties/6a_CP-SCF.ipynb)

11. **electricFieldNuclear(atoms, gauge)**

	parameters - *atoms* is an array of atom objects (the molecular atom set) and *gauge* is the type of gauge origin. Calls gaugeCenter (atom) and returns the nuclear components of the electric field and potential. Returned array [E<sub>x</sub>, E<sub>y</sub>, E<sub>z</sub>, V].

12. **buildElectricField(atoms, bases, density, gauge):**

	parameters - *atoms* is an array of atom objects (the molecular atom set) and *bases* an array of basis objects (the molecular basis), *density* is the final density matrix. *gaugeOrigin* is the gauge type string. Calls electricField and returns an array of electric field components (field first potential second, [E<sub>x</sub>, E<sub>y</sub>, E<sub>z</sub>, V, E<sub>nx</sub>, E<sub>ny</sub>, E<sub>nz</sub>, V<sub>n</sub>]. 

13. **hyperPolarizabilities(atoms, bases, c, eri, e, f, nOccupied, gauge)**

	parameters - *atoms* is an array of atom objects (the molecular atom set) and *bases* an array of basis objects (the molecular basis),
		             *c* are the final eigenvectors, *eri* are the 2-electron replusion integrals, *e* are the orbital energies, *f* the final fock matrix, *nOccupied* the number of occupied orbitals and *gauge* is the type of gauge origin. Returns the (static) hyperpolarizations in the axial directions, the total amplitude, the parallel and perpendicular components with respect to z-axis and the hyperpolarizability tensor. Based on the psi4numpy tutorial [here](https://github.com/psi4/psi4numpy/blob/master/Tutorials/06_Molecular_Properties/6b_first_hyperpolarizability.ipynb). Defined parameters calculated as per [3.200.7](http://web.mit.edu/multiwfn_v3.3.8/Manual_3.3.8.pdf). For the static case these parameters are
                 
	+ **&beta;**<sub>i</sub> = &Sigma;<sub>j</sub> &beta;<sub>ijj</sub>+&beta;<sub>jji</sub>+&beta;<sub>jij</sub>  i in {x,y,z}. j = {x,y,z} - {i}
	+ **&beta;**<sub>tot</sub> = (**&beta;**<sub>x</sub><sup>2</sup>+**&beta;**<sub>y</sub><sup>2</sup>+**&beta;**<sub>z</sub><sup>2</sup>)<sup>0.5</sup>
	+ **&beta;**<sub>&#8869;z</sub> = 0.2 **&beta;**<sub>z</sub>
	+ **&beta;**<sub>&#8741;</sub> = 0.6 **&beta;**<sub>z</sub>

14. **bondOrder(D, S, atoms, basis)**

	parameters - *D* is the final density matrix, *S* the overlap matrix, *atoms* is an array of atom objects (the molecular atom
	set) and *bases* an array of basis objects (the molecular basis). Computes the Mayer bond orders as B<sub>AB</sub> = &Sigma; &Sigma; (DS)<sub>&mu;&nu;</sub>(DS)<sub>&nu;&mu;</sub> where &mu; are orbitals centered on atom A and &nu; are orbitals centered on atom B. The valency V<sub>A</sub> is calculated simply as &Sigma; B<sub>AX</sub>. Returns bond orders and atom valencies.
