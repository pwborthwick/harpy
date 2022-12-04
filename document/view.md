# View Module - output results to HTML file

Theres nothing really interesting here. Results are passed from the computational algorithms to this module for formatting into an HTML file suitable for viewing in a browser. Each run is written to a harpy.html which unless saved elsewhere will be overwritten on each run, the file will be created if doesn't exist. 

1.	**pre(name, atoms, type)**

	parameters - *name* is the molecule name as read for the project file and *type* is the level of the calculation 'rhf' or 'uhf'. This function must be called as it writes essential HTML to the output file.

2. 	**geometry(atoms)**

	parameters - *atoms* is an array of atom objects (the molecular atom set). Outputs

			             the number ao atoms,
			             the atomic input data,
			             the bond lengths,
			             the bond angles,
			             the out-of-plane angles,
			             the dihedral angles,
			             the center of mass,
			             the connection matrix,
			             the inertia tensor,
			             the principal moments  of inertia,
			             the rotor type,
			             the rotational constants,
			             the nuclear repulsion energy.

3. 	**orbitals(atoms, charge, name, bases)**

	parameters - *atoms* is an array of atom objects (the molecular atom set), *charge* the molecular charge, *name* is name of the basis and *bases* is an array of basis objects (the molecular basis). Outputs

			             the number of electrons,
			             the name of basis,
			             the information from the basis exchange file about the basis,
			             the orbital information - number of orbitals, occupied orbitals, virtual orbitals,
			             the list of orbitals with their atom centers and momenta,
			             the aufbau occupancy of the atoms in the molecule,

4.	**uhfOrbitals(alpha, beta, multiplicity)**

	parameters - *alpha* and *beta* are the number of alpha spin and beta spin electrons. *multiplicity* is the multiplicity of the molecule. Outputs

				     the number of alpha spin electrons/orbitals, 
				     the number of beta spin electrons/orbitals,
			         the multiplicity.
						 
5. 	**preSCF(S,K,J,ERI,U, FO, D, IE, guess)**

	parameters - *S* is the overlap matrix, *K* is the exchange matrix, *J* is the coulomb matrix, *ERI* are the 2-electron repulsion integrals, *U* is the orthogonal transform matrix, *F0* is the initial Fock guess, *D* is the initial density matrix, *IE* is the initial electronic energy and guess is either 'core' or 'gwh' for the type of initial Fock used. Outputs

				     the initial guess type, 
				     the overlap matrix, 
				     the exchange matrix, 
				     the coulomb matrix, 
				     the 2-electron repulsion integrals, 
				     the orthogonal transform matrix, 
				     the initial orthogonal Fock matrix, 
				     the initial density matrix, 
				     the initial electronic energy.

6. 	**SCF(e, de, dd, cycle, diis, iterations, convergence)**

	parameters - *e* the current electronic energy, *de* delta energy during the SCF cycle, *dd* rms delta density during the SCF cycle, *cycle* the current iteration, *diis* whether diis is being employed, *iterations* is the maximum number of iterations allowed and *convergence* is the convergence threshold. Outputs
  the maximum allowed iterations, convergence threshold and diis status, for each cycle delta energy, rms delta density, electronic energy and iteration number. 

7. 	**postSCF(data, type)**

	parameters - *data* a list of values, *type* an identifier for the 'data' list. Outputs...

		     'eigen'             - number of cycles convergence took, final total energy and the orbital
                                   energies, the final eigenvectors and the final density matrix, 
             'uhf-post'          - spin statistics for UHF
             'uhf-mull'          - Mulliken population and spin analysis,
		     'mulliken'          - the Mulliken charge analysis, 
		     'lowdin'            - the Lowdin charge analysis,
		     'bonds'             - Mayer bond orders and valency, 
		     'energy'            - analysis of the energy partitioning, 
		     'dipole'            - components and resultant of the dipole, 
		     'mp2-dipole'        - components of the mp2 level dipole momemnt.
		     'quad'              - components, amplitude and asymmetry of quadrupole.
		     'mp'                - moller-plesset 2nd and 3rd order corrections, 
		     'omp'               - orbital optimised moller-plesset 2, 
		     'mplp'              - Laplace transform mp2, 
		     'mbpt'              - many-body perturbation theory,
		     'ci' and 'rpa'      - configuation interaction and random phase approximation, 
		     'fci'               - full configuration interaction,
		     'cisd'              - configuration interaction singles and doubles,
		     'cis'               - configuration interaction singles
		     'cisas' and 'cisat' - spin-adapted configuration interaction, 
		     'ju'                - CI energy level contributions and significant jumps, 
		     'bd'                - block Davidson - 1st five eigenvalues, 
		     'ccsd'              - coupled-cluster singles and doubles, 
		     'ccsd(t)'           - coupled-cluster singles and doubles perturbative triples,
		     'lccd'              - linear coupled-cluster doubles,
		     'ccd'               - coupled-cluster doubles,
		     'cc2'               - coupled-cluster 2 approximation,
		     '+c'                - fast coupled-cluster 
		     'fa'                - intra-molecular forces on atom - analytic, 
		     'fn'                - intra-molecular forces on atom - numeric, 
		     'ep'                - electron propagator (2)(spatial), 
		     'eps'               - electron propagator (2)(spin), 
		     'ep3'               - electron propagator (3)(spin), 
		     'gfa'               - approximate Green's function correction
		     'po'                - polarizabilities
		     'hyper'             - hyperpolarizabilities
		     'resp'              - restrained electrostatic potential charges
		     'eom'               - equation of motion ccsd and mbpt
		     'cogus'             - Cluster Operator Genrator Using Sympy
						 
8.	**post(exit = True)**

	parameters - *exit* if not true will write error messages else complete HTML script and close output file. Must be called as last output function.

9. 	**showMatrix(title, matrix, f, precision = '%.4f')**

	parameters - *title* is caption for HTML table, *matrix* the values to be displayed in table form, *f* is the output file and *precision* is the format of the displayed numbers.

10. **matrixHeatPlot(a, title = '')**

    parameter - *a* is a 2-dimensional matrix and *title* is a title for the plot. Displays a heatplot of the matrix.

We can run this as eg
```python
    #do ccsd computation
    _, ts, _ = cc.ccsd(molAtom, rhf.C, 0, rhf.fock, rhf.ERI, 50, 1e-8, eSCF, 'on')

    maxAmplitudes = eom.maximumAmplitudes(ts, 5, 0)
    print(maxAmplitude)

    import view
    view.matrixHeatPlot(eomEVec, 'eom-mbpt2')

```
to give 
```
['0.021778 (7, 11)', '0.021778 (6, 10)', '0.00329 (2, 10)', '0.00329 (3, 11)', '-0.002501 (5, 13)']
```
10. **matrixHeatPlot(a, title = '')**

    parameter - *a* is a 2-dimensional matrix and *title* is a title for the plot. Displays a heatplot of the matrix with values normalised to inteval \[0,1].

We can run this as eg
```python
    #do ccsd computation
    _, ts, _ = cc.ccsd(molAtom, rhf.C, 0, rhf.fock, rhf.ERI, 50, 1e-8, eSCF, 'on')

    maxAmplitudes = eom.maximumAmplitudes(ts, 5, 0)
    print(maxAmplitude)

    import view
    view.matrixHeatPlot(eomEVec, 'eom-mbpt2')

```
to give 
```
['0.021778 (7, 11)', '0.021778 (6, 10)', '0.00329 (2, 10)', '0.00329 (3, 11)', '-0.002501 (5, 13)']
```
![image](https://user-images.githubusercontent.com/73105740/120977873-bf4f3780-c76b-11eb-979e-37652bd60b80.png)

11. **evaluateGaussian(iBasis, x ,y , plane, z, normal)**

    parameters - *iBasis* is a basis object, *x* and *y* are coordinates in the plane in which the Gaussian is to be evaluated, *plane* is 'xy'|'yz'|'zx' which defines the plane, *z* is the distance from the plane at which to evaluate Gaussian, *normal* is a boolean flag - if True Gaussian will be normalised. Returns the Gaussian density at point (x,y,z) (or y,z,x or z,x,y depending on *plane*).

12. **plotGaussianOverlap(iBasis, jBasis, plane, z, extent, grid, atoms, options = \[False, 20])**

    parameters - *iBasis* and *jBasis* are basis objects, *plane* is 'xy'|'yz'|'zx' - the plane in which to display contour, *z* distance (bohr) above (+) or below (-) plane in which to draw contour, *extent* is a list of maximum and minimum extents of the area to be contoured, *grid* is the mesh size of the grid, *atoms* are the molecular atom objects, *options* are a list \[normalise, number of contours]. Plots a contour of the overlap between orbitals iBasis and jBasis. If jBasis is *None* singlem orbital is plotted.

13. **plotMO(C, orbital, plane, z, extent, grid, atoms, bases, options = \[True, 60, 1e-8])**

    parameters - *C* are the eigenvectors from a converged SCF calculation, *orbital* is the number of the orbital to plot (0,1,...),  *plane* is 'xy'|'yz'|'zx' - the plane in which to display contour, *z* distance (bohr) above (+) or below (-) plane in which to draw contour, *extent* is a list of maximum and minimum extents of the area to be contoured, *grid* is the mesh size of the grid, *atoms* are the molecular atom objects, *bases* is the molecular basis and *options* is a list \[normalise, number of contour lines, reject absolute values below this]. Plots a contour of the molecular orbital.

For water in STO-3g (in xy-plane) we can run
```python
	graphic.plotGaussianOverlap(molBasis[3], None,'xy', 0, [-4, 4,-4, 4], 100, molAtom, [False, 20])
	graphic.plotGaussianOverlap(molBasis[2], molBasis[6],'xy', 0, [-4, 4,-4, 4], 100, molAtom, [False, 20])
	graphic.plotMO(rhf.C, 5, 'xy', 0, [-5,5,-5,5],100, molAtom, molBasis, [False, 60, 1e-6])
```
![image](https://user-images.githubusercontent.com/73105740/120989336-58378000-c777-11eb-9143-dca37f19a3da.png) ![image](https://user-images.githubusercontent.com/73105740/120988845-d8111a80-c776-11eb-9dd3-7e7ba9a37b03.png) ![image](https://user-images.githubusercontent.com/73105740/120989996-07745700-c778-11eb-8e76-04690c0e4c25.png)


