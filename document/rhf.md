# Restricted Hartree-Fock Module - SCF calculations

A discussion of the algorithm can be found in [Crawford projects](https://github.com/CrawfordGroup/ProgrammingProjects/tree/master/Project%2303). You will need from numpy - sqrt, zeros, dot, from numpy.linalg - fractional_matrix_power, eigh, from math - pow. All functions are imported from their modules explicitly so you can see where they come from.

1.	**rms(Da, Db)**

	parameters - *Da*, *Db* two consecutive cycles of the density matrix. Returns square root of the square of the difference of the two matrices.

2.	**mol(show, file = 'project.hpf', molecule='')**

	parameters - *show* is a list of information to be output to HTML file, \['geometry',| 'orbitals'], \[&nbsp;] will surpress output. *File* is by default 'project.hpf' but can be any file following the format given in a) below. The routine...
	
+	a) reads the file containing the job to be processed. Name of file is 'project.hpf'. The file has the following type of format...

		name=h2o
		matrix=c
		post={ch,di}

		O1 8 0.000000000000 -0.143225816552 0.000000000000
		H1 1 1.638036840407 1.136548822547 -0.000000000000
		H2 1 -1.638036840407 1.136548822547 -0.000000000000

		end
		#=============================================
		name=h2o-z
		matrix=z
		units=bohr

		O
		H   1	OH
		H   1	OH    2	   HOH

		OH  2.078698587436746 
		HOH 104

		end
		#----------------------------------------------

	Each molecule definition is terminated by an **end** statement. **#** lines and blank lines are ignored (except in a z-matrix block). The following keywords, options and defaults are recognised, listed as keyword \[default\](options)

			name         	[]                                  name of molecule
			basis        	[sto-3g]                            name of basis
			diis 	     	[on](off)                           use diis or not
			engine       	[aello](native)                     use cython or python integral engine.
			gauge        	[origin](nuclear | mass)            gauge center
			guess        	[core](gwh)                         hamiltonian guess
			charge       	[0](integer)                        molecular charge
			multiplicity 	[1](integer)                        multiplicity
			cycles       	[100](integer)                      maximum iterations
			capacity     	[8](integer)                        diis holding capacity
			tolerance    	[1e-8](float)                       convergence tolerance
			uhfmix       	[0.0]                               uhf orbital mixing parameter
			matrix       	[](c | z)                           use cartesian or z-matrix atom definition 
			post         	[]                                  list of post-SCF routines to run
			units        	[bohr](angstrom)                    units used in atom definitions
			cogus        	[]                                  symbolic cluster codes
			no_mint_density                                     don't use initial guess density if available
			end                                                 marks end of molecule definition
These keywords are dictionary keys in molData.

**Note that only *matrix* and *end* are mandatory**
The file may contain multiple sections as above seperated by a blank or *#* line, the program will execute the **first** such section. It is the first section that is your effective project molecule, other entries in this file are just for storage until they are required when they will have to be moved into position one. All uses of rhf.mol during execution refer to the molecule in position one. Having read the geometry and options an array of atom objects is made. The key *post* has the form

	        post={ch,di,...,po}

where the two letter code represents a post-SCF procedure to execute. The codes are **ch**arges (ch), **bo**nd order (bo), **en**ergy analysis (en), **di**pole (di), **qu**adrupole (qu), **m**oller-**p**lesset (mp), **o**rbital-optimised **m**p2 (om), **m**p2 by **l**aplace transform (ml), **m**any-**b**ody perturbation theory (mb), **c**onfiguation **i**nteraction singles (ci), **f**ull **c**onfiguration interaction (fc), configuration interaction **s**ingles and **d**oubles (sd), **c**onfiguration interaction **s**ingles (slater determinants) (cs), ci **s**pin-adapted **s**ingles(ss), ci **s**pin-adapted **t**riples(st), **r**adom **p**hase approximation (rp), **c**oupled-cluster sd(**t**) (ct), **c**c**2** approximation (c2), **l**inear coupled-cluster singles and **d**oubles (ld).
**l**inear **c**oupled-cluster doubles (lc), **c**oupled-cluster **d**oubles (cd), **fo**rces (fo), **e**lectron **p**ropagator 2 (ep), 
**e**lectron propagator **2** spin-adapted (e2), **e**lectron propagator **3** spin-adapted (e3), **ko**opman associated Greens function correction (ko),
**po**larizabilities (po), **hy**perpolarizabilities (hy). There are also 'fast' (einsum) coupled-cluster options '+s' - cc**s**d, '+d' - cc**d**, '+t' - ccsd(**t**), '+2' - cc**2**, '+l' - **l**ccd, '+L' - **L**ccsd and '+^ - **lambda**. Coupled-cluster triples variants are available as '-T' - ccsd**T**, '-t' - ccsd(**t**), '-1a' - ccsdt-**1a**, '-1b' - ccsd-**1b**, '-2' - ccsdt-**2**, '-3' - ccsdt-**3**, '-4' - ccsdt-**4**.

The *post* keyword will only have effect if the *postSCF* option appears in the **show** list of the scf function call eg

		rhf.scf(molAtom, molBasis, molData , [ 'preSCF', 'SCF', 'postSCF', 'mints'], density)
Running the code from harpy.py includes the postSCF option.

The *cogus* keyword has a similar form to *post*

	    cogus={d,cd(t)}

where the codes represent a coupled-cluster code produced by the Cluster Operator Generator Using Sympy program. The codes are **d**  - CCD, **sd** - CCSD, **sd(t)** - CCSD(T), **sdt** - CCSDT, **2** - CC2, **3** - CC3, **ld** - LCCD, **lsd** - LCCSD.

The *molecule* parameter is any molecule identifier from an .hpf file, that is something on the right of a 'name=' specifier.

+	b) Builds the molecular basis. Uses buildBasis. Returns the atom and basis objects and the parameters defining the run as a list.

3.	**rebuildCenters(molAtom, molBasis, geo)**

	parameters - *molAtom* is an array of atom objects, *molBasis* is an array of basis objects and *geo* is an array of cartesian coordinates. This routine is used to copy a new coordinate set represented by *geo* into *molAtom* and *molBasis*.center attributes. Returns the rebuilt object arrays which can then be used as arguements for scf to calculate an energy for a changed geometry.
			
4.	**scf(molAtom, molBasis, run, show, cptDensity = None)**
	parameters - *molAtom* is an array of atom objects, *molBasis* is an array of basis objects, *run* is the list of option parameters returned by *mol* and *show* is a list of information to be output to HTML file ['preSCF',| 'SCF',| 'postSCF',| 'mints'], and *cptDensity* is a density matrix used to start an SCF calculation instead of a matrix of zeros. If a mints file for the molecule and basis exists it will be used to start the computation (unless no_mint_density is stated in input file). This is useful in geometry optimisation and molecular dynamics where the density from the nth calculation can be used to start the (n+1)th. The following objects are declared global - C, density, fock, ERI, totalEnergy, S, e and coreH. The algorithm is briefly...

	a) Build overlap, exchange, coulomb, core Hamiltonian and 2-electron repulsion integrals.
	   Uses	buildOverlap, buildKinetic, buildCoulomb, buildHamiltonian, buildEri. There are two integral engines 1) the normal python which is invoked with 'native' and a cython engine invoked with 'aello'. 

	b) Define convergence parameters - maximum iterations allowed, convergence threashold, diis switch. Initialise density matrix or use check-point one.

	c) Define storage for, and size of diis mechanism.

	d) Get the inverse of the square root of the overlap matrix.

	e) Start the iteration cycle

	f) Initial Fock is Hamiltonian guess, subsequently H core + G.

	g) If diis and not 1st cycle use diis to compute the next Fock matrix. Uses diisFock.

	h) Orthogonalise the Fock matrix.

	i) Diagonalise the orthogonal Fock matrix.

	j) Transform the eigenvectors of l) back to non-orthogonal atomic basis.

	k) Build density matrix (over occupied orbitals). Uses buildDensity.

	l) Compute the SCF electronic energy.

	m) Check for converged solution. Delta energy and rms density must be satisfied and maximum iterations not exceeded.
	   if not converged go to step i).

	n) If converged solution do post-SCF (if selected) \
			charges \
			energyPartition \
			dipoles (ground state and mp2)\
			quadrupoles \
			moller-plesset \
			cis \
			full configuration interaction \
			configuration interaction singles and doubles \
			cis spin-adapted \
			random phase approximation \
			coupled-cluster singles and doubles and perturbative triples. \
			coupled-cluster doubles \
			linear coupled-cluster \
			cc2 \
			linear couple-cluster singles and doubles \
			forces \
			electron propogator \
			orbital optimised mp \
			laplace transform mp \
			polarizabilities \
			hyperpolarizabilities \
		or
			Symbolic generated cluster code

	o) if 'mints' is in show string a compressed .npz file is saved of the molecular integrals. The file is named 
	   '<molecule_name>-<basis_name>-mints.npz'. The files are read as 

	   		data = load(<file_name>)
	   		<integral> = data[<key_word>]

,where <key_word> is 

**s** - overlap, **k** - kinetic, **j** - coulomb, **i** - 2-electron repulsion, **f** - fock,\
**d** - density, **c** - eigenvectors, **e** - eigenvalues(orbital energies) and **E** - total energy, **m** -[charge, molecule name, basis name, nuclearRepulsion],**a** - list of atomic numbers of atoms in molecule.
