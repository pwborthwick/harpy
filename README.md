# harpy                                                                                                                              ![](../main/media/aello.jpeg)

## Hartree-Fock Python

This is not a *program*, it will run as a program but it wasn't designed with any program structure in mind. It works as a program mainly to facilitate testing but really it is just a collection of quantum chemistry codes. This is not written in Python, the python interpreter understands the code but the style is in no way 'pythonic'. The aim is to write code that can be understood not just by a Python programmer but anyone who understands pseudocode. This means that some of the more syntactically cryptic elements of the langauge have been avoided. For example, there is no einsum. Einsum is very concise and faster than basic Python loops, which are *very* slow, but if you want speed you probably will be programming in another language or using Cython in which case you'll want the explicit loops.

There are 6 main directories:
* **source**  - This contains the .py and .pyx files. Currently they are
    1.  adc     - algebraic diagrammatic construction
    2.  aello   - Cython version of the molecular integrals.
    3.  atom    - atom class.
    4.  basis   - basis class.
    5.  bomd    - Born-Oppenheimer molecular dynamics
    6.  cc      - coupled cluster directory (scc, fcc.ucc and rcc)
    7.  ci      - configuration interaction.
    8.  cogus   - cluster operator generator using Sympy
    9.  diis    - direct inversion of iterative subspace.
    10. eom     - equation of motion ccsd and mbpt
    11. ep      - electron pair approximation.
    12. fci     - full configuration interaction 
    13. force   - integral derivatives and forces.
    14. h       - hydrogenic orbitals
    15. harpy   - main run program
    16. integral- molecular integrals and transforms.
    17. mbpt    - many-body perturbation diagrams
    18. mp      - Moller-Plesset perturbation theory.
    19. ocypete - Cython version of integral derivatives.
    20. post    - post SCF, Mulliken, Lowdin, Dipole, Polarizabilities.
    21. project - data files for run. Coordinate geometry.
    22. rESP    - restrained Electrostatic Potential
    23. rhf     - main resticted Hartree-Fock code.
    24. rohf    - restricted open-shell Hartree-Fock
    25. tdhf    - time dependent Hartree-Fock
    26. uhf     - main unrestricted Hartree-Fock code
    27. view    - output control writes to harpy.html.
     
* **document**  - This contains a .md file for each code module.
* **test**      - This has HTML output files for water in sto-3g, dz bases, methane and HHe+ in sto-3g.
    These files are provided for comparision with existing programs and for developers wanting to compare their own program results. There are also corresponding .md files which compare harpy results to other programs. 
    
* **mints**   - This contains .npz files containing results from calculation in numpy compressed files. Details of the file contents and how to access them are given in \document\rhf.md
    
* **basis**   - Basis files from Basis Set Exchange.

* **codes**   - Python code for various cluster methods produced by *cogus*

For the theory of the code I recommend the book 'Modern Quantum Chemistry' by Szabo & Ostlund, on Github [the Crawford Projects](https://github.com/CrawfordGroup/ProgrammingProjects) and [psi4Numpy](https://github.com/psi4/psi4numpy). My interest in quantum chemistry was renewed by [Josh Goings blog](https://joshuagoings.com/blog/) particularly his article on McMurchie-Davidson integral evaluation, you should also checkout his program on Github [McMurchie-Davidson](https://github.com/jjgoings/McMurchie-Davidson). I hope something here will be useful to you.

Finally, if you really want to run this as a program then 1) copy (as a minimum) source and basis directories, 2) from **source** directory run setup.py as **python3 setup.py build_ext --inplace install --user** 3) I have my PYTHONPATH set to /harpy/source 4) run harpy.py, again from source directory, as for example **python3 harpy.py -rhf**. You'll need python3, cython, numpy and scipy. The harpy.html output file will also be written to source directory. You can run **python3 harpy.py** with parameters **-rhf|-uhf|-rohf** - if not specified **-rhf** is run, **-v** for a verbose HTML output - if not specified a minimal output is produced, **-m** to produce a mints .npz file - if not specified no file is produced. By default harpy reads the geometry and run specification from the file *project.hpf* in the source directory this can be overridden by specifying an alternative file on the command line as eg **-..\test\water_geometry.hpf** - the file must nave an .hpf extension which must be specified on the command line. The molecule run is by default the first one in the .hpf file, this can be overridden using the command line option **-name**, where *name* is the title of a molecule in the hpf file as it appears in the *name=* specifier.
    
__What you will find here__
+ ADC  - Algebraic Diagrammatic Construction - 2nd order PP, IP and EA, 1st order PP and properties
+ aello - cython McMurchie-Davidson integrals, overlap, kinetic, coulomb, 2-electron repulsion, dipole, angular momentum, nabla
+ Atom - Geometry calculations, inertia tensor, principal moments, rotor type, gauge centers, z-matrix
+ Basis - reading BSE files to create basis, normalisation, aufbau orbital assignment
+ BOMD - velocity-Verlet, Beeman or Adams-Moulton integrators, auto-correlation
+ cc(scc) - coupled cluster, CCSD(t), LCCD, CCD, CCSD-&Lambda;, CC2, LCCSD (for loops)
+ cc(fcc) - coupled cluster, CCD, CCSD, CCSD(t), CC2, QCISD, LCCD, LCCSD (einsum) and EOM (einsum and intermediates)
+ cc(ucc) - unitary coupled-cluster (UCC2, UCC3 and UCC(4))
+ cc(cctn) - coupled-cluster triples, CCSDT-1a, CCSDT-1b, CCSDT-2, CCSDT-3, CCSDT-4, CCSD(T) and CCSDT
+ cc(rcc) -restricted (spin-summed) CCD and CCSD
+ ci - configuration interaction singles, spin-adapted singles/triples, random phase approximation, block Davidson, CIS(D)
+ cogus - code to run COGUS automatically generated codes for CCSD, CCSD(T), CCSDT, CCD, LCCD, LCCSD, CC2, CC3 and CCSD-&Lambda; (einsum)
+ diis - Direct Inversion of Iterative Subspace acceleration for SCF (RHF and UHF) and CCSD
+ eom - equation of motion EOM-CCSD, EOM-MBPT2 (for loops)
+ ep - electron propogator EP2 (spatial and spin), EP3 (spin), Koopman correction by Approximate Green's Function
+ fci - full configuration interaction, determinant representation by string and boolean types
+ force - McMurchie-Davidson 1st derivative integrals (for loops), numerical derivative, geometry optimization
+ h - hydrogenic orbitals, spherical harmonics, Laguerre, Numerov, finite difference. Monte-Carlo
+ integral - McMurchie-Davidson, overlap, kinetic, coulomb, 2-electron repulsion, dipole, angular, nable, electric field and quadrupole (for loops), AO -> MO
+ mbpt - MPn via automatic generation of Hugenholtz diagrams and corresponding Python code to evaluate (einsum)
+ mp - Moller-Plesset, mp2(spin-scaled and natural orbitals), mp3 (for loops), orbital-optimized mp2, Laplace Transform mp2, mp2 unrelaxed density
+ ocepete - cython force integrals
+ post - Mulliken and Lowden charge analysis, Mayer bond order, energy partitioning, dipole, mp2 dipole, quadrupole, polarizabilities, electric field, hyper-polarizability
+ rESP - restrained electrostatic potential
+ rhf - restricted Hartree-Fock, SCF procedure
+ rohf - restricted open-shell Hartree-Fock
+ tdhf - time-dependent Hartree-Fock, Magnus 2 and 4, Pade spectrum, calculation of transition properties, spectrum display, td-CCSD
+ uhf - unrestricted Hartree-Fock
+ view - HTML output, orbital plots

 
    

