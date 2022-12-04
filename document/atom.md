# Atom Module - basic geometry

Much of this module follows the properties considered in 
[Crawford projects]  (https://github.com/CrawfordGroup/ProgrammingProjects/tree/master/Project%2301)

1. weight                       - list of atomic weights (H - Xe)
2. covalentRadius               - list of covalent radii (H - Xe). Used to infer joins
3. symbol                       - list of atomic symbols (H - Xe)
4. __init__                     - instantiate atom class object.\
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; id     - identifier, eg *H1*\
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; number - atomic number, eg *4*\
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; center - atomic center, \[x,y,z]\
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; joins  - connected atoms, \[]

5. **seperation(atoms, i, j, unit=*b*)**

   parameters - *atoms* is array of atom class objects, *i*, *j* are atom objects, *unit = b* for Bohr
        *a* for Angstrom. Returns the interatom distance between atoms i and j.

6. **angle(atoms, i, j, k)**

   parameters - *atoms* is array of atom class objects, *i*, *j*, *k* are atom objects. Returns the
                     angle between vectors **ji** and **jk** calculated as arccos of the dot product of 
                     the normalised vectors. Angle is in degrees.

7. **oopAngle(atoms, i, j, k, l)**

   parameters - *atoms* is array of atom class objects, *i*, *j*, *k*, *l* are atom objects. Returns
                     the angle between **ik** and normal to plane **jkl**, this is the out-of-plane angle.
                     Calculated as arcsin of dot product of **ik** and the cross-product **jk** X **lk**.
                     Angle is in degrees.

8. **dihedral(atoms, i, j, k, l)**

   parameters - *atoms* is array of atom class objects, *i*, *j*, *k*, *l* are atom objects. Returns
                     the angle planes **ijk** and **jkl**, this is the dihedral or torsional angle. Angle
                     is in degrees.

10. **massCenter(atoms)**

    parameters - *atoms* is array of atom class objects. Calculates the coordinates of the center of mass
                     of the system. Needs weight list. Returns a list [x,y,z].

11. **isBond(atoms, i,j)**

    parameters - *atoms* is array of atom class objects, *i*, *j* are atom objects. Using the covalent 
                     radii determines if atom i is within bond range of atom j. Factor 1.6 is to give a bit
                     of leeway in distance. Returns a boolean. Returns **False** if *i* equals **j**.

12. **bondMatrix(atoms)**

    parameters - *atoms* is array of atom class objects. Returns a connection matrix. Cell \[i,j] is 0 if no
                     connection, 1 otherwise.

13. **inertiaTensor(atoms)**

     parameters - *atoms* is array of atom class objects.. Returns a matrix of the inertia Tensor. Uses 
                      massCenter function.

14. **principalMoments(atoms)**

    parameters - *atoms* is array of atom class objects. Returns the principal moments of inertia. Uses 
                     inertiaTensor function which is diagonalised by numpy.eig. Returns list of principal 
                     moments sorted by argsort in ascending order. 

15. **rotor(atoms)**

    parameters - *atoms* is array of atom class objects. Returns the rotor type as a string. Uses 
                     principalMoments function.

16. **rotationalConstants(atoms)**

    parameters - *atoms* is array of atom class objects. Returns the rotational constants, uses principalMoments
                     function. Returns [A,B,C] 

17. **nuclearRepulsion(atoms, charge)**

    parameters - *atoms* is array of atom class objects, and *charge* is the molecular charge (remember closed shell 
                      only for RHF. Uses *atoms*.number property. Returns the nuclear repulsion energy as a float.

18. **nuclearChargeCenter(atoms)**

    parameters - *atoms* is array of atom class objects. Returns the center of the molecules nuclear charges as list
                     [x,y,z], Uses *atoms*.number property.

19. **getMass(atoms, atom)**

    parameters - *atoms* is array of atom class objects and *atom* is a member of the atom class. Returns the atomic weight of *atom*.

20. **getNumber(atomicSymbol)**

    parameters - *atomicSymbol* is the recognised symbol for the atom type. Return the atomic number corresponding to the symbol.

21. **gaugeCenter(atoms, mode = 'origin')**

     parameters - *atoms* is array of atom class objects and *mode* is the type of gauge center 
['origin' | 'mass' | 'nuclear charge']. 'origin' is the origin of the Cartesian coordinate system, 'mass' is center of molecular mass and 'nuclear charge' is the center of the positive molecular charges. Default is 'origin'. If *mode* is passed as an array then the routine will pass this point on and evaluation will occur at that point rather than at a named origin.

22. **getConstant(unit):**

      parameters - *unit* is a string representing the conversions. *unit* can be  | 'bohr->angstom' | 'picometre->bohr' | 'radian->degree' | 'planck' | 'bohr->cm' | 'c' | 'dalton->gm' | 'em2->amu' | 'atu->femtosecond' | 'hartree-eV' | 'au->debye' | 'alpha' | 'eV[-1]->nm' | 'Eh' | 'avogardo' | 'electric constant' | 'e' | 'bohr magneton' |. Returns the value of the constant. The constants are 
      
|  mnemonic   |  value    |   description   |
|-------------|-----------|-----------------|
| *bohr->angstrom* | 0.52917721092  | conversion factor atomic units to Angstrom   |
| *picometre->bohr* | 0.018897261339213 | 10<sup>-12</sup> metres in atomic units |
| *radian->degree* | 180/&pi; | radian angles to degrees |
| *planck* | 6.62607015e-34 | Plancks constant (*h*) |
| *bohr->cm* |  0.529177249e-10 | atomic units to centimetres |
| *c* | 2.99792458e10 | speed of light in cm |
| *dalton->gm* | 1.6605402e-27 | unified mass units to gm |
| *em2->amu* | 1822.8884850 | electron mass<sup>2</sup> to atomic mass units |
| *atu->femtosecond* | 0.02418884254 | atomic time units to 10<sup>-15</sup> seconds |
| *hartree->eV* | 27.21138505 | atomic energy units to electron volts |
| *au->debye* | 2.541580253 | atomic electric dipole moment units to Debye |
| *alpha* | 0.00729735256 | fine-structure constant |
| *Ev[-1]->nm* | 1239.841701 | reciprocal electron volts to nanometres (10<sup>-9</sup>) |
| *Eh* | 4.359744722207e-18 | atomic unit of energy in Joules |
| *avogadro* | 6.022140857e+23 | Avogadros's number mol<sup>-1</sup> |
| *electric constant* | 8.854187817e-12 | electric constant (&epsilon;<sub>0</sub>) Fm<sup>-1</sup> |
| *e* | 1.6021766208e-19 | elementary charge in Coulombs |
| *bohr megneton* | 9.274009994e-24 | bohr magneton &mu;<sub>B</sub> JT<sup>-1</sup> |
| *rydberg->eV* | 13.6056980659 | 1 Rydberg in electron volts |

23. **zMatrix(input)**

    parameters - *input* is a list of lines representing the z-matrix input. Dummy atoms represented by **X** are supported and symbolic stretches, bends and dihedrals are allowed. Contains subroutines:

    + rodriguez(axis, theta) - generates rotation matrix for rotation about axis (unit) vector by angle theta (radians).

    + getSymbolValue(s, input) - finds the corresponding numeric value of symbol *s* in input stream *input*.

    + isSymbol(s) - determines if input stream item *s* is a symbol. Returns boolean.

    + getValue(item, type, input) - returns the numeric value of *item* of *type* stretch, bend or dihedral in *input* stream. Angles on input are degrees and are converted to radians by this routine. Calls isSymbol and getSymbolValue for symbolic representations.

    + clean(val, mode) - in mode 1 removes dummy atoms (X) from final coordinate list, in mode 2 strips leading and trailing blank lines from input stream. Internal blank lines are left as delimiters between atoms and symbols.

    + processAtom(atom, input, geo=None) - processes a line of *input* stream representing *atom*. *geo* is the current list of coordinates (needed for number of atoms > 2).
    Returns tuple of atomic number of *atom* and coordinate list.

    The *input* stream is processed for each atom, dummy atoms are removed and list [id, Z, x, y, z] is returned. *id* is a sequence number base 1.

24. **atomList(molAtom):**

    parameters - *molAtom* is an array of atom objects. Returns a list of the atomic numbers of the atoms in the molecule.



