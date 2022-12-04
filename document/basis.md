# Basis Module - set up the molcular basis


1. subShell                     - dictionary of subShell momenta {s:p:d:f}
2. __init__                     - instantiate basis class object.\
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  atom  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;- atomic center of orbital\
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  symbol  &nbsp; &nbsp;&nbsp; &nbsp; - subShell designation of orbitals\
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  center  &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; - atomic center, \[x,y,z]\
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  momentum - subShell item for the orbital\
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  ex &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  - list of Gaussian exponents\
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  co  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; - list of Gaussian coefficients\
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  normal  &nbsp; &nbsp; &nbsp; &nbsp;   - list of normalisation factors for coefficients

3. **electronCount(atoms, charge)**

   parameters - *atoms* is array of atom class objects, *charge* is molecular charge. Calculated as sum of
		             atom numbers minus the charge. Returns an integer.

4. **species(atoms)**

   parameters - *atoms* is array of atom class objects. Constructs a list of unique atom types contained in
		             the molecule. The returned list is sorted in ascending order so that basis file can be searched
		             sequentially. Returns list.

5. **checkBasis(name, atoms)**

   parameters - *name* is the name of the basis eg *6-32g*, *atoms* is array of atom class objects. Firstly the 
		             routine checks for a file *name*.gbf (Gaussian Basis Functions). Then the atom types returned by species are then checked against the range covered by the basis. If all checks are  passed returns True else False.

6. **principalQuantumNumber(basis)**

   parameters - *basis* is a basis object. Determines the principal quantum number (n) of the orbital. This is
		             the sum of the momentum components of the orbital. Returns an integer.

7. **basisNormalise(basis)**

   parameters - *basis* is a basis object. Calulates the normalisation factor for *basis* and populates 
			     *basis*.normal. Returns the basis object. Uses principalQuantumNumber function.
		             [see for details of calculation](https://content.wolfram.com/uploads/sites/19/2012/02/Ho.pdf)

8. **buildBasis(atoms, name)**

   parameters - *atoms* is array of atom class objects, name is the string containing name of basis. Open basis
		   	     file (uses psi4 format) and cycle through unique atom list from species function. Note each atom 
			     is in file seperated with ******, followed by an atom header eg *Br &nbsp; &nbsp;   0*. File is read 
			     sequentially to locate relevant atom header then reads blocks of exponents and coefficients eg 
			     *SP   3   1.0*. Routine constructs an atomic basis set for the atom type then copies to every instance of that 
			     type in molecule. Returns a molecular basis.

9. **orbitalType(subShell)**

   parameters - *subShell* is an element from subShell list eg \[0,1,0]. Returns a subshell type eg for \[0,1,0] 
			     returns *py*. Returns a string.

10. **aufbau(atom)**

    parameters - *atom* is an object. Returns the aufbau occupancy of the atom object. So for eg Oxygen returns 
			     *1s2 2s2 2p4* but formatted for HTML output. so 1s<sup>2</sup>2s<sup>2</sup>2p<sup>6</sup>.

11. **orbitalShell(molBasis)**

    parameters - *molBasis* is the molecular basis array of basis objects. Prefixes the orbital type with the shell ie px->2px and writes result back to symbol 			 property of the basis object.





