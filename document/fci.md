# full configuration interaction and singles and doubles

Josh Goings pointed me in the direction of [this paper](https://hal.archives-ouvertes.fr/hal-01539072/document). It uses bit manipulation to implement Slater-Condon rules. The essential results of the Slater-Condon rules are:

1.   The full N! terms that arise in the N-electron Slater determinants do not have to be treated explicitly, nor do the N!(N! + 1)/2 Hamiltonian matrix elements among
     the N! terms of one Slater determinant and the N! terms of the same or another Slater determinant
2.   All such matrix elements, for any one- and/or two-electron operator can be expressed in terms of one- or two-electron integrals over the spin-orbitals that appear 
     in the determinants.
3.   The integrals over orbitals are three or six dimensional integrals, regardless of how many electrons N there are.
4.   These integrals over mo's can, through the LCAO-MO expansion, ultimately be expressed in terms of one- and two-electron integrals over the primitive atomic 
     orbitals. It is only these ao-based integrals that can be evaluated explicitly (on high speed computers for all but the smallest systems).

For single excitations:</br>
  <&#916;|&#937;<sub>1</sub>|&#916;<sub>i</sub><sup>j</sup>> = <&#981;<sub>i</sub>|&#937;<sub>1</sub>|&#981;<sub>j</sub>> , where &#937;<sub>n</sub> is an n-body operator</br>
  <&#916;|&#937;<sub>2</sub>|&#916;<sub>i</sub><sup>j</sup>> = &#931; <&#981;<sub>i</sub>&#981;<sub>k</sub>|&#937;<sub>2</sub>|&#981;<sub>j</sub>&#981;<sub>k</sub>>  - <&#981;<sub>i</sub>&#981;<sub>k</sub>|&#937;<sub>2</sub>|&#981;<sub>k</sub>&#981;<sub>j</sub>> 
  
For double excitations:</br>
  <&#916;|&#937;<sub>1</sub>|&#916;<sub>ik</sub><sup>jl</sup>> = 0</br>
  <&#916;|&#937;<sub>2</sub>|&#916;<sub>ik</sub><sup>jl</sup>> = &#931; <&#981;<sub>i</sub>&#981;<sub>k</sub>|&#937;<sub>2</sub>|&#981;<sub>j</sub>&#981;<sub>l</sub>>  - <&#981;<sub>i</sub>&#981;<sub>k</sub>|&#937;<sub>2</sub>|&#981;<sub>l</sub>&#981;<sub>j</sub>> 
  
All higher excitations are 0. The paper states to implement the rules you need</br>
1.  to find the number of spin-orbital substitutions between two determinants
2.  to find  which spin-orbitals are  involved in the substitution
3.  to compute the phase factor if a reordering of the spin-orbitals has occured.

The paper uses bit manipulation which is certainly efficient, but does it need to be? Surely the evaluation of the integrals will be the critical stage computationally. We could probably do the 3 steps above using strings. Python strings are limited in length only by the machine RAM so you do not need to worry about stitching variables together when 64 bits are used as in the bit method. Let's see how we go...
Using binary variable a slater determinant will be represented as 0b1101 to represent particles at 0&#593;1&#593;1&#946;, that is the lower states are to the right. We shall write this as '1011' that is lower states to the left as this seems more natural. We are not constrained by the little-endian storage of binary data. We could store both the un-excited and excited determinants on the same string as say '1101:1011' but for now we'll save as two separate strings.

Let's take two determinants (single excitation example) 0b111 and 0b1000101. In our notation these will be '111' and '1010001'</br>
Firstly, lets get strings to same shape '111' -> '1110000'. Now we need to find out how many excitations our determinants represent. Let's put our determinants on top of each other</br>
'1110000'</br>
'1010001'</br>
Everywhere a '1' has become a zero represents an excitation (and where a '0' has become a '1').  We have the following routine to count the number of excitations 

1.  **excitations(da, db)** \
     parameters - *da* and *db* are string representations of determinants ie strings of '1's and '0's. Returns the number of excitations between the two determinants.

Now we need to find the excitation jumps. We can see (for single excitation) we need to find where the (first) '1' in determinant 1 has become a '0', then where the (first) '0' has become a '1' in determinant 2. For double excitations we just need to find the second occurrence of the preceding rule too. Once we have found the excitation in the second determinant we must remove it to stop counting it twice, this means we must save a copy of second determinant to restore on exit. This is done in

2.  **levels(da, db)** \
     parameters - *da* and *db* are string representations of determinants. Returns a list of the excitation jumps eg [[0,2], [3,7]]

Now for the phase. The paper states *The phase is calculated as −1<sup>Nperm</sup>, where Nperm is the number permutations necessary to bring the spin-orbitals on which the holes are made to the positions of the particles. This number is equal to the number of occupied spin-orbitals between these two positions.*</br> So let's look at our example,</br>
'1110000'</br>
'1010001'</br>
The number of occupied spin orbitals between where the hole is (1) and where the particle is (6) is 1 (2) so the phase is -1<sup>1</sup> = -1. Let's look at a second example  0b111 -> 0b101010, which we would write '111000' -> '010101'. This is obvoiusly a double excitation (0->3 and 2->5).</br> The phase is </br>
'111000'</br>
'010101'</br>
As above the first hole is (0) and the first particle is (3) and there is 1 occupied orbital between them (1), but there is a second hole at (2) and a second particle at (5) with 1 occupied orbital between them (3) - **but this is an excitation**. The paper notes *For a double excitation, if the realization of the first excitation introduces a new orbital between the hole and the particle of the second excitation  (crossing of the two excitations), an additional permutation is needed...* 
So we have 1+1+1=3 and the phase is -1<sup>3</sup> = -1. </br>
Since n+1 has the same parity n-1 (odd or even) instead of adding an extra permutation we could just not count the excited state in the first place. So we would say we have a hole at (0) and an excitation at (3) with one occupied state between, (3) now becomes '0' as we've dealt with it. We have a second hole at (2) and a particle at (5) with no occupied states between, so permutations are 1 and phase is -1. We have a routine to calculate this

3.  **phase(da, db)** \
     parameters - *da* and *db* are string representations of determinants. Returns the phase of the excitation, either 1 or -1.

We could get the number of permutations of the determinants by using scipy.special.comb to get the total number of combinations. As an example with H<sub>2</sub> in 3-21g basis there are 2 electrons in 4 basis functions, so
```python
print('For hydrogen molecule in 3-21g basis with 2 electrons and 8 (spin) basis functions there are.')
print(scipy.special.comb(2,8), ' determinants')
```
which gives an answer of 28 ie 8 things taken 2 at a time <sup>8</sup>C<sub>2</sub> = !8/!2 !(8-2) = 8.7/2 = 28 \
But we will write our own routine

4.  **determinantCount(m, k)** \
    parameters - *m* is the total number of things to be taken *k* at a time. Returns <sup>m</sup>C<sub>k</sub>.

We now have to find what these combinations are. For example for <sup>8</sup>C<sub>2</sub> (H<sub>2</sub> in 3-21g basis will have 8  spin orbitals and 2 electrons) we will get \
(0,7)(0,6)(0,5)(0,4)(0,3)(0,2)(0,1)</br>
(1,7)(1,6)(1,5)(1,4)(1,3)(1,2)</br>
(2,7)(2,6)(2,5)(2,4)(2,3)</br>
(3,7)(3,6)(3,5)(3,4)</br>
(4,7)(4,6)(4,5)</br>
(5,7)(5,6)</br>
(6,7)</br>

The routine to calculate these combinations is

5.  **combinationList(combs, group, start, stop, level)** \
    parameters - *combs* is an empty list that will contain the combinations, *group* is an empty string, *start* is the beginning of the sequence of items - usually 0, *stop* is the number of items - 1 and *level* is the nmber of selections - 1. For the H<sub>2</sub> example we would run
```python
combs = []
n = 8
k = 2
combinations = combinationList(combs, '', 0, n-1, k-1)
```
Returns a list of combinations.

These combinations must be converted to binary strings. How do we generate the binary strings? \
Well (0,1) we say is 2<sup>0</sup>+2<sup>1</sup> = 3 -> '11', as (1,3) would be 2<sup>1</sup>+2<sup>3</sup> = 2+8 = 10 = '0101' ie a '1' in positions 1 and 3</br>
For water ((0, 1, 2, 3, 4, 5, 7, 10, 11, 13) state would be 11111101001101. We do this using 

6.  **binaryString(comb, nOrbitals)** \
     parameters - *comb* is an element of the combination list (usually a two element list itself) and *nOrbitals* is the length of the returned string.

Examples of the conversion are
(0,1) = '11'</br>
(0,2) = '101'</br>
(2,6) = '0010001'</br>
(2,7) = '00100001'</br>
(5,6) = '0000011'</br>
(5,7) = '00000101'</br>

These binary strings representing the combinations of excited determinants are then passed to

7.  **buildFCIhamiltonian(binary, eriMOspin, coreMOspin)** \
     parameters - *binary* is the list of binary strings representing the excited determinants, *eriMOspin* are the 2-electron repulsion integrals in the molecular spin basis and *coreMOspin* is the core Hamiltonian also in the molecular spin basis. Returns the excited Hamiltonian.

Now to define hamiltonianElement. Firstly it must call *excitations* to get the degree of the excitation, then if degree is <= 2 continue to process. It must now call *levels* to get the excitations themselves and finally call *phase* for the phase. </br>
If it's a double excitation we will have 4 values as say, \[0,3] and \[1,5] this is interpreted as the phase times <01||35>. </br>
For a single excitation say,\[1,6] this will be phase times &#931; <1n||6n>. Where n are common elements between the determinants. We need to calculate the common elements now. 

8.  **commonStates(da, db)** \
     parameters - *da* and *db* are string representations of determinants. Returns a list of the states the two determinants have in common.

For single excitations there is also a one-body contribution which for \[1,6] would be H<sub>sc</sub>\[1,6], where H<sub>sc</sub> is the molecular spin core Hamltonian. Finally the zero degree exitations. These are m are the common states (namely anywhere there is a '1' in either determinant) so there is a H<sub>sc</sub>\[m,m] contribution and if m=n there is a phase times a half times &#931; <mn||mn> for all combinations. These Hamiltonian elements are computed in

9.  **hamiltonianElement(da, db, eriMOspin, Hp)** \
     parameters -  *da* and *db* are string representations of determinants, *eriMOspin* are the 2-electron repulsion integrals in the molecular spin basis and *Hp* is the core Hamiltonian in the molecular spin basis. Returns an element of the excited Hamiltonian.

10. **fci(molAtom, molBasis, charge, c, ERI, coreH)** \
     parameters - *molAtom* is an array of atom objects (the molecular atoms), *molBasis* is an array of basis objects (the molecular basis), *charge* is the molecular charge, *c* are the converged eigenvectors, *ERI* the 2-electron repulsion integrals in atomic basis and *coreH* is the core Hamiltonian. Sends number of electrons, number of spin orbitals, number of determinants, SCF energy, FCI (electronic) energy and FCI correction to postSCF (view module). Returns electronic FCI energies.

For CISD we need a different approach. For H2 we have two parts, the first part is the nElectron ground state eg '11'. Aside from the ground state we want all single '0' substitutions ie '01' and '10', then all double substitutions ie '00'. The second part is the nOrbitals - nElectron part. This must be all combinations of the nOrbitals - nElectron taken k at a time, where k is a number to make total electron count nElectrons. Let's look at an example,
For H2, \
'11' + '000000' \
'01', '10' + '000001', '000010', '000100', '001000', '010000', '100000' \
'00' + '000011', '000101', '001001', '010001', '100001', '000110', '001010', '010010', '100010', '001100', '010100', '100100', '011000', '101000' \
total 28. We have code to do this 

11. **configurations(nElectrons, nOrbitals, type = 'S')** \
     parameters - *nElectrons* are the number of electrons, *nOrbitals* are the number of spin orbitals and *type* can be one or a combination of 'G' - ground state, 'S' - singles, 'D' - doubles, 'T' - triples and 'Q' - quadruples. Returns a list of determinants. Eg 'S' for singles (CIS), 'GSD' for singles and doubles (CISD), 'GD' for doubles (CID) etc.


12. **cisd(molAtom, molBasis, charge, c, ERI, coreH)**
     parameters - *molAtom* is an array of atom objects (the molecular atoms), *molBasis* is an array of basis objects (the molecular basis), *charge* is the molecular charge, *c* are the converged eigenvectors, *ERI* the 2-electron repulsion integrals in atomic basis and *coreH* is the core Hamiltonian. Sends number of electrons, number of spin orbitals, number of determinants, SCF energy, CISD energy and CISD correction to postSCF (view module).Returns transition energies.
  
13. **ciss(molAtom, molBasis, charge, c, ERI, coreH)** \
     parameters - *molAtom* is an array of atom objects (the molecular atoms), *molBasis* is an array of basis objects (the molecular basis), *charge* is the molecular charge, *c* are the converged eigenvectors, *ERI* the 2-electron repulsion integrals in atomic basis and *coreH* is the core Hamiltonian. Sends number of electrons, number of spin orbitals, number of determinants and excitations energy (eV) to postSCF (view module). Returns exitations (Hartree). Returns transition energies and eigenvectors which can be re-shaped for transition dipole and oscillator computations.

We have the following test results - for water in STO-3G (all values checked against McMurchie-Davidson program),

|            |            |
|------------|------------|
|number of electrons    |10 |
|number of spin orbitals    |14 |
|number of determinants|    1001 |
|SCF energy|    -74.942080  |
|FCI energy|    -75.012980 |
|FCI correction|    -0.070900  |
|            |            |
|CISD energy|   -75.011223  |
|CISD correction|   -0.069143  |

and for CI singles using slater determinants
|           |        |
|--------|--------|
|number of electrons   | 10|
|number of spin orbitals | 14 |
|number of determinants | 40 |

| in eV  |   |  |   |   | |
|------|-----|--------|----------|---------|---------|
|7.816620 | 7.816620 |  7.816620    |9.372282 | 9.372282|   9.372282 |
|9.699819 | 9.959068 |  9.959068    |9.959068 | 10.735267 | 10.735267 |

and for H<sub>2</sub>, FCI and CISD should be same as only double excitations possible for 2 electrons

|            |            |
|------------|------------|
|number of electrons    |2 |
|number of spin orbitals    |8 |
|number of determinants|    28 |
|SCF energy|    -1.122940 |
|FCI energy|    -1.147813 |
|FCI correction|    -0.024873 |
|            |            |
|CISD energy|   -1.147813  |
|CISD correction|   -0.024873  |

and for CI singles using slater determinants
|           |        |
|--------|--------|
|number of electrons   | 2|
|number of spin orbitals | 8 |
|number of determinants | 12 |

| in eV  |   |  |   |   | |
|------|-----|--------|----------|---------|---------|
|10.395356 |  10.395356 |  10.395356 |  15.755380  | 25.908408  | 25.908408 |
|25.908408  | 32.127966  | 39.998167 |  39.998167 |  39.998167 |  46.622756 |

14. **spinStates(da, nBasis)**
    parameters - *da* is a spin-intermngled determinant and *nBasis* are the number of doubly occupied basis functions. Returns two string representations of the alpha and beta spin states. 

15. **occupancy(da, nBasis)**
    parameters - *da* is a spin-intermingled determinant and *nBasis* are the number of doubly occupied basis functions. Returns a string representing the occupany of the spatial orbitals. A string entry can be one of 0,1,2 for vacant, singly and doubly occupied orbitals.
- - -
### Bit manipulation versions

16. **bString(da, spinOrbitals)**

    parameters - *da* is a spin-intermingled determinant and *spinOrbitals* are the number of spin orbitals. Return the integer representation of the bit string representing *da*. <mark> 0b101010 -> 42 </mark>

17. **bCombinationList(spinOrbitals, nElectrons)**

    parameters - *spinOrbitals* are the number of spin orbitals and *nElectrons* are the number of electrons. Returns a list of all combinations of putting *nElectrons* into *spinOrbital* orbitals. The list is integers produced by *bstring*. <mark>(1,4,5,8) -> 0b010011001 -> 153</mark>

18. **bRightZeros(n)**

    parameter - *n* is an integer that the number of zeros to it's right of its binary representation is returned. Returns integer. <mark> 88 -> 0b1011000 -> 3</mark>

19. **bSetZero(da, n)**

    parameters - *da* is a spin-intermingled determinant and *n* is the bit that will be cleared (set to 0). Returns integer representing modified determinant. <mark>22 -> 0b10110 -> n=2 -> 0b10010 -> 18</mark>

20. **bOccupancy(da, db, type = 'h')**

    parameters - *da* and *db* are a pair of determinants and *type* is either 'h' for holes or 'p' for particles. Returns a list of the positions where holes have appeared in the excitation or the position where particles have appeared. <mark> 0b111 and 0b101010 -> for 'h' [1], for 'p' [5, 3]</mark>

21. **bExcitations(da, db)**

    parameters - *da* and *db* are a pair of determinants. Returns the number of excitations between two determinants. <mark> 0b111 and 0b101010 -> 2</mark>

22. **bSingleExcitations(da, db)**

    parameters - *da* and *db* are a pair of determinants. Returns the matrix element for a pair of determinants with a single excitation between them.

23. **bDoubleExcitations(da, db)**

    parameters - *da* and *db* are a pair of determinants. Returns the matrix element for a pair of determinants with a double excitation between them.

24. **bBuildFCIhamiltonian(determinants, eriMOspin, coreH)**

    parameters - *determinants* are a list a combinations of determinants, *eriMOspin* are the double-bar 2-electron repulsion integrals and *coreH* is the core Hamiltonian. Returns the FCI Hamiltonian matrix.

25. **bFci(molAtom, molBasis, charge, c, ERI, coreH)**

    parameters - *molAtom* is an array of atom objects, *molBasis* is an array of basis objects, *c* are the converged scf eigenvectors, *ERI* are the 2-electron repulsion integrals and *coreH* is the core Hamiltonian. Returns the FCI eigenvalues, eigenvectors and the number of determinants processed.

26. **bCommonStates(da, db)**

    parameters - *da* and *db* are a pair of determinants. Returns the states that the determinants have in common. <mark> 0b111 and 0b101010 -> [1]</mark>

27. **bHamiltonianElement(da, db, eriMOspin, coreH)**

    parameters - *da* and *db* are a pair of determinants, *eriMOspin* are the double-bar 2-electron repulsion integrals and *coreH* is the core Hamiltonian. Returns the matrix element for determinants *da* and *db*.

28. **bResidues(da, spinOrbitals)**

    parameters - *da* is a spin-intermingled determinant and *spinOrbitals* are the number of spin orbitals. Returns a list of residues - *The residues are a set of determinants that are generated by removing two electrons in all possible ways from the reference determinant*.

29. **bSetResidues(residues, spinOrbitals)**

    parameters - *residues* is a list of residues and *spinOrbitals* are the number of spin orbitals. Returns a list of the residues with every combination of 2 orbitals filled.

30. **bCisd(molAtom, molBasis, charge, c, ERI, coreH)**

    parameters - *molAtom* is an array of atom objects, *molBasis* is an array of basis objects, *c* are the converged scf eigenvectors, *ERI* are the 2-electron repulsion integrals and *coreH* is the core Hamiltonian. Returns the CISD eigenvalues and the number of determinants processed.




