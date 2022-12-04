# H<sub>2</sub> k-edge spectra

The K-edge spectrum is a spectrum obtained when an X-Ray excites a *1s* inner core electron. The resulting excitation state can be below the ionization potential (IP) of the *1s* orbital (**pre-edge**) or above the IP (**above-edge**). For one electron, the spectrum arises because of transitions from the *1s* core orbital to excited molecular orbitals. Here we use Koopman's theorem to do a crude k-edge analysis of H<sub>2</sub>O so an approximation of excitation energy is obtained by taking the sum of the absolute value of the two orbital energies involved in the transition. These are the routines used...

1. **getMolecule(basis)**

   parameters - *basis* is the molecular basis object as returned by *rhf.mol*. We start by defining the water molecule in a code object and writing that to temporary file to be read by rhf.mol. We run an SCF compuation so we have access to integrals and return the atom, basis and data objects along with the converged SCF energy.

2. **getOrbitalEnergies(fock, S)**

   parameters - *fock* the Fock matrix from a converged RHF calculation and *S* is the overlap matrix from an SCF calculation. Orthogonalizes the Fock matrix and gets the eigensolution. Returns the orbital energies.

3. **getOrbitalCharacter(orbital, basis, atom)**

   parameters - *orbital* is a column of the eigenvectors matrix (dimension\[nbf,1]), *basis* is an array of basis objects and *atom* is an array of atom objects. Finds the predominant contributing AO and returns a tuple of the AO's atom center id and it's orbital symbol eg (2px', 'O1').

4. **getMOwithCharacter(character, c, basis, atom)**

   parameters - *character* is a tuple as returned by *getOrbitalCharacter*, *c* are the molecular eigenvectors, *basis* is an array of basis objects and *atom* is an array of atom objects. Finds the MO which most matches the character requested. Returns an integer.

5. **getKedgeExcitations(k, n, c, basis, atom, eMO)**

   parameters - *k* is the starting AO for the excitations as a tuple (here it is ('1s','O1')), *n* the number of excitations required, *c* are the molecular eigenvectors, *basis* is an array of basis objects, *atom* is an array of atom objects and *eMO* is an array of orbital energies. This finds the LUMO and returns the *n* energy differences between *k* and lumo+0, lumo+1,...lumo+*n*.

6. **getDipoles(atom, basis)**

   parameters - *atom* is an array of atom objects and *basis* is an array of basis objects.Returns the components matrices of the dipoles in the 'origin' gauge.

7. **getTransitionProperties(mu, c, i, a)**

   parameters - *mu* are the dipole component matrices, *c* are the orbital eigenvectors, *i* is an occupied MO and *a* in an unoccupied MO - the excitation id *i*->*a*. Computes the components of the transition dipole moment and the oscillator strength.

8. **lorentzian(e0, e, tau)**

   parameters - *e0* is the excitation energy, *e* are the energies over the broadening range and *tau* is the life-time of the shape. The life-time (&tau;) is the reciprocal of the full-width at half-maximum of the shape (FWHM). We use a *tau* of 40au corresponding to 10<sup>-15</sup>s, a reasonable value for core excitations. Returns an array of points representing the broadened line shape.


The procedure is to first call **getMolecule** to run an RHF SCF computation so integrals are calculated. Next from **getOrbitalEnergies** we calculate the orbital energies, which are used by **getKedgeExcitations** to get the energy differences between the *1s* oxygen dominant MO and, say, 5 levels (in 6-31g basis with 13 basis functions 5-occupied and 8-virtual) at LUMO and above (eg 0->5, 0->6, 0->7, 0->8 nd 0->9). We then use **getTransitionProperties** to calculate the transition dipoles and oscillator strengths for the 5 transitions. The oscillator strengths are then broadened and plotted. This gives the following plot

![image](https://user-images.githubusercontent.com/73105740/130357664-df7a0970-62f6-4b72-9c8f-34a8b3e01bd7.png)


The broadened line shapes are in black and the orange bars represent the unbroadened values (normalised). The MO are numbered at the top and you can see the biggest strength is from 1s to MO<sub>9</sub>. This MO is the oxygen 2p<sub>z</sub> orbital.

