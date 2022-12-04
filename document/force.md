# Forces Module - integral derivatives and force

This module computes the integrals need for calculating the first derivatives of the molecular energy and hence the intra-molecular forces as F = -dE/dx. The integrals routines are written for compactness and are not in any way efficient. It should be easy to expand them to see what is happening. *x* is the direction, either 0=x, 1=y or 2=z and center is an atomic center, either *a*, *b*, *c* or *d*. The main routine is forces which calculates an analytic solution and then a numeric one based on the the central difference formula. Most of the procedures mirror similar ones in integral module without the *fx* postfix. See also the ocypete module which contains Cython versions of these integrals.

1.	**e(ia, ja, type, r, ie, je, n, x)**

2.	**efx(ia, ja, type, r, ie, je, n, x, p, s)**

3.	**overlapfx(ia, ja, ie, je, ir, jr, n, origin, x, center)**

4.	**sfx(iBasis, jBasis, center, x = 0, n = [0,0,0] , origin = [0,0,0])**

5.	**buildOverlapfx(atom, direction, bases)**

6.	**kineticfx(ia, ja, ie, je, ir, jr, n, origin, x, center)**

7.	**kfx(iBasis, jBasis, center, x = 0, n = [0,0,0] , origin = [0,0,0])**

8.	**buildKineticfx(atom, direction, bases)**

9.	**coulombfxs(ia, ja, ie, je, ir, jr, nucleus, n, origin, x, center)**

10.	**jfx(iBasis, jBasis, nucleus, center = 'a', x = 0, n = [0,0,0] , origin = [0,0,0])**

11.	**buildCoulombfx(atom, direction, molAtom, bases)**

12.	**coulombfxh(ia, ja, ie, je, ir, jr, nucleus, n, origin, x)**

13.	**jfh(iBasis, jBasis, nucleus, x = 0,  n = [0,0,0] , origin = [0,0,0])**

14.	**buildCoulombfh(atom, direction, molAtom, bases)**

15.	**buildNuclearfx(atom, direction, molAtom)**

16.	**electronRepulsionfx(ia, ja, ka, la, ie, je, ke, le, ir, jr, kr, lr, ra, rb, origin, x = 0, center = 'a')**

17.	**ERIfx(iBasis, jBasis, kBasis, lBasis, center, x = 0, n = [0,0,0], nu = [0,0,0], origin = [0,0,0])**

18.	**buildERIfx(atom, direction, molAtom, bases)**

19.	**buildFockfx(atom, direction, molAtom, molBasis, density)**

20. **gradient(molAtom, molBasis, molData)**

    parameters - *molAtom* is an array of atom objects, *molBasis* an array of basis objects and *molData* is the array of *charge*, molecule name and basis name which is returned by rhf.mol. This routine calculates the analytic gradient and returns a [number of atoms, 3] array of gradient components.

21. **efxNumeric(atom, direction, molAtom, molBasis, molData)**

    parameters - *atom* is the atom for which the gradient is being calculated, *direction* is the cartesian axis along which the gradient is to be calculated, *molAtom* is an array of atom objects, *molBasis* an array of basis objects and *molData* is the array of *charge*, molecule name and basis name which is returned by rhf.mol. This routine calculates the approximate gradient by central differences and returns the gradient of *atom* in *direction*.


22. **forces(molAtom, molBasis, density, fock, engine = 'aello', type='analytic')**

    parameters - *molAtom* is an array of atom objects, *molBasis* an array of basis objects and *density* is the converged density matrix, *fock* is the converged Fock matrix and *engine* is the integral engine to be used default is 'aello' which uses the **ocypete** cython engine. The final option *type* can be one of 'analytic' or 'both'. This routine calculates the analytic forces (negative of potential derivative with respect to coordinates) and returns a [number of atoms, 3] array of force components and optionally the numeric forces obtained via efxNumeric by a central difference formula.

It is possible to do some geometry optimisation, the function optimiseGeometry is a suggestion of an approach...

23. **optimiseGeometry(f, q0, text):**

     parameters - *f* is a function which evaluates the molecular energy with respect to a number of geometry parameters supplied to the function. *q0* is an initial point in the optimisation space and *text* is a list of descriptions of the geometry items being optimised. The routine calls scipy.minimise using Nelder-Mead simplex method [here](https://en.wikipedia.org/wiki/Nelder–Mead_method) .

	This is an example of optimising the H-O-H angle of water in the 6-31G basis. Define a function to calculate the molecular energy depending on the value of the changing parameter (angle)...
```python
def f(q, molAtom, molBasis, molData):

	import numpy as np
	import rhf

	geo = np.zeros((len(molAtom), 3))
	geo[1,0] = -2.079  
	geo[2,0] = 2.079 * np.cos(q*180/np.pi)
	geo[2,1] = 2.079 * np.sin(q*180/np.pi)
	molAtom, molBasis = rhf.rebuildCenters(molAtom, molBasis, geo)

	return rhf.scf(molAtom, molBasis, molData, [])
```
Run as...
```python

import force

force.optimiseGeometry(f, [90], ['H-O-H angle'])

```
which gives an output of...

H-O-H angle    :   103.5

Optimal energy :  -75.95250973385882

Cycles :  20

True

We can run a full optimisation of both bonds and angle, using
```python
def f(x, molAtom, molBasis, molData):

	import numpy as np
	import rhf
	import math

	geo = np.zeros((len(molAtom), 3))
	a = x[2]*np.pi/180
	geo[1,0] = -x[0]
	geo[2,0] = x[1] * math.cos(a)
	geo[2,1] = x[1] * math.sin(a)
	molAtom, molBasis = rhf.rebuildCenters(molAtom, molBasis, geo)

	return rhf.scf(molAtom, molBasis, molData, [])

```
Run as...
```python

import force

force.optimiseGeometry(f, [1.5, 1.5, 90], ['O-H bond', 'O-H bond', 'H-O-H angle'])

```

with an output of...

H-O-H angle  :  68.5 (111.5 is HOH angle)

H1-O1 bond   :  1.79

H2-O1 bond   :  1.79

Optimal energy :  -75.98535916927149

Cycles :  79

True

--when doing a sequence of similar calculations as in geometry optimisation or molecular dynamics then the eigenvectors of the last computation are likely to be close to those of the next one. It is therefore possible to use the eigenvectors of one calculation as the initial guess for the next one thus reducing the computational cost significantly --







