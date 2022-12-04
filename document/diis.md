# Direct Inversion of the Iterative Subspace

A discussion of the algorithm can be found in [Crawford projects]  (https://github.com/CrawfordGroup/ProgrammingProjects/tree/master/Project%2308) for Fock convergence and [here](https://github.com/CrawfordGroup/ProgrammingProjects/tree/master/Project%2310) for the coupled cluster iteration. DIIS requires an 'error vector', a quantity that approaches zero as the equations converge. For the Fock iteration that quantity is a two-dimensional matrix while for coupled cluster it is made up of a 2-dimensional array and a 4-dimensional array. In the latter case the arrays are flattened and concatenated to make the diis object. It is possible to flatten the Fock iteration quantities and in practice use a single routine for both cases. Here there are two routines, one for Fock matrix and one for coupled cluster linear arrays as an example of both ways of doing a diis procedure. They are implemented as classes.

1.	**diis_f(diisCapacity)**

	parameters - *diisCapacity* is the (maximum) size of diis buffer. *diisStatus* is either 'on' or 'off'. Returns a new Fock matrix. The class is initiated with the capacity being passed. The other call is *build(fock, density, s, x)* which builds the fock and error buffers and resturns the extrapolated fock. See rhf.py for an example of use.


2. **diis_c(diisCapacity, [ts, td, tt, ...])**

	parameters - *ts, td, tt, ...*  are arrays of the quantities on which the diis procedure is based, in this case the singles, doubles, triples etc amplitudes.  *diisCapacity* is the (maximum) size of diis buffer and *spinOrbitals* are the number of alpha plus beta spin orbitals.The class is initiated with the initial amplitudes and the capacity being passed. *refresh_store([ts, td, ...])* is used to update a buffer of this iteration and the last iteration values for use in convergence determination. *build([ts, td. ...])* builds the amplitude and error buffers and returns the extrapolated amplitudes. See either scc.py or fcc.py for examples of how to use this class.
