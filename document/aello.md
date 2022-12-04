# aello Module - cython fast integrals

This project was not designed to be fast, the emphasis was on clarity rather than conciseness or speed. However this is an exercise on speeding up an implementation. Note it is a *bolt-on* to existing code and not how perhaps one would design a cython project from the beginning. The module is called *aello* who was one of the Harpies in Greek mythology the name means *storm-swift*. The aim of any python-to-cython project is to reduce as much as possible the python interaction with the code leaving c to do the most work possible. The approach I took here was to convert the python basis and atom class arrays (numpy) to c-memory views and use those. This is done by the following method...

        me = np.empty([nb,ng], dtype = np.double)   #define a variable, in this case for the exponent array
        cdef double[:,:] alo_e = me                 #equate it to a memory view
        
Then populate the view and use it for all calculations. Use np.empty rather than np.zeros, it save a bit of time and helps with debugging. I've also reduced the function calls by expanding some of the integral module functions in-line. (Because the exponent, coefficient and normal array length varies within a basis set there is an extra array which holds the number of primatives for each basis function.) If you compile aello via **cython aello.pyx -a** and get an html file it shows how the code is interacting with the python interpreter you will see that most of the calculation is in white (good) with only the python->c code in yellow (bad). The loops are free of yellow which is key to a successful conversion. The routines are

1.  **cdef double cye(int ia, int ja, int type, double r, double ie, double je)**

    parameters - *ia, ja* are momenta values, *type* determines the calculation mode, *r* is the seperation vector between 
        atoms i and j, and *ie, je* are the exponents of atom centers i and j. Cython version of e from integral module.

2.  **cdef double ovlp(int ia0, int ia1, int ia2, int ja0, int ja1, int ja2, int type, double r0, double r1, double r2, double ie, double je):**

    parameters - *ia0-ja2* are momenta values of atoms i and j, *type* as before, *r0-r2* are components of vector between 
        atoms i and j, and *ie, je* are the exponents of atom centers i and j, Cython version of overlap from integral module.

3.  **cdef double clmb(int l, int m, int n, int bf, double p, double r0, double r1, double r2)**

    parameters - *l, m, n* components of sum of momenta on atoms i and j plus 1, *bf* Boys function parameter, *r0-r2* 
        components of vector from atom i to atom j. Cython version of j from integral module.

4.  **cdef double boys(double m,double T)**

    Direct copy of similar routine from integral module. Use *from scipy.special.cython_special cimport hyp1f1*

5.  **cdef double tei(int al0, int al1, int al2, int al3, short[:,:] aa, double[:,:] an, double[:,:] ac,double[:,:] ae, double[:,:] ao, int i, int j, int k, int l):**

    parameters - *al0-al3* number of primatives on basis on atoms i,j,k and l, *aa* array of primative lengths, *an* array 
        of normals, *ac* array of coefficients, *ae* array of exponents, *ao* array of basis atom centers and *i,j,k,l* basis 
        numbers ie  <ij|kl>. Note by array is meant a memory view.

6.  **cdef double mu(int[3] ia, int[3] ja, double ie, double je, double[3] ir, double[3] jr, double[3] kr, int direction):**

    parameters - *ia, ja* are momenta on atoms i and j, *ie, je* are expononts, *ir, jr* are atom centers, *kr* is charge center and *direction* the cartesian axis             being computed.
   
7.  **cdef double ang(int[3] ia, int[3] ja, double ie, double je, double[3] ir, double[3] jr, double[3] kr, int direction):**

    parameters - *ia, ja* are momenta on atoms i and j, *ie, je* are expononts, *ir, jr* are atom centers, *kr* is charge center and *direction* the cartesian axis             being computed.


8.  **def aello(molAtom, molBasis, mode = 'scf', density = None, gauge = None):**

    parameters - *molAtom* array of atom objects, *molBasis* array of basis objects. *mode* is type of integrals returned, either ['scf'|'dipole']. Returns S, K, J and I (overlap,kinetic, coulomb and eri) in 'scf' mode, dipole component matrices in 'dipole' mode.

Does it work? Water DZ basis without aello - 1m 42s, with aello - 11s and\
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Water 6-31g &nbsp; &nbsp; &nbsp; without aello - 1m 16s, with aello - 9s\
so yes it's faster but still quite slow because speed has not been a consideration in writing the code.
The integral routines are quite efficient, eg a aug-cc-pvdz integral comparison with McMurchie-Davidson (Josh Goings)...\
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; harpy  &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp;      McMurchie-Davidson\
1-electron  &nbsp; &nbsp; &nbsp;    0.046   &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;         3.164\
2-electron   &nbsp; &nbsp; &nbsp;  30.147   &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;        35.482

 9. **cpdef aelloDipole(double[:,:] alo_n, double[:,:] alo_c, double[:,:] alo_e, short[:,:] alo_a, double[:,:] alo_o, short[:] alo, short[:] alo_z, double[:,:] alo_x, int na, int nb, object molAtom, double[:,:] density):** 

    parameters - *alo* \_n, \_c, \_e, \_o are basis normal, coefficients, exponents and centers. *alo* is the number of primatives, *alo* \_z and \_x are the atomic numbers and centers. *na* and *nb* the number of atoms and bases, *molAtom* is the molecular atom object and *density* the converged final density matrix. Returns a vector of dipole components.
    
10. **cpdef aelloAngular(double[:,:] alo_n, double[:,:] alo_c, double[:,:] alo_e, short[:,:] alo_a, double[:,:] alo_o, short[:] alo, short[:] alo_z, double[:,:] alo_x, int na, int nb, object molAtom, gauge):** 

    parameters - *alo* \_n, \_c, \_e, \_o are basis normal, coefficients, exponents and centers. *alo* is the number of primatives, *alo* \_z and \_x are the atomic numbers and centers. *na* and *nb* the number of atoms and bases, *molAtom* is the molecular atom object and  *gauge* is the gauge type as string. Returns a vector of angular components.

The other major bottleneck is the construction of the fock matrix. There are two aello cython routines to accelerate this.

11. **iEri(long i, long j, long k, long l)**

    parameters - *i,j,k,l* are indexes into the eri tensor.

12. **aelloFock(long n, double[:,:] H, double[:] eri, double[:,:] D))**

    parameters - *n* is the number of basis functions, *H* the core hamiltonian, *eri* the 2-electron repulsion integrals and *D* the density matrix.

13. **tdhfFock(long n, double complex[:,:] D, double complex[:,:] H, double complex[:] eri):**

    parameters - *n* is the number of basis functions, *H* the core hamiltonian, *eri* the 2-electron repulsion integrals and *D* the density matrix. This is a complex arithmetic version of **aelloFock** above. Especially written for the TDHF module. It is an example of how to use complex with Cython.


It's still not fast but it is a lot faster. The file *aello.html* produced by *cython aello.pyx -a* is included in document directory.
