# ocypete Module - cython fast integrals derivatives

As I mentioned before this project was not designed to be fast, the emphasis was on clarity rather than conciseness or speed. However the **aello** module was an example of how to speed up an implementation with Cython. **Aello** is a cython version of the integral module whereas **ocypete** is a cython version of (essentially) the force module, and so concentrates on integral derivatives and forces. The module is called **ocypete** who was one of the Harpies in Greek mythology the name means *swift-wing*. The approach again taken was to convert the python basis and atom class arrays (numpy) to c-memory views and use those.  If you compile **ocypete** via	'cython ocypete.pyx -a' and get an html file which shows how the code is interacting with the python interpreter you will see that most of the calculation is in white (good) with only the python->c code in yellow (bad). The loop are free of yellow which is key to a successful conversion. A copy of ocypete.html is to be found in the document folder. The routines are

1.	**cdef double cye(int ia, int ja, int type, double r, double ie, double je, int n, double x)**\
	  Cython version of e from integral module.

2.	**cdef double efx(int ia,int ja,int type, double r, double ie, double je, int n = 0, double x = 0.0, int p = 0, int s = 0):**\
	  Cython version  of force.efx.

3. 	**cdef double ovlpfx(int ia0, int ia1, int ia2, int ja0, int ja1, int ja2, double ie, double je, \
                   double ir0, double ir1, double ir2, double jr0, double jr1, double jr2, int[3] n, double[3] origin, int x, int center):**\
    Cython version of force.overlapfx

4.	**cdef double kntcfx(int[3] ia, int[3] ja, double ie, double je, double[3] ir, double[3] jr, \
               int[3] n, double[3] origin, int x, int center):**\
    Cython version of kineticfx

5.	**cdef double boys(double m,double T):**\
    Cython version of force.boys

6.	**cdef double clmb(int l, int m, int n, int bf, double p, double r0, double r1, double r2):**\
    Cython version of integral.j

7.	**cdef double clmbsfx(int[3] ia, int[3] ja, double ie, double je, double[3] ir, double[3] jr, \
               double[:] nucleus, int[3] n, double[3] origin, int x, int center):**\
    Cython version of force.coulombfxs

8.	**cdef double clmbhfx(int[3] ia, int[3] ja, double ie, double je, double[3] ir, double[3] jr, \
               double[:] nucleus, int[3] n, double[3] origin, int x):**\
    Cython version of force.coulombfxh

9.	**cdef double teifx(short[:] ia, short[:] ja, short[:] ka, short[:] la, double ie, double je, double ke, double le, \
	              double[:] ir, double[:] jr, double[:] kr, double[:] lr, int[3] ra, int[3] rb, double[3] origin, int x, int center):**\
	Cython version of force.electronRepulsionfx

10.	**cdef double erifx(short[:] ng, int p, int q, int r, int s, double[:,:] im, double[:,:] ic, double[:,:] ie, short[:,:] ia, double[:,:] io, \
	              int x, int center):**\
	Cython version of force.ERIfx

11.	**cpdef ocypete(object molAtom,object molBasis, double[:,:] density, double[:,:] fock):**\
	Main routine. Main loops are over all atoms and cartesian directions. Routines from force called *build* are combined with *kfx*, *jfx* and *jfh* within the main loops to form the one-electron contribution to the fock derivative. two electron derivatives are then calculated and a fock derivative matrix formed. Overlap are then incorporated along with the nuclear contribution. A cut-off of 1e-12 is applied to the final force tensor (any element < cut-off set to 0). Returns the force tensor \[atoms, directions].

		Timings against McMurchie-Davidson program (also using Cython)(JJGoings) on water...
		dz basis   Harpy 20s     McMurchie-Davidson  2m 55s
		sto-3g     Harpy  3s     McMurchie-Davidson  20s
