#cython: boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False, cdivision=True
from libc.math cimport exp, pow, tgamma, sqrt, abs
from scipy.special.cython_special cimport hyp1f1 
from atom import gaugeCenter

import numpy as np
cimport numpy as np

cdef double pi = 3.141592653589793238462643383279

#from integral - e
cdef double cye(int ia,int ja,int type, double r, double ie, double je, int n = 0, double x = 0.0):

    cdef:
        double p = ie + je
        double q = ie*je / p

    if n == 0:
        if (type < 0) or (type > (ia + ja)):
            return 0.0
        elif (ia + ja + type) == 0:
            return exp(-q*r*r)
        elif ja == 0:
            return (1/(2 * p)) * cye(ia-1,ja,type-1,r,ie,je) - (q*r/ie) * cye(ia-1,ja,type,r,ie,je) + \
                         (type+1) * cye(ia-1,ja,type+1,r,ie,je)
        else:
            return (1/(2 * p)) * cye(ia,ja-1,type-1,r,ie,je) + (q*r/je) * cye(ia,ja-1,type,r,ie,je) + \
                         (type+1) * cye(ia,ja-1,type+1,r,ie,je)
    else:
        return cye(ia+1,ja,type,r,ie,je,n-1,x) + x * cye(ia,ja,type,r,ie,je,n-1,x)

cdef double ovlp(int ia0, int ia1, int ia2, int ja0, int ja1, int ja2, int type, \
                           double r0, double r1, double r2, double ie, double je):
    cdef double s 
    s =  cye(ia0, ja0, type, r0, ie, je)
    s *= cye(ia1, ja1, type, r1, ie, je)
    s *= cye(ia2, ja2, type, r2, ie, je)

    return s * pow(pi/(ie+je),1.5)

cdef double clmb(int l, int m, int n, int bf, double p, double r0, double r1, double r2):

    cdef double t, s, nm
    nm = sqrt(r0*r0 + r1*r1 + r2*r2)
    t = p * nm * nm

    s = 0.0
    if (l+m+n)  == 0:
        s += pow(-2*p, bf) * boys(bf, t)
    elif (l+m) == 0:
        if n > 1:
            s +=(n-1) * clmb(l,m,n-2,bf+1,p,r0,r1,r2)
        s += r2 * clmb(l,m,n-1,bf+1,p,r0,r1,r2)
    elif l == 0:
        if m > 1:
            s +=(m-1) * clmb(l,m-2,n,bf+1,p,r0,r1,r2)
        s += r1 * clmb(l,m-1,n,bf+1,p,r0,r1,r2)
    else:
        if l > 1:
            s +=(l-1) * clmb(l-2,m,n,bf+1,p,r0,r1,r2)
        s += r0 * clmb(l-1,m,n,bf+1,p,r0,r1,r2)

    return s


#boys function
cdef double boys(double m,double T):
    return hyp1f1(m+0.5,m+1.5,-T)/(2.0*m+1.0) 

cdef double tei(int al0, int al1, int al2, int al3, short[:,:] aa, double[:,:] an, double[:,:] ac, \
                double[:,:] ae, double[:,:] ao, int i, int j, int k, int l):
    
    cdef:
        double s = 0.0
        int mu, nu, vu, su, tu, psi, phi, chi, alpha, beta, gamma
        double f, p, q, t1, s1, s2
        double t2[3]

    for mu in range(0, al0):
        for nu in range(0, al1):
            for vu in range(0, al2):
                for su in range(0, al3):
                    f =  an[i,mu] * an[j,nu] * an[k,vu] * an[l,su] * ac[i,mu] * ac[j,nu] * ac[k,vu] * ac[l,su]
                    p = ae[i,mu] + ae[j,nu]
                    q = ae[k,vu] + ae[l,su]
                    t1 = p*q/(p+q)
                    for tu in range(0, 3):
                        t2[tu] = (ae[i,mu]*ao[i,tu] + ae[j,nu]*ao[j,tu])/p - (ae[k,vu]*ao[k,tu] + ae[l,su]*ao[l,tu])/q

                    s1 = 0.0
                    for psi in range(0, aa[i,0]+aa[j,0]+1):
                        for phi in range(0, aa[i,1]+aa[j,1]+1):
                            for chi in range(0, aa[i,2]+aa[j,2]+1):
                                for alpha in range(0, aa[k,0]+aa[l,0]+1):
                                    for beta in range(0, aa[k,1]+aa[l,1]+1):
                                        for gamma in range(0, aa[k,2]+aa[l,2]+1):
                
                                            s2 = cye(aa[i,0],aa[j,0],psi, ao[i,0]-ao[j,0],ae[i,mu],ae[j,nu]) *           \
                                                 cye(aa[i,1],aa[j,1],phi, ao[i,1]-ao[j,1],ae[i,mu],ae[j,nu]) *           \
                                                 cye(aa[i,2],aa[j,2],chi, ao[i,2]-ao[j,2],ae[i,mu],ae[j,nu]) 
                                            s2*= cye(aa[k,0],aa[l,0],alpha, ao[k,0]-ao[l,0],ae[k,vu],ae[l,su]) *         \
                                                 cye(aa[k,1],aa[l,1],beta, ao[k,1]-ao[l,1],ae[k,vu],ae[l,su])  *         \
                                                 cye(aa[k,2],aa[l,2],gamma, ao[k,2]-ao[l,2],ae[k,vu],ae[l,su])
                                            s2*= pow(-1, alpha+beta+gamma) * clmb(psi+alpha, phi+beta, chi+gamma, 0, t1, \
                                                                                  t2[0],t2[1],t2[2])
                                            s1 += s2
                    s1 *= 2 * pow(pi, 2.5) / ((p*q) * sqrt(p+q))
                    s += s1 * f
    return s

#|-------------------------------------dipole helper-------------------------------------|

cdef double mu(int[3] ia, int[3] ja, double ie, double je, double[3] ir, double[3] jr, double[3] kr, int direction):
    # dipole moment
    cdef:
        double p = ie + je
        double[3] q, ijr
        int i
        double u, v, t

    for i in range(3):
        q[i] = ((ie*ir[i] + je*jr[i])/p) - kr[i]
        ijr[i] = ir[i] - jr[i]

    if direction == 1:
        u = cye(ia[0], ja[0], 1, ijr[0], ie, je) + q[0]* cye(ia[0], ja[0], 0, ijr[0], ie, je) 
        v = cye(ia[1], ja[1], 0, ijr[1], ie, je) 
        t = cye(ia[2], ja[2], 0, ijr[2], ie, je) 
        return u * v * t * pow(pi/p, 1.5)
    if direction == 2:
        u = cye(ia[0], ja[0], 0, ijr[0], ie, je)  
        v = cye(ia[1], ja[1], 1, ijr[1], ie, je) + q[1]* cye(ia[1], ja[1], 0, ijr[1], ie, je)
        t = cye(ia[2], ja[2], 0, ijr[2], ie, je) 
        return u * v * t * pow(pi/p, 1.5)
    if direction == 3:
        u = cye(ia[0], ja[0], 0, ijr[0], ie, je)  
        v = cye(ia[1], ja[1], 0, ijr[1], ie, je) 
        t = cye(ia[2], ja[2], 1, ijr[2], ie, je) + q[2]* cye(ia[2], ja[2], 0, ijr[2], ie, je)
        return u * v * t * pow(pi/p, 1.5)

#|---------------------------------end dipole helper-------------------------------------|

#|------------------------------------momentum helper------------------------------------|
cdef double ang(int[3] ia, int[3] ja, double ie, double je, double[3] ir, double[3] jr, double[3] kr, int direction):
    # angular momentum
    cdef:
        double p = ie + je
        double[3] ijr
        int i
        double u, v, t
        double sd[3][3]

    for i in range(3):
        ijr[i] = ir[i] - jr[i]

    for i in range(3):
        sd[0][i] = cye(ia[i], ja[i], 0, ijr[i], ie, je)
        sd[1][i] = cye(ia[i], ja[i], 0, ijr[i], ie, je, 1, ir[i]-kr[i])
        sd[2][i] = (ja[i] * cye(ia[i], ja[i]-1, 0, ijr[i], ie, je)) - (2.0 * je * cye(ia[i], ja[i]+1, 0, ijr[i], ie, je))

    if direction == 1:
        return -sd[0][0] * (sd[1][1] * sd[2][2] - sd[1][2] * sd[2][1]) * pow(pi/p, 1.5)
    elif direction == 2:
        return -sd[0][1] * (sd[1][2] * sd[2][0] - sd[1][0] * sd[2][2]) * pow(pi/p, 1.5)
    elif direction == 3:
        return -sd[0][2] * (sd[1][0] * sd[2][1] - sd[1][1] * sd[2][0]) * pow(pi/p, 1.5)

#|--------------------------------end momentum helper------------------------------------|

#get the atom and basis classes
def aello(molAtom, molBasis, mode = 'scf', density = None, gauge = None):

    cdef:
        int na = len(molAtom)
        int nb = len(molBasis)
        int ng = len(molBasis[0].co)
        int  i, j, k, l, m, n, p, q

    #get largest primative length
    for i in range(nb):
        j = len(molBasis[i].co)
        if j > ng:
            ng = j

    #convert atom class properties to c views
    mx = np.empty([na,3], dtype = np.double)
    mz = np.empty([na], dtype = np.short)
    cdef:
        double[:,:] alo_x = mx
        short[:]   alo_z = mz
    for p in range(0, na):
        for q in range(0, 3):   
            alo_x[p,q] = molAtom[p].center[q]
        alo_z[p] = molAtom[p].number

    #convert basis class properties to c-variables
    me = np.empty([nb,ng], dtype = np.double)
    mc = np.empty([nb,ng], dtype = np.double)
    mn = np.empty([nb,ng], dtype = np.double)
    ma = np.empty([nb,3],  dtype = np.short)
    mo = np.empty([nb,3],  dtype = np.double)
    ml = np.empty([nb],    dtype = np.short)
    cdef: 
        double[:,:] alo_e = me
        double[:,:] alo_c = mc
        double[:,:] alo_n = mn
        short[:,:]  alo_a = ma
        double[:,:] alo_o = mo
        short[:]    alo   = ml

    for p in range(0, nb):
        alo[p] = len(molBasis[p].co)
        for q in range(0, len(molBasis[p].co)):
            alo_e[p,q] = molBasis[p].ex[q]
            alo_c[p,q] = molBasis[p].co[q]
            alo_n[p,q] = molBasis[p].normal[q]
        for q in range(0, 3):
            alo_a[p,q] = molBasis[p].momentum[q]
            alo_o[p,q] = molBasis[p].center[q]

    if mode == 'dipole':
        return aelloDipole(alo_n, alo_c, alo_e, alo_a, alo_o, alo, alo_z, alo_x, na, nb, molAtom, density, gauge)

    if mode == 'angular':
        return aelloAngular(alo_n, alo_c, alo_e, alo_a, alo_o, alo, alo_z, alo_x, na, nb, molAtom, gauge)

#-------------------------------------Begin Overlap---------------------------------------|
    S = np.empty([nb,nb], dtype = np.double)
    cdef:
        double [:,:] overlap  = S
        double s, f

    for p in range(0, nb):
        for q in range(p, nb):

            s = 0.0
            for i in range(0, alo[p]):
                for j in range(0, alo[q]):
                    f = alo_n[p,i] * alo_n[q,j] * alo_c[p,i] * alo_c[q,j]
                    s += ovlp(alo_a[p,0], alo_a[p,1], alo_a[p,2], alo_a[q,0], alo_a[q,1], alo_a[q,2],       \
                              0 ,alo_o[p,0] - alo_o[q,0], alo_o[p,1] - alo_o[q,1], alo_o[p,2] - alo_o[q,2], \
                              alo_e[p,i], alo_e[q,j]) * f

            overlap[p,q] = s
            if p != q:
                overlap[q,p] = overlap[p,q]
#----------------------------------------End Overlap----------------------------------------|

#---------------------------------------Begin Kinetic---------------------------------------|
    K = np.empty([nb,nb], dtype = np.double)
    cdef:
        double[:,:] kinetic = K
        double t1, t2, t3

    for p in range(0, nb):
        for q in range(p, nb):

            s = 0.0
            for i in range(0, alo[p]):
                for j in range(0, alo[q]):
                    f = alo_n[p,i] * alo_n[q,j] * alo_c[p,i] * alo_c[q,j]
                    t1 = alo_e[q,j] * (2*(alo_a[q,0] + alo_a[q,1] + alo_a[q,2]) + 3) *                      \
                         ovlp(alo_a[p,0], alo_a[p,1], alo_a[p,2], alo_a[q,0], alo_a[q,1], alo_a[q,2],       \
                              0 ,alo_o[p,0] - alo_o[q,0], alo_o[p,1] - alo_o[q,1], alo_o[p,2] - alo_o[q,2], \
                              alo_e[p,i], alo_e[q,j])   

                    t2 = -2 * alo_e[q,j] * alo_e[q,j] * (                                                   \
                         ovlp(alo_a[p,0], alo_a[p,1], alo_a[p,2], alo_a[q,0]+2, alo_a[q,1], alo_a[q,2],     \
                              0 ,alo_o[p,0] - alo_o[q,0], alo_o[p,1] - alo_o[q,1], alo_o[p,2] - alo_o[q,2], \
                              alo_e[p,i], alo_e[q,j])   +                                                   \
                         ovlp(alo_a[p,0], alo_a[p,1], alo_a[p,2], alo_a[q,0], alo_a[q,1]+2, alo_a[q,2],     \
                              0 ,alo_o[p,0] - alo_o[q,0], alo_o[p,1] - alo_o[q,1], alo_o[p,2] - alo_o[q,2], \
                              alo_e[p,i], alo_e[q,j])   +                                                   \
                         ovlp(alo_a[p,0], alo_a[p,1], alo_a[p,2], alo_a[q,0], alo_a[q,1], alo_a[q,2]+2,     \
                              0 ,alo_o[p,0] - alo_o[q,0], alo_o[p,1] - alo_o[q,1], alo_o[p,2] - alo_o[q,2], \
                              alo_e[p,i], alo_e[q,j])  )


                    t3 = alo_a[q,0] * (alo_a[q,0] - 1) *                                                    \
                         ovlp(alo_a[p,0], alo_a[p,1], alo_a[p,2], alo_a[q,0]-2, alo_a[q,1], alo_a[q,2],     \
                              0 ,alo_o[p,0] - alo_o[q,0], alo_o[p,1] - alo_o[q,1], alo_o[p,2] - alo_o[q,2], \
                              alo_e[p,i], alo_e[q,j])
                    t3 +=alo_a[q,1] * (alo_a[q,1] - 1) *                                                    \
                         ovlp(alo_a[p,0], alo_a[p,1], alo_a[p,2], alo_a[q,0], alo_a[q,1]-2, alo_a[q,2],     \
                              0 ,alo_o[p,0] - alo_o[q,0], alo_o[p,1] - alo_o[q,1], alo_o[p,2] - alo_o[q,2], \
                              alo_e[p,i], alo_e[q,j])
                    t3 +=alo_a[q,2] * (alo_a[q,2] - 1) *                                                    \
                         ovlp(alo_a[p,0], alo_a[p,1], alo_a[p,2], alo_a[q,0], alo_a[q,1], alo_a[q,2]-2,     \
                              0 ,alo_o[p,0] - alo_o[q,0], alo_o[p,1] - alo_o[q,1], alo_o[p,2] - alo_o[q,2], \
                              alo_e[p,i], alo_e[q,j])

                    s += (t1 + t2 - 0.5*t3) * f

            kinetic[p,q] = s
            if p != q:
                kinetic[q,p] = kinetic[p,q]
#----------------------------------------End Kinetic----------------------------------------|

#---------------------------------------Begin Coulomb---------------------------------------|
    J = np.empty([nb,nb], dtype = np.double)
    cdef:
        double[:,:] coulomb = J
        double r[3]
        double cp

    for p in range(0, nb):
        for q in range(p, nb):

            t1 = 0.0
            for k in range(0, na):

                s = 0.0
                for i in range(0, alo[p]):
                    for j in range(0, alo[q]):
                        f = alo_n[p,i] * alo_n[q,j] * alo_c[p,i] * alo_c[q,j]
                        cp = alo_e[p,i] + alo_e[q,j]
                        for n in range(0, 3):
                            r[n] = ((alo_e[p,i] * alo_o[p,n]) + (alo_e[q,j] * alo_o[q,n]))/cp - alo_x[k,n]
                            
                        t2 = 0.0
                        for l in range(0, alo_a[p,0]+alo_a[q,0]+1):
                            for m in range(0, alo_a[p,1]+alo_a[q,1]+1):
                                for n in range(0, alo_a[p,2]+alo_a[q,2]+1):
                                    t2 += cye(alo_a[p,0], alo_a[q,0], l, alo_o[p,0]- alo_o[q,0], alo_e[p,i], alo_e[q,j]) * \
                                          cye(alo_a[p,1], alo_a[q,1], m, alo_o[p,1]- alo_o[q,1], alo_e[p,i], alo_e[q,j]) * \
                                          cye(alo_a[p,2], alo_a[q,2], n, alo_o[p,2]- alo_o[q,2], alo_e[p,i], alo_e[q,j]) * \
                                          clmb(l, m, n, 0, cp, r[0], r[1], r[2])

                        t2 = t2 * pi * 2.0 / cp
                        s += t2 * f
                t1 -= s * alo_z[k]
            coulomb[p,q] = t1
            if p != q:
                coulomb[q,p] = coulomb[p,q]

#----------------------------------------End Coulomb----------------------------------------|

#----------------------------------Begin electron repulsion---------------------------------|
    i = int(nb*(nb-1)/2 + nb)
    j = (i*(i+1)/2)
    I = np.empty([j], dtype=np.double)
    cdef:
        double[:] eri = I
        int idx
    for i in range(0, nb):
        for j in range(0, i+1):
            m = i * (i+1)/2 + j
            for k in range(0, nb):
                for l in range(0, k+1):
                    n = k*(k+1)/2 + l
                    if m >= n:
                        idx = int(m*(m+1)/2 + n)
                        I[idx] = tei(alo[i], alo[j], alo[k], alo[l], alo_a, alo_n, alo_c, alo_e, alo_o, i, j, k, l)
#|----------------------------------End electron repulsion----------------------------------|

    return S, K, J, I

#---------------------------------------Begin Dipole----------------------------------------|
cpdef aelloDipole(double[:,:] alo_n, double[:,:] alo_c, double[:,:] alo_e, short[:,:] alo_a, double[:,:] alo_o, \
                        short[:] alo, short[:] alo_z, double[:,:] alo_x, int na, int nb, object molAtom, double[:,:] density, str gauge): 

    D = np.empty([nb,nb], dtype = np.double)
    cdef:
        double[:,:] dipole = D
        double[3] gaugeOrigin = gaugeCenter(molAtom, gauge)
        int direction, p, q, i, j
        double s, f
        double[3] dipoleComponent

    for direction in range(1,4):
        #electronic component
        for p in range(0, nb):
            for q in range(p, -1, -1):

                s = 0.0
                for i in range(0, alo[p]):
                    for j in range(0, alo[q]):
                        f = alo_n[p,i] * alo_n[q,j] * alo_c[p,i] * alo_c[q,j]
                        s += mu([alo_a[p,0], alo_a[p,1], alo_a[p,2]], [alo_a[q,0], alo_a[q,1], alo_a[q,2]],   \
                                 alo_e[p,i], alo_e[q,j],                                                      \
                                  [alo_o[p,0], alo_o[p,1], alo_o[p,2]], [alo_o[q,0], alo_o[q,1], alo_o[q,2]], \
                                  gaugeOrigin, direction) * f

                dipole[p, q] = s
                if p != q:
                    dipole[q,p] = dipole[p,q]

        s =0.0
        for p in range(0, nb):
            for i in range(0, nb):
                s += -2.0 * density[p,i] * dipole[i,p]

        #nuclear component and charge center adjustment
        for i in range(0, na):
                s += alo_z[i] * (alo_x[i, direction-1] - gaugeOrigin[direction-1])

        dipoleComponent[direction-1] = s 

    return dipoleComponent
#|-----------------------------------------End Dipole---------------------------------------|

#----------------------------------------Begin Angular--------------------------------------|

cpdef aelloAngular(double[:,:] alo_n, double[:,:] alo_c, double[:,:] alo_e, short[:,:] alo_a, double[:,:] alo_o, \
                        short[:] alo, short[:] alo_z, double[:,:] alo_x, int na, int nb, object molAtom, str gauge): 

    A = np.empty([3,nb,nb], dtype = np.double)
    cdef:
        double[:,:,:] angular = A
        double[3] gaugeOrigin = gaugeCenter(molAtom, gauge)
        int direction, p, q, i, j
        double s, f

    for direction in range(0, 3):
        #electronic component
        for p in range(0, nb):
            for q in range(0, p+1):

                s = 0.0
                for i in range(0, alo[p]):
                    for j in range(0, alo[q]):
                        f = alo_n[p,i] * alo_n[q,j] * alo_c[p,i] * alo_c[q,j]
                        s += ang([alo_a[p,0], alo_a[p,1], alo_a[p,2]], [alo_a[q,0], alo_a[q,1], alo_a[q,2]],  \
                                 alo_e[p,i], alo_e[q,j],                                                      \
                                  [alo_o[p,0], alo_o[p,1], alo_o[p,2]], [alo_o[q,0], alo_o[q,1], alo_o[q,2]], \
                                  gaugeOrigin, direction+1) * f

                angular[direction, p, q] = s
                if p != q:
                    angular[direction,q,p] = -angular[direction,p,q]



    return A

#|----------------------------------------End Angular---------------------------------------|

#|---------------------------------------fock build-----------------------------------------|
cdef long iEri(long i, long j, long k, long l):
    #eri indexing
    cdef:
        long p = max(i*(i+1)//2 + j, j*(j+1)//2 + i)
        long q = max(k*(k+1)//2 + l, l*(l+1)//2 + k)

    return  long(max(p*(p+1)//2 + q, q*(q+1)//2 + p))


cpdef aelloFock(long n, double[:,:] H, double[:] eri, double[:,:] D):
    #fast Fock build
    fock = np.empty([n,n], dtype=np.double)
    g    = np.empty([n,n], dtype=np.double)
    cdef:
        long i, m, k, l
        double[:,:] F = fock
        double[:,:] G = g

    for i in range(0, n):
        for m in range(0, n):
            G[i,m] = 0.0
            for k in range(0, n):
                for l in range(0, n):
                    G[i,m] += D[k,l] * ( 2* eri[iEri(i,m,k,l)] - eri[iEri(i,k,m,l)])
            F[i,m] = H[i,m] + G[i,m]

    return fock, G

#|------------------------------------End Fock Build-------------------------------------|

#|----------------------------------Complex Fock Build-----------------------------------|
cpdef tdhfFock(long n, double complex[:,:] D, double complex[:,:] H, double complex[:] eri):
    #complex fock build for tdhf
    fock = np.empty([n,n], dtype=np.cdouble)
    g    = np.empty([n,n], dtype=np.cdouble)
    cdef:
        long i, m, k, l
        double complex[:,:] F = fock
        double complex[:,:] G = g

    for i in range(0, n):
        for m in range(0, n):
            G[i,m] = 0.0
            for k in range(0, n):
                for l in range(0, n):
                    G[i,m] = G[i,m] + D[k,l] * ( 2.0 * eri[iEri(i,m,k,l)] - eri[iEri(i,k,m,l)])
            F[i,m] = H[i,m] + G[i,m]

    return fock




