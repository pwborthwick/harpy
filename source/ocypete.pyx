#cython: boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False, cdivision=True
from libc.math cimport exp, pow, tgamma, sqrt, abs
from scipy.special.cython_special cimport hyp1f1 

import numpy as np
cimport numpy as np

cdef double pi = 3.141592653589793238462643383279

#---------------------------------------Shared Routines -----------------------------------|

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


cdef double efx(int ia,int ja,int type, double r, double ie, double je, int n = 0, double x = 0.0, int p = 0, int s = 0):
    if p == 1:
        return 2.0 * ie * cye(ia+1, ja, type, r, ie, je, n, x) - ia * cye(ia-1, ja, type, r, ie, je, n, x)
    elif s == 1:
        return 2.0 * je * cye(ia, ja+1, type, r, ie, je, n, x) - ja * cye(ia, ja-1, type, r, ie, je, n, x)

cdef double ovlpfx(int ia0, int ia1, int ia2, int ja0, int ja1, int ja2, double ie, double je, \
     double ir0, double ir1, double ir2, double jr0, double jr1, double jr2, int[3] n, double[3] origin, int x, int center):

    cdef:
        int pa = 0
        int pb
        double t = 0.0
        double r0, r1, r2

    if center == 0:
        pa = 1
    pb = (pa+1) % 2

    r0 = ir0 - jr0
    r1 = ir1 - jr1
    r2 = ir2 - jr2 

    if x == 0:
        t =  efx(ia0, ja0 , 0, r0, ie, je, n[0], ir0 - origin[0],pa ,pb )
        t *= cye(ia1, ja1 , 0, r1, ie, je, n[1], ir1 - origin[1])
        t *= cye(ia2, ja2 , 0, r2, ie, je, n[2], ir2 - origin[2])
    elif x == 1:
        t =  cye(ia0, ja0 , 0, r0, ie, je, n[0], ir0 - origin[0])
        t *= efx(ia1, ja1 , 0, r1, ie, je, n[1], ir1 - origin[1],pa ,pb )
        t *= cye(ia2, ja2 , 0, r2, ie, je, n[2], ir2 - origin[2])
    elif x == 2:
        t =  cye(ia0, ja0 , 0, r0, ie, je, n[0], ir0 - origin[0])
        t *= cye(ia1, ja1 , 0, r1, ie, je, n[1], ir1 - origin[1])
        t *= efx(ia2, ja2 , 0, r2, ie, je, n[2], ir2 - origin[2],pa ,pb )

    return t * pow(pi/(ie+je), 1.5)

cdef double kntcfx(int[3] ia, int[3] ja, double ie, double je, double[3] ir, double[3] jr, \
     int[3] n, double[3] origin, int x, int center):

    #cases for center 'a' and center 'b'
    cdef:
        int pa = 0
        int pb, i
        double[3] t
        double[3] mu, nu, vu

    if center == 0:
        pa = 1
    pb = (pa+1) % 2

    for i in range(0, 3):
        mu[i] = (2*ja[i] + 1) * je
        nu[i] = -2*pow(je,2)
        vu[i] = -0.5 * ja[i]* (ja[i]-1)
        t[i] = 0.0

    for i in range(0, 3):

        if i == x:
            t[x] = mu[x] * efx(ia[x], ja[x] , 0, ir[x] - jr[x], ie, je, n[x], ir[x] - origin[x],pa ,pb ) +     \
                   nu[x] * efx(ia[x], ja[x] + 2 , 0, ir[x] - jr[x], ie, je, n[x], ir[x] - origin[x],pa ,pb ) + \
                   vu[x] * efx(ia[x], ja[x] - 2, 0, ir[x] - jr[x], ie, je, n[x], ir[x] - origin[x],pa ,pb )

        else:
            t[i] = mu[i] * cye(ia[i], ja[i] , 0, ir[i] - jr[i], ie, je, n[i], ir[i] - origin[i]) +    \
                   nu[i] * cye(ia[i], ja[i] + 2, 0, ir[i] - jr[i], ie, je, n[i], ir[i] - origin[i]) + \
                   vu[i] * cye(ia[i], ja[i] - 2, 0, ir[i] - jr[i], ie, je, n[i], ir[i] - origin[i])

    for i in range(0, 3):

        if i == x:
            t[(x+1) % 3] *= efx(ia[x], ja[x] , 0, ir[x] - jr[x], ie, je, n[x], ir[x] - origin[x],pa ,pb ) 
            t[(x+2) % 3] *= efx(ia[x], ja[x] , 0, ir[x] - jr[x], ie, je, n[x], ir[x] - origin[x],pa ,pb )
        else:
            t[(i+1) % 3] *= cye(ia[i], ja[i], 0, ir[i] - jr[i], ie, je, n[i], ir[i] - origin[i])
            t[(i+2) % 3] *= cye(ia[i], ja[i], 0, ir[i] - jr[i], ie, je, n[i], ir[i] - origin[i])

    return (t[0] + t[1] + t[2]) * pow(pi/(ie+je), 1.5)

cdef double boys(double m,double T):
    return hyp1f1(m+0.5,m+1.5,-T)/(2.0*m+1.0) 

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

cdef double clmbsfx(int[3] ia, int[3] ja, double ie, double je, double[3] ir, double[3] jr, \
     double[:] nucleus, int[3] n, double[3] origin, int x, int center):
    #generalised coulomb derivatives dV(ab^(0,0,0))/dx terms

    cdef:
        double p = ie + je
        double[3] q, r
        int i, mu, nu, vu, pa, pb
        int tau[3]
        double sum, val
        int[3] xi

    for i in range(0, 3):
        q[i] = (ie*ir[i] + je*jr[i])/p
        tau[i] = ia[i] + ja[i] + n[i] + 1
        r[i] = q[i] - nucleus[i]

    tau[x] += 1

    pa = 0
    if center == 0:
        pa = 1
    pb = (pa+1) % 2

    sum = 0.0
    val = 1.0

    for mu in range(0, tau[0]):
        for nu in range(0, tau[1]):
            for vu in range(0, tau[2]):
                val = 1.0
                xi = [mu,nu,vu]
                for i in range(0, 3):
                    if i == x:
                        val *= efx(ia[x], ja[x], xi[x], ir[x]-jr[x], ie, je, n[x], ir[x]-nucleus[x], pa, pb)
                    else:
                        val *=   cye(ia[i], ja[i], xi[i], ir[i]-jr[i], ie, je, n[i], ir[i]-nucleus[i])

                sum += val * clmb(mu,nu,vu, 0, p, r[0], r[1], r[2] )

    return sum * 2 * pi/p

cdef double clmbhfx(int[3] ia, int[3] ja, double ie, double je, double[3] ir, double[3] jr, \
     double[:] nucleus, int[3] n, double[3] origin, int x):
    #generalised coulomb derivatives dV(ab^(0,0,0))/dx terms

    cdef:
        double p = ie + je
        double[3] q, r
        int i, mu, nu, vu, pa, pb
        int[3] tau
        double sum, val
        int[3] xi

    for i in range(0, 3):
        q[i] = (ie*ir[i] + je*jr[i])/p
        tau[i] = ia[i] + ja[i] + n[i] + 1
        r[i] = q[i] - nucleus[i]

    sum = 0.0
    val = 1.0

    for mu in range(0, tau[0]):
        for nu in range(0, tau[1]):
            for vu in range(0, tau[2]):
                val = 1.0
                xi = [mu,nu,vu]
                for i in range(0, 3):
                    val *=   cye(ia[i], ja[i], xi[i], ir[i]-jr[i], ie, je, n[i], ir[i]-nucleus[i])

                xi[x] += 1

                sum -= val * clmb(xi[0],xi[1],xi[2], 0, p, r[0], r[1], r[2] )

    return sum * 2 * pi/p

cdef double teifx(short[:] ia, short[:] ja, short[:] ka, short[:] la, double ie, double je, double ke, double le, \
                  double[:] ir, double[:] jr, double[:] kr, double[:] lr, int[3] ra, int[3] rb, double[3] origin, int x, int center):

    cdef:
        double p = ie + je
        double q = ke + le
        double rho = p*q/(p + q)
        double[3] P, Q, r
        int i, pa, pb, mu, nu, vu, psi, phi, chi
        int[3] xia, xib, tau, sigma
        double val = 0.0
        double term

    for i in range(0, 3):
        P[i] = (ie*ir[i] + je*jr[i])/p
        Q[i] = (ke*kr[i] + le*lr[i])/q
        r[i] = P[i] - Q[i]

        tau[i] = ia[i] + ja[i] + 1 + ra[i]
        sigma[i] = ka[i] + la[i] + 1 + rb[i]

    if (center == 0) or (center == 1):
        tau[x] += 1
    else:
        sigma[x] += 1

    pa = 0
    if (center == 0) or (center == 2):
        pa = 1
    pb = (pa+1) % 2

    for mu in range(tau[0]):
        for nu in range(tau[1]):
            for vu in range(tau[2]):
                for psi in range(sigma[0]):
                    for phi in range(sigma[1]):
                        for chi in range(sigma[2]):
                            xia = [mu, nu, vu]
                            xib = [psi, phi, chi] 
                            term = 1.0
                            for i in range(0, 3):
                                if (i == x):
                                    if (center == 0 or center == 1):
                                        term *= efx(ia[x],ja[x],xia[x],ir[x]-jr[x],ie,je,ra[x],ir[x] - origin[x], pa, pb)
                                        term *= cye(ka[x],la[x],xib[x],kr[x]-lr[x],ke,le,rb[x],kr[x] - origin[x]) 
                                    elif (center == 2 or center == 3):
                                        term *= cye(ia[x],ja[x],xia[x],ir[x]-jr[x],ie,je,ra[x],ir[x] - origin[x])
                                        term *= efx(ka[x],la[x],xib[x],kr[x]-lr[x],ke,le,rb[x],kr[x] - origin[x], pa, pb) 

                                else:
                                    term *= cye(ia[i],ja[i],xia[i],ir[i]-jr[i],ie,je,ra[i],ir[i] - origin[i])
                                    term *= cye(ka[i],la[i],xib[i],kr[i]-lr[i],ke,le,rb[i],kr[i] - origin[i]) 

                            term *= pow(-1, (psi+phi+chi)) * clmb(mu+psi,nu+phi,vu+chi,0, rho,r[0], r[1], r[2]) 
                            val += term

    return val*2*pow(pi,2.5)/(p*q*sqrt(p+q)) 


cdef double erifx(short[:] ng, int p, int q, int r, int s, double[:,:] im, double[:,:] ic, double[:,:] ie, short[:,:] ia, double[:,:] io, \
                  int x, int center):

    cdef:
        double sum = 0.0
        int i, j, k, l

    for i in range(0, ng[p]):
        for j in range(0, ng[q]):
            for k in range(0, ng[r]):
                for l in range(0, ng[s]):
                    sum += im[p,i]*im[q,j]*im[r,k]*im[s,l] * ic[p,i]*ic[q,j]*ic[r,k]*ic[s,l] *   \
                           teifx(ia[p], ia[q], ia[r], ia[s], ie[p,i], ie[q,j], ie[r,k], ie[s,l], \
                                 io[p], io[q], io[r], io[s], [0,0,0], [0,0,0], [0,0,0], x, center)

    return sum

#-----------------------------------End Shared Routines -----------------------------------|


cpdef ocypete(object molAtom,object molBasis, double[:,:] density, double[:,:] fock):
 
    cdef:
        int na = len(molAtom)
        int nb = len(molBasis)
        int ng = len(molBasis[0].co)
        int i, j, k, l, m, n, p, q, r, s

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
    mp = np.empty([nb],    dtype = np.short)

    cdef: 
        double[:,:] alo_e = me
        double[:,:] alo_c = mc
        double[:,:] alo_n = mn
        short[:,:]  alo_a = ma
        double[:,:] alo_o = mo
        short[:]    alo   = ml
        short[:]    ala   = mp

    for p in range(0, nb):
        alo[p] = len(molBasis[p].co)
        ala[p] = molBasis[p].atom
        for q in range(0, len(molBasis[p].co)):
            alo_e[p,q] = molBasis[p].ex[q]
            alo_c[p,q] = molBasis[p].co[q]
            alo_n[p,q] = molBasis[p].normal[q]
        for q in range(0, 3):
            alo_a[p,q] = molBasis[p].momentum[q]
            alo_o[p,q] = molBasis[p].center[q]

    #matrix definitions
    cdef:
        double ss, sk, sj, sh, si, sf, f, ra, rb, force
        int cart
    Sx = np.zeros([nb,nb], dtype = np.double)
    Ix = np.zeros([nb,nb,nb,nb], dtype = np.double)
    Hx = np.zeros([nb,nb], dtype = np.double)
    Fx = np.zeros([nb,nb], dtype = np.double)
    Wx = np.zeros([nb,nb], dtype = np.double)
    Ex = np.zeros([nb,nb], dtype = np.double)

    Vx = np.zeros([na,3], dtype = np.double)
    cdef:
        double[:,:]  overlapfx = Sx
        double[:,:,:,:]  teifx = Ix
        double[:,:]      oeifx = Hx
        double[:,:]      fockfx= Fx
        double[:,:]      weightedEnergy = Wx
        double[:,:]      energy = Ex
        double[:,:]      vires = Vx

    #----------------------------------------Begin derivatives ------------------------------------|

    #---------------------------------------Begin one electron-------------------------------------|
    #don't need overlapfx for fock but included anyway for completeness
    for atom in range(0, na):
        for cart in range(0, 3):

            for p in range(0, nb):
                for q in range(0, p+1):

                    ss = 0.0
                    sk = 0.0
                    sj = 0.0
                    sh = 0.0
                    for i in range(0, alo[p]):
                        for j in range(0, alo[q]):

                            f = alo_n[p,i] * alo_n[q,j] * alo_c[p,i] * alo_c[q,j]
                            if ala[p] == atom:
                                ss += ovlpfx(alo_a[p,0], alo_a[p,1], alo_a[p,2], alo_a[q,0], alo_a[q,1], alo_a[q,2],                         \
                                             alo_e[p,i], alo_e[q,j], alo_o[p,0], alo_o[p,1], alo_o[p,2], alo_o[q,0], alo_o[q,1], alo_o[q,2], \
                                             [0,0,0], [0,0,0], cart, 0   ) * f

                                sk += kntcfx([alo_a[p,0], alo_a[p,1], alo_a[p,2]], [alo_a[q,0], alo_a[q,1], alo_a[q,2]],                          \
                                              alo_e[p,i], alo_e[q,j], [alo_o[p,0], alo_o[p,1], alo_o[p,2]], [alo_o[q,0], alo_o[q,1], alo_o[q,2]], \
                                              [0,0,0], [0,0,0], cart, 0   ) * f

                            if ala[q] == atom:
                                ss += ovlpfx(alo_a[p,0], alo_a[p,1], alo_a[p,2], alo_a[q,0], alo_a[q,1], alo_a[q,2],                         \
                                             alo_e[p,i], alo_e[q,j], alo_o[p,0], alo_o[p,1], alo_o[p,2], alo_o[q,0], alo_o[q,1], alo_o[q,2], \
                                             [0,0,0], [0,0,0], cart, 1  ) * f

                                sk += kntcfx([alo_a[p,0], alo_a[p,1], alo_a[p,2]], [alo_a[q,0], alo_a[q,1], alo_a[q,2]],                          \
                                              alo_e[p,i], alo_e[q,j], [alo_o[p,0], alo_o[p,1], alo_o[p,2]], [alo_o[q,0], alo_o[q,1], alo_o[q,2]], \
                                              [0,0,0], [0,0,0], cart, 1  ) * f

                            sh -= clmbhfx([alo_a[p,0], alo_a[p,1], alo_a[p,2]], [alo_a[q,0], alo_a[q,1], alo_a[q,2]],                          \
                                           alo_e[p,i], alo_e[q,j], [alo_o[p,0], alo_o[p,1], alo_o[p,2]], [alo_o[q,0], alo_o[q,1], alo_o[q,2]], \
                                           alo_x[atom], [0,0,0], [0,0,0], cart  ) * f * alo_z[atom]

                    for r in range(0, na):
                        for i in range(0, alo[p]):
                            for j in range(0, alo[q]):
                                f = alo_n[p,i] * alo_n[q,j] * alo_c[p,i] * alo_c[q,j]
                                if ala[p] == atom:
                                    sj -= clmbsfx([alo_a[p,0], alo_a[p,1], alo_a[p,2]], [alo_a[q,0], alo_a[q,1], alo_a[q,2]],                          \
                                                   alo_e[p,i], alo_e[q,j], [alo_o[p,0], alo_o[p,1], alo_o[p,2]], [alo_o[q,0], alo_o[q,1], alo_o[q,2]], \
                                                   alo_x[r], [0,0,0], [0,0,0], cart, 0  ) * f * alo_z[r]

                                if ala[q] == atom:
                                    sj -= clmbsfx([alo_a[p,0], alo_a[p,1], alo_a[p,2]], [alo_a[q,0], alo_a[q,1], alo_a[q,2]],                          \
                                                   alo_e[p,i], alo_e[q,j], [alo_o[p,0], alo_o[p,1], alo_o[p,2]], [alo_o[q,0], alo_o[q,1], alo_o[q,2]], \
                                                   alo_x[r], [0,0,0], [0,0,0], cart, 1  ) * f * alo_z[r]

                    
                    oeifx[p,q] = oeifx[q,p] = sk + sj + sh
                    overlapfx[p,q] = overlapfx[q,p] = ss

    #----------------------------------------End one electron--------------------------------------|

    #---------------------------------------Begin two electron-------------------------------------|

            for p in range(0, nb):
                for q in range(0, p+1):

                    si = 0.0

                    i = p*(p+1)//2 + q
                    for r in range(0, nb):
                        for s in range(0, r+1):

                            j = r*(r+1)//2 + s

                            if i >= j:
                                si = 0.0
                                if ala[p] == atom:
                                    si += erifx(alo , p ,q ,r ,s ,alo_n ,alo_c ,alo_e ,alo_a , alo_o, cart, 0)                                  
                                if ala[q] == atom:
                                    si += erifx(alo , p ,q ,r ,s ,alo_n ,alo_c ,alo_e ,alo_a , alo_o, cart, 1)                                  
                                if ala[r] == atom:
                                    si += erifx(alo , p ,q ,r ,s ,alo_n ,alo_c ,alo_e ,alo_a , alo_o, cart, 2)                                  
                                if ala[s] == atom:
                                    si += erifx(alo , p ,q ,r ,s ,alo_n ,alo_c ,alo_e ,alo_a , alo_o, cart, 3)                                  

                            teifx[p,q,r,s] = teifx[p,q,s,r] = teifx[q,p,s,r] = teifx[q,p,r,s] = \
                            teifx[r,s,p,q] = teifx[r,s,q,p] = teifx[s,r,q,p] = teifx[s,r,p,q] = si

    #----------------------------------------End two electron--------------------------------------|

    #-------------------------------------------build Fock-----------------------------------------|

            for p in range(0, nb):
                for q in range(0, nb):
                    sf = 0.0
                    for r in range(0, nb):
                        for s in range(0, nb):
                            sf += (2.0 * teifx[p,q,r,s] - teifx[p,s,q,r]) * density[s,r]

                    fockfx[p,q] = oeifx[p,q] + sf

    #|----------------------------------------build energy-----------------------------------------|

            force = 0.0
            for p in range(0, nb):
                for q in range(0, nb):
                    force -= density[p,q] * (fockfx[q,p] + oeifx[q,p])

    #|-----------------------------------density weighted energy-----------------------------------|
            for p in range(0, nb):
                for q in range(0, nb):
                    energy[p,q] = 0.0
                    for r in range(0, nb):
                        energy[p,q] += fock[p,r] * density[r,q]
            for p in range(0, nb):
                for q in range(0, nb):
                    weightedEnergy[p,q] = 0.0
                    for r in range(0, nb):
                        weightedEnergy[p,q] += density[p,r] * energy[r,q]

    #|-------------------------------------overlap contribution------------------------------------|

            for p in range(0, nb):
                for q in range(0, nb):
                    force += 2.0 * overlapfx[p,q] * weightedEnergy[q,p]

    #|---------------------------------------nuclear repulsion-------------------------------------|

            for i in range(0, na):
                ra = sqrt((alo_x[atom,0] - alo_x[i,0])*(alo_x[atom,0] - alo_x[i,0]) + (alo_x[atom,1] - alo_x[i,1])*(alo_x[atom,1] - alo_x[i,1]) + \
                         (alo_x[atom,2] - alo_x[i,2])*(alo_x[atom,2] - alo_x[i,2]))
                rb = alo_x[atom,cart] - alo_x[i,cart]
                if ra != 0:
                    force += rb * alo_z[i] * alo_z[atom]/(ra*ra*ra)
    #|------------------------------------------final forces---------------------------------------|
            if abs(force) > 1e-12:
                vires[atom,cart] = force

    return Vx
