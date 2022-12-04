from __future__ import division
from numpy import zeros, dot, all
from numpy.linalg import norm
from math import exp, pow, pi, sqrt
from scipy.special import hyp1f1
from atom import gaugeCenter

def e(ia, ja, type, r, ie, je, n = 0, x = 0.0):
    #recursive definition of Hermite Gaussian 
    # i,j - Gaussian 'i' 'j' ,  a - angular momentum  , e - exponent
    #type number of Hermite node

    p = ie + je
    q = ie * je / p
    #check bound of type
    if n == 0:
        if (type < 0) or (type > (ia + ja)):
            return 0.0
        elif (ia + ja + type) == 0:
            return exp(-q*r*r)
        elif ja == 0:
            return (1/(2 * p)) * e(ia-1,ja,type-1,r,ie,je) - (q*r/ie) * e(ia-1,ja,type,r,ie,je) + \
                         (type+1) * e(ia-1,ja,type+1,r,ie,je)
        else:
            return (1/(2 * p)) * e(ia,ja-1,type-1,r,ie,je) + (q*r/je) * e(ia,ja-1,type,r,ie,je) + \
                         (type+1) * e(ia,ja-1,type+1,r,ie,je)
    else:
        return e(ia+1,ja,type,r,ie,je,n-1,x) + x * e(ia,ja,type,r,ie,je,n-1,x)

def overlap(ia, ja, ie, je, ir, jr):
    #overlap between two Gaussians 
    # i,j - Gaussian 'i' 'j' ,  a - angular momentum  , e - exponent, r - seperation
    #0 type number of Hermite node
    s = zeros(3)
    for dim in range(0,3):
        s[dim] = e(ia[dim], ja[dim], 0, ir[dim]-jr[dim], ie, je)

    return s[0]*s[1]*s[2] * pow(pi/(ie+je),1.5)

def s(iBasis, jBasis):
    #overlap between contracted Gaussians (S)

    sum = 0.0
    for i in range(0, len(iBasis.co)):
        for j in range(0, len(jBasis.co)):
            sum += iBasis.normal[i]*jBasis.normal[j]*iBasis.co[i]*jBasis.co[j]*overlap(iBasis.momentum, jBasis.momentum, \
                   iBasis.ex[i], jBasis.ex[j], iBasis.center, jBasis.center)

    return sum

def buildOverlap(bases):
    #compute the overlap matrix
    n = len(bases)
    overlap = zeros((n,n))

    for iBasis in range(0, n):
        for jBasis in range(0, iBasis+1):
            overlap[iBasis,jBasis] = s(bases[iBasis], bases[jBasis])
            #symmetrize
            if iBasis != jBasis:
                overlap[jBasis, iBasis] = overlap[iBasis, jBasis]

    return overlap

def kinetic(ia, ja, ie, je, ir, jr):
    #kinetic between two Gaussians
    # i,j - Gaussian 'i' 'j' ,  a - angular momentum  , e - exponent, r - seperation

    t = zeros(3)

    t[0] = je * (2*(ja[0] + ja[1] + ja[2]) + 3) * overlap(ia,ja,ie,je,ir,jr)
    t[1] = -2 * je * je * (overlap(ia, [ja[0]+2, ja[1], ja[2]], ie, je, ir, jr) + \
                           overlap(ia, [ja[0], ja[1]+2, ja[2]], ie, je, ir, jr) + \
                           overlap(ia, [ja[0], ja[1], ja[2]+2], ie, je, ir, jr)) 
    t[2] =  ja[0] * (ja[0] - 1) * overlap(ia, [ja[0]-2, ja[1], ja[2]], ie, je, ir, jr)
    t[2] += ja[1] * (ja[1] - 1) * overlap(ia, [ja[0], ja[1]-2, ja[2]], ie, je, ir, jr)
    t[2] += ja[2] * (ja[2] - 1) * overlap(ia, [ja[0], ja[1], ja[2]-2], ie, je, ir, jr)

    return t[0] + t[1] - 0.5*t[2]

def k(iBasis, jBasis):
    #kinetic between contracted Gaussians (K)

    sum = 0.0
    for i in range(0, len(iBasis.co)):
        for j in range(0, len(jBasis.co)):
            sum += iBasis.normal[i]*jBasis.normal[j]*iBasis.co[i]*jBasis.co[j]*kinetic(iBasis.momentum, jBasis.momentum, \
                   iBasis.ex[i], jBasis.ex[j], iBasis.center, jBasis.center)

    return sum


def buildKinetic(bases):
    #compute the kinetic (eN) matrix
    n = len(bases)
    kinetic = zeros((n,n))

    for iBasis in range(0, n):
        for jBasis in range(0, iBasis+1):
            kinetic[iBasis,jBasis] = k(bases[iBasis], bases[jBasis])
            #symmetrize
            if iBasis != jBasis:
                kinetic[jBasis, iBasis] = kinetic[iBasis, jBasis]

    return kinetic

def j(v, n, p, r, rnorm):
    #Coulomb auxillary integrals
    #v[] - order of Coulomb Hermite derivatives
    #n Boys function order ,   p  sum of exponents
    #r distance between Gaussians composite center and nuclear center,
    #rnorm - norm of r

    t = p * rnorm * rnorm
    sum =0.0
    if (v[0]+v[1]+v[2])  == 0:
        sum += pow(-2*p, n) * boys(n, t)
    elif (v[0]+v[1]) == 0:
        if v[2] > 1:
            sum +=(v[2]-1) * j([v[0],v[1],v[2]-2],n+1,p,r,rnorm)
        sum += r[2] * j([v[0],v[1],v[2]-1],n+1,p,r,rnorm)
    elif v[0] == 0:
        if v[1] > 1:
            sum +=(v[1]-1) * j([v[0],v[1]-2,v[2]],n+1,p,r,rnorm)
        sum += r[1] * j([v[0],v[1]-1,v[2]],n+1,p,r,rnorm)
    else:
        if v[0] > 1:
            sum +=(v[0]-1) * j([v[0]-2,v[1],v[2]],n+1,p,r,rnorm)
        sum += r[0] * j([v[0]-1,v[1],v[2]],n+1,p,r,rnorm)

    return sum
    


def coulomb(ia, ja, ie, je, ir, jr, kr):
    #Coulomb energy between two Gaussians
    # i,j - Gaussian 'i' 'j' ,  a - angular momentum  , e - exponent, r - seperation

    #p is composite center
    p = ie + je
    q = (ie*ir + je*jr)/p

    r = q - kr

    sum = 0.0
    for i in range(0, ia[0]+ja[0]+1):
        for k in range(0, ia[1]+ja[1]+1):
            for l in range(0, ia[2]+ja[2]+1):
                sum += e(ia[0], ja[0], i, ir[0]-jr[0], ie, je) * \
                       e(ia[1], ja[1], k, ir[1]-jr[1], ie, je) * \
                       e(ia[2], ja[2], l, ir[2]-jr[2], ie, je) * \
                       j([i,k,l], 0, p, r, norm(r))

    return sum * pi * 2 / p


def v(iBasis, jBasis, r):
    #contracted Gaussian
    sum = 0.0

    for i in range(0, len(iBasis.co)):
        for j in range(0, len(jBasis.co)):
            sum += iBasis.normal[i]*jBasis.normal[j]*iBasis.co[i]*jBasis.co[j]*coulomb(iBasis.momentum, jBasis.momentum, \
                   iBasis.ex[i], jBasis.ex[j], iBasis.center,jBasis.center, r)

    return sum

def buildCoulomb(atoms, bases):
    #compute the Coulomb (eN) matrix
    n = len(bases)
    coulomb = zeros((n,n))

    for iBasis in range(0,n):
        for jBasis in range(0, iBasis+1):

            sum = 0.0
            for i in range(0, len(atoms)):
                r = atoms[i].center
                sum -= atoms[i].number * v(bases[iBasis], bases[jBasis], r)
            coulomb[iBasis, jBasis] = sum

            if iBasis != jBasis:
                coulomb[jBasis, iBasis] = coulomb[iBasis, jBasis]

    return coulomb

def er(ie, je, ke, le, ia, ja, ka, la, ir, jr, kr, lr):
    #two electron repulsion integral Gaussians

    p = ie + je
    q = ke + le

    alpha = p*q/(p+q)
    beta = (ie*ir + je*jr)/p - \
           (ke*kr + le*lr)/q
    betaNorm = norm(beta)

    sum = 0.0

    for a in range(0, ia[0] + ja[0] + 1):
        for b in range(0, ia[1] + ja[1] + 1):
            for c in range(0, ia[2] + ja[2] + 1):

                for t in range(0, ka[0] + la[0] + 1):
                    for u in range(0, ka[1] + la[1] + 1):
                        for v in range(0, ka[2] + la[2] + 1):
                            factor =  e(ia[0],ja[0], a, ir[0]-jr[0], ie, je) * \
                                      e(ia[1],ja[1], b, ir[1]-jr[1], ie, je) * \
                                      e(ia[2],ja[2], c, ir[2]-jr[2], ie, je)  
                            factor *= e(ka[0],la[0], t, kr[0]-lr[0], ke, le) * \
                                      e(ka[1],la[1], u, kr[1]-lr[1], ke, le) * \
                                      e(ka[2],la[2], v, kr[2]-lr[2], ke, le) 
                            factor *= pow(-1, t+u+v) * j([a+t,b+u,c+v], 0, alpha, beta, betaNorm)

                            sum += factor

    sum *=  2 * pow(pi, 2.5) / ((p*q) * sqrt(p+q))

    return sum

def eri(iBasis, jBasis, kBasis, lBasis):
    #two electron repulsion integrals
    sum = 0.0

    for i in range(0, len(iBasis.co)):
        for j in range(0, len(jBasis.co)):
            for k in range(0, len(kBasis.co)):
                for l in range(0, len(lBasis.co)):
                    factor =  iBasis.normal[i]*jBasis.normal[j]*kBasis.normal[k]*lBasis.normal[l] *  \
                              iBasis.co[i]*jBasis.co[j]*kBasis.co[k]*lBasis.co[l]
                    factor *= er(iBasis.ex[i], jBasis.ex[j], kBasis.ex[k], lBasis.ex[l],             \
                                 iBasis.momentum, jBasis.momentum, kBasis.momentum, lBasis.momentum, \
                                 iBasis.center, jBasis.center, kBasis.center, lBasis.center)

                    sum += factor
    return sum

def buildEri(bases):
    #build all the two electron integrals into linear array
    #using symmetry only store upper triangle of ij,kl matrix

    #get dimension of array
    n = len(bases) * (len(bases) - 1)/2 + len(bases)
    n = n*(n+1)/2

    linearEri = zeros(int(n))
    for i in range(0, len(bases)):
        for j in range(0, i+1):
            ij = i * (i + 1)/2 + j
            for k in range(0, len(bases)):
                for l in range(0, k+1):
                    kl = k * (k + 1)/2 + l
                    if ij >= kl:
                        ijkl = ij * (ij + 1)/2 + kl
                        linearEri[int(ijkl)] = eri(bases[i], bases[j], bases[k], bases[l])

    return linearEri

def iEri(i,j,k,l):
    #index into the four-index eri integrals
    p = max(i*(i+1)/2 + j, j*(j+1)/2 + i)
    q = max(k*(k+1)/2 + l, l*(l+1)/2 + k)

    return  int(max(p*(p+1)/2 + q, q*(q+1)/2 + p))


def buildHamiltonian(type, S, K, J):
    #build the intial Hamiltonian guess either
    #core H = K + J
    if type == 'core':
        return (K + J), (K + J)

    #generalised Wolfberg-Helmholtz
    if type == 'gwh':
        n = S.shape[0]
        H = zeros((n,n))
        for i in range(0, n):
            for j in range( i, n):
                H[i,j] = 1.75 * S[i,j] * ((K+J)[i,i] + (K+J)[j,j])/2
                if i != j:
                    H[j,i] = H[i,j]

        return (K + J), H

def buildDensity(n, occupiedOrbitals, C):
    density = zeros(( n, n))
    #initial density matrix
    for i in range(0, n):
        row = zeros(n)
        for j in range(0, n):
            for l in range(0, occupiedOrbitals):
                row[j] += C[i,l] * C[j,l]
        density[i,:] = row[:]

    return density

def buildFock(H, eri, D, engine = 'aello'):
    #build the fock matrix
    n = H.shape[0]
    fock = zeros((n,n))
    G = zeros((n,n))

    if (all(D==0)):
        #initially set Fock to core Hamiltonian
        fock = H
    else:
        #subsequently Fock is core Hamiltonian + G ->[D.(2.<ij|kl> - <ik|jl>)]
        if engine == 'native':
            for i in range(0, n):
                for m in range(0, n):
                    G[i,m] = 0.0
                    for k in range(0, n):
                        for l in range(0, n):
                            G[i,m] += D[k,l] * ( 2* eri[iEri(i,m,k,l)] - eri[iEri(i,k,m,l)])
                    fock[i,m] = H[i,m] + G[i,m]
        elif engine == 'aello':
            from aello import aelloFock
            fock, G = aelloFock(n, H, eri, D)

    return fock, G 

def boys(n,T):
    return hyp1f1(n+0.5 ,n+1.5, -T)/(2.0*n + 1.0) 

def mu(iBasis, jBasis, kr, direction):
    #collects dipole values

    sum = 0.0
    for i in range(0, len(iBasis.co)):
        for j in range(0, len(jBasis.co)):
            sum += iBasis.normal[i] * jBasis.normal[j] * iBasis.co[i] * jBasis.co[j] * \
                   dipole(iBasis.momentum, jBasis.momentum, iBasis.ex[i], jBasis.ex[j], iBasis.center,  jBasis.center, kr, direction)

    return sum

def dipole(ia, ja, ie, je, ir, jr, kr, direction):
    # dipole moment
    p = ie + je
    q = ((ie*ir + je*jr)/p) - kr
    ijr = ir - jr

    if direction == 'x':
        u = e(ia[0], ja[0], 1, ijr[0], ie, je) + q[0]* e(ia[0], ja[0], 0, ijr[0], ie, je) 
        v = e(ia[1], ja[1], 0, ijr[1], ie, je) 
        t = e(ia[2], ja[2], 0, ijr[2], ie, je) 
        return u * v * t * pow(pi/p, 1.5)
    if direction == 'y':
        u = e(ia[0], ja[0], 0, ijr[0], ie, je)  
        v = e(ia[1], ja[1], 1, ijr[1], ie, je) + q[1]* e(ia[1], ja[1], 0, ijr[1], ie, je)
        t = e(ia[2], ja[2], 0, ijr[2], ie, je) 
        return u * v * t * pow(pi/p, 1.5)
    if direction == 'z':
        u = e(ia[0], ja[0], 0, ijr[0], ie, je)  
        v = e(ia[1], ja[1], 0, ijr[1], ie, je) 
        t = e(ia[2], ja[2], 1, ijr[2], ie, je) + q[2]* e(ia[2], ja[2], 0, ijr[2], ie, je)
        return u * v * t * pow(pi/p, 1.5)

def buildEriMO( eigenVectors, ERI):
    #transform eri from AO basis to MO basis
    n = eigenVectors.shape[0]
    eriMatrix_a = zeros((n,n))
    eriMatrix_b = zeros((n,n))
    eriMatrix_t = zeros((int(n*(n+1)/2), int(n*(n+1)/2)))
    eriMO = zeros(len(ERI))

    ij = 0
    for i in range(0, n):
        for j in range(0 , i+1):
            kl = 0
            for k in range(0, n):
                for l in range(0, k+1):
                    eriMatrix_a[l,k] = ERI[iEri(i,j,k,l)]
                    eriMatrix_a[k,l] = eriMatrix_a[l,k]
                    kl += 1

            #transform eriMatrix_a to MO basis
            eriMatrix_b = dot(eigenVectors.T, dot(eriMatrix_a, eigenVectors))

            kl = 0
            for k in range(0, n):
                for l in range(0, k+1):
                    eriMatrix_t[kl,ij] = eriMatrix_b[k,l]
                    kl += 1

            ij += 1

    kl = 0
    for k in range(0, n):
        for l in range(0, k+1):
            #reset eriMatrix_a and _b
            eriMatrix_a = zeros((n,n))
            eriMatrix_b = zeros((n,n))

            ij = 0
            for i in range(0, n):
                for j in range(0, i+1):
                    eriMatrix_a[j,i] = eriMatrix_t[kl,ij]
                    eriMatrix_a[i,j] = eriMatrix_a[j,i]
                    ij += 1

            #transform eriMatrix_a to MO basis
            eriMatrix_b = dot(eigenVectors.T, dot(eriMatrix_a, eigenVectors))

            for i in range(0, n):
                for j in range(0, i+1):
                    eriMO[iEri(k,l,i,j)] = eriMatrix_b[i,j]

            kl += 1

    return eriMO

def buildFockMOspin(spinOrbitals, eigenVectors, fock):
    #transform Fock -> MO -> spin basis

    fockspin = zeros((spinOrbitals, spinOrbitals))
    eigenspin = zeros((spinOrbitals, spinOrbitals))

    for p in range(0, spinOrbitals):
        for q in range(0, spinOrbitals):
            fockspin[p,q]  = fock[int(p/2), int(q/2)] * ((p % 2) == (q % 2))
            eigenspin[p,q] = eigenVectors[int(p/2), int(q/2)] * ((p % 2) == (q % 2))

    return dot(eigenspin.T, dot(fockspin, eigenspin))

def buildEriSingleBar(spinOrbitals, eriMO):
    #construct the spin two-electron repulsion integrals - < | >
    moSpinSingle = zeros((spinOrbitals, spinOrbitals, spinOrbitals, spinOrbitals))

    for p in range(0, spinOrbitals):
        for q in range(0, spinOrbitals):
            for r in range(0, spinOrbitals):
                for s in range(0, spinOrbitals):
                    moSpinSingle[p,q,r,s] = eriMO[iEri(int(p/2),int(q/2),int(r/2),int(s/2))] * \
                    ((p % 2) == (q % 2)) * ((r % 2) == (s % 2))

    return moSpinSingle

def buildEriDoubleBar(spinOrbitals, eriMO):
    #construct the spin two-electron repulsion integrals - < || >
    moSpinDouble = zeros((spinOrbitals, spinOrbitals, spinOrbitals, spinOrbitals))

    #get the sigle bar integrals
    moSpinSingle = buildEriSingleBar(spinOrbitals, eriMO)

    for p in range(0, spinOrbitals):
        for q in range(0, spinOrbitals):
            for r in range(0, spinOrbitals):
                for s in range(0, spinOrbitals):
                    moSpinDouble[p,q,r,s] = moSpinSingle[p,r,q,s] - moSpinSingle[p,s,q,r]

    return moSpinDouble

def eriTransform(eri):
    #transform between chemist [ij|kl] -> physicist<ik|jl> notation
    return eri.transpose(0, 2, 1, 3)

def expandEri(erimo, nBasis):
    #convert linear eri in MO basis to tensor form
    tensor = zeros((nBasis, nBasis, nBasis, nBasis))

    for p in range(0, nBasis):
        for q in range(0, nBasis):
            for r in range(0, nBasis):
                for s in range(0, nBasis):
                    tensor[p,q,r,s] = erimo[iEri(p,q,r,s)]
    return tensor

def d(iBasis, jBasis, direction):
    #construct Nabla

    sum = 0.0
    for i in range(0, len(iBasis.co)):
        for j in range(0, len(jBasis.co)):
            sum += iBasis.normal[i] * jBasis.normal[j] * iBasis.co[i] * jBasis.co[j] * \
                   nabla(iBasis.momentum, jBasis.momentum, iBasis.ex[i], jBasis.ex[j], iBasis.center,  jBasis.center, direction)

    return sum

def nabla(ia, ja, ie, je, ir, jr, direction):
    #Nabla
    p = ie + je
    ijr = ir - jr

    sx = e(ia[0], ja[0], 0, ijr[0], ie, je)
    sy = e(ia[1], ja[1], 0, ijr[1], ie, je)
    sz = e(ia[2], ja[2], 0, ijr[2], ie, je)

    dx = ja[0]*e(ia[0], ja[0]-1, 0, ijr[0], ie, je) - 2*je*e(ia[0], ja[0]+1, 0, ijr[0], ie, je)
    dy = ja[1]*e(ia[1], ja[1]-1, 0, ijr[1], ie, je) - 2*je*e(ia[1], ja[1]+1, 0, ijr[1], ie, je)
    dz = ja[2]*e(ia[2], ja[2]-1, 0, ijr[2], ie, je) - 2*je*e(ia[2], ja[2]+1, 0, ijr[2], ie, je)

    if direction == 'x':
        return dx * sy * sz * pow(pi/p , 1.5)
    if direction == 'y':
        return sx * dy * sz * pow(pi/p , 1.5)
    if direction == 'z':
        return sx * sy * dz * pow(pi/p , 1.5)

def buildNabla(atoms, bases, direction):
    #build Nabla matrix
    n = len(bases)
    delOperator = zeros((n,n))

    for iBasis in range(0,n):
        for jBasis in range(0, iBasis+1):
            delOperator[iBasis,jBasis] = d(bases[iBasis], bases[jBasis], direction)

            if iBasis != jBasis:
                delOperator[jBasis, iBasis] = -delOperator[iBasis, jBasis]

    return delOperator

def ang(ia, ja, ie, je, ir, jr, kr, direction):
    # angular momentum
    p = ie + je
    ijr = ir - jr

    sd = zeros((3,3))
    for i in range(3):
        sd[0,i] = e(ia[i], ja[i], 0, ijr[i], ie, je)
        sd[1,i] = e(ia[i], ja[i], 0, ijr[i], ie, je, 1, ir[i]-kr[i])
        sd[2,i] = (ja[i] * e(ia[i], ja[i]-1, 0, ijr[i], ie, je)) - (2.0 * je * e(ia[i], ja[i]+1, 0, ijr[i], ie, je))

    if direction == 'x':
        return -sd[0,0] * (sd[1,1] * sd[2,2] - sd[1,2] * sd[2,1]) * pow(pi/p, 1.5)
    elif direction == 'y':
        return -sd[0,1] * (sd[1,2] * sd[2,0] - sd[1,0] * sd[2,2]) * pow(pi/p, 1.5)
    elif direction == 'z':
        return -sd[0,2] * (sd[1,0] * sd[2,1] - sd[1,1] * sd[2,0]) * pow(pi/p, 1.5)

def a(iBasis, jBasis, kr, direction):
    #construct angular momentum
    sum = 0.0
    for i in range(0, len(iBasis.co)):
        for j in range(0, len(jBasis.co)):
            sum += iBasis.normal[i] * jBasis.normal[j] * iBasis.co[i] * jBasis.co[j] * \
                   ang(iBasis.momentum, jBasis.momentum, iBasis.ex[i], jBasis.ex[j], iBasis.center,  jBasis.center, kr, direction)

    return sum


def buildAngular(atoms, bases, direction, gaugeOrigin):
    #build the angular momentum integrals
    n = len(bases)
    angular = zeros((n,n))

    #get gauge center
    gauge = gaugeCenter(atoms, gaugeOrigin)

    for iBasis in range(0,n):
        for jBasis in range(0, iBasis+1):
            angular[iBasis,jBasis] = a(bases[iBasis], bases[jBasis], gauge, direction)

            if iBasis != jBasis:
                angular[jBasis, iBasis] = -angular[iBasis, jBasis]

    return angular

def quadrupole(ia, ja, ie, je, ir, jr, kr, direction):
    # quadrupole moment
    p = ie + je
    q = ((ie*ir + je*jr)/p) - kr
    ijr = ir - jr

    sx = e(ia[0], ja[0], 0, ijr[0], ie, je)
    sy = e(ia[1], ja[1], 0, ijr[1], ie, je) 
    sz = e(ia[2], ja[2], 0, ijr[2], ie, je) 

    tx = e(ia[0], ja[0], 1, ijr[0], ie, je) + q[0]* e(ia[0], ja[0], 0, ijr[0], ie, je) 
    ty = e(ia[1], ja[1], 1, ijr[1], ie, je) + q[1]* e(ia[1], ja[1], 0, ijr[1], ie, je)
    tz = e(ia[2], ja[2], 1, ijr[2], ie, je) + q[2]* e(ia[2], ja[2], 0, ijr[2], ie, je)

    if direction == 'xx':
        u = 2.0 * e(ia[0], ja[0], 2, ijr[0], ie, je) + 2.0 * q[0]* e(ia[0], ja[0], 1, ijr[0], ie, je)  \
            + (q[0]*q[0] + (0.5 / p)) * e(ia[0], ja[0], 0, ijr[0], ie, je)
        return u * sy * sz * pow(pi/p, 1.5)
    if direction == 'yy':
        u = 2.0 * e(ia[1], ja[1], 2, ijr[1], ie, je) + 2.0 * q[1]* e(ia[1], ja[1], 1, ijr[1], ie, je)  \
            + (q[1]*q[1] + (0.5 / p)) * e(ia[1], ja[1], 0, ijr[1], ie, je) 
        return sx * u * sz * pow(pi/p, 1.5)
    if direction == 'zz':
        u = 2.0 * e(ia[2], ja[2], 2, ijr[2], ie, je) + 2.0 * q[2]* e(ia[2], ja[2], 1, ijr[2], ie, je)  \
            + (q[2]*q[2] + (0.5 / p)) * e(ia[2], ja[2], 0, ijr[2], ie, je) 
        return sx * sy * u * pow(pi/p, 1.5)
    if direction == 'xy':
        return  tx * ty * sz * pow(pi/p, 1.5)
    if direction == 'yz':
        return  sx * ty * tz * pow(pi/p, 1.5)
    if direction == 'zx':
        return  tx * sy * tz * pow(pi/p, 1.5)

def q(iBasis, jBasis, kr, direction):
    #collects quadrupole values

    sum = 0.0
    for i in range(0, len(iBasis.co)):
        for j in range(0, len(jBasis.co)):
            sum += iBasis.normal[i] * jBasis.normal[j] * iBasis.co[i] * jBasis.co[j] * \
                   quadrupole(iBasis.momentum, jBasis.momentum, iBasis.ex[i], jBasis.ex[j], iBasis.center,  jBasis.center, kr, direction)

    return sum

def electricField(atoms, bases, direction, gauge):
    #compute components of the electric field

    n = len(bases)
    electric = zeros((n,n))

    #get gauge center
    gaugeOrigin = gaugeCenter(atoms, gauge)

    for iBasis in range(0,n):
        for jBasis in range(0, iBasis+1):
            electric[iBasis,jBasis] = ef(bases[iBasis], bases[jBasis], gaugeOrigin, direction)

            if iBasis != jBasis:
                electric[jBasis, iBasis] = electric[iBasis, jBasis]

    return electric

def ef(iBasis, jBasis, kr, direction):
    #construct electric field 
    sum = 0.0
    for i in range(0, len(iBasis.co)):
        for j in range(0, len(jBasis.co)):
            sum += iBasis.normal[i] * jBasis.normal[j] * iBasis.co[i] * jBasis.co[j] * \
                   electric(iBasis.momentum, jBasis.momentum, iBasis.ex[i], jBasis.ex[j], iBasis.center,  jBasis.center, kr, direction)

    return sum

def electric(ia, ja, ie, je, ir, jr, kr, direction):
    # electric field
    #Coulomb energy between two Gaussians
    # i,j - Gaussian 'i' 'j' ,  a - angular momentum  , e - exponent, r - seperation

    #p is composite center
    p = ie + je
    q = (ie*ir + je*jr)/p

    r = q - kr

    #component
    if direction == 'x': ix, iy, iz = [1,0,0]
    if direction == 'y': ix, iy, iz = [0,1,0]
    if direction == 'z': ix, iy, iz = [0,0,1]
    #potential
    if direction == 'p': ix, iy, iz = [0,0,0]

    sum = 0.0
    for i in range(0, ia[0]+ja[0]+1):
        for k in range(0, ia[1]+ja[1]+1):
            for l in range(0, ia[2]+ja[2]+1):
                sum += e(ia[0], ja[0], i, ir[0]-jr[0], ie, je) * \
                       e(ia[1], ja[1], k, ir[1]-jr[1], ie, je) * \
                       e(ia[2], ja[2], l, ir[2]-jr[2], ie, je) * \
                       j([i+ix,k+iy,l+iz], 0, p, r, norm(r))
                       
    sign = pow(-1, (ix+iy+iz))
    return sign* sum * pi * 2 / p
