from __future__ import division
from math import pow, pi, sqrt
from numpy import zeros, ones, dot, array
from numpy.linalg import norm
from view import postSCF
import integral

def e(ia, ja, type, r, ie, je, n, x):
    #recursive definition of Hermite Gaussian for derivative at x
    # i,j - Gaussian 'i' 'j' ,  a - angular momentum  , e - exponent
    #type number of Hermite node

    if n == 0:
        return integral.e(ia, ja, type, r, ie, je)
    else:
        return e(ia+1, ja, type, r, ie, je, n-1, x) + x * e(ia, ja, type, r, ie, je, n-1, x)

def efx(ia, ja, type, r, ie, je, n, x, p, s):

    if p == 1:
        return 2.0 * ie * e(ia+1, ja, type, r, ie, je, n, x) - ia * e(ia-1, ja, type, r, ie, je, n, x)
    elif s == 1:
        return 2.0 * je * e(ia, ja+1, type, r, ie, je, n, x) - ja * e(ia, ja-1, type, r, ie, je, n, x)

def overlapfx(ia, ja, ie, je, ir, jr, n, origin, x, center):
    #ia, ja momentum vectors, ie, je exponent values, ir, jr center vectors

    pa = 0
    if center == 'a':
        pa = 1
    pb = (pa+1) % 2

    t = 0.0

    if x == 0:
        t =  efx(ia[0], ja[0] , 0, ir[0] - jr[0], ie, je, n[0], ir[0] - origin[0],pa ,pb )
        t *= e(ia[1], ja[1] , 0, ir[1] - jr[1], ie, je, n[1], ir[1] - origin[1])
        t *= e(ia[2], ja[2] , 0, ir[2] - jr[2], ie, je, n[2], ir[2] - origin[2])
    elif x == 1:
        t =  e(ia[0], ja[0] , 0, ir[0] - jr[0], ie, je, n[0], ir[0] - origin[0])
        t *= efx(ia[1], ja[1] , 0, ir[1] - jr[1], ie, je, n[1], ir[1] - origin[1],pa ,pb )
        t *= e(ia[2], ja[2] , 0, ir[2] - jr[2], ie, je, n[2], ir[2] - origin[2])
    elif x == 2:
        t =  e(ia[0], ja[0] , 0, ir[0] - jr[0], ie, je, n[0], ir[0] - origin[0])
        t *= e(ia[1], ja[1] , 0, ir[1] - jr[1], ie, je, n[1], ir[1] - origin[1])
        t *= efx(ia[2], ja[2] , 0, ir[2] - jr[2], ie, je, n[2], ir[2] - origin[2],pa ,pb )

    return t * pow(pi/(ie+je), 1.5)

def sfx(iBasis, jBasis, center, x = 0, n = [0,0,0] , origin = [0,0,0]):
    #derivatives of generalised overlap for basis centered on 'center'
    #iBasis, jBasis are basis objects
    sum = 0.0
    for i in range(0, len(iBasis.co)):
        for j in range(0, len(jBasis.co)):
            sum += iBasis.normal[i]*jBasis.normal[j]*iBasis.co[i]*jBasis.co[j]*overlapfx(iBasis.momentum, jBasis.momentum, \
                   iBasis.ex[i], jBasis.ex[j], iBasis.center, jBasis.center, n ,  origin, x, center)

    return sum

def buildOverlapfx(atom, direction, bases):
  #overlap derivative for 'atom' in 'direction'

  n = len(bases)
  Sx = zeros((n,n))

  for i in range(0, len(bases)):
    for j in range(0, (i+1)):
      sum = 0.0
      if bases[i].atom == bases[j].atom: continue

      #use translational invariance - only need calculate one center
      f = sfx(bases[i], bases[j], 'a', direction)

      if bases[i].atom == atom: sum += f
      if bases[j].atom == atom: sum -= f

      Sx[i,j] = Sx[j,i] = sum
  
  return Sx

def kineticfx(ia, ja, ie, je, ir, jr, n, origin, x, center):
    #ia, ja momentum vectors, ie, je exponent values, ir, jr center vectors

    #cases for center 'a' and center 'b'
    pa = 0
    if center == 'a':
        pa = 1
    pb = (pa+1) % 2

    mu = (2*ja + 1) * je
    nu = -2*pow(je,2) * ones(3)
    vu = -0.5 * ja* (ja-1)

    t = zeros(3)
    for i in range(0, 3):

        if i == x:
            t[x] = mu[x] * efx(ia[x], ja[x] , 0, ir[x] - jr[x], ie, je, n[x], ir[x] - origin[x],pa ,pb ) +     \
                   nu[x] * efx(ia[x], ja[x] + 2 , 0, ir[x] - jr[x], ie, je, n[x], ir[x] - origin[x],pa ,pb ) + \
                   vu[x] * efx(ia[x], ja[x] - 2, 0, ir[x] - jr[x], ie, je, n[x], ir[x] - origin[x],pa ,pb )

        else:
            t[i] = mu[i] * e(ia[i], ja[i] , 0, ir[i] - jr[i], ie, je, n[i], ir[i] - origin[i]) +    \
                   nu[i] * e(ia[i], ja[i] + 2, 0, ir[i] - jr[i], ie, je, n[i], ir[i] - origin[i]) + \
                   vu[i] * e(ia[i], ja[i] - 2, 0, ir[i] - jr[i], ie, je, n[i], ir[i] - origin[i])

    for i in range(0, 3):

        if i == x:
            t[(x+1) % 3] *= efx(ia[x], ja[x] , 0, ir[x] - jr[x], ie, je, n[x], ir[x] - origin[x],pa ,pb ) 
            t[(x+2) % 3] *= efx(ia[x], ja[x] , 0, ir[x] - jr[x], ie, je, n[x], ir[x] - origin[x],pa ,pb )
        else:
            t[(i+1) % 3] *= e(ia[i], ja[i], 0, ir[i] - jr[i], ie, je, n[i], ir[i] - origin[i])
            t[(i+2) % 3] *= e(ia[i], ja[i], 0, ir[i] - jr[i], ie, je, n[i], ir[i] - origin[i])

    return (t[0] + t[1] + t[2]) * pow(pi/(ie+je), 1.5)

def kfx(iBasis, jBasis, center, x = 0, n = [0,0,0] , origin = [0,0,0]):
    #derivatives of generalised exchange integral for basis centered on 'center'
    #iBasis, jBasis are basis objects
    sum = 0.0
    for i in range(0, len(iBasis.co)):
        for j in range(0, len(jBasis.co)):
            sum += iBasis.normal[i]*jBasis.normal[j]*iBasis.co[i]*jBasis.co[j] * kineticfx(iBasis.momentum, jBasis.momentum, \
                   iBasis.ex[i], jBasis.ex[j], iBasis.center, jBasis.center, n ,  origin, x, center)

    return sum

def buildKineticfx(atom, direction, bases):
  #overlap derivative for 'atom' in 'direction'

  n = len(bases)
  Kx = zeros((n,n))

  for i in range(0, len(bases)):
    for j in range(0, (i+1)):
      sum = 0.0
      if bases[i].atom == bases[j].atom: continue

      #use translational invariance - only need calculate one center
      f = kfx(bases[i], bases[j], 'a', direction)

      if bases[i].atom == atom: sum += f
      if bases[j].atom == atom: sum -= f

      Kx[i,j] = Kx[j,i] = sum
  
  return Kx

def coulombfxs(ia, ja, ie, je, ir, jr, nucleus, n, origin, x, center):
    #generalised coulomb derivatives dV(ab^(0,0,0))/dx terms

    #center of four Gaussians
    p = ie + je
    q = (ie*ir + je*jr)/p
    r = q - nucleus

    #loop limits
    tau = ia + ja + n + 1
    tau[x] += 1

    pa = 0
    if center == 'a':
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
                        val *=   e(ia[i], ja[i], xi[i], ir[i]-jr[i], ie, je, n[i], ir[i]-nucleus[i])

                sum += val * integral.j(xi, 0, p, r, norm(r))

    return sum * 2 * pi/p

def jfx(iBasis, jBasis, nucleus, center = 'a', x = 0, n = [0,0,0] , origin = [0,0,0]):
    #derivatives of generalised coulomb integral for basis centered on 'center'
    #iBasis, jBasis are basis objects
    sum = 0.0
    for i in range(0, len(iBasis.co)):
        for j in range(0, len(jBasis.co)):
            sum += iBasis.normal[i]*jBasis.normal[j]*iBasis.co[i]*jBasis.co[j] * coulombfxs(iBasis.momentum, jBasis.momentum, \
                   iBasis.ex[i], jBasis.ex[j], iBasis.center, jBasis.center, nucleus, n ,  origin, x, center)

    return sum

def buildCoulombfx(atom, direction, molAtom, bases):
    #coulomb derivative for 'atom' in 'direction'

    n = len(bases)
    Jx = zeros((n,n))

    for i in range(0, len(bases)):
        for j in range(0, (i+1)):               
            sum = 0.0
            for k in range(0, len(molAtom)):
                if bases[i].atom == atom:
                    sum -= molAtom[k].number * jfx(bases[i], bases[j], molAtom[k].center, 'a', direction)
                if bases[j].atom == atom:
                    sum -= molAtom[k].number * jfx(bases[i], bases[j], molAtom[k].center, 'b', direction)

                Jx[i,j] = Jx[j,i] = sum

    return Jx

def coulombfxh(ia, ja, ie, je, ir, jr, nucleus, n, origin, x):
    #generalised coulomb derivatives dV(ab^(0,0,0))/dx terms

    #center of two Gaussians
    p = ie + je
    q = (ie*ir + je*jr)/p
    r = q - nucleus

    #loop limits
    tau = ia + ja + n + 1

    sum = 0.0
    val = 1.0

    for mu in range(0, tau[0]):
        for nu in range(0, tau[1]):
            for vu in range(0, tau[2]):
                val = 1.0
                xi = [mu,nu,vu]
                for i in range(0, 3):
                    val *=   e(ia[i], ja[i], xi[i], ir[i]-jr[i], ie, je, n[i], ir[i]-nucleus[i])

                xi[x] += 1
                sum -= val * integral.j(xi, 0, p, r, norm(r))


    return sum * 2 * pi/p

def jfh(iBasis, jBasis, nucleus, x = 0,  n = [0,0,0] , origin = [0,0,0]):
    #derivatives of generalised coulomb integral for basis centered on 'center'
    #iBasis, jBasis are basis objects
    sum = 0.0
    for i in range(0, len(iBasis.co)):
        for j in range(0, len(jBasis.co)):

            sum += iBasis.normal[i]*jBasis.normal[j]*iBasis.co[i]*jBasis.co[j] * coulombfxh(iBasis.momentum, jBasis.momentum, \
                   iBasis.ex[i], jBasis.ex[j], iBasis.center, jBasis.center, nucleus, n , origin, x)

    return sum

def buildCoulombfh(atom, direction, molAtom, bases):
    #coulomb derivative for 'atom' in 'direction' - Hellman-Feynmann term

    n = len(bases)
    Fx = zeros((n,n))

    for i in range(0, len(bases)):
        for j in range(0, (i+1)):
            Fx[j,i] = Fx[i,j] = -molAtom[atom].number * jfh(bases[i], bases[j], molAtom[atom].center, direction)


    return Fx

def buildNuclearfx(atom, direction, molAtom):
    #nuclear repulsion derivative

    sum = 0.0
    for i in range(0, len(molAtom)):
        r = norm(molAtom[atom].center - molAtom[i].center)
        #vector component i-atom in 'direction'
        vr = molAtom[atom].center[direction] - molAtom[i].center[direction]
        if r != 0:
            sum -= vr * molAtom[i].number * molAtom[atom].number / (r * r * r)

    return sum

def electronRepulsionfx(ia, ja, ka, la, ie, je, ke, le, ir, jr, kr, lr, ra, rb, origin, x = 0, center = 'a'):

    p = ie + je
    q = ke + le
    rho = p*q/(p + q)
    P = (ie*ir + je*jr)/p
    Q = (ke*kr + le*lr)/q
    r = P - Q

    tau = ia + ja + 1 + ra
    sigma = ka + la + 1 + rb
    if (center == 'a') or (center == 'b'):
        tau[x] += 1
    else:
        sigma[x] += 1

    pa = 0
    if (center == 'a') or (center == 'c'):
        pa = 1
    pb = (pa+1) % 2

    val = 0.0
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
                                    if (center == 'a' or center == 'b'):
                                        term *= efx(ia[x],ja[x],xia[x],ir[x]-jr[x],ie,je,ra[x],ir[x] - origin[x], pa, pb)
                                        term *= e(ka[x],la[x],xib[x],kr[x]-lr[x],ke,le,rb[x],kr[x] - origin[x]) 
                                    elif (center == 'c' or center == 'd'):
                                        term *= e(ia[x],ja[x],xia[x],ir[x]-jr[x],ie,je,ra[x],ir[x] - origin[x])
                                        term *= efx(ka[x],la[x],xib[x],kr[x]-lr[x],ke,le,rb[x],kr[x] - origin[x], pa, pb) 

                                else:
                                    term *= e(ia[i],ja[i],xia[i],ir[i]-jr[i],ie,je,ra[i],ir[i] - origin[i])
                                    term *= e(ka[i],la[i],xib[i],kr[i]-lr[i],ke,le,rb[i],kr[i] - origin[i]) 

                            term *= pow(-1, (psi+phi+chi)) * integral.j([mu+psi,nu+phi,vu+chi],0, rho,r,norm(r)) 
                            val += term

    return val*2*pow(pi,2.5)/(p*q*sqrt(p+q)) 

def ERIfx(iBasis, jBasis, kBasis, lBasis, center, x = 0, n = [0,0,0], nu = [0,0,0], origin = [0,0,0]):
    #2-electrons derivative
    sum = 0.0

    for i in range(0, len(iBasis.co)):
        for j in range(0, len(jBasis.co)):
            for k in range(0, len(kBasis.co)):
                for l in range(0, len(lBasis.co)):
                    sum += iBasis.normal[i]*jBasis.normal[j]*kBasis.normal[k]*lBasis.normal[l] * \
                           iBasis.co[i]*jBasis.co[j]*kBasis.co[k]*lBasis.co[l] *                 \
                           electronRepulsionfx(iBasis.momentum, jBasis.momentum, kBasis.momentum, lBasis.momentum, \
                                               iBasis.ex[i], jBasis.ex[j], kBasis.ex[k], lBasis.ex[l],             \
                                               iBasis.center, jBasis.center, kBasis.center, lBasis.center,         \
                                               n, nu, origin, x, center)

    return sum

def buildERIfx(atom, direction, molAtom, bases):
    #derivatives of 2-electron repulsion integrals

    n = len(bases)
    ERIx = zeros((n,n,n,n))

    for i in range(0, len(bases)):
        for j in range(0, (i+1)):
            ij = i*(i+1)/2 + j
            for k in range(0, len(bases)):
                for l in range(0, (k+1)):
                    kl = k*(k+1)/2 + l
                    if ij >= kl:
                        sum = 0.0
                        if bases[i].atom == atom:
                            sum += ERIfx(bases[i], bases[j], bases[k], bases[l], 'a', direction)
                        if bases[j].atom == atom:
                            sum += ERIfx(bases[i], bases[j], bases[k], bases[l], 'b', direction)
                        if bases[k].atom == atom:
                            sum += ERIfx(bases[i], bases[j], bases[k], bases[l], 'c', direction)
                        if bases[l].atom == atom:
                            sum += ERIfx(bases[i], bases[j], bases[k], bases[l], 'd', direction)

                        ERIx[i,j,k,l] = ERIx[i,j,l,k] = ERIx[j,i,l,k] = ERIx[j,i,k,l] = \
                        ERIx[k,l,i,j] = ERIx[k,l,j,i] = ERIx[l,k,j,i] = ERIx[l,k,i,j] = sum

    return ERIx

def buildFockfx(atom, direction, molAtom, molBasis, density):
    #compute the derivative of the fock matrix

    #core hamiltonian
    hfx  = buildKineticfx(atom , direction, molBasis) +         \
           buildCoulombfx(atom, direction, molAtom, molBasis) + \
           buildCoulombfh(atom, direction, molAtom, molBasis)

    ERIx = buildERIfx(atom, direction, molAtom, molBasis)

    #get g-matrix jx+kx
    n = len(molBasis)
    jx = zeros((n,n))
    for p in range(0, n):
        for q in range(0, n):
            sum = 0.0
            for r in range(0, n):
                for s in range(0, n):
                    sum += ERIx[p,q,r,s] * density[s, r]

            jx[p,q] = sum

    kx = zeros((n,n))
    for p in range(0, n):
        for q in range(0, n):
            sum = 0.0
            for r in range(0, n):
                for s in range(0, n):
                    sum += ERIx[p,s,q,r] * density[s, r]

            kx[p,q] = sum

    fockfx = hfx + 2.0 * jx - kx

    return fockfx , hfx

def gradient(molAtom, molBasis, molData):
    import rhf
    #compute gradient
    gfx = zeros((len(molAtom), 3))

    #equilibrium geometry - restore after each perturbation
    referenceGeometry = zeros((len(molAtom), 3))
    for i in range(0, referenceGeometry.shape[0]):
        referenceGeometry[i,:] = molAtom[i].center

    for atom in range(0, gfx.shape[0]):
        for direction in range(0 , gfx.shape[1]):
            gfx[atom,direction] = -1.0 * efxNumeric(atom, direction, molAtom, molBasis, molData)
            rhf.rebuildCenters(molAtom, molBasis, referenceGeometry)

    return gfx

def efxNumeric(atom, direction, molAtom, molBasis, molData):
    #derivative of energy in 'direction' using central difference
    #after changing centers must update molAtom and molBasis
    import rhf
    dq = 1e-4
    
    #make copy of coordinates for perturbation
    geo = zeros((len(molAtom), 3))
    for i in range(0, geo.shape[0]):
        geo[i,:] = molAtom[i].center

    #perturb -> +
    geo[atom, direction] += dq 
    molAtom, molBasis = rhf.rebuildCenters(molAtom, molBasis, geo)
    ep = rhf.scf(molAtom, molBasis, molData, [])
    
    #perturb -> -
    geo[atom, direction] -= 2 * dq 
    molAtom, molBasis = rhf.rebuildCenters(molAtom, molBasis, geo)
    em = rhf.scf(molAtom, molBasis, molData, [])

    deltaE = (ep - em)/ (2 * dq)

    return deltaE 

def forces(molAtom, molBasis, density, fock, engine = 'aello', type = 'analytic'):
    #get force aor all atoms in all direction
    n = len(molBasis)

    if ('analytic,both'.find(type)) != -1 :

        molForces = zeros((len(molAtom), 3))
        if engine != 'aello':
            for atom in range(0, len(molAtom)):
                for direction in range(0, 3):
                    fx, hx = buildFockfx(atom, direction, molAtom, molBasis, density )

                    ex = fx + hx
                    force = 0.0
                    for p in range(0, n):
                        for q in range(0, n):
                            force -= density[p,q] * ex[q,p]

                    #density weighted energy and overlap derivative
                    eWeighted = dot(density, dot(fock, density))
                    sx = buildOverlapfx(atom, direction, molBasis)

                    for p in range(0, n):
                        for q in range(0, n):
                            force += 2.0 * sx[p,q] * eWeighted[q,p]

                    #nuclear repulsion term
                    force -= buildNuclearfx(atom, direction, molAtom)
                    #F = -dE/dx
                    molForces[atom, direction] = force
        else:

            from ocypete import ocypete
            molForces = ocypete(molAtom, molBasis, density, fock)

        postSCF([molForces, molAtom], 'fa')

    if type == 'both' :
        import rhf      
        #numeric derivatives
        _, _, molData = rhf.mol([])
        postSCF([gradient(molAtom, molBasis, molData), molAtom], 'fn')

    return molForces

    
def optimiseGeometry(f, q, text):
    #optimise geometry using Nelder-Mead
    from scipy.optimize import minimize
    import rhf
    
    molAtom, molBasis, molData = rhf.mol([])

    optimumGeometry = minimize(f, q, method="Nelder-Mead", args = (molAtom, molBasis, molData))

    for i in range(0, len(text)):
        print(text[i],' : ', optimumGeometry.x[i])

    print("Optimal energy : ", optimumGeometry.fun)
    print("Cycles : ", optimumGeometry.nit)
    print(optimumGeometry.success)

