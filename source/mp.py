from __future__ import division
from integral import buildEriMO, iEri, buildEriDoubleBar, buildFockMOspin, expandEri
from numpy.polynomial.laguerre import laggauss as quadrature
from basis import electronCount
from view import postSCF
from numpy import zeros, exp, diag, ones, einsum, newaxis, fill_diagonal
from numpy.linalg import eigh

def mollerPlesset(atoms, charge, bases, eigenVectors, eigenValues, fock, ERI, e):
    #compute mp2 and mp3
    ps = aps = mp = 0.0
    ps , aps = mp2(atoms, charge, bases, eigenVectors, eigenValues, ERI)
    mp = mp3(atoms, charge, bases, eigenVectors, fock, ERI)

    postSCF([ps, aps, mp, e], 'mp')


def mp2(atoms, charge, bases, eigenVectors, eigenValues, ERI, type = 'p-ap'):
    #moller-plesset perturbation theory 2nd order
    eriMO = buildEriMO(eigenVectors, ERI)

    nOccupied = int(electronCount(atoms, charge)/2)
    n = len(bases)

    mpParallel = mpAntiParallel = 0
    for i in range(0, nOccupied):
        for a in range(nOccupied, n):

            for j in range(0, nOccupied):
                for b in range(nOccupied, n):
                    u = eigenValues[i] + eigenValues[j] - eigenValues[a] - eigenValues[b]
                    v = eriMO[iEri(i,a,j,b)]

                    mpParallel     += (v*(v- eriMO[iEri(i,b,j,a)]))/u
                    mpAntiParallel += (v*v)/u

    if type == 'p-ap':

        #parallel and anti-parallel components
        return mpParallel , mpAntiParallel

    elif type == 'scs':

        #do a spin-component scaled mp2 correction

        scs = mpParallel/3.0 + 1.2 * mpAntiParallel
        return scs

    elif type == 'no':

        #get mp2 natural orbitals

        nVirtual = n - nOccupied
        ga = zeros((nOccupied, nVirtual, nOccupied, nVirtual))
        gb = zeros((nOccupied, nVirtual, nOccupied, nVirtual))

        for i in range(nOccupied):
            for a in range(nOccupied, n):
                for j in range(nOccupied):
                    for b in range(nOccupied, n):
                        ga[i,a-nOccupied,j,b-nOccupied] = eriMO[iEri(i,a,j,b)] + (eriMO[iEri(i,a,j,b)] - eriMO[iEri(i,b,j,a)])
                        gb[i,a-nOccupied,j,b-nOccupied] = eriMO[iEri(i,a,j,b)] / (eigenValues[i] + eigenValues[j] - eigenValues[a] - eigenValues[b])

        co = zeros((nOccupied, nOccupied))
        for i in range(nOccupied):
            for j in range(nOccupied):
                for a in range(nOccupied, n):
                    for k in range(nOccupied):
                        for b in range(nOccupied, n):
                            co[i,j] += ga[i,a-nOccupied,k,b-nOccupied] * gb[j,a-nOccupied,k,b-nOccupied]

        cv = zeros((nVirtual, nVirtual))
        for i in range(nOccupied):
            for c in range(nOccupied, n):
                for a in range(nOccupied, n):
                    for j in range(nOccupied):
                        for b in range(nOccupied, n):
                            cv[c-nOccupied,a-nOccupied] += ga[i,a-nOccupied,j,b-nOccupied] * gb[i,c-nOccupied,j,b-nOccupied]

        #construct symmetric mp2 density matrix
        mp2Density = zeros((n, n))
        po = 0.25 * (co + co.T) + diag(ones(nOccupied)) * 2
        pv = -0.25 * (cv + cv.T)

        mp2Density[:nOccupied, :nOccupied] = po
        mp2Density[nOccupied:, nOccupied:] = pv

        #diagonalise
        noE, noC = eigh(mp2Density)

        return noE, noC

def mp3(atoms, charge, bases, eigenVectors, fock, ERI):
    #moller-plesset perturbation theory 3rd order - spin orbitals

    nElectrons = int(electronCount(atoms, charge))
    spinOrbitals = len(bases) * 2

    #spin MO eris
    eriMO = buildEriMO(eigenVectors, ERI)
    eriMOspin = buildEriDoubleBar(spinOrbitals, eriMO)


    #spin MO fock
    fs = buildFockMOspin(spinOrbitals, eigenVectors, fock)

    mp3 = 0.0
    for i in range(0, nElectrons):
        for j in range(0, nElectrons):
            for k in range(0, nElectrons):
                for l in range(0, nElectrons):
                    for a in range(nElectrons,spinOrbitals):
                        for b in range(nElectrons,spinOrbitals):
                            u = (fs[i,i] + fs[j,j] - fs[a,a] - fs[b,b]) * \
                                (fs[k,k] + fs[l,l] - fs[a,a] - fs[b,b])

                            mp3 += 0.125 * eriMOspin[i,j,a,b] * eriMOspin[k,l,i,j] * eriMOspin[a,b,k,l] / u

    for i in range(0, nElectrons):
        for j in range(0, nElectrons):
            for a in range(nElectrons,spinOrbitals):
                for b in range(nElectrons,spinOrbitals):
                    for c in range(nElectrons,spinOrbitals):
                        for d in range(nElectrons,spinOrbitals):
                            u = (fs[i,i] + fs[j,j] - fs[a,a] - fs[b,b]) * \
                                (fs[i,i] + fs[j,j] - fs[c,c] - fs[d,d])

                            mp3 += 0.125 * eriMOspin[i,j,a,b] * eriMOspin[a,b,c,d] * eriMOspin[c,d,i,j] / u

    for i in range(0, nElectrons):
        for j in range(0, nElectrons):
            for k in range(0, nElectrons):
                for a in range(nElectrons,spinOrbitals):
                    for b in range(nElectrons,spinOrbitals):
                        for c in range(nElectrons,spinOrbitals):
                            u = (fs[i,i] + fs[j,j] - fs[a,a] - fs[b,b]) * \
                                (fs[i,i] + fs[k,k] - fs[a,a] - fs[c,c])

                            mp3 += eriMOspin[i,j,a,b] * eriMOspin[k,b,c,j] * eriMOspin[a,c,i,k] / u

    return mp3

def orbitalOptimisedMP2(eigenVectors, h, e, molBasis, eNuclear, ERI, nElectrons):
    import scipy.linalg as la
    from numpy import zeros, dot, zeros_like, append

    def hSpinBlock(h, spinOrbitals, nBasis):
        #spin blocking a square matrix
        sb = zeros((spinOrbitals, spinOrbitals))

        for p in range(0, nBasis):
            for q in range(0, nBasis):
                sb[p,q] = sb[p+nBasis,q+nBasis] = h[p,q]

        return sb

    def eriSpinBlock(eri, spinOrbitals, nBasis):
        #spin blocking the eri
        sbEri = zeros((spinOrbitals, spinOrbitals, spinOrbitals, spinOrbitals))

        for p in range(0, nBasis):
            for q in range(0, nBasis):
                for r in range(0, nBasis):
                    for s in range(0, nBasis):
                        sbEri[p,q,r,s] = sbEri[p,q,r+nBasis, s+nBasis] = eri[iEri(p,q,r,s)]

        for p in range(0, spinOrbitals):
            for q in range(0, spinOrbitals):
                for r in range(0, nBasis):
                    for s in range(0, nBasis):
                        sbEri[p,q,r,s] = sbEri[p,q,r+nBasis, s+nBasis] = sbEri[s,r,q,p]

        aoSpinDoubleBar = zeros((spinOrbitals, spinOrbitals, spinOrbitals, spinOrbitals))
        for p in range(0, spinOrbitals):
            for q in range(0, spinOrbitals):
                for r in range(0, spinOrbitals):
                    for s in range(0, spinOrbitals):
                        aoSpinDoubleBar[p,q,r,s] = sbEri[p,r,q,s] - sbEri[p,s,q,r]

        return aoSpinDoubleBar

    def eriSpinMO(gao, eigenVectors):
        #change eri from spin AO to spin MO
        gmoA = zeros((spinOrbitals, spinOrbitals, spinOrbitals, spinOrbitals))
        gmo = zeros_like(gmoA)
        for a in range(0, spinOrbitals):
            for q in range(0, spinOrbitals):
                for r in range(0, spinOrbitals):
                    for s in range(0, spinOrbitals):
                        for p in range(0, spinOrbitals):
                            gmoA[a,q,r,s] += eigenVectors[p,a] * gao[p,q,r,s]
        for a in range(0, spinOrbitals):
            for b in range(0, spinOrbitals):
                for r in range(0, spinOrbitals):
                    for s in range(0, spinOrbitals):
                        for q in range(0, spinOrbitals):
                            gmo[a,b,r,s] += eigenVectors[q,b] * gmoA[a,q,r,s]
        gmoA = zeros_like(gmoA)
        for a in range(0, spinOrbitals):
            for b in range(0, spinOrbitals):
                for c in range(0, spinOrbitals):
                    for s in range(0, spinOrbitals):
                        for r in range(0, spinOrbitals):
                            gmoA[a,b,c,s] += eigenVectors[r,c] * gmo[a,b,r,s]
        gmo = zeros_like(gmo)
        for a in range(0, spinOrbitals):
            for b in range(0, spinOrbitals):
                for c in range(0, spinOrbitals):
                    for d in range(0, spinOrbitals):
                        for s in range(0, spinOrbitals):
                            gmo[a,b,c,d] += eigenVectors[s,d] * gmoA[a,b,c,s]

        return gmo

    #loop control
    iterations = 40
    tolerance = 1.0e-8

    #metrics
    nBasis = len(molBasis)        
    nOccupied = nElectrons 
    spinOrbitals = 2 * nBasis
    nVirtual = spinOrbitals - nOccupied

    #eri in spin atomic basis and double bar
    gao = eriSpinBlock(ERI,spinOrbitals, nBasis)
    #h in spin atomic basis
    hao = hSpinBlock(h,spinOrbitals, nBasis)

    #get orbital energies (o,v)+(o,v)
    eps = append(e, e)

    #get orbital coefficients, block (not spin block), and sort by eps (oo,vv)
    c = zeros((spinOrbitals, spinOrbitals))
    for p in range(0, spinOrbitals):
        for q in range(0, spinOrbitals):
            c[p,q] = eigenVectors[p % nBasis, q % nBasis]
            if ((p < nBasis) and (q >= nBasis)) or ((p >= nBasis) and (q < nBasis)):
                c[p,q] = 0.0

    c = c[:, eps.argsort()]

    # Transform gao and hao into MO basis
    hmo = dot(c.T, dot(hao,c))
    gmo = eriSpinMO(gao, c)

    #initialise the fock matrix, amplitudes, correlation and reference
    #one-particle density matrix and energy, two-particle density matrix
    fock = zeros((spinOrbitals, spinOrbitals))
    td = zeros((nOccupied, nOccupied, nVirtual, nVirtual)) 
    tpdmCor = zeros((spinOrbitals, spinOrbitals, spinOrbitals, spinOrbitals))
    preOMP = 0.0 

    # Build the reference one particle density matrix
    opdmRef = zeros((spinOrbitals, spinOrbitals))
    for i in range(0, nOccupied):
        opdmRef[i,i] = 1.0

    # Initialize the rotation matrix parameter 
    rot = zeros((spinOrbitals, spinOrbitals))

    for cycle in range(0, iterations):

        # Build the Fock matrix
        for p in range(0, spinOrbitals):
            for q in range(0, spinOrbitals):
                fock[p,q] = hmo[p,q]
                for i in range(0, nOccupied):
                    fock[p,q] += gmo[p,i,q,i]

        # Build off-diagonal Fock Matrix and orbital energies
        fprime = zeros((spinOrbitals, spinOrbitals))
        for p in range(0, spinOrbitals):
            for q in range(0, spinOrbitals):
                if p != q:
                    fprime[p,q] = fock[p,q]
                else:
                    eps[p] = fock[p,p]

        # Update t amplitudes
        t = zeros_like(td)
        for i in range(0, nOccupied):
            for j in range(0, nOccupied):
                for a in range(0, nVirtual):
                    for b in range(0, nVirtual):
                        t[i,j,a,b] = gmo[i,j,a+nOccupied,b+nOccupied]
                        for d in range(0, nVirtual):
                            t[i,j,a,b] += fprime[a+nOccupied,d+nOccupied] * td[i,j,d,b] - \
                                          fprime[b+nOccupied,d+nOccupied] * td[i,j,d,a]
                        for k in range(0, nOccupied):
                            t[i,j,a,b] += -fprime[k,i] * td[k,j,a,b] + fprime[k,j] * td[k,i,a,b]
                        t[i,j,a,b] /= (eps[i] + eps[j] - eps[a+nOccupied] - eps[b+nOccupied])
        td = t.copy() 

        #build one particle density matrix
        opdmCor = zeros((spinOrbitals, spinOrbitals))
        for i in range(0, nOccupied):
            for j in range(0, nOccupied):
                for a in range(0, nVirtual):
                    for b in range(0, nVirtual):
                        for d in range(0, nVirtual):
                            opdmCor[b+nOccupied, a+nOccupied] += 0.5 * td[j,i,d,a] * td[i,j,b,d]
                        for k in range(0, nOccupied):
                            opdmCor[j,i] -= 0.5 * td[k,j,b,a] * td[i,k,a,b]
        opdm = opdmCor + opdmRef 

        #build two particle density matrix
        tpdm = zeros_like(tpdmCor)
        tpdm1 = zeros_like(tpdmCor)
        for i in range(0, nOccupied):
            for j in range(0, nOccupied):
                for a in range(0, nVirtual):
                    for b in range(0, nVirtual):
                        tpdmCor[a+nOccupied,b+nOccupied,i,j] = td[i,j,a,b]
                        tpdmCor[i,j,a+nOccupied,b+nOccupied] = td[j,i,b,a]
        for r in range(0, spinOrbitals):
            for s in range(0, spinOrbitals):
                for p in range(0, spinOrbitals):
                    for q in range(0, spinOrbitals):
                        tpdm[r,s,p,q] += opdmCor[r,p] * opdmRef[s,q]
                        tpdm[s,r,q,p]   = tpdm[r,s,p,q]
                        tpdm[s,r,p,q]  = tpdm[r,s,q,p] = -tpdm[r,s,p,q]

                        tpdm1[r,s,p,q] += opdmRef[r,p] * opdmRef[s,q]
                        tpdm1[s,r,p,q] = - tpdm1[r,s,p,q] 
        tpdm += tpdmCor + tpdm1

        # Newton-Raphson step
        generalFock = zeros_like(fock)
        for p in range(0, spinOrbitals):
            for q in range(0, spinOrbitals):
                for r in range(0, spinOrbitals):
                    generalFock[p,q] += hmo[p,r] * opdm[r,q]
                    for s in range(0, spinOrbitals):
                        for t in range(0, spinOrbitals):
                            generalFock[p,q] += 0.5 * gmo[p,r,s,t] * tpdm[s,t,q,r]

        for i in range(0, nOccupied):
            for a in range(0, nVirtual):
                rot[a+nOccupied,i] = (generalFock - generalFock.T)[a+nOccupied, i]/(eps[i] - eps[a+nOccupied])

        # Build Newton-Raphson orbital rotation matrix
        U = la.expm(rot - rot.T)
        
        # Rotate spin-orbital coefficients
        c = c.dot(U)

        # Transform one and two electron integrals using new C
        hmo = dot(c.T, dot(hao,c))
        gmo = eriSpinMO(gao, c)

        # Compute the energy
        omp = eNuclear
        for p in range(0, spinOrbitals):
            for q in range(0, spinOrbitals):
                omp += hmo[p,q] * opdm[q,p]
        for p in range(0, spinOrbitals):
            for q in range(0, spinOrbitals):
                for r in range(0, spinOrbitals):
                    for s in range(0, spinOrbitals):
                        omp += 0.25 * gmo[p,q,r,s] * tpdm[r,s,p,q]

        if (abs(omp - preOMP)) < tolerance:
            break

        # Updating values
        preOMP = omp

    postSCF([cycle, omp, omp-preOMP],'omp')

    return omp

def eriMOpartition(eri, co, cv, nBasis, no, nv):
    #split eri [:no,:nv,:,no,:nv]
    #[no,:,:,:]
    eriReduced = zeros((nBasis, nBasis, nBasis, nBasis))
    for p in range(0, nBasis):
        for q in range(0, nBasis):
            for r in range(0, nBasis):
                for s in range(0, nBasis):
                    for i in range(0, no):
                        eriReduced[i,q,r,s] += co[p,i] * eri[p,q,r,s]
    eriReduced = eriReduced[:no,:,:,:]
    #[no,nv,:,:]
    eriIntermediate = zeros((no, nBasis, nBasis, nBasis))
    for p in range(0, no):
        for q in range(0, nBasis):
            for r in range(0, nBasis):
                for s in range(0, nBasis):
                    for a in range(0, nv):
                        eriIntermediate[p,a,r,s] += cv[q,a] * eriReduced[p,q,r,s]
    eriReduced = eriIntermediate[:,:nv,:,:]
    #[no,nv,no,:]
    eriIntermediate = zeros((no, nv, nBasis, nBasis))
    for p in range(0, no):
        for q in range(0, nv):
            for r in range(0, nBasis):
                for s in range(0, nBasis):
                    for j in range(0, no):
                        eriIntermediate[p,q,j,s] += co[r,j] * eriReduced[p,q,r,s]
    eriReduced = eriIntermediate[:,:,:no,:]
    #[no,nv,no,nv]
    eriIntermediate = zeros((no, nv, no, nBasis))
    for p in range(0, no):
        for q in range(0, nv):
            for r in range(0, no):
                for s in range(0, nBasis):
                    for b in range(0, nv):
                        eriIntermediate[p,q,r,b] += cv[s,b] * eriReduced[p,q,r,s]
    eriReduced = eriIntermediate[:,:,:,:nv]

    return eriReduced

def mp2LaplaceTransform(molBasis, c, ERI, eps, nOccupied, eSCF, meshSize=40):
    #Laplace transform Moller-Plesset (2) - restricted reference implementation

    #metrics
    nBasis = len(molBasis)
    nVirtual = nBasis - nOccupied

    #slices of orbital eigens
    epsOccupied = eps[:nOccupied]
    epsVirtual  = eps[nOccupied:]
    cOccupied   = c[:,:nOccupied]
    cVirtual    = c[:,nOccupied:]

    #sliced eri in MO basis
    eriTensor = expandEri(ERI, nBasis)
    eriMOslice = eriMOpartition(eriTensor, cOccupied, cVirtual, nBasis, nOccupied, nVirtual)

    #Gauss-Leguerre quadrature
    mesh, weights = quadrature(meshSize)
    #use exponential weights
    weights *= exp(mesh)
    #energies for mp2 parallel and anti-parallel spins
    eMP = [0.0,0.0]

    #cycle over mesh
    for cycle in range(0, meshSize):

        #compute amplitudes
        ampOccupied = exp(mesh[cycle] * epsOccupied)
        ampVirtual  = exp(-mesh[cycle] * epsVirtual)
        ampEri = zeros((nOccupied, nVirtual, nOccupied, nVirtual))
        for i in range(0, nOccupied):
            for a in range(0, nVirtual):
                for j in range(0, nOccupied):
                    for b in range(0, nVirtual):
                        ampEri[i,a,j,b] = ampOccupied[i] * ampVirtual[a] * ampOccupied[j] * ampVirtual[b] * eriMOslice[i,a,j,b]

        #mp2 energies
        eMPcontraction = [0.0,0.0]
        for i in range(0, nOccupied):
            for a in range(0, nVirtual):
                for j in range(0, nOccupied):
                    for b in range(0, nVirtual):
                        eMPcontraction[0] += ampEri[i,a,j,b] * eriMOslice[i,a,j,b]
                        eMPcontraction[1] += ampEri[i,a,j,b] * (eriMOslice[i,a,j,b] - eriMOslice[i,b,j,a])

        for e in range(0, 2):
            eMP[e] -= eMPcontraction[e] * weights[cycle]

    postSCF([eMP, eSCF], 'mplp')

    return eMP

def mp2UnrelaxedDensity(c, e, eri, nbf, nOccupied):
    #compute the mp2 level unrelaxed density matrix

    #transform to mo basis
    eriMO = buildEriMO(c, eri)
    g     = expandEri(eriMO, nbf)

    #slices
    o = slice(None, nOccupied) ; v = slice(nOccupied, None) ; n = newaxis

    #energy denominator
    dd = 1.0/(e[o, n, n, n] + e[n, o, n, n] - e[n, n, v, n] - e[n, n, n, v])

    #doubles amplitudes
    t2 = -einsum('iajb,ijab->ijab', g[o,v,o,v], dd, optimize=True)

    #double Lagrange multipliers
    l2 = -2.0 * einsum('iajb,ijab->ijab', 2.0*g[o,v,o,v] - g[o,v,o,v].swapaxes(3, 1), dd, optimize=True)

    oo = -einsum('kiab,kjab->ij', l2, t2, optimize=True)
    vv =  einsum('ijbc,ijac->ab', l2, t2, optimize=True)

    #enforce symmetry
    oo = 0.5 * (oo + oo.T)
    vv = 0.5 * (vv + vv.T)

    #HF and mp2 contributions
    mp2Density = zeros((nbf, nbf))
    fill_diagonal(mp2Density[:nOccupied, :nOccupied], 2.0)
    mp2Density[:nOccupied, :nOccupied] += oo
    mp2Density[nOccupied:, nOccupied:] += vv

    return mp2Density
    