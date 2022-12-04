from __future__ import division
from numpy import zeros, power
from integral import eriTransform, buildEriMO, iEri, buildEriDoubleBar, expandEri
from view import postSCF
from atom import getConstant

def electronPropagator2(molBasis, c, ERI, eps, nOccupied, startOrbital = 2, nOrbitals = 4):
    #electron propagator (2)

    eV = getConstant('hartree->eV')
    threshold = 1e-4
    iterations = 20

    nBasis = len(molBasis)

    #get 2-electron integral tensor in AO basis
    eri = buildEriMO(c, ERI)
    eriMO = expandEri(eri, nBasis)

    #physists notation
    eriMO = eriTransform(eriMO)

    #get occupied and virtual span
    nVirtual = c.shape[0] - nOccupied

    if nOrbitals > nOccupied:
        nOrbitals = nOccupied

    ep2 = []
    for orbital in range(startOrbital + 1, startOrbital + nOrbitals + 1):

        e = eps[orbital]
        converged = False

        for cycle in range(0, iterations):
            ePre = e

            # Compute sigmas - first term
            sigmaA  = 0.0
            dSigmaA = 0.0
            for r in range(nOccupied, nBasis):
                for s in range(nOccupied, nBasis):
                    for a in range(0, nOccupied):
                        sigmaA  += eriMO[r,s,orbital,a] \
                                  * (2*eriMO[orbital,a,r,s]-eriMO[a,orbital,r,s]) / (e+eps[a]-eps[r]-eps[s])
                        dSigmaA += sigmaA/ (e+eps[a]-eps[r]-eps[s])

            #second term
            sigmaB = 0.0
            dSigmaB = 0.0
            for r in range(nOccupied, nBasis):
                for a in range(0, nOccupied):
                    for b in range(0, nOccupied):
                        sigmaB  += eriMO[a,b,orbital,r] \
                                  * (2*eriMO[orbital,r,a,b]-eriMO[r,orbital,a,b])/ (e+eps[r]-eps[a]-eps[b])
                        dSigmaB += sigmaB/ (e+eps[r]-eps[a]-eps[b])

            #new sigma total
            eNow = eps[orbital] + sigmaA + sigmaB

            # Break if below threshold
            if abs(eNow - ePre) < threshold:
                converged= True
                ep2.append(eNow * eV)
                break

            dSigma = -1 * (dSigmaA + dSigmaB)
            e = ePre - (ePre - eNow) / (1-dSigma)

        if not converged:
            ep2.append(eNow * eV)
            print('warn: ep2 for orbital HOMO - %d did not converged' % (nOccupied/2 - orbital/2 - 1))

    postSCF([startOrbital, ep2, eps, eV, nOccupied], 'ep')

    return ep2

def electronPropagator2spin(molBasis, c, ERI, eigenValues, nOccupied, nOrbitals = 5):
    #spin version of electron propagator 2

    eV = getConstant('hartree->eV')
    threshold = 1e-4
    iterations = 50

    nBasis = len(molBasis)
    spinOrbitals = nBasis * 2

    #compute eri in molecular spin basis
    eriMO = buildEriMO(c, ERI)
    eriMOspin = buildEriDoubleBar(spinOrbitals, eriMO)

    #recalculate occupation with respect to spin basis
    nOccupied = nOccupied * 2
    nVirtual = spinOrbitals - nOccupied

    #spin eps tensor
    eps = zeros(2*len(eigenValues))
    for p in range(0, spinOrbitals):
        eps[p] = eigenValues[p//2]

    if nOrbitals > nOccupied:
        nOrbitals = nOccupied

    ep2 = []
    for orbital in range(nOccupied-nOrbitals*2, nOccupied, 2):
        e = eps[orbital]
        converged = False

        for cycle in range(iterations):
            ePre = e

            # Compute sigmas - first term
            sigmaA = 0.0
            for r in range(nOccupied, spinOrbitals):
                for s in range(nOccupied, spinOrbitals):
                    for a in range(0, nOccupied):
                        sigmaA += 0.5 * eriMOspin[r,s,orbital,a] * eriMOspin[orbital,a,r,s] / (e+eps[a]-eps[r]-eps[s])
            #second term
            sigmaB = 0.0
            for r in range(nOccupied, spinOrbitals):
                for a in range(0, nOccupied):
                    for b in range(0, nOccupied):
                        sigmaB += 0.5 * eriMOspin[a,b,orbital,r] * eriMOspin[orbital,r,a,b] / (e+eps[r]-eps[a]-eps[b])

            #new sigma total
            eNow = eps[orbital] + sigmaA + sigmaB

            # Break if below threshold
            if abs(eNow - ePre) < threshold:
                converged= True
                ep2.append(eNow * eV)
                break

            # Build derivatives
            dSigmaA = 0.0
            for r in range(nOccupied, spinOrbitals):
                for s in range(nOccupied, spinOrbitals):
                    for a in range(0, nOccupied):
                        dSigmaA += eriMOspin[r,s,orbital,a] * eriMOspin[orbital,a,r,s] / \
                                   (power((e+eps[a]-eps[r]-eps[s]), 2))
            dSigmaB = 0.0
            for r in range(nOccupied, spinOrbitals):
                for a in range(0, nOccupied):
                    for b in range(0, nOccupied):
                        dSigmaB += eriMOspin[a,b,orbital,r] * eriMOspin[orbital,r,a,b] /  \
                                   (power((e+eps[r]-eps[a]-eps[b]), 2))

            dSigma = 1 + (dSigmaA + dSigmaB)
            e = ePre - (ePre - eNow) / dSigma

        if not converged:
            ep2.append(eNow * eV)
            print('warn: ep2 for orbital HOMO - %d did not converged' % (nOccupied - orbital/2 - 1))

    postSCF([ep2, eps, eV ], 'eps')

    return ep2


def electronPropagator3spin(molBasis, c, ERI, eigenValues, nOccupied, nOrbitals = 5):
    #electron propagator 3
    eV = getConstant('hartree->eV')
    threshold = 1e-4
    iterations = 50

    nBasis = len(molBasis)
    spinOrbitals = nBasis * 2

    #compute eri in molecular spin basis
    eriMO = buildEriMO(c, ERI)
    eriMOspin = buildEriDoubleBar(spinOrbitals, eriMO)

    # Update nOccupied and nvirt
    nOccupied = nOccupied * 2
    nVirtual = spinOrbitals - nOccupied

    #spin eps tensor
    eps = zeros(2*len(eigenValues))
    for p in range(0, spinOrbitals):
        eps[p] = eigenValues[p//2]

    if nOrbitals > nOccupied:
        nOrbitals = nOccupied

    ep2 = []
    ep3 = []
    for orbital in range(nOccupied - nOrbitals * 2, nOccupied, 2):
        e = eps[orbital]
        converged = False

        #electron propagator (2)
        for cycle in range(iterations):
            ePre = e

            # Compute sigmas - first term
            sigmaA = 0.0
            for r in range(nOccupied, spinOrbitals):
                for s in range(nOccupied, spinOrbitals):
                    for a in range(0, nOccupied):
                        sigmaA += 0.5 * eriMOspin[r,s,orbital,a] \
                                      * eriMOspin[orbital,a,r,s] / (e+eps[a]-eps[r]-eps[s])
            #second term
            sigmaB = 0.0
            for r in range(nOccupied, spinOrbitals):
                for a in range(0, nOccupied):
                    for b in range(0, nOccupied):
                        sigmaB += 0.5 * eriMOspin[a,b,orbital,r] \
                                      * eriMOspin[orbital,r,a,b] / (e+eps[r]-eps[a]-eps[b])

            #new sigma total
            eNow = eps[orbital] + sigmaA + sigmaB

            # Break if below threshold
            if abs(eNow - ePre) < threshold:
                converged= True
                ep2.append(eNow * eV)
                break

            # Build derivatives
            dSigmaA = 0.0
            for r in range(nOccupied, spinOrbitals):
                for s in range(nOccupied, spinOrbitals):
                    for a in range(0, nOccupied):
                        dSigmaA += eriMOspin[r,s,orbital,a] * eriMOspin[orbital,a,r,s] / \
                                            (power((e+eps[a]-eps[r]-eps[s]), 2))
            dSigmaB = 0.0
            for r in range(nOccupied, spinOrbitals):
                for a in range(0, nOccupied):
                    for b in range(0, nOccupied):
                        dSigmaB += eriMOspin[a,b,orbital,r] * eriMOspin[orbital,r,a,b] /  \
                                            (power((e+eps[r]-eps[a]-eps[b]), 2))

            dSigma = 1 + (dSigmaA + dSigmaB)
            e = ePre - (ePre - eNow) / dSigma

        if not converged:
            ep2.append(eNow * eV)
            print('warn: ep2 for orbital HOMO - %d did not converged' % (nOccupied/2 - orbital/2 - 1))

        #ep2 value for orbital
        orbitalEp2 = e

        #electron propagator (3)
        f = 0.0
        for a in range(0, nOccupied):
            for b in range(0, nOccupied):
                for p in range(nOccupied, spinOrbitals):
                    for q in range(nOccupied, spinOrbitals):
                        for r in range(nOccupied, spinOrbitals):
                            f += 0.5 * eriMOspin[orbital,p,orbital,a] * eriMOspin[q,r,p,b] * eriMOspin[a,b,q,r] / \
                                       ((eps[a]-eps[p])*(eps[a]+eps[b]-eps[q]-eps[r]))
        for a in range(0, nOccupied):
            for b in range(0, nOccupied):
                for c in range(0, nOccupied):
                    for p in range(nOccupied, spinOrbitals):
                        for q in range(nOccupied, spinOrbitals):
                            f -= 0.5 * eriMOspin[orbital,p,orbital,b] * eriMOspin[b,q,a,c] * eriMOspin[a,c,p,q] / \
                                       ((eps[b]-eps[p])*(eps[a]+eps[c]-eps[p]-eps[q]))
        for a in range(0, nOccupied):
            for b in range(0, nOccupied):
                for p in range(nOccupied, spinOrbitals):
                    for q in range(nOccupied, spinOrbitals):
                        for r in range(nOccupied, spinOrbitals):
                            f += 0.5 * eriMOspin[p,r,a,b] * eriMOspin[orbital,q,orbital,p] * eriMOspin[a,b,q,r] / \
                                       ((eps[a]+eps[b]-eps[p]-eps[r])*(eps[a]+eps[b]-eps[q]-eps[r]))
        for a in range(0, nOccupied):
            for b in range(0, nOccupied):
                for c in range(0, nOccupied):
                    for p in range(nOccupied, spinOrbitals):
                        for q in range(nOccupied, spinOrbitals):
                            f -= 0.5 * eriMOspin[p,q,b,c] * eriMOspin[orbital,b,orbital,a] * eriMOspin[a,c,p,q] / \
                                       ((eps[b]+eps[c]-eps[p]-eps[q])*(eps[a]+eps[c]-eps[p]-eps[q]))
        for a in range(0, nOccupied):
            for b in range(0, nOccupied):
                for p in range(nOccupied, spinOrbitals):
                    for q in range(nOccupied, spinOrbitals):
                        for r in range(nOccupied, spinOrbitals):
                            f += 0.5 * eriMOspin[p,r,a,b] * eriMOspin[q,b,p,r] * eriMOspin[orbital,a,orbital,q] / \
                                       ((eps[a]+eps[b]-eps[p]-eps[r])*(eps[a]-eps[q]))
        for a in range(0, nOccupied):
            for b in range(0, nOccupied):
                for c in range(0, nOccupied):
                    for p in range(nOccupied, spinOrbitals):
                        for q in range(nOccupied, spinOrbitals):
                            f -= 0.5 * eriMOspin[p,q,b,c] * eriMOspin[b,c,a,q] * eriMOspin[orbital,a,orbital,p] / \
                                       ((eps[b]+eps[c]-eps[p]-eps[q])*(eps[a]-eps[p]))

        #koopman-ep2 average guess for ep3 calculation
        e = (orbitalEp2 + eps[orbital]) / 2

        converged = False
        for cycle in range(iterations):
            ePre = e

            eNow = eps[orbital] + f
            de = 0

            #ep3 - quadratic eri terms
            for a in range(0, nOccupied):
                for p in range(nOccupied, spinOrbitals):
                    for q in range(nOccupied, spinOrbitals):
                        termA = 0.5 * eriMOspin[orbital,a,p,q] * eriMOspin[p,q,orbital,a]
                        termB = (e+eps[a]-eps[p]-eps[q])
                        eNow += termA/termB
                        de   += -1.0 * termA/(termB*termB)
            for a in range(0, nOccupied):
                for b in range(0, nOccupied):
                    for p in range(nOccupied, spinOrbitals):
                        termA = 0.5 * eriMOspin[orbital,p,a,b] * eriMOspin[a,b,orbital,p]
                        termB = (e+eps[p]-eps[a]-eps[b])
                        eNow += termA/termB
                        de   += -1.0 * termA/(termB*termB)

            #ep3 - cubic eri terms(1)
            for a in range(0, nOccupied):
                for p in range(nOccupied, spinOrbitals):
                    for q in range(nOccupied, spinOrbitals):
                        for r in range(nOccupied, spinOrbitals):
                            for s in range(nOccupied, spinOrbitals):
                                termA = 0.25 * eriMOspin[orbital,a,q,s] * eriMOspin[q,s,p,r] * eriMOspin[p,r,orbital,a] 
                                termB = (e+eps[a]-eps[p]-eps[r]) * (e+eps[a]-eps[q]-eps[s])
                                eNow += termA/termB
                                de += -termA/(termB * (e+eps[a]-eps[p]-eps[r])) -termA/(termB * (e+eps[a]-eps[q]-eps[s]))
            for a in range(0, nOccupied):
                for b in range(0, nOccupied):
                    for p in range(nOccupied, spinOrbitals):
                        for q in range(nOccupied, spinOrbitals):
                            for r in range(nOccupied, spinOrbitals):
                                termA = eriMOspin[orbital,a,q,r] * eriMOspin[q,b,p,a] * eriMOspin[p,r,orbital,b] 
                                termB = (e+eps[b]-eps[p]-eps[r]) * (e+eps[a]-eps[q]-eps[r])
                                eNow -= termA/termB
                                de   += termA/(termB * (e+eps[b]-eps[p]-eps[r])) + termA/(termB * (e+eps[a]-eps[q]-eps[r]))
            #(2)
            for a in range(0, nOccupied):
                for b in range(0, nOccupied):
                    for p in range(nOccupied, spinOrbitals):
                        for q in range(nOccupied, spinOrbitals):
                            for r in range(nOccupied, spinOrbitals):
                                termA = eriMOspin[orbital,r,a,q] * eriMOspin[a,b,p,r] * eriMOspin[p,q,orbital,b] 
                                termB = (e+eps[b]-eps[p]-eps[q]) * (eps[a]+eps[b]-eps[p]-eps[r])
                                eNow -= termA/termB
                                de   += termA/(termB * (e+eps[b]-eps[p]-eps[r])) 
            for a in range(0, nOccupied):
                for b in range(0, nOccupied):
                    for c in range(0, nOccupied):
                        for p in range(nOccupied, spinOrbitals):
                            for q in range(nOccupied, spinOrbitals):
                                termA = 0.25 * eriMOspin[orbital,c,a,b] * eriMOspin[a,b,p,q] * eriMOspin[p,q,orbital,c] 
                                termB = (e+eps[c]-eps[p]-eps[q]) * (eps[a]+eps[b]-eps[p]-eps[q])
                                eNow += termA/termB
                                de   -= termA/(termB * (e+eps[c]-eps[p]-eps[q])) 
            for a in range(0, nOccupied):
                for b in range(0, nOccupied):
                    for p in range(nOccupied, spinOrbitals):
                        for q in range(nOccupied, spinOrbitals):
                            for r in range(nOccupied, spinOrbitals):
                                termA = eriMOspin[orbital,b,p,r] * eriMOspin[p,q,a,b] * eriMOspin[a,r,orbital,q] 
                                termB = (e+eps[b]-eps[p]-eps[r]) * (eps[a]+eps[b]-eps[p]-eps[q])
                                eNow -= termA/termB
                                de   += termA/(termB * (e+eps[c]-eps[p]-eps[q])) 
            for a in range(0, nOccupied):
                for b in range(0, nOccupied):
                    for c in range(0, nOccupied):
                        for p in range(nOccupied, spinOrbitals):
                            for q in range(nOccupied, spinOrbitals):
                                termA = 0.25 * eriMOspin[orbital,b,p,q] * eriMOspin[p,q,a,c] * eriMOspin[a,c,orbital,b] 
                                termB = (e+eps[b]-eps[p]-eps[q]) * (eps[a]+eps[c]-eps[p]-eps[q])
                                eNow += termA/termB
                                de   -= termA/(termB * (e+eps[c]-eps[p]-eps[q])) 

            #(3)
            for a in range(0, nOccupied):
                for b in range(0, nOccupied):
                    for p in range(nOccupied, spinOrbitals):
                        for q in range(nOccupied, spinOrbitals):
                            for r in range(nOccupied, spinOrbitals):
                                termA = 0.25 * eriMOspin[orbital,q,a,b] * eriMOspin[a,b,p,r] * eriMOspin[p,r,orbital,q]
                                termB = (e+eps[q]-eps[a]-eps[b]) * (eps[p]+eps[r]-eps[a]-eps[b])
                                eNow -= termA/termB
                                de   += termA/(termB *  (e+eps[q]-eps[a]-eps[b]))
            for a in range(0, nOccupied):
                for b in range(0, nOccupied):
                    for c in range(0, nOccupied):
                        for p in range(nOccupied, spinOrbitals):
                            for q in range(nOccupied, spinOrbitals):
                                termA = eriMOspin[orbital,q,a,c] * eriMOspin[a,b,p,q] * eriMOspin[p,c,orbital,b]
                                termB = (e+eps[q]-eps[a]-eps[c]) * (eps[p]+eps[q]-eps[a]-eps[b])
                                eNow += termA/termB
                                de   -= termA/(termB *  (e+eps[q]-eps[a]-eps[c]))
            for a in range(0, nOccupied):
                for b in range(0, nOccupied):
                    for p in range(nOccupied, spinOrbitals):
                        for q in range(nOccupied, spinOrbitals):
                            for r in range(nOccupied, spinOrbitals):
                                termA = 0.25 * eriMOspin[orbital,r,p,q] * eriMOspin[p,q,a,b] * eriMOspin[a,b,orbital,r]
                                termB = (e+eps[r]-eps[a]-eps[b]) * (eps[p]+eps[q]-eps[a]-eps[b])
                                eNow -= termA/termB
                                de   += termA/(termB *  (e+eps[r]-eps[a]-eps[b]))
            for a in range(0, nOccupied):
                for b in range(0, nOccupied):
                    for c in range(0, nOccupied):
                        for p in range(nOccupied, spinOrbitals):
                            for q in range(nOccupied, spinOrbitals):
                                termA = eriMOspin[orbital,c,p,b] * eriMOspin[p,q,a,c] * eriMOspin[a,b,orbital,q]
                                termB = (e+eps[q]-eps[a]-eps[b]) * (eps[p]+eps[q]-eps[a]-eps[c])
                                eNow += termA/termB
                                de   -= termA/(termB *  (e+eps[q]-eps[a]-eps[b]))
            #(4)
            for a in range(0, nOccupied):
                for b in range(0, nOccupied):
                    for c in range(0, nOccupied):
                        for p in range(nOccupied, spinOrbitals):
                            for q in range(nOccupied, spinOrbitals):
                                termA = eriMOspin[orbital,p,b,c] * eriMOspin[b,q,a,p] * eriMOspin[a,c,orbital,q]
                                termB = (e+eps[p]-eps[b]-eps[c]) * (e+eps[q]-eps[a]-eps[c])
                                eNow += termA/termB
                                de   -= termA/(termB *  (e+eps[p]-eps[b]-eps[c]))
                                de   -= termA/(termB *  (e+eps[q]-eps[a]-eps[c]))
            for a in range(0, nOccupied):
                for b in range(0, nOccupied):
                    for c in range(0, nOccupied):
                        for d in range(0, nOccupied):
                            for p in range(nOccupied, spinOrbitals):
                                termA = 0.25 * eriMOspin[orbital,p,b,d] * eriMOspin[b,d,a,c] * eriMOspin[a,c,orbital,p]
                                termB = (e+eps[p]-eps[b]-eps[d]) * (e+eps[p]-eps[a]-eps[c])
                                eNow -= termA/termB
                                de   += termA/(termB *  (e+eps[p]-eps[b]-eps[d]))
                                de   += termA/(termB *  (e+eps[p]-eps[a]-eps[c]))

            # Break if below threshold
            if abs(eNow - ePre) < 1.e-4:
                ep3.append(eNow * 27.21138505)
                converged = True
                break

            # Newton-Raphson update
            e = ePre - (ePre - eNow) / (1 - de)

        if converged is False:
            ep3.append(e)
            print('warn: ep3 for orbital HOMO - %d did not converged' % (nOccupied/2 - orbital/2 - 1))

    postSCF([ep3, eps, eV ], 'ep3')

    return ep3

def koopmanAGFcorrection(molBasis ,c ,ERI, eigenValues, nOccupied, nOrbitals = 5):
    #Approximate Green's function approximation correction to koopman IP

    eV = getConstant('hartree->eV')
    threshold = 1e-4
    iterations = 50

    nBasis = len(molBasis)
    spinOrbitals = nBasis * 2

    #compute eri in molecular spin basis
    eriMO = buildEriMO(c, ERI)
    eriMOspin = buildEriDoubleBar(spinOrbitals, eriMO)

    #recalculate occupation with respect to spin basis
    nOccupied = nOccupied * 2
    nVirtual = spinOrbitals - nOccupied

    #spin eps tensor
    eps = zeros(2*len(eigenValues))
    for p in range(0, spinOrbitals):
        eps[p] = eigenValues[p//2]

    if nOrbitals > nOccupied:
        nOrbitals = nOccupied

    agf = []
    #starts at nOrbitals below HOMO
    cycleControl = -1
    for orbital in range(nOccupied-nOrbitals*2, nOccupied, 2):

        #orbital relaxation component
        orx = 0.0
        for a in range(0, nOccupied):
            for r in range(nOccupied, spinOrbitals):
                orx += power(eriMOspin[a,orbital,orbital,r],2) / (eps[r]-eps[a])

        #pair relaxation component
        prx = 0.0
        for a in range(0, nOccupied):
            for b in range(0, nOccupied):
                for r in range(nOccupied, spinOrbitals):
                    if (a != orbital) and (b != orbital):
                        prx += 0.5 * power(eriMOspin[a,b,orbital,r],2) / (eps[orbital]+eps[r]-eps[a]-eps[b])

        #pair removal component
        prm = 0.0
        for a in range(0, nOccupied):
            for r in range(nOccupied, spinOrbitals):
                for s in range(nOccupied, spinOrbitals):
                    prm += 0.5 * power(eriMOspin[r,s,orbital,a],2) / (eps[a]+eps[orbital]-eps[r]-eps[s])
         
        deltaIP = -orx - prx + prm

        #IP -ve of orbital energy
        koopman = -eps[orbital]

        #send results to postSCF for output - cycleControl [-1] first,[0] :, [1]  last
        if (orbital + 2 == nOccupied):
            cycleControl = 1

        postSCF([cycleControl, orbital/2, koopman*eV, orx*eV ,prx*eV, prm*eV, deltaIP*eV, (koopman+deltaIP)*eV],'gfa')
        agf.append([koopman*eV, deltaIP*eV])
        cycleControl = 0

    return agf