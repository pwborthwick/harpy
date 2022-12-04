from __future__ import division
import numpy as np
from integral import buildFockMOspin, buildEriMO, buildEriDoubleBar
from basis import electronCount
from math import sqrt
from view import postSCF
from diis import diis_c

#'** references to Stanton et al
#'** J.Chem.Phys 94(6), 15 March 1991

def tauSpin(i, j, a, b, ts, td):
    #equation (9)
    return td[a,b,i,j] + 0.5 * (ts[a,i] * ts[b,j] - ts[b,i] * ts[a,j])

def tau(i, j, a, b, ts, td):
    #equation (10)
    return td[a,b,i,j] + ts[a,i] * ts[b,j] - ts[b,i] * ts[a,j]

def updateIntermediates(fs, ts, td, eriMOspin, nElectrons):
    #fs - fockSpin  ts - tsingle cluster operator   td - tdouble cluster operator
    #eriMOspin - electron repulsion integrals in the MO spin

    def amplitudesT1():
        # equation (1)
        TS = np.zeros((spinOrbitals, spinOrbitals))

        for i in range(0, nElectrons):
            for a in range(nElectrons, spinOrbitals):
                TS[a,i] += fs[a,i]
                for e in range(nElectrons, spinOrbitals):
                    TS[a,i] += ts[e,i] * fu[e,a]
                for m in range(0, nElectrons):
                    TS[a,i] -= ts[a,m] * fo[i,m]
                    for e in range(nElectrons, spinOrbitals):
                        TS[a,i] += td[a,e,i,m] * fm[e,m]
                        for f in range(nElectrons, spinOrbitals):
                            TS[a,i] -= 0.5 * td[e,f,i,m] * eriMOspin[m,a,e,f]
                        for n in range(0, nElectrons):
                            TS[a,i] -= 0.5 * td[a,e,m,n] * eriMOspin[n,m,e,i]

                for n in range(0, nElectrons):
                    for f in range(nElectrons, spinOrbitals):
                        TS[a,i] -= ts[f,n] * eriMOspin[n,a,i,f]
                TS[a,i] /= (fs[i,i] - fs[a,a])

        return TS

    def amplitudesT2():
        # equation (2)
        TD = np.zeros((spinOrbitals, spinOrbitals, spinOrbitals, spinOrbitals))

        for i in range(0, nElectrons):
            for j in range(0, nElectrons):
                for a in range(nElectrons, spinOrbitals):
                    for b in range(nElectrons, spinOrbitals):
                        TD[a,b,i,j] +=eriMOspin[i,j,a,b]
                        for e in range(nElectrons, spinOrbitals):
                            TD[a,b,i,j] += td[a,e,i,j] * fu[e,b] - td[b,e,i,j] * fu[e,a]
                            for m in range(0, nElectrons):
                                TD[a,b,i,j] += -0.5 * td[a,e,i,j] * ts[b,m] * fm[e,m] + 0.5 * td[b,e,i,j] * ts[a,m] * fm[e,m]
                            for f in range(nElectrons, spinOrbitals):
                                TD[a,b,i,j] += 0.5 * wu[e,f,a,b] * tau(i,j,e,f,ts,td)
                        for m in range(0, nElectrons):
                            TD[a,b,i,j] -= td[a,b,i,m] * fo[j,m] - td[a,b,j,m] * fo[i,m]
                            for e in range(nElectrons, spinOrbitals):
                                TD[a,b,i,j] += -0.5 * td[a,b,i,m] * ts[e,j] * fm[e,m] + 0.5 * td[a,b,j,m] * ts[e,i] * fm[e,m]
                            for n in range(0, nElectrons):
                                TD[a,b,i,j] += 0.5 * tau(m,n,a,b,ts,td) * wo[i,j,m,n]
                        for m in range(0, nElectrons):
                            for e in range(nElectrons, spinOrbitals):
                                TD[a,b,i,j] += td[a,e,i,m] * wm[e,j,m,b] - ts[e,i] * ts[a,m] * eriMOspin[m,b,e,j]
                                TD[a,b,i,j] -= td[a,e,j,m] * wm[e,i,m,b] - ts[e,j] * ts[a,m] * eriMOspin[m,b,e,i]
                                TD[a,b,i,j] -= td[b,e,i,m] * wm[e,j,m,a] - ts[e,i] * ts[b,m] * eriMOspin[m,a,e,j]
                                TD[a,b,i,j] += td[b,e,j,m] * wm[e,i,m,a] - ts[e,j] * ts[b,m] * eriMOspin[m,a,e,i]
                        for e in range(nElectrons, spinOrbitals):
                            TD[a,b,i,j] += eriMOspin[a,b,e,j] * ts[e,i]
                            TD[a,b,i,j] -= eriMOspin[a,b,e,i] * ts[e,j]
                        for m in range(0, nElectrons):
                            TD[a,b,i,j] -= eriMOspin[m,b,i,j] * ts[a,m]
                            TD[a,b,i,j] += eriMOspin[m,a,i,j] * ts[b,m]
                        TD[a,b,i,j] /= (fs[i,i] + fs[j,j] - fs[a,a] - fs[b,b])

        return TD

    spinOrbitals = fs.shape[0]

    fo = np.zeros((spinOrbitals, spinOrbitals))
    fu = np.zeros((spinOrbitals, spinOrbitals))
    fm = np.zeros((spinOrbitals, spinOrbitals))

    wo = np.zeros((spinOrbitals, spinOrbitals, spinOrbitals, spinOrbitals))
    wu = np.zeros((spinOrbitals, spinOrbitals, spinOrbitals, spinOrbitals))
    wm = np.zeros((spinOrbitals, spinOrbitals, spinOrbitals, spinOrbitals))

    #equation (3)
    for a in range(nElectrons, spinOrbitals):
        for e in range(nElectrons, spinOrbitals):
            fu[e,a] += (1 - (a == e)) * fs[e,a]
            for m in range(0, nElectrons):
                fu[e,a] -= 0.5 * fs[e,m] * ts[a,m]
                for f in range(nElectrons, spinOrbitals):
                    fu[e,a] += ts[f,m] * eriMOspin[m,a,f,e]
                    for n in range(0, nElectrons):
                        fu[e,a] -= 0.5 * tauSpin(m,n,a,f,ts,td) * eriMOspin[m,n,e,f]
    #equation (4)
    for m in range(0, nElectrons):
        for i in range(0, nElectrons):
            fo[i,m] += (1 - (m == i)) * fs[i,m]
            for e in range(nElectrons, spinOrbitals):
                fo[i,m] += 0.5 * fs[e,m] * ts[e,i]
                for n in range(0, nElectrons):
                    fo[i,m] += ts[e,n] * eriMOspin[m,n,i,e]
                    for f in range(nElectrons, spinOrbitals):
                        fo[i,m] += 0.5 * tauSpin(i,n,e,f,ts,td) * eriMOspin[m,n,e,f]
    #equation (5)
    for m in range(0, nElectrons):
        for e in range(nElectrons, spinOrbitals):
            fm[e,m] += fs[e,m]
            for n in range(0, nElectrons):
                for f in range(nElectrons, spinOrbitals):
                    fm[e,m] += ts[f,n] * eriMOspin[m,n,e,f]
    #equation (6)
    for m in range(0, nElectrons):
        for n in range(0, nElectrons):
            for i in range(0, nElectrons):
                for j in range(0, nElectrons):
                    wo[i,j,m,n] +=eriMOspin[m,n,i,j]
                    for e in range(nElectrons, spinOrbitals):
                        wo[i,j,m,n] += ts[e,j] * eriMOspin[m,n,i,e] - ts[e,i] * eriMOspin[m,n,j,e]
                        for f in range(nElectrons, spinOrbitals):
                            wo[i,j,m,n] += 0.25 * tau(i,j,e,f,ts,td) * eriMOspin[m,n,e,f]
    #equation (7)
    for a in range(nElectrons, spinOrbitals):
        for b in range(nElectrons, spinOrbitals):
            for e in range(nElectrons, spinOrbitals):
                for f in range(nElectrons, spinOrbitals):
                    wu[e,f,a,b] += eriMOspin[a,b,e,f]
                    for m in range(0, nElectrons):
                        wu[e,f,a,b] += -ts[b,m] * eriMOspin[a,m,e,f] + ts[a,m] * eriMOspin[b,m,e,f]
                        for n in range(0, nElectrons):
                            wu[e,f,a,b] += 0.25 * tau(m,n,a,b,ts,td) * eriMOspin[m,n,e,f]
    #equation (8)
    for m in range(0, nElectrons):
        for b in range(nElectrons, spinOrbitals):
            for e in range(nElectrons, spinOrbitals):
                for j in range(0, nElectrons):
                    wm[e,j,m,b] += eriMOspin[m,b,e,j]
                    for f in range(nElectrons, spinOrbitals):
                        wm[e,j,m,b] += ts[f,j] * eriMOspin[m,b,e,f]
                    for n in range(0, nElectrons):
                        wm[e,j,m,b] -= ts[b,n] * eriMOspin[m,n,e,j]
                        for f in range(nElectrons, spinOrbitals):
                            wm[e,j,m,b] -= (0.5 * td[f,b,j,n] + ts[f,j] * ts[b,n]) * eriMOspin[m,n,e,f]

    ts = amplitudesT1()
    td = amplitudesT2()

    return ts, td

def amplitudesT3(fs, ts, td, eriMOspin, nElectrons):
    # perturbative triples correction

    spinOrbitals = fs.shape[0]
    tttd = 0.0
    et = 0.0

    for i in range(0, nElectrons):
        for j in range(0, nElectrons):
            for k in range(0, nElectrons):

                for a in range(nElectrons, spinOrbitals):
                    for b in range(nElectrons, spinOrbitals):
                        for c in range(nElectrons, spinOrbitals):

                            fockDenominator = fs[i,i] + fs[j,j] + fs[k,k] - fs[a,a] - fs[b,b] - fs[c,c]
                            
                            tttd = (ts[a,i] * eriMOspin[j,k,b,c] - ts[a,j] * eriMOspin[i,k,b,c] - ts[a,k] * eriMOspin[j,i,b,c] - \
                                    ts[b,i] * eriMOspin[j,k,a,c] + ts[b,j] * eriMOspin[i,k,a,c] + ts[b,k] * eriMOspin[j,i,a,c] - \
                                    ts[c,i] * eriMOspin[j,k,b,a] + ts[c,j] * eriMOspin[i,k,b,a] + ts[c,k] * eriMOspin[j,i,b,a])
                            tttd /= fockDenominator

                            tttc = 0.0
                            for e in range(nElectrons, spinOrbitals):
                                tttc += (td[a,e,j,k] * eriMOspin[e,i,b,c] - td[a,e,i,k] * eriMOspin[e,j,b,c] - td[a,e,j,i] * eriMOspin[e,k,b,c] \
                                       - td[b,e,j,k] * eriMOspin[e,i,a,c] + td[b,e,i,k] * eriMOspin[e,j,a,c] + td[b,e,j,i] * eriMOspin[e,k,a,c] \
                                       - td[c,e,j,k] * eriMOspin[e,i,b,a] + td[c,e,i,k] * eriMOspin[e,j,b,a] + td[c,e,j,i] * eriMOspin[e,k,b,a])

                            for m in range(0, nElectrons):
                                tttc -= (td[b,c,i,m] * eriMOspin[m,a,j,k] - td[b,c,j,m] * eriMOspin[m,a,i,k] - td[b,c,k,m] * eriMOspin[m,a,j,i] \
                                       - td[a,c,i,m] * eriMOspin[m,b,j,k] + td[a,c,j,m] * eriMOspin[m,b,i,k] + td[a,c,k,m] * eriMOspin[m,b,j,i] \
                                       - td[b,a,i,m] * eriMOspin[m,c,j,k] + td[b,a,j,m] * eriMOspin[m,c,i,k] + td[b,a,k,m] * eriMOspin[m,c,j,i])
                            tttc /= fockDenominator

                            et += tttc * (tttc + tttd) * fockDenominator / 36.0

    return et

def ccsd(atoms, eigenVectors, charge, fock, ERI, iterations, convergence, SCFenergy, diisStatus):
    #compute the charged coupled single and doubles

    spinOrbitals = (fock.shape[0]) * 2
    nElectrons = electronCount(atoms, charge)

    #get fock in MO spin basis
    fockMOspin = buildFockMOspin(spinOrbitals, eigenVectors, fock)

    #get two-electron repulsion integrals in MO basis
    eriMO = buildEriMO(eigenVectors, ERI)
 
    #transform eri from MO to spin basis
    eriMOspin = buildEriDoubleBar(spinOrbitals, eriMO)

    #amplitude array for singles and doubles
    ts = np.zeros((spinOrbitals, spinOrbitals))
    td = np.zeros((spinOrbitals, spinOrbitals, spinOrbitals, spinOrbitals))

    #get mp2
    mp2 = 0.0
    for i in range(0, nElectrons):
        for j in range(0, nElectrons):
            for a in range(nElectrons, spinOrbitals):
                for b in range(nElectrons, spinOrbitals):
                    Dijab = fockMOspin[i,i] + fockMOspin[j,j] - fockMOspin[a,a] - fockMOspin[b,b]
                    td[a,b,i,j] = eriMOspin[i,j,a,b] / Dijab
                    mp2 += 0.25 * eriMOspin[i,j,a,b] * td[a,b,i,j]

    #diis variables
    if diisStatus == 'on':
        diisCapacity = 6
        diis = diis_c(diisCapacity, [ts, td])

    #start the convergence iterations
    energy = 0.0
    rmsAmplitudes = 0.0
    for cycle in range(1 , iterations):
        
        diis.refresh_store([ts, td])
        ts,td = updateIntermediates(fockMOspin, ts, td, eriMOspin, nElectrons)

        energy = ccsdEnergy(fockMOspin, ts, td, eriMOspin, nElectrons)

        #test convergence
        rms = max(sqrt(np.sum(ts*ts)) ,sqrt(np.sum(td*td)))
        if cycle != 1:
            deltaEnergy = abs(preEnergy - energy)
            deltaAmplitudes = abs(rmsAmplitudes - rms)
            postSCF([energy, deltaEnergy, deltaAmplitudes, cycle, diisStatus], 'diis-sd')

            if (deltaEnergy < convergence) and (deltaAmplitudes < convergence):
                break

        preEnergy = energy
        rmsAmplitudes = rms
        #diis
        if diisStatus == 'on':
            ts, td = diis.build([ts, td])

    postSCF([cycle,energy, amplitudesT3(fockMOspin, ts, td, eriMOspin, nElectrons), mp2, SCFenergy], 'ccsd(t)')

    return energy, ts, td

def ccsdEnergy(fs, ts, td, eriMOspin, nElectrons):
    #compute the ccsd energy
    energy = 0.0
    mp2 = 0
    spinOrbitals = fs.shape[0]

    for i in range(0, nElectrons):
        for a in range(nElectrons, spinOrbitals):
            energy += fs[i,a] * ts[i,a]
            for j in range(0, nElectrons):
                for b in range(nElectrons, spinOrbitals):
                    mp2 =  0.25 * eriMOspin[i,j,a,b] * td[a,b,i,j]
                    energy += mp2 + 0.5 * eriMOspin[i,j,a,b] * ts[a,i] * ts[b,j]
    return energy

def lccd(atoms, eigenVectors, charge, fock, ERI, iterations, convergence, SCFenergy):
    #coupled electron pairs approximation zero order

    spinOrbitals = (fock.shape[0]) * 2
    nElectrons = electronCount(atoms, charge)

    #get fock in MO spin basis
    fockMOspin = buildFockMOspin(spinOrbitals, eigenVectors, fock)

    #get two-electron repulsion integrals in MO basis
    eriMO = buildEriMO(eigenVectors, ERI)

    #transform eri from MO to spin basis
    eriMOspin = buildEriDoubleBar(spinOrbitals, eriMO)

    #amplitude array for singles and doubles
    td = np.zeros((spinOrbitals, spinOrbitals, spinOrbitals, spinOrbitals))

    #start the convergence iterations
    energy = 0.0
    rmsAmplitudes = 0.0
    diisStatus = 'off'
    
    for cycle in range(1 , iterations):

        td = lccdAmplitudes(td, fockMOspin, eriMOspin, nElectrons, spinOrbitals)

        energy = ccdEnergy(td, eriMOspin, nElectrons, spinOrbitals)

        rms = sqrt(np.sum(td*td))
        if cycle != 1:
            deltaEnergy = abs(preEnergy - energy)
            deltaAmplitudes = abs(rmsAmplitudes - rms)
            postSCF([energy, deltaEnergy, deltaAmplitudes, cycle, diisStatus], 'diis-ld')

            if (deltaEnergy < convergence) and (deltaAmplitudes < convergence):
                break

        preEnergy = energy
        rmsAmplitudes = rms


    postSCF([cycle, energy, SCFenergy], 'lccd')

    return energy
                        

def lccdAmplitudes(td, fs, eriMOspin, nElectrons, spinOrbitals):
    #compute coupled electron pair approximation

    TD = np.zeros((spinOrbitals, spinOrbitals, spinOrbitals, spinOrbitals))
    for i in range(0, nElectrons):
        for j in range(0, nElectrons):
            for a in range(nElectrons, spinOrbitals):
                for b in range(nElectrons, spinOrbitals):

                    TD[a,b,i,j] = eriMOspin[i,j,a,b]                                               #[D1]

                    for e in range(nElectrons, spinOrbitals):
                        TD[a,b,i,j] += fs[e,b] * (1 - (b == e)) * td[a,e,i,j] - \
                                       fs[e,a] * (1 - (a == e)) * td[b,e,i,j]                      #[D2a]

                        for f in range(nElectrons, spinOrbitals):
                            TD[a,b,i,j] += 0.5 * eriMOspin[e,f,a,b] * td[e,f,i,j]                  #[D2c]


                    for m in range(0, nElectrons):
                        TD[a,b,i,j] += -fs[j,m] * (1 - (m == j)) *td[a,b,i,m] + \
                                        fs[i,m] * (1 - (m == i)) *td[a,b,i,m]                      #[D2b]

                        for n in range(0, nElectrons):
                            TD[a,b,i,j] += 0.5 * eriMOspin[i,j,m,n] * td[a,b,m,n]                  #[D2d]

                    for m in range(0, nElectrons):
                        for e in range(nElectrons, spinOrbitals):
                            TD[a,b,i,j] += eriMOspin[e,j,m,b] * td[a,e,i,m] - \
                                           eriMOspin[e,i,m,b] * td[a,e,j,m] - \
                                           eriMOspin[e,j,m,a] * td[b,e,i,m] + \
                                           eriMOspin[e,i,m,a] * td[b,e,j,m]                        #[D2e]
                        
                    fockDenominator = fs[i,i] + fs[j,j]  - fs[a,a] - fs[b,b] 

                    TD[a,b,i,j] /= fockDenominator

    return TD

def ccdEnergy( td, eriMOspin, nElectrons, spinOrbitals):

    energy = 0.0
    for i in range(0, nElectrons):
        for j in range(0, nElectrons):
            for a in range(nElectrons, spinOrbitals):
                for b in range(nElectrons, spinOrbitals):
                    energy +=  0.25 * eriMOspin[a,b,i,j] * td[a,b,i,j]

    return energy

def ccd(atoms, eigenVectors, charge, fock, ERI, iterations, convergence, SCFenergy):
    #coupled cluster doubles 

    spinOrbitals = (fock.shape[0]) * 2
    nElectrons = electronCount(atoms, charge)

    #get fock in MO spin basis
    fockMOspin = buildFockMOspin(spinOrbitals, eigenVectors, fock)

    #get two-electron repulsion integrals in MO basis
    eriMO = buildEriMO(eigenVectors, ERI)

    #transform eri from MO to spin basis
    eriMOspin = buildEriDoubleBar(spinOrbitals, eriMO)

    #amplitude array for singles and doubles
    td = np.zeros((spinOrbitals, spinOrbitals, spinOrbitals, spinOrbitals))

    #start the convergence iterations
    energy = 0.0
    rmsAmplitudes = 0.0
    diisStatus = 'off'

    for cycle in range(1 , iterations):
        
        td = ccdAmplitudes(td, fockMOspin, eriMOspin, nElectrons, spinOrbitals)

        #can reuse cepaEnergy as algorithm the same
        energy = ccdEnergy(td, eriMOspin, nElectrons, spinOrbitals)

        rms = sqrt(np.sum(td*td))
        if cycle != 1:
            deltaEnergy = abs(preEnergy - energy)
            deltaAmplitudes = abs(rmsAmplitudes - rms)
            postSCF([energy, deltaEnergy, deltaAmplitudes, cycle, diisStatus], 'diis-sd')

            if (deltaEnergy < convergence) and (deltaAmplitudes < convergence):
                break

        preEnergy = energy
        rmsAmplitudes = rms
        
    postSCF([cycle, energy, SCFenergy], 'ccd')
    
    return energy
                        

def ccdAmplitudes(td, fs, eriMOspin, nElectrons, spinOrbitals):
    #compute coupled electron pair approximation

    TD = np.zeros((spinOrbitals, spinOrbitals, spinOrbitals, spinOrbitals))
    for i in range(0, nElectrons):
        for j in range(0, nElectrons):
            for a in range(nElectrons, spinOrbitals):
                for b in range(nElectrons, spinOrbitals):

                    #linear
                    TD[a,b,i,j] = eriMOspin[i,j,a,b]                                                                     #[D1]

                    for e in range(nElectrons, spinOrbitals):
                        TD[a,b,i,j] += fs[e,b] * (1 - (b == e)) * td[a,e,i,j] - \
                                       fs[e,a] * (1 - (a == e)) * td[b,e,i,j]                                            #[D2a]

                        for f in range(nElectrons, spinOrbitals):
                            TD[a,b,i,j] += 0.5 * eriMOspin[e,f,a,b] * td[e,f,i,j]                                        #[D2c]


                    for m in range(0, nElectrons):
                        TD[a,b,i,j] += -fs[j,m] * (1 - (m == j)) *td[a,b,i,m] + \
                                        fs[i,m] * (1 - (m == i)) *td[a,b,i,m]                                            #[D2b]

                        for n in range(0, nElectrons):
                            TD[a,b,i,j] += 0.5 * eriMOspin[i,j,m,n] * td[a,b,m,n]                                        #[D2d]

                    for m in range(0, nElectrons):
                        for e in range(nElectrons, spinOrbitals):
                            TD[a,b,i,j] += eriMOspin[e,j,m,b] * td[a,e,i,m] - \
                                           eriMOspin[e,i,m,b] * td[a,e,j,m] - \
                                           eriMOspin[e,j,m,a] * td[b,e,i,m] + \
                                           eriMOspin[e,i,m,a] * td[b,e,j,m]                                              #[D2e]
   
                    #quadratic
                    for m in range(0, nElectrons):
                        for n in range(0, nElectrons):
                            for e in range(nElectrons, spinOrbitals):
                                for f in range(nElectrons, spinOrbitals):
                                    TD[a,b,i,j] += -0.5 * eriMOspin[e,f,m,n] * td[f,b,i,j] * td[a,e,n,m] + \
                                                    0.5 * eriMOspin[e,f,m,n] * td[f,a,i,j] * td[b,e,n,m]                 #[D3d]

                                    TD[a,b,i,j] += -0.5 * eriMOspin[e,f,m,n] * td[a,b,n,j] * td[f,e,i,m] + \
                                                    0.5 * eriMOspin[e,f,m,n] * td[a,b,n,i] * td[f,e,j,m]                 #[D3c]

                                    TD[a,b,i,j] += 0.25 * eriMOspin[e,f,m,n] * td[e,f,i,j] * td[a,b,m,n]                 #[D3a]

                                    TD[a,b,i,j] += eriMOspin[e,f,m,n] * td[a,e,i,m] * td[b,f,j,n] - \
                                                   eriMOspin[e,f,m,n] * td[a,e,j,m] * td[b,f,i,n]                        #[D3a]

                        

                    fockDenominator = fs[i,i] + fs[j,j]  - fs[a,a] - fs[b,b] 

                    TD[a,b,i,j] /= fockDenominator

    return TD

def ccsd_lambda(fs, eriMOspin, ts, td, nElectrons, iterations, tolerance):
    #CCSD lambda equations - Gauss and Stanton J. Chem. Phys.103(9), 1 September 1995

    spinOrbitals = fs.shape[0]

    #initial guess from lambda amplitudes
    ls = ts * 0.0
    ld = td.copy()
    ld.transpose(2,3,0,1)

    def tauSpin(i, j, a, b, ts, td):
        # table III(d)
        return td[a,b,i,j] + 0.5 * (ts[a,i] * ts[b,j] - ts[b,i] * ts[a,j])

    def tau(i, j, a, b, ts, td):
        #table III(d)
        return td[a,b,i,j] + ts[a,i] * ts[b,j] - ts[b,i] * ts[a,j]

    def buildLambdaIntermediates(type = ''):
        #fs - fockSpin  ts - tsingle cluster operator   td - tdouble cluster operator
        #eriMOspin - electron repulsion integrals in the MO spin



        fi = np.zeros((spinOrbitals, spinOrbitals))
        wi = np.zeros((spinOrbitals, spinOrbitals, spinOrbitals, spinOrbitals))

        if type == 'ae':
            #table III(a)[1] (b)[1]                                                                                      F[ae] 
            fi[:,:] = 0.0 
            for a in range(nElectrons, spinOrbitals):
                for e in range(nElectrons, spinOrbitals):
                    fi[a,e] += (1 - (a == e)) * fs[a,e]
                    for m in range(0, nElectrons):
                        fi[a,e] -= fs[e,m] * ts[a,m]
                        for f in range(nElectrons, spinOrbitals):
                            fi[a,e] += ts[f,m] * eriMOspin[a,m,e,f]
                            for n in range(0, nElectrons):
                                fi[a,e] -= 0.5 * tauSpin(m,n,a,f,ts,td) * eriMOspin[m,n,e,f]
                                fi[a,e] -= 0.5 * ts[a,m] * ts[f,n] * eriMOspin[m,n,e,f]
            return fi

        if type == 'mi':
            #table III(a)[2] (b)[2]                                                                                      F[mi]
            fi[:,:] = 0.0
            for m in range(0, nElectrons):
                for i in range(0, nElectrons):
                    fi[m,i] += (1 - (m == i)) * fs[m,i]
                    for e in range(nElectrons, spinOrbitals):
                        fi[m,i] += fs[e,m] * ts[e,i]
                        for n in range(0, nElectrons):
                            fi[m,i] += ts[e,n] * eriMOspin[m,n,i,e]
                            for f in range(nElectrons, spinOrbitals):
                                fi[m,i] += 0.5 * tauSpin(i,n,e,f,ts,td) * eriMOspin[m,n,e,f]
                                fi[m,i] += 0.5 * ts[e,i] * ts[f,n] * eriMOspin[m,n,e,f]
            return fi

        if type == 'me':
            #table III(a)[3] (b)[3]                                                                                      F[me]
            fi[:,:] = 0.0
            for m in range(0, nElectrons):
                for e in range(nElectrons, spinOrbitals):
                    fi[m,e] += fs[m,e]
                    for n in range(0, nElectrons):
                        for f in range(nElectrons, spinOrbitals):
                            fi[m,e] += ts[f,n] * eriMOspin[m,n,e,f]
            return fi

        if type == 'mnij':
            #table III(a)[4] (b)[4]                                                                                      W[mnij]
            wi[:,:,:,:] = 0.0                    
            for m in range(0, nElectrons):
                for n in range(0, nElectrons):
                    for i in range(0, nElectrons):
                        for j in range(0, nElectrons):
                            wi[m,n,i,j] +=eriMOspin[m,n,i,j]
                            for e in range(nElectrons, spinOrbitals):
                                wi[m,n,i,j] += ts[e,j] * eriMOspin[m,n,i,e] - ts[e,i] * eriMOspin[m,n,j,e]
                                for f in range(nElectrons, spinOrbitals):
                                    wi[m,n,i,j] += 0.5 * tau(i,j,e,f,ts,td) * eriMOspin[m,n,e,f]
            return wi

        if type == 'abef':
            #table III(a)[5] (b)[5]                                                                                      W[abef]
            wi[:,:,:,:] = 0.0
            for a in range(nElectrons, spinOrbitals):
                for b in range(nElectrons, spinOrbitals):
                    for e in range(nElectrons, spinOrbitals):
                        for f in range(nElectrons, spinOrbitals):
                            wi[a,b,e,f] += eriMOspin[a,b,e,f]
                            for m in range(0, nElectrons):
                                wi[a,b,e,f] += -ts[b,m] * eriMOspin[a,m,e,f] + ts[a,m] * eriMOspin[b,m,e,f]
                                for n in range(0, nElectrons):
                                    wi[a,b,e,f] += 0.5 * tau(m,n,a,b,ts,td) * eriMOspin[m,n,e,f]
            return wi

        if type == 'mbej':
            #table III(a)[6] (b)[6]                                                                                      W[mbej]
            wi[:,:,:,:] = 0.0
            for m in range(0, nElectrons):
                for b in range(nElectrons, spinOrbitals):
                    for e in range(nElectrons, spinOrbitals):
                        for j in range(0, nElectrons):
                            wi[m,b,e,j] += eriMOspin[m,b,e,j]
                            for f in range(nElectrons, spinOrbitals):
                                wi[m,b,e,j] += ts[f,j] * eriMOspin[m,b,e,f]
                            for n in range(0, nElectrons):
                                wi[m,b,e,j] -= ts[b,n] * eriMOspin[m,n,e,j]
                                for f in range(nElectrons, spinOrbitals):
                                    wi[m,b,e,j] -= (td[f,b,j,n] + ts[f,j] * ts[b,n]) * eriMOspin[m,n,e,f]
            return wi

        if type == 'abei':
            wu = buildLambdaIntermediates('abef')
            fm = buildLambdaIntermediates('me')
            #table III(b)[10]                                                                                            W[abei]
            wi[:,:,:,:] = 0.0
            for a in range(nElectrons, spinOrbitals):
                for b in range(nElectrons, spinOrbitals):
                    for e in range(nElectrons, spinOrbitals):
                        for i in range(0, nElectrons):
                            wi[a,b,e,i] += eriMOspin[a,b,e,i]
                            for m in range(0, nElectrons):
                                wi[a,b,e,i] -= td[a,b,m,i] * fm[m,e]
                                wi[a,b,e,i] += -ts[a,m] * eriMOspin[m,b,e,i] + ts[b,m] * eriMOspin[m,a,e,i]
                                for n in range(0, nElectrons):
                                    wi[a,b,e,i] += 0.5 * tau(m,n,a,b,ts,td) * eriMOspin[m,n,e,i]
                                    for f in range(nElectrons, spinOrbitals):
                                        wi[a,b,e,i] += ts[a,m] * td[b,f,n,i] * eriMOspin[m,n,e,f] - ts[b,m] * td[a,f,n,i] * eriMOspin[m,n,e,f]
                                for f in range(nElectrons, spinOrbitals):
                                    wi[a,b,e,i] += -td[a,f,m,i] * eriMOspin[m,b,e,f] + td[b,f,m,i] * eriMOspin[m,a,e,f]
                            for f in range(nElectrons, spinOrbitals):
                                wi[a,b,e,i] += ts[f,i] * wu[a,b,e,f]
            return wi

        if type == 'mbij':
            wo = buildLambdaIntermediates('mnij')
            fm = buildLambdaIntermediates('me')
            #table III(b)[9]                                                                                             W[mbij]
            wi[:,:,:,:] = 0.0
            for m in range(0, nElectrons):
                for b in range(nElectrons, spinOrbitals):
                    for i in range(0, nElectrons):
                        for j in range(0, nElectrons):
                            wi[m,b,i,j] += eriMOspin[m,b,i,j]
                            for e in range(nElectrons, spinOrbitals):
                                wi[m,b,i,j] -= td[b,e,i,j] * fm[m,e]
                                wi[m,b,i,j] += ts[e,i] * eriMOspin[m,b,e,j] - ts[e,j] * eriMOspin[m,b,e,i] 
                                for f in range(nElectrons, spinOrbitals):
                                    wi[m,b,i,j] += 0.5 * tau(i,j,e,f,ts,td) * eriMOspin[m,b,e,f]
                                    for n in range(0, nElectrons):
                                        wi[m,b,i,j] += -ts[e,i] * td[b,f,n,j] * eriMOspin[m,n,e,f] + ts[e,j] * td[b,f,n,i] * eriMOspin[m,n,e,f] 
                                for n in range(0, nElectrons):
                                    wi[m,b,i,j] += td[b,e,j,n] * eriMOspin[m,n,i,e] - td[b,e,i,n] * eriMOspin[m,n,j,e]
                            for n in range(0, nElectrons):
                                wi[m,b,i,j] -= ts[b,n] * wo[m,n,i,j]
            return wi

        if type == 'amef':
            #table III(b)[8]                                                                                             W[amef]
            wi[:,:,:,:] = 0.0
            for a in range(nElectrons, spinOrbitals):
                for m in range(0, nElectrons):
                    for e in range(nElectrons, spinOrbitals):
                        for f in range(nElectrons, spinOrbitals):
                            wi[a,m,e,f] += eriMOspin[a,m,e,f]
                            for n in range(0, nElectrons):
                                wi[a,m,e,f] -= ts[a,n] * eriMOspin[n,m,e,f]
            return wi

        if type == 'mnie':
            #table III(b)[7]                                                                                             W[mnie]                                     
            wi[:,:,:,:] = 0.0
            for m in range(0, nElectrons):
                for n in range(0, nElectrons):
                    for i in range(0, nElectrons):
                        for e in range(nElectrons, spinOrbitals):
                            wi[m,n,i,e] += eriMOspin[m,n,i,e]
                            for f in range(nElectrons, spinOrbitals):
                                wi[m,n,i,e] += ts[f,i] * eriMOspin[m,n,f,e]
            return wi

    def buildGintermediates(type = ''):
        #build the G lambda intermediates

        gi = np.zeros((spinOrbitals, spinOrbitals))

        if type == 'ae':
            #table III(c)[1]
            gi[:,:] = 0.0
            for a in range(nElectrons, spinOrbitals):
                for e in range(nElectrons, spinOrbitals):
                    for m in range(0, nElectrons):
                        for n in range(0, nElectrons):
                            for f in range(nElectrons, spinOrbitals):
                                gi[a,e] -= 0.5 * td[e,f,m,n] * ld[m,n,a,f]
            return gi

        if type == 'mi':
            #table III(c)[1]
            gi[:,:] = 0.0
            for m in range(0, nElectrons):
                for i in range(0, nElectrons):
                    for n in range(0, nElectrons):
                        for e in range(nElectrons, spinOrbitals):
                            for f in range(nElectrons, spinOrbitals):
                                gi[m,i] += 0.5 * td[e,f,m,n] * ld[i,n,e,f]
            return gi


    #evaluate intermediates
    fme = buildLambdaIntermediates('me')
    fae = buildLambdaIntermediates('ae')
    fmi = buildLambdaIntermediates('mi')
    wmnij = buildLambdaIntermediates('mnij')
    wabef = buildLambdaIntermediates('abef')
    wmbej = buildLambdaIntermediates('mbej')

    wmnie = buildLambdaIntermediates('mnie')
    wamef = buildLambdaIntermediates('amef')
    wmbij = buildLambdaIntermediates('mbij')
    wabei = buildLambdaIntermediates('abei')
    
    preLambdaCorrelationEnergy = 0.0

    #start energy loop
    for cycle in range(iterations):

        #build lambda intermediates
        gae = buildGintermediates('ae')
        gmi = buildGintermediates('mi')

        #make the lambda singles Table II(a)
        LS = np.zeros_like(ls)
        for i in range(0, nElectrons):
            for a in range(nElectrons, spinOrbitals):
                LS[i,a] += fme[i,a]
                for e in range(nElectrons, spinOrbitals):
                    LS[i,a] += ls[i,e] * fae[e,a]
                    for f in range(nElectrons, spinOrbitals):
                        LS[i,a] -= gae[e,f] * wamef[e,i,f,a]
                for m in range(0, nElectrons):
                    LS[i,a] -= ls[m,a] * fmi[i,m]
                    for n in range(0, nElectrons):
                        LS[i,a] -= gmi[m,n] * wmnie[m,i,n,a]
                    for e in range(nElectrons, spinOrbitals):
                        LS[i,a] += ls[m,e] * wmbej[i,e,a,m]
                        for f in range(nElectrons, spinOrbitals):
                            LS[i,a] += 0.5 * ld[i,m,e,f] * wabei[e,f,a,m]
                        for n in range(0, nElectrons):
                            LS[i,a] -= 0.5 * ld[m,n,a,e] * wmbij[i,e,m,n]

                LS[i,a] /= (fs[i,i] - fs[a,a])

        #make the lambda doubles Table II(b)
        LD = np.zeros_like(ld)
        for i in range(0, nElectrons):
            for a in range(nElectrons, spinOrbitals):
                for j in range(0, nElectrons):
                    for b in range(nElectrons, spinOrbitals):
                        LD[i,j,a,b] += eriMOspin[i,j,a,b]
                        LD[i,j,a,b] += ls[i,a] * fme[j,b] - ls[i,b] * fme[j,a] - \
                                       ls[j,a] * fme[i,b] + ls[j,b] * fme[i,a]
                        for e in range(nElectrons, spinOrbitals):
                            LD[i,j,a,b] += ld[i,j,a,e] * fae[e,b] - ld[i,j,b,e] * fae[e,a]
                            LD[i,j,a,b] += ls[i,e] * wamef[e,j,a,b] - ls[j,e] * wamef[e,i,a,b]
                            LD[i,j,a,b] += eriMOspin[i,j,a,e] * gae[b,e] - eriMOspin[i,j,b,e] *gae[a,e]
                            for f in range(nElectrons, spinOrbitals):
                                LD[i,j,a,b] += 0.5 * ld[i,j,e,f] * wabef[e,f,a,b]
                            for m in range(0, nElectrons):
                                LD[i,j,a,b] += ld[i,m,a,e] * wmbej[j,e,b,m] - ld[i,m,b,e] * wmbej[j,e,a,m] -\
                                               ld[j,m,a,e] * wmbej[i,e,b,m] + ld[j,m,b,e] * wmbej[i,e,a,m]
                        for m in range(0, nElectrons):
                            LD[i,j,a,b] += -ld[i,m,a,b] * fmi[j,m] + ld[j,m,a,b] * fmi[i,m]
                            LD[i,j,a,b] += -ls[m,a] * wmnie[i,j,m,b] + ls[m,b] * wmnie[i,j,m,a]
                            LD[i,j,a,b] += -eriMOspin[i,m,a,b] * gmi[m,j] + eriMOspin[j,m,a,b] * gmi[m,i]
                            for n in range(0, nElectrons):
                                LD[i,j,a,b] += 0.5 * ld[m,n,a,b] * wmnij[i,j,m,n]

                        LD[i,j,a,b] /= (fs[i,i] + fs[j,j] - fs[a,a] - fs[b,b])

        ls = LS
        ld = LD

        #compute lambda pseudo-energy
        lambdaCorrelationEnergy = 0.0
        for i in range(0, nElectrons):
            for a in range(nElectrons, spinOrbitals):
                lambdaCorrelationEnergy += fs[a,i] * ls[i,a]
                for j in range(0, nElectrons):
                    for b in range(nElectrons, spinOrbitals):
                        lambdaCorrelationEnergy += 0.25 * eriMOspin[i,j,a,b] * ld[i,j,a,b]

        #convergence control
        if abs(lambdaCorrelationEnergy - preLambdaCorrelationEnergy) < tolerance:
            break

        preLambdaCorrelationEnergy = lambdaCorrelationEnergy
        
    #return lambda intermediates
    ims = [fae, fmi, fme, wmnij, wabef, wmbej, wmnie, wamef, wmbij, wabei]

    return lambdaCorrelationEnergy, ls, ld, ims

def cc2_updateIntermediates(fs, ts, td, eriMOspin, nElectrons):
    #fs - fockSpin  ts - single cluster operator   td - double cluster operator
    #eriMOspin - electron repulsion integrals in the MO spin

    def cc2_amplitudesT1():
        # equation (1) CC2 T1 amplitude (same as CCSD)
        #diagram labels Shavitt & Bartlett (Many Body methods in Chemistry & Physics Table 10.1)

        TS = np.zeros((spinOrbitals, spinOrbitals))

        for i in range(0, nElectrons):
            for a in range(nElectrons, spinOrbitals):
                TS[a,i] += fs[a,i]                                                                                       #[S1]

                for e in range(nElectrons, spinOrbitals):
                    TS[a,i] += ts[e,i] * (1 - (a == e)) * fs[e,a]                                                        #[S3a]

                    for m in range(0, nElectrons):
                        TS[a,i] -= ts[e,i] * ts[a,m] * fs[e,m]                                                           #[S5a]
                        TS[a,i] += td[a,e,i,m] * fs[e,m]                                                                 #[S2a]
                        TS[a,i] += ts[e,m] * eriMOspin[a,m,i,e]                                                          #[S3c]

                        for n in range(0, nElectrons):
                            TS[a,i] -= ts[a,m] * ts[e,n] * eriMOspin[m,n,i,e]                                            #[S5c]
                            TS[a,i] -= 0.5 * td[a,e,m,n] * eriMOspin[m,n,i,e]                                            #[S2c]

                        for f in range(nElectrons, spinOrbitals):
                            TS[a,i] += ts[e,i] * ts[f,m] * eriMOspin[a,m,e,f]                                            #[S5b]
                            TS[a,i] += 0.5 * td[e,f,i,m] * eriMOspin[a,m,e,f]                                            #[S2b]

                            for n in range(0, nElectrons):
                                TS[a,i] -= 0.5 * ts[e,i] * td[a,f,m,n] * eriMOspin[m,n,e,f]                              #[S4b]
                                TS[a,i] -= ts[e,i] * ts[a,m] * ts[f,n] * eriMOspin[m,n,e,f]                              #[S6]

                                TS[a,i] -= 0.5 * ts[a,m] * td[e,f,i,n] * eriMOspin[m,n,e,f]                              #[S4b]
                                TS[a,i] += td[a,e,i,m] * ts[f,n] * eriMOspin[m,n,e,f]                                    #[S4c]

                for m in range(0, nElectrons):
                    TS[a,i] -= ts[a,m] *(1 - (m == i)) * fs[i,m]                                                         #[S3b]

                TS[a,i] /= (fs[i,i] - fs[a,a])

        return TS

    def cc2_amplitudesT2():
        # equation (2) CC2 T2 amplitude (T1H + T2F)
        #diagram labels Shavitt & Bartlett (Many Body methods in Chemistry & Physics Table 10.2)
        TD = np.zeros((spinOrbitals, spinOrbitals, spinOrbitals, spinOrbitals))

        for i in range(0, nElectrons):
            for j in range(0, nElectrons):
                for a in range(nElectrons, spinOrbitals):
                    for b in range(nElectrons, spinOrbitals):
                        TD[a,b,i,j] +=eriMOspin[i,j,a,b]                                                                 #[D1]

                        for m in range(0, nElectrons):
                            TD[a,b,i,j] += -ts[a,m] * eriMOspin[m,b,i,j] + ts[b,m] * eriMOspin[m,a,i,j]                  #[D4b]
                            TD[a,b,i,j] += -td[a,b,i,m] * fs[j,m] * (1 - (m == j)) + td[a,b,j,m] * fs[i,m] * (1 - (m == i))    #[D2b]

                            for n in range(0, nElectrons):
                                TD[a,b,i,j] += ts[a,m] * ts[b,n] * eriMOspin[m,n,i,j]                                    #[D6b]

                                for e in range(nElectrons, spinOrbitals):
                                    TD[a,b,i,j] += ts[a,m] * ts[b,n] * ts[e,i] * eriMOspin[m,n,e,j] - \
                                                   ts[a,m] * ts[b,n] * ts[e,j] * eriMOspin[m,n,e,i]                      #[D8b]

                                    for f in range(nElectrons, spinOrbitals):
                                        TD[a,b,i,j] += ts[a,m] * ts[b,n] * ts[e,i] * ts[f,j] * eriMOspin[m,n,e,f]        #[D9]

                        for e in range(nElectrons, spinOrbitals):
                            TD[a,b,i,j] += ts[e,i] * eriMOspin[a,b,e,j] - ts[e,j] * eriMOspin[a,b,e,i]                   #[D4a]
                            TD[a,b,i,j] += td[a,e,i,j] * fs[e,b] *  (1 - (b == e))                                       #[D2a]

                            for m in range(0, nElectrons):
                                TD[a,b,i,j] += -ts[e,i] * ts[a,m] * eriMOspin[m,b,e,j] + \
                                                ts[e,i] * ts[b,m] * eriMOspin[m,a,e,j] + \
                                                ts[e,j] * ts[a,m] * eriMOspin[m,b,e,i] - \
                                                ts[e,j] * ts[b,m] * eriMOspin[m,a,e,i]                                   #[D6c]

                            for f in range(nElectrons, spinOrbitals):
                                TD[a,b,i,j] += ts[e,i] * ts[f,j] * eriMOspin[a,b,e,f]                                    #[D6a]

                                for m in range(0, nElectrons):
                                    TD[a,b,i,j] += -ts[e,i] * ts[f,j] * ts[a,m] * eriMOspin[m,b,e,f] + \
                                                   ts[e,i] * ts[f,j] * ts[b,m] * eriMOspin[m,a,e,f]                      #[D8a]

                        TD[a,b,i,j] /= (fs[i,i] + fs[j,j] - fs[a,a] - fs[b,b])

        return TD

    spinOrbitals = fs.shape[0]

    ts = cc2_amplitudesT1()
    td = cc2_amplitudesT2()

    return ts, td


def cc2(atoms, eigenVectors, charge, fock, ERI, iterations, convergence, SCFenergy, diisStatus):
    #compute the charged coupled single and doubles

    spinOrbitals = (fock.shape[0]) * 2
    nElectrons = electronCount(atoms, charge)

    #get fock in MO spin basis
    fockMOspin = buildFockMOspin(spinOrbitals, eigenVectors, fock)

    #get two-electron repulsion integrals in MO basis
    eriMO = buildEriMO(eigenVectors, ERI)

    #transform eri from MO to spin basis
    eriMOspin = buildEriDoubleBar(spinOrbitals, eriMO)

    #amplitude array for singles and doubles
    ts = np.zeros((spinOrbitals, spinOrbitals))
    td = np.zeros((spinOrbitals, spinOrbitals, spinOrbitals, spinOrbitals))

    # #previous iteration amplitudes needed for diis
    # preTs = np.zeros_like(ts)
    # preTd = np.zeros_like(td)

    #get mp2
    mp2 = 0.0
    for i in range(0, nElectrons):
        for j in range(0, nElectrons):
            for a in range(nElectrons, spinOrbitals):
                for b in range(nElectrons, spinOrbitals):
                    Dijab = fockMOspin[i,i] + fockMOspin[j,j] - fockMOspin[a,a] - fockMOspin[b,b]
                    td[a,b,i,j] = eriMOspin[i,j,a,b] / Dijab
                    mp2 += 0.25 * eriMOspin[i,j,a,b] * td[a,b,i,j]

    diisCapacity = 6
    diis = diis_c(diisCapacity, [ts, td])

    #start the convergence iterations
    energy = 0.0
    rmsAmplitudes = 0.0
    for cycle in range(1 , iterations):
        
        diis.refresh_store([ts, td])
        ts,td = cc2_updateIntermediates(fockMOspin, ts, td, eriMOspin, nElectrons)

        energy = ccsdEnergy(fockMOspin, ts, td, eriMOspin, nElectrons)

        #test convergence
        rms = max(sqrt(np.sum(ts*ts)) ,sqrt(np.sum(td*td)))
        if cycle != 1:
            deltaEnergy = abs(preEnergy - energy)
            deltaAmplitudes = abs(rmsAmplitudes - rms)
            postSCF([energy, deltaEnergy, deltaAmplitudes, cycle, diisStatus], 'diis-c2')

            if (deltaEnergy < convergence) and (deltaAmplitudes < convergence):
                break

        preEnergy = energy
        rmsAmplitudes = rms
        #diis
        if diisStatus == 'on':
            ts, td = diis.build([ts, td])

    postSCF([cycle, energy, mp2, SCFenergy], 'cc2')

    return energy, ts, td

def lccsd(atoms, eigenVectors, charge, fock, ERI, iterations, convergence, diisStatus, SCFenergy):
    #coupled electron pairs approximation zero order

    spinOrbitals = (fock.shape[0]) * 2
    nElectrons = electronCount(atoms, charge)

    #get fock in MO spin basis
    fockMOspin = buildFockMOspin(spinOrbitals, eigenVectors, fock)

    #get two-electron repulsion integrals in MO basis
    eriMO = buildEriMO(eigenVectors, ERI)

    #transform eri from MO to spin basis
    eriMOspin = buildEriDoubleBar(spinOrbitals, eriMO)

    #amplitude array for singles and doubles
    ts = np.zeros((spinOrbitals, spinOrbitals))
    td = np.zeros((spinOrbitals, spinOrbitals, spinOrbitals, spinOrbitals))

    #previous iteration amplitudes needed for diis
    preTs = np.zeros_like(ts)
    preTd = np.zeros_like(td)

    #diis variables
    if diisStatus == 'on':
        diisCapacity = 6
        diis = diis_c(diisCapacity, [ts, td])

    #start the convergence iterations
    energy = 0.0
    rmsAmplitudes = 0.0
    for cycle in range(1 , iterations):

        diis.refresh_store([ts, td])
        ts, td = lccsd_updateIntermediates(ts, td, fockMOspin, eriMOspin, nElectrons, spinOrbitals)

        energy = lccsdEnergy(fockMOspin, ts, td, eriMOspin, nElectrons, spinOrbitals)

        #test convergence
        rms = max(sqrt(np.sum(ts*ts)) ,sqrt(np.sum(td*td)))
        if cycle != 1:
            deltaEnergy = abs(preEnergy - energy)
            deltaAmplitudes = abs(rmsAmplitudes - rms)
            postSCF([energy, deltaEnergy, deltaAmplitudes, cycle, diisStatus], 'diis-ls')

            if (deltaEnergy < convergence) and (deltaAmplitudes < convergence):
                break

        preEnergy = energy
        rmsAmplitudes = rms

        #diis
        if diisStatus == 'on':
            ts, td = diis.build([ts, td])

    postSCF([cycle, energy, SCFenergy], 'lccsd')

    return energy, ts, td
                        

def lccsd_updateIntermediates(ts, td, fs, eriMOspin, nElectrons, spinOrbitals):
    #fs - fockSpin  ts - tsingle cluster operator   td - double cluster operator
    #eriMOspin - electron repulsion integrals in the MO spin

    def lccsd_amplitudesT1():
        # using S1, S2a, S2b, S2c, S3a, S3b and S3c
        TS = np.zeros((spinOrbitals, spinOrbitals))

        for i in range(0, nElectrons):
            for a in range(nElectrons, spinOrbitals):
                TS[a,i] += fs[a,i]                                                                                       #[S1]

                for e in range(nElectrons, spinOrbitals):
                    TS[a,i] += ts[e,i] * (1 - (a == e)) * fs[e,a]                                                        #[S3a]

                    for m in range(0, nElectrons):
                        TS[a,i] += td[a,e,i,m] * fs[e,m]                                                                 #[S2a]
                        TS[a,i] += ts[e,m] * eriMOspin[a,m,i,e]                                                          #[S3c]

                        for n in range(0, nElectrons):
                            TS[a,i] -= 0.5 * td[a,e,m,n] * eriMOspin[m,n,i,e]                                            #[S2c]

                        for f in range(nElectrons, spinOrbitals):
                            TS[a,i] += 0.5 * td[e,f,i,m] * eriMOspin[a,m,e,f]                                            #[S2b]

                for m in range(0, nElectrons):
                    TS[a,i] -= ts[a,m] *(1 - (m == i)) * fs[i,m]                                                         #[S3b]

                TS[a,i] /= (fs[i,i] - fs[a,a])

        return TS

    def lccsd_amplitudesT2():
        # using D1, D2a, D2b, D2c, D2d, D2e, D4a, D4b

        TD = np.zeros((spinOrbitals, spinOrbitals, spinOrbitals, spinOrbitals))
        for i in range(0, nElectrons):
            for j in range(0, nElectrons):
                for a in range(nElectrons, spinOrbitals):
                    for b in range(nElectrons, spinOrbitals):

                        TD[a,b,i,j] = eriMOspin[i,j,a,b]                                                                 #[D1]

                        for e in range(nElectrons, spinOrbitals):
                            TD[a,b,i,j] += fs[e,b] * (1 - (b == e)) * td[a,e,i,j] - \
                                           fs[e,a] * (1 - (a == e)) * td[b,e,i,j]                                        #[D2a]
                            TD[a,b,i,j] += ts[e,i] * eriMOspin[a,b,e,j] - ts[e,j] * eriMOspin[a,b,e,i]                   #[D4a]

                            for f in range(nElectrons, spinOrbitals):
                                TD[a,b,i,j] += 0.5 * eriMOspin[e,f,a,b] * td[e,f,i,j]                                    #[D2c]


                        for m in range(0, nElectrons):
                            TD[a,b,i,j] += -fs[j,m] * (1 - (m == j)) *td[a,b,i,m] + \
                                            fs[i,m] * (1 - (m == i)) *td[a,b,j,m]                                        #[D2b]
                            TD[a,b,i,j] += -ts[a,m] * eriMOspin[m,b,i,j] + ts[b,m] * eriMOspin[m,a,i,j]                  #[D4b]

                            for n in range(0, nElectrons):
                                TD[a,b,i,j] += 0.5 * eriMOspin[i,j,m,n] * td[a,b,m,n]                                    #[D2d]

                        for m in range(0, nElectrons):
                            for e in range(nElectrons, spinOrbitals):
                                TD[a,b,i,j] += eriMOspin[e,j,m,b] * td[a,e,i,m] - \
                                               eriMOspin[e,i,m,b] * td[a,e,j,m] - \
                                               eriMOspin[e,j,m,a] * td[b,e,i,m] + \
                                               eriMOspin[e,i,m,a] * td[b,e,j,m]                                          #[D2e]
                            
                        fockDenominator = fs[i,i] + fs[j,j]  - fs[a,a] - fs[b,b] 

                        TD[a,b,i,j] /= fockDenominator

        return TD

    ts = lccsd_amplitudesT1()
    td = lccsd_amplitudesT2()

    return ts, td

def lccsdEnergy(fs, ts, td, eriMOspin, nElectrons, spinOrbitals):

    energy = 0.0
    for i in range(0, nElectrons):
        for a in range(nElectrons, spinOrbitals):
            energy += fs[a,i] * ts[a,i]
            for j in range(0, nElectrons):
                for b in range(nElectrons, spinOrbitals):
                    energy +=  0.25 * eriMOspin[a,b,i,j] * td[a,b,i,j]

    return energy
