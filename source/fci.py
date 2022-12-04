from __future__ import division
from integral import buildEriDoubleBar, buildEriMO, buildFockMOspin
from atom import nuclearRepulsion
from basis import electronCount
import numpy as np
import rhf
from view import postSCF

from scipy.special import comb
from itertools import combinations


def determinantCount(m, k):
    #compute number of combinations of n things taken k at a time
    
    def fact(n):
        #compute factorial
        
        f = 1
        if n < 0:
            return -1
        elif n == 0:
            return 1
        else:
            for i in range(1,n + 1):
                f *= i
        return f
        
    return  int(fact(m)/(fact(k)*fact(m-k)))

def combinationList(combs, group, start, stop, level):
    #compute the combinations for taking n things k at a time
    
    for i in range(start, stop+1):

        if level == 0: 
            s = (group + ',' + str(i))[1:]
            combs.append(list(map(int, s.split(','))))
        
        combinationList(combs, group + ',' + str(i), i+1, stop, level-1)
    
    return combs

def binaryString(comb, nOrbitals):
    #compute a binary string representation of combination as list
    
    sbinary = ''
    for i in range(max(comb)+1):
        if i in comb: sbinary = sbinary + '1'
        else: sbinary = sbinary + '0'
        
    return sbinary + '0' * (nOrbitals - len(sbinary))

def buildFCIhamiltonian(determinants, eriMOspin, Hp, singles=False):
    #compute the full FCI Hamiltonian

    nH = len(determinants)
    fciH = np.zeros((nH, nH))

    for i in range(len(determinants)):
        for j in range(0, i+1):

            da = determinants[i]
            db = determinants[j]

            element = hamiltonianElement(da, db, eriMOspin, Hp, singles)
            fciH[i,j] = fciH[j,i] = element

    return fciH

def excitations(da, db):
    #compute the total excitations
    
    excite = 0
    for i in range(len(da)):
        if (da[i] == '1') and (db[i] == '0') : excite += 1
    
    return excite

def phase(da, db):
    #compute the phase between the determinants
    
    t = db
    ld = len(da)
    occupied = 0

    for i in range(ld):
        if da[i] == '1' and db[i] == '0':

        #hole has appeared
            for j in range(ld):
                if db[j] == '1' and da[j] == '0':

                    #particle has appeared - get occupied between
                    #hole and particle
                    m = min(i,j)
                    n = max(j,i)
                    occupied += db.count('1', m+1, n)

                    db = db[:j] + '0' + db[j+1:]
                    break
     
    db = t  

    return pow(-1,occupied)

def levels(da, db):
    #compute the jumps
    
    ld = len(da)
    jumps = []
    t = db

    for i in range(ld):
        if da[i] == '1' and db[i] == '0':

            #hole has appeared
            for j in range(ld):
                if db[j] == '1' and da[j] == '0':

                #particle has appeared               
                    jumps.append([i,j])

                    #set excited state to zero so don't count again
                    db = db[:j] + '0' + db[j+1:]
                    break
    db = t

    return jumps

def commonStates(da, db):
    #compute common states between determinants
    ld = min(len(da), len(db))
    
    common = []
    for i in range(ld):
        if da[i] == '1' and db[i] == '1': common.append(i)
        
    return common

def hamiltonianElement(da, db, eriMOspin, coreH, singles):
    #compute an individual Hamiltonian element

    #get number of excitions
    excites = excitations(da, db)

    if excites > 2 : return 0.0

    theta = phase(da, db)
    jump = levels(da, db)
    if excites == 2:
        if singles:
            #alpha must come before beta
            if (jump[0][0] % 2) == 1 and (jump[1][0] % 2) != 1: jump[0][0], jump[1][0] = jump[1][0], jump[0][0]
            if (jump[0][1] % 2) == 1 and (jump[1][1] % 2) != 1: jump[0][1], jump[1][1] = jump[1][1], jump[0][1]

        return theta * eriMOspin[jump[0][0], jump[1][0], jump[0][1], jump[1][1]]

    elif excites == 1:
        common = commonStates(da, db)
        f = coreH[jump[0][0],jump[0][1]]
        for i in common:
            f += eriMOspin[jump[0][0], i, jump[0][1], i]

        return theta * f

    elif excites == 0:
        common = commonStates(da, db)
        f = 0.0
        for i in common:
            f += coreH[i, i]
        for i in common:
            for j in common:
             f += 0.5 * eriMOspin[i, j, i, j]

        return theta * f

def configurations(nElectrons, nOrbitals, type = 'S'):

    determinants = []
    pad = nOrbitals - nElectrons

    def subDeterminant(n, k, bit):
        #components of full determinant

        sub = []
        comb = []
        combinationList(comb, '', 0, n-1, k)

        for i in comb:
            s = bit[0] * n
            for j in range(k+1):
                s = s[:i[j]] + bit[1] + s[i[j]+1:]
            sub.append(s)

        return sub

    #groundstate
    if 'G' in type:
        determinants.append('1' * nElectrons + '0' * pad)

    #generate groundstate single excitations
    if ('S' in type) and (nElectrons > 0):

        pre = subDeterminant(nElectrons, 0, '10')
        post = subDeterminant(pad, 0, '01')

        for i in pre:
            for j in post:
                determinants.append(i+j)

    #generate groundstate double excitations
    if ('D' in type) and (nElectrons > 1):

        pre = subDeterminant(nElectrons, 1, '10')
        post = subDeterminant(pad, 1, '01')

        for i in pre:
            for j in post:
                determinants.append(i+j)

    #generate groundstate triple excitations
    if ('T' in type) and (nElectrons > 2):

        pre = subDeterminant(nElectrons, 2, '10')
        post = subDeterminant(pad, 2, '01')

        for i in pre:
            for j in post:
                determinants.append(i+j)

    #generate groundstate quadruples excitations
    if ('Q' in type) and (nElectrons > 3):

        pre = subDeterminant(nElectrons, 3, '10')
        post = subDeterminant(pad, 3, '01')

        for i in pre:
            for j in post:
                determinants.append(i+j)

    #sort determinants into major alpha order
    determinants.sort()

    return determinants

def fci(molAtom, molBasis, charge, c, ERI, coreH):
    #compute a full configuration interaction

    nElectrons = electronCount(molAtom, charge)
    nBasis = len(molBasis)
    nOrbitals = nBasis * 2

    #get all combinations of orbitals taken electrons at a time
    combinations = []
    combinations = combinationList(combinations, '', 0, nOrbitals-1, nElectrons-1)

    #convert combinations to determinant binary strings
    binary = []
    for det in combinations:
        binary.append(binaryString(det, nOrbitals))
    binary.sort()

    #spin molecular integrals
    coreMOspin = buildFockMOspin(nOrbitals, c, coreH)
    eriMO = buildEriMO(c, ERI)
    eriMOspin = buildEriDoubleBar(nOrbitals, eriMO)

    Hp = buildFCIhamiltonian(binary, eriMOspin, coreMOspin)

    e, _ = np.linalg.eigh(Hp)

    postSCF([nElectrons, nOrbitals, determinantCount(nOrbitals, nElectrons), rhf.SCFenergy, e[0], nuclearRepulsion(molAtom)],'fci')

    return e

def cisd(molAtom, molBasis, charge, c, ERI, coreH):
    #compute configuration up to doubles

    nElectrons = electronCount(molAtom, charge)
    nBasis = len(molBasis)
    nOrbitals = nBasis * 2

    coreMOspin = buildFockMOspin(nOrbitals, c, coreH)
    eriMO = buildEriMO(c, ERI)
    eriMOspin = buildEriDoubleBar(nOrbitals, eriMO)

    determinants = configurations(nElectrons, nOrbitals, 'GSD')

    Hp = buildFCIhamiltonian(determinants, eriMOspin, coreMOspin)

    e, _ = np.linalg.eigh(Hp)

    postSCF([nElectrons, nOrbitals, determinantCount(nOrbitals, nElectrons), rhf.SCFenergy, e[0], nuclearRepulsion(molAtom)],'cisd')

    return e

    
def ciss(molAtom, molBasis, charge, c, ERI, coreH):
    #compute configuration single using slater determinants

    nElectrons = electronCount(molAtom, charge)
    nBasis = len(molBasis)
    nOrbitals = nBasis * 2

    coreMOspin = buildFockMOspin(nOrbitals, c, coreH)
    eriMO = buildEriMO(c, ERI)
    eriMOspin = buildEriDoubleBar(nOrbitals, eriMO)

    determinants = configurations(nElectrons, nOrbitals, 'S')

    determinants = list(zip(*(iter(determinants),) * (nOrbitals - nElectrons)))
    determinants = sum([list(i)[::-1] for i in determinants], [])

    Hp = buildFCIhamiltonian(determinants, eriMOspin, coreMOspin, singles=True) 
    Hp -= np.eye(Hp.shape[0]) * (rhf.SCFenergy - nuclearRepulsion(molAtom))

    e , v = np.linalg.eigh(Hp)

    postSCF([nElectrons, nOrbitals, len(determinants), e],'cis')

    return e, v

def spinStates(da, nBasis):
    #get the alpha and beta spin states

    alpha = da[0::2]
    beta  = da[1::2]

    return  alpha + '0'*(nBasis - len(alpha)) , beta + '0'*(nBasis - len(beta))

def occupancy(da, nBasis):
    #get the unoccupied, singe and double occupancy (spatial) orbitals

    alpha , beta = spinStates(da, nBasis)

    occupancy = ''
    for i in range(len(alpha)):
        occupancy += str(int(alpha[i]) + int(beta[i]))

    return occupancy

def bString(da, spinOrbitals):
    #convert determinant to a bit string integer

    bits = '0b'
    for i in range(spinOrbitals-1, -1, -1): 
        if i in da: bits += '1'
        else: bits += '0'

    return int(bits,2)

def bCombinationList(spinOrbitals, nElectrons):
    #create list of all combinations of nElectrons in spinOrbital orbitals

    nVirtual = spinOrbitals - nElectrons
    determinantList = []

    for determinant in combinations(range(spinOrbitals), nElectrons):

        determinantList.append(bString(determinant, spinOrbitals))

    return determinantList


def bRightZeros(n):
    #compute the number of rightmost zero bits

    return (n & -n).bit_length() - 1


def bSetZero(da, n):
    #set the nth bit to zero 0-based

    return ~(1 << n) & da

def bOccupancy(da, db, type = 'h'):
    #compute the holes or particles between determinants

    k = 0
    hp = []
    if type == 'h': h = (da ^ db) & da
    else:  h = (da ^ db) & db

    while h != 0:
        p = bRightZeros(h)
        hp.append(p)

        h = bSetZero(h, p)
        k += 1

    return hp

def bExcitations(da, db):
    #the number of excitation between the two determinants

    return (bin(da ^ db).count('1')) >> 1

def bSingleExcitations(da, db):
    #compute the single excitation da->db

    excite = np.zeros((2,2), dtype=object)

    if da == db: return excite, 0

    #get hole and particle
    aorb = da ^ db
    hole = aorb & da
    particle = aorb & db

    #get positions and excitation degree=1
    excite[0,0] = 1
    excite[1,0] = bRightZeros(hole)
    excite[0,1] = 1
    excite[1,1] = bRightZeros(particle)

    #get phase
    lo = min(excite[1,0], excite[1,1])
    hi = max(excite[1,0], excite[1,1])

    nPerm = bin(da &  (~(1 << lo+1)+1  & (1 << hi)-1)).count('1')
    phases = [1, -1]
    phase = phases[nPerm & 1]

    return excite, phase

def bDoubleExcitations(da, db):
    #compute the double excitation da->db

    excite = np.zeros((3,2), dtype=object)

    if da == db: return excite, 0

    aorb = da ^ db
    hole     = aorb & da
    particle = aorb & db

    #the holes
    iHole = 0
    while hole != 0:
        iHole += 1
        excite[iHole,0] = bRightZeros(hole)
        excite[0,0] += 1
    
        hole = hole & (hole - 1)

    #the particles
    iParticle = 0
    while particle != 0:
        iParticle += 1
        excite[iParticle,1] = bRightZeros(particle)
        excite[0,1] += 1
    
        particle = particle & (particle - 1)
    
    #phase
    nPerm = 0
    for excitation in [1,2]:
        lo = min(excite[excitation,0],excite[excitation,1])
        hi = max(excite[excitation,0],excite[excitation,1])

        nPerm += bin(da &  (~(1 << lo+1)+1  & (1 << hi)-1)).count('1')
    
    #orbital crossings
    i = min(excite[1,0], excite[1,1])
    a = max(excite[1,0], excite[1,1])
    j = min(excite[2,0], excite[2,1])
    b = max(excite[2,0], excite[2,1])

    if (j>i) and (j<a) and (b>a) : 
        nPerm += 1

    phases = [1,-1]
    phase = phases[nPerm & 1]

    return excite, phase

def bBuildFCIhamiltonian(determinants, eriMOspin, coreH):
    #compute the full FCI Hamiltonian

    nH = len(determinants)
    fciH = np.zeros((nH, nH))

    for i in range(len(determinants)):
        for j in range(0, i+1):

            da = determinants[i]
            db = determinants[j]

            element = bHamiltonianElement(da, db, eriMOspin, coreH)
            fciH[i,j] = fciH[j,i] = element

    return fciH

def bFci(molAtom, molBasis, charge, c, ERI, coreH):
    #bit version of fci

    nElectrons = electronCount(molAtom, charge)
    nBasis = len(molBasis)
    spinOrbitals = nBasis * 2

    #get all combinations of orbitals taken electrons at a time
    combinations = []
    combinations = bCombinationList(spinOrbitals, nElectrons)

    #spin molecular integrals
    coreMOspin = buildFockMOspin(spinOrbitals, c, coreH)
    eriMO = buildEriMO(c, ERI)
    eriMOspin = buildEriDoubleBar(spinOrbitals, eriMO)

    Hp = bBuildFCIhamiltonian(combinations, eriMOspin, coreMOspin)

    e, c = np.linalg.eigh(Hp)

    return e, c, comb(spinOrbitals, nElectrons, exact=True)

def bCommonStates(da, db):
    #get the common orbitals

    common = []
    ab = da & db

    for i in range(ab.bit_length()):
        idx = ab & (1 << i)
        if idx: common.append(bRightZeros(1<<i))

    return common

def bHamiltonianElement(da, db, eriMOspin, coreH):
    #compute an individual Hamiltonian element

    #get number of excitions
    excites = bExcitations(da, db)

    if excites > 2 : return 0.0

    if excites == 2:

        jump, phase = bDoubleExcitations(da, db)

        return phase * eriMOspin[jump[1,0], jump[2,0], jump[1,1], jump[2,1]]

    elif excites == 1:

        common = bCommonStates(da, db)
        jump, phase = bSingleExcitations(da, db)

        f = coreH[jump[1,0], jump[1,1]]

        for i in common:
            f += eriMOspin[jump[1,0], i, jump[1,1], i]

        return phase * f

    elif excites == 0:

        common = bCommonStates(da, db)
        f = 0.0
        for i in common:
            f += coreH[i, i]
        for i in common:
            for j in common:
             f += 0.5 * eriMOspin[i, j, i, j]

        return  f

def bResidues(da, spinOrbitals):
    #get the residues of determinant da

    residues = []
    occupancy = bin(da).count('1')

    for i in range(spinOrbitals):
        mask1 = 1 << i
        for j in range(i):
            mask2 = 1 << j
            reducedOccupancy = da & ~(mask1 ^ mask2)
    
            if bin(reducedOccupancy).count('1') == (occupancy - 2):
                residues.append(reducedOccupancy)

    return residues

def bSetResidue(residues, spinOrbitals):
    #populate residues with two particles

    determinants = []

    for residue in residues:

        for i in range(spinOrbitals):
            mask1 = 1 << i
            if not bool(residue & mask1):
                p = residue | mask1

                for j in range(i):
                    mask2 = 1 << j
                    if not bool(p & mask2):
                        q = p | mask2

                        determinants.append(q)

    return list(set(determinants))

def bCisd(molAtom, molBasis, charge, c, ERI, coreH):
    #compute configuration up to doubles

    nElectrons = electronCount(molAtom, charge)
    nBasis = len(molBasis)
    spinOrbitals = nBasis * 2

    coreMOspin = buildFockMOspin(spinOrbitals, c, coreH)
    eriMO = buildEriMO(c, ERI)
    eriMOspin = buildEriDoubleBar(spinOrbitals, eriMO)

    groundState = (2**nElectrons-1)

    residueList = bResidues(groundState, spinOrbitals)

    particles = bSetResidue(residueList, spinOrbitals)

    Hp = bBuildFCIhamiltonian(particles, eriMOspin, coreMOspin)

    e, c = np.linalg.eigh(Hp)

    return e, len(particles)
    