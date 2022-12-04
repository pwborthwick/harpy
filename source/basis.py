from __future__ import division
from numpy import sqrt, sort, zeros, array
from math import sqrt, pi, pow
from atom import symbol
from scipy.special import factorial2 as df

subshell = {'s': [[0, 0, 0]], \
            'p': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  \
            'd': [[2, 0, 0], [1, 1, 0], [1, 0, 1], [0, 2, 0], [0, 1, 1], [0, 0, 2]],  \
            'f': [[3, 0, 0], [2, 1, 0], [2, 0, 1], [1, 2, 0], [1, 1, 1], [1, 0, 2], [0, 3, 0], [0, 2, 1], [0, 1, 2], [0, 0, 3]]}

class basis(object):

    def __init__(self, atom, symbol, center= zeros(3), momentum = zeros(0), ex = zeros(0), co = zeros(0) , normal = zeros(0) ):
        self.atom = atom
        self.symbol = symbol
        self.center = array(center)
        self.momentum = array(momentum)
        self.ex = ex
        self.co = co
        self.normal = normal

def electronCount(atoms, charge):
    #compute the number of electrons

    n = 0
    for i in range(0, len(atoms)):
        n += atoms[i].number

    #adjust for change
    n -= charge

    return n

def species(atoms):
    #get a list of different atom types
    z = zeros(len(atoms))
    for i in range(0, len(atoms)):
        z[i] = atoms[i].number

    #order
    z = sort(z)
    #unique list
    unique = []
    unique.append(int(z[0]))
    for i in range(1, len(z)):
        if z[i] != unique[-1]:
            unique.append(int(z[i]))
    
    return unique

def checkBasis(name, atoms):
    #is basis supported
    f = open('../basis/' + name + '.gbf','r') 
    try:
        f.close()
    except FileNotFoundError:
        return False

    #does basis support atom type, last types entry is greatest z
    types = species(atoms)
    if (name in ['sto-3g', '3-21g']) and (types[-1] > 54):
        print('Atom not supported for ' + name + ' basis set')
        return False
    if (name == 'dz') and (types[-1] > 17):
        print('Atom not supported for ' + name + ' basis set')
        return False
    if (name in ['6-31g','aug-cc-pvdz','cc-pvdz']) and (types[-1] > 36):
        print('Atom not supported for ' + name + ' basis set')
        return False

    return True

def principalQuantumNumber(basis):
    #compute principal quantum number of subshell
    n = 1
    for i in range(0, 3):
        n += basis.momentum[i]

    return n

def basisNormalise(basis):
    #normalise the primatives

    #get principal quantum number
    n = principalQuantumNumber(basis) - 1

    normalisedBasis = zeros(len(basis.ex))

    for i in range(0, len(basis.ex)):
        num = pow(2*basis.ex[i]/pi, 0.75) * pow(4*basis.ex[i], n/2)
        den = sqrt(df(2*basis.momentum[0]-1) * df(2*basis.momentum[1]-1) * df(2*basis.momentum[2]-1))
        normalisedBasis[i] = num/den

    #normalise coefficients
    prefactor = (pi ** 1.5) * df(2*basis.momentum[0] - 1) * \
                              df(2*basis.momentum[1] - 1) * \
                              df(2*basis.momentum[2] - 1) / (2 ** n)
    
    s = 0.0
    for i in range(0,len(basis.ex)):
        for j in range(0,len(basis.ex)):
            s += normalisedBasis[i] * normalisedBasis[j] * basis.co[i] * basis.co[j] \
                / pow( (basis.ex[i] + basis.ex[j]) , (n + 1.5))

    s *= prefactor
    s = 1/sqrt(s)

    for i in range(0, len(basis.co)):
        basis.co[i] *= s

    basis.normal = normalisedBasis
    
    return basis

def buildBasis(atoms,name):
    #read basis information  for each atom type in molecule
    types = species(atoms)

    #molecular basis
    molBasis = []

    #open 'g'aussian 'b'asis 'f'ile
    f = open('../basis/' + name + '.gbf', 'r')
    data = []

    #construct basis for atom type - construct atomic basis
    atomBasis = []

    #cycle through unique atoms list
    for i in range(0, len(types)):
        sym = symbol[types[i]-1]

        #find atomic symbol
        while f:
            line = f.readline().strip()
            if line[:2].strip() == sym: 
                #S could be element or shell - element lines eg S   0
                data = line.split()
                if data[1] == '0':
                    break

        momentum = []
        primatives = []
        exponents = []
        coefficients = []
        coefficientp = []

        while True:
            #get momentum header eg S   3   1.0
            data = f.readline().split()
            if data[0] == '****':
                break
            momentum.append(data[0].lower())
            primatives.append(int(data[1]))

            #read primatives
            for j in range(0, int(primatives[-1])):
                data = f.readline().split()

                exponents.append(float(data[0].replace('D','e')))
                if 's,p,d,f'.rfind(momentum[-1]) >= 0:  
                    coefficients.append(float(data[1].replace('D','e')))
                else:
                    coefficients.append(float(data[1].replace('D','e')))
                    coefficientp.append(float(data[2].replace('D','e')))

        start = 0
        pStart = 0
        for j in range(0, len(momentum)):
            end = start + primatives[j]
            if momentum[j] != 'sp':
                for angular in range(0, len(subshell[momentum[j]])):
                    atomBasis.append(basis(types[i], orbitalType(subshell[momentum[j]][angular]),          \
                                            [0,0,0] ,subshell[momentum[j]][angular], exponents[start:end], \
                                            coefficients[start:end]))
            else:
                atomBasis.append(basis(types[i], orbitalType(subshell['s'][0]) , \
                                       [0,0,0], subshell['s'][0], exponents[start:end], coefficients[start:end]))

                for angular in range(0, len(subshell['p'])):
                    atomBasis.append(basis(types[i], orbitalType(subshell['p'][angular]) , \
                                     [0,0,0], subshell['p'][angular], exponents[start:end], coefficientp[pStart:pStart+primatives[j]]))
                pStart = primatives[j]
            start = end

    #assign atomBasis to each atom of that type - construct molecular basis
    for i in range(0, len(atoms)):
        for j in range(0, len(atomBasis)):
            if atoms[i].number == atomBasis[j].atom:
                molBasis.append(basis(i, atomBasis[j].symbol, atoms[i].center, atomBasis[j].momentum,    \
                                       atomBasis[j].ex, atomBasis[j].co))
                #add normalisation factor 
                molBasis[-1] = basisNormalise(molBasis[-1])

    #prefix orbital type with shell and sort
    orbitalShell(molBasis)
    molBasis.sort(key=lambda b: (b.atom, 'spdf'.index(b.symbol[1]),  b.symbol[0]))

    return molBasis

def orbitalType(subShell):
    #get the orbital symbol from momentum tuple
    name = { '[000]' : 's' , \
             '[100]' : 'px', '[010]' : 'py' , '[001]' : 'pz' , \
             '[200]' : 'dxx', '[110]' : 'dxy', '[101]' : 'dxz', '[020]' : 'dyy', '[011]' : 'dyz', '[002]' : 'dzz'  ,  \
             '[300]' : 'fxxx', '[210]' : 'fxxy' , '[201]' : 'fxxz' , '[120]' : 'fxyy' , '[111]' : 'fxyz' ,  '[102]' : 'fxzz' , \
             '[030]' : 'fyyy' , '[021]' : 'fyyz' , '[012]' : 'fyzz' , '[003]' : 'fzzz' }

    return  name['[' + ''.join(str(e) for e in subShell) + ']']

def aufbau(atom):
    #implementation to f-orbitals {s,p,d,f} so shells {K,L,M,N}, n {1,2,3,4}

    principal = []   # list principal quantum number
    azimuthal = []   # list of azimuthal quantum numbers, n[i], l[i] give eg the 2, p respectively designation
                     # for orbital i
    order = []
    for n in range(1, len(subshell)+1):
        for a in range(0,n):
            principal.append(n)
            azimuthal.append(a)
            #aufbau -> order is (n+l) and where (n+l)'s are equal then smallest n applies
            order.append(n+a)

    #create hash of 10*(n+l) +n this is the correct aufbau order
    for i in range(0 , len(order)):
        order[i] = 10*order[i] + principal[i]

    #sort 'order' into ascending order and we have aufbau occupancy order
    order.sort()

    #reverse hash to generate eg 1s, 2p etc
    orbitalCode = ['s' , 'p', 'd', 'f']
    orbitals = []
    for i in range(0 , len(order)):
        n = order[i] - (order[i]//10) * 10
        l = order[i]//10 - n
        orbitals.append(str(n) + orbitalCode[l])


    orbitalCapacity = { 's' : 2 , 'p' : 2+4 , 'd' : 2+4+4 , 'f' : 2+4+4+4 }

    #loop over the orbitals in order accumulating total electrons until get to atomic number of 'atom'
    electronCount = 0
    effectiveAN = atom.number
    orbitalSequence = ''

    for i in range(0 , len(order)):
        e = orbitalCapacity[orbitals[i][-1]]
        electronCount += e
        #build sequence
        orbitalSequence += orbitals[i]

        if electronCount >= effectiveAN:
            #enough orbitals
            orbitalSequence += '<sup>' + str(effectiveAN) + '</sup>'
            break
        else:
            # need (at least) another orbital
            orbitalSequence += '<sup>' + str(e) + '</sup>'
            effectiveAN -= e
            electronCount = 0

    return orbitalSequence

def orbitalShell(molBasis):
    #prefix orbital type with shell

    atom = -1
    lowest = {'s':1,'p':2,'d':3,'f':4}

    for mol in molBasis:
        if mol.atom != atom:
            atom = mol.atom
            n = {}
        sym = mol.symbol
        if not sym in n.keys(): n[sym] = lowest[sym[0]]
        mol.symbol = str(n[sym]) + sym
        n[sym] += 1



