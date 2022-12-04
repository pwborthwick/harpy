from __future__ import division
from numpy import zeros, cross, dot, array, sqrt, pi, asarray, ndarray
from numpy.linalg import norm, eig
from math import asin, acos, atan2, cos, sin

weight = array([1.00784, 4.002602, 6.938, 9.0121831, 10.806, 12.0096, 14.00643, 15.99903, 18.998403163, 20.1797,
  22.98976928, 24.304, 26.9815384, 28.084, 30.973761998, 32.059, 35.446, 39.792, 39.0983, 40.078, 44.955908,
  47.867, 50.9415, 51.9961, 54.938043, 55.845, 58.933194, 58.6934, 63.546, 65.38, 69.723, 72.630, 74.921595,
  78.971, 79.901, 83.798, 85.4678, 87.62, 88.90584, 91.224, 92.90637, 95.95, 98, 101.07, 102.90549, 106.42, 
  107.8682, 112.414, 114.818, 118.710, 121.760, 127.60, 126.90447, 131.293, 132.90545196, 137.327, 138.90547, 
  140.116, 140.90766, 144.242, 145, 150.36, 151.964, 157.25, 158.925354, 162.500, 164.930328, 167.259, 168.934218,
  173.045, 174.9668, 178.486,  180.94788, 183.84, 186.207, 190.23, 192.217, 195.084, 196.966570, 200.592, 204.382,
  207.2, 208.98040, 209, 210, 222])

covalentRadius = array([31, 28, 128, 96, 84, 73, 71, 66, 57, 58, 166, 141, 121, 111, 107, 105, 102, 106, 
  203, 176, 170, 160, 153, 139, 139, 132, 126, 124, 132, 122, 122, 120, 119, 120, 120, 116, 220, 195, 190,
  175, 164, 154, 147, 146, 142, 139, 145, 144, 142, 139, 139, 138, 139, 140, 224, 215, 207, 204, 203, 201,
  199, 198, 198, 196, 194, 192, 192, 189, 190, 187, 187, 187, 175, 162, 151, 144, 141, 136, 136, 132, 145, 
  146, 148, 140, 150, 150])

symbol = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 
  'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
  'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu','Rb', 
  'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Hf', 
  'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn']

class atom(object):

    def __init__(self,id,number,center=zeros(3)):      
        self.id = id
        self.number = number
        self.center = array(center)

def seperation(atoms, i, j, unit='b'):
    #length of vector ij

    d =0.0
    for dim in range(0,3):
        d += (atoms[i].center[dim] - atoms[j].center[dim])**2
    d = sqrt(d)

    if unit == 'a':
        return d * getConstant('bohr->angstrom')    
    else:
        return d

def angle(atoms,i,j,k):
    #angle between vectors ij and jk 

    u = zeros(3)
    v = zeros(3)

    for dim in range(0,3):
        u[dim] = atoms[j].center[dim] - atoms[i].center[dim]
        v[dim] = atoms[j].center[dim] - atoms[k].center[dim]

    return acos(dot(u,v)/(norm(v)*norm(u)))*getConstant('radian->degree')

def oopAngle(atoms, i, j, k, l):
    #for i the angle between vector ik and normal to plane jkl

    u = zeros(3)
    v = zeros(3)
    w = zeros(3)

    for dim in range(0,3):
        u[dim] = atoms[i].center[dim] - atoms[k].center[dim]
        v[dim] = atoms[j].center[dim] - atoms[k].center[dim]
        w[dim] = atoms[l].center[dim] - atoms[k].center[dim]

    u /= norm(u)
    v /= norm(v)
    w /= norm(w)

    vw = cross(v,w)

    angle = dot(vw,u)/norm(vw)
    return asin(angle)*getConstant('radian->degree')

def dihedral(atoms,i,j,k,l):
    #angle between planes ijk and jkl

    u = zeros(3)
    v = zeros(3)
    w = zeros(3)


    for dim in range(0,3):
        u[dim] = atoms[i].center[dim] - atoms[j].center[dim]    
        v[dim] = atoms[j].center[dim] - atoms[k].center[dim]
        w[dim] = atoms[k].center[dim] - atoms[l].center[dim]

    u /= norm(u)
    v /= norm(v)
    w /= norm(w)

    uv = cross(u,v)
    vw = cross(v,w)
    uvv = cross(uv,v)

    return atan2(dot(uvv,vw),dot(uv,vw))*getConstant('radian->degree')

def massCenter(atoms):
     #total mass of system
     w = 0.0
     for i in range(0 , len(atoms)):
        w += weight[atoms[i].number-1]

     com = zeros(3)
     for dim in range(0,3):
        for i in range(0, len(atoms)):
            com[dim] += atoms[i].center[dim] * weight[atoms[i].number-1]/w

     return com

def isBond(atoms,i,j):
    #if sum of covalent radii of i and j is less than seperation

    if i == j: return False
    
    sumRadii = (covalentRadius[atoms[i].number-1] + covalentRadius[atoms[j].number-1]) * getConstant('picometre->bohr')

    if seperation(atoms,i,j) < 1.6 * sumRadii:
        return True
    else:
        return False

def bondMatrix(atoms):
    # construct connection matrix for molecule

    joins = zeros([len(atoms), len(atoms)])

    for i in range(0, len(atoms)):
        for j in range(i+1, len(atoms)):
            joins[i,j] = isBond(atoms,i,j)
    #symmetrise
    joins = joins + joins.T

    return joins

def inertiaTensor(atoms):
    #compute the moment of inertia tensor for molecule

    inertiaTensor = zeros([3,3])
    comFrame = zeros([len(atoms),3])

    com = massCenter(atoms)
    #transfer to center of mass frame
    for i in range(0,len(atoms)):
        for dim in range(0, 3):
            comFrame[i,dim] = atoms[i].center[dim] - com[dim]

    for i in range(0,len(atoms)):
        w = weight[atoms[i].number-1]
        for dim in range(0,3):
            #diagonal elements
            inertiaTensor[dim,dim] += w * (comFrame[i, (dim+1) % 3]**2 + comFrame[i, (dim+2) % 3]**2)
            #off-diagonal elements
            inertiaTensor[dim, (dim+1) % 3] -= w * comFrame[i, dim]*comFrame[i, (dim+1) % 3]
            #symmetrize
            inertiaTensor[(dim+1) % 3, dim] = inertiaTensor[dim, (dim+1) % 3] 
    
    return inertiaTensor

def principalMoments(atoms):
    #compute principal moments of inertia (amu bohr2)

    tensor = inertiaTensor(atoms)

    principalMoments, V = eig(tensor)
    #order ascending
    idx = principalMoments.argsort()

    return principalMoments[idx]

def rotor(atoms):
    #compute rotor type

    epsilon = 1e-8
    pm = principalMoments(atoms)

    type = ''

    if len(atoms) == 2:
        return 'diatomic'
    elif pm[0] < epsilon:
        return 'linear'
    elif (abs(pm[0] - pm[1]) < epsilon) and (abs(pm[1] - pm[2]) < epsilon):
        return 'spherical top'
    elif (abs(pm[0] - pm[1]) < epsilon) and (abs(pm[1] - pm[2]) < epsilon):
        return 'spherical top'
    elif (abs(pm[0] - pm[1]) < epsilon) and (abs(pm[1] - pm[2]) > epsilon):
        return 'oblate spherical top'
    elif (abs(pm[0] - pm[1]) > epsilon) and (abs(pm[1] - pm[2]) < epsilon):
        return 'prolate spherical top'
    else:
        return 'asymmetric top'

def rotationalConstants(atoms):
    #compute the rotational constants (/cm)

    rotor = zeros(3)

    #principal moments
    pm = principalMoments(atoms)

    for dim in range(0, 3):
        rotor[dim] = getConstant('planck')/(8.0*pi*pi*getConstant('c') * \
                     getConstant('bohr->cm') * getConstant('bohr->cm'))
        if pm[dim] != 0:
            rotor[dim] /= pm[dim]* getConstant('dalton->gm')

    return rotor

def nuclearRepulsion(atoms):
    #compute the repulsion of the nucleii

    energy = 0.0
    for i in range(0, len(atoms)):
        for j in range(i+1 , len(atoms)):
            energy += atoms[i].number*atoms[j].number/seperation(atoms,i,j)

    return energy

def nuclearChargeCenter(atoms):
    #compute the center of the nuclear charge
    #total charge
    charge = 0.0
    for i in range(0, len(atoms)):
        charge += atoms[i].number

    chargeCenter = zeros(3)
    for dim in range(0, 3):
        for i in range(0, len(atoms)):
            chargeCenter[dim] += atoms[i].center[dim] * atoms[i].number/charge

    return chargeCenter

def getMass(molAtom, atom):
    #return the atom weight
    
    return weight[molAtom[atom].number-1]

def getNumber(atomicSymbol):
    #get the atomic number from symbol
    i = symbol.index(atomicSymbol)

    return i+1

def gaugeCenter(atoms, mode = 'origin'):
    #set the gauge center

    if isinstance(mode, str):
        if mode == 'origin':
            return array([0,0,0])
        elif mode == 'mass':
            return massCenter(atoms)
        elif mode == 'nuclear charge':
            return nuclearChargeCenter(atoms)
            
    elif isinstance(mode, ndarray):
        return mode

def getConstant(unit):
    unitDict = {'bohr->angstrom' : 0.52917721092, 'picometre->bohr' : 0.018897261339213, 'radian->degree': 180.0/pi,                   \
                'planck' : 6.62607015e-34, 'bohr->cm' : 0.529177249e-10, 'c' : 2.99792458e10 , 'dalton->gm' : 1.6605402e-27,           \
                'em2->amu' : 1822.8884850, 'atu->femtosecond' : 0.02418884254, 'hartree->eV' : 27.21138505, 'au->debye' : 2.541580253, \
                'alpha' : 0.00729735256, 'eV[-1]->nm' : 1239.841701, 'Eh' : 4.359744722207e-18, 'avogadro' : 6.022140857e+23,          \
                'electric constant' : 8.854187817e-12 , 'e' : 1.6021766208e-19, 'bohr magneton' : 9.274009994e-24, 'rydberg->eV':13.6056980659}

    return unitDict[unit]

def zMatrix(input):
    #compute cartesians from z-matrix

    def rodriguez(axis, theta):
        #rotation matrix generator

        axis /= norm(axis)
        psi = cos(theta/2)
    
        i,j,k = -axis * sin(theta/2)
    
        return asarray([[psi*psi+i*i-j*j-k*k,2*(i*j-psi*k),2*(i*k+psi*j)],      \
                           [2*(i*j+psi*k), psi*psi+j*j-i*i-k*k, 2*(j*k-psi*i)], \
                           [2*(i*k-psi*j),2*(j*k+psi*i),psi*psi+k*k-i*i-j*j]])


    def getSymbolValue(s, input):
        #find numeric replacement for symbol

        stream = len(input) -1
        for i in range(stream, -1, -1):
            data = input[i].split()
            if len(data)==0:continue
            if data[0] == s:
                return data[1]
     
        return 'failed to find symbol [', s , ']'

    def isSymbol(s):
        #check if arguement(s) is representation of number [false] or symbol [True]

        sym = False
        try:
            float(s)
        except:
            sym = True
        
        return sym

    def getValue(item, type, input):
        #return numeric value of stretch, bend or dihedral

        if isSymbol(item):
            val = getSymbolValue(item, input)
        else:
            val = item  
        #return as float and in radians for angular type
        val = float(val)
        if type in 'bt': val *= pi/180.0

        return val

    def clean(val, mode):
    #strip out 'X' dummy atoms and blank line at ends

        if mode == 1:
            while True:
                b = False
                for i in range(len(val)):
                    if val[i][1] == -1:
                        del val[i]
                        b = True
                    if b: break
                if not b:break

        if mode == 2:
            for i in range(len(val)):
                if len(val[i]) != 0: break
                del val[i]

            for i in range(len(val)-1, -1):
                if len(val[i]) != 0: break
                del val[i]

        return val

    def processAtom(atom, input, geo=None):
        #process the input line for each atom

        data = input[atom].split()
        z = data[0]

        #non-general atoms
        if atom >=1:
            a = data[1]
            stretch = getValue(data[2],'s', input)
        if atom >= 2:
            b = data[3]
            bend = getValue(data[4],'b', input)
        if atom >= 3:
            t = data[5]
            dihedral = getValue(data[6],'t', input)

        if atom == 0: return z, [0,0,0]
        if atom == 1: return z, [stretch,0,0]
        if atom == 2:   
    
            a = int(data[1])-1
            b = int(data[3])-1
        
            ap = coordinates[a,:]
            bp = coordinates[b,:]

            u = bp - ap
        
            w = stretch*u/norm(u)
            w = dot(rodriguez([0,0,1],bend),w)
        
            return z, w + ap
        
        #general atom processing
        if atom >= 3:
        
            a = int(data[1])-1
            b = int(data[3])-1
            c = int(data[5])-1
    
            ap = coordinates[a,:]
            bp = coordinates[b,:] 
            cp = coordinates[c,:]

            u = bp-ap
            v = bp-cp

            #vector a->b length s
            w = stretch*u/norm(u)

            #normal to plane abc
            n = cross(u, v)
            #rotation of w by b about n
            w = dot(rodriguez(n,bend),w)
            #rotation of w by dihedral about u
            w = dot(rodriguez(u,dihedral),w)

        #return atomic symbol and coordinates
        return z, w + ap    

    #strip control character '\t' and calculate number of atoms
    bAtom = True
    atoms = 0

    input = clean(input,2) #strip leading and trailing blank lines
    for i in range(len(input)):
        input[i] = input[i].replace('\t','    ')
        
        #first blank line is start of symbols and end of atom definitions
        if bAtom and (len(input[i]) == 0):
            bAtom = False
            atoms = i
    #no symbols
    if bAtom: atoms = len(input)

    #define array for geometry
    coordinates = zeros((atoms,3))
    z = []

    for i in range(atoms):
        s, coordinates[i,:] = processAtom(i, input, coordinates)
        if s != 'X': n = getNumber(s)
        else: n = -1
        z.append([str(i+1), n, coordinates[i,0],coordinates[i,1],coordinates[i,2]])

    #remove dummy atoms from final list and return [id, Z, x, y, z]

    return clean(z,1)

def atomList(molAtom):
    #generate a list of atom type for each atom in molecule
    atoms = []
    for a in molAtom:
        atoms.append(a.number)

    return atoms