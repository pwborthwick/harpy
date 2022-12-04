from __future__ import division
import rhf
from numpy import zeros,pi, asarray, trace, dot, savez, load, all, sqrt
from numpy.linalg import norm, solve
from atom import getConstant, getNumber, bondMatrix
from post import buildElectricField
from view import pre, post, postSCF
import os

def writerESP(mData, rData, qc, qr, metrics, stage):
    #write results to 'harpy.html'

    if stage == '1':
        name = mData['name']
        pre(name, 'rhf')
        postSCF([mData, rData, qc, qr, metrics, stage], 'resp')
    if stage == '2':
        postSCF([mData, rData, qc, qr, metrics, stage], 'resp')


def deltaRMS(esp, qr , invRadii):
    #calculate root mean square difference between classical restrained charges 
    #and qm derived charges

    rms = 0.0
    for p in range(invRadii.shape[0]):

        # q/r classical
        q = 0.0
        for atom in range(invRadii.shape[1]):
            q += qr[atom] * invRadii[p, atom]

        rms += pow(q - esp[p],2)

    return sqrt(rms/esp.shape[0])



def equiDistribution(points=45, variant='con'):

    from math import cos, sin, pi, sqrt

    surfacePoints = []

    if variant == 'fib':
        #based on fibonacci sphere algorithm

        dTheta = pi*(3.0-sqrt(5.0))
        dZ   =  2.0/points
        theta =  0.0
        z    =  1.0 - dZ/2.0

        for i in range(points): 

            r = sqrt(1.0 - z*z)
            p = (cos(theta)*r, sin(theta)*r, z)
            surfacePoints.append(p)

            #update positions
            z    = z - dZ
            theta += dTheta

    elif variant == 'con':
        #connolly algorithm

        nxy = int(sqrt(pi*points))
        nz = int(nxy/2)

        tol = 1e-10
        nu = 0

        for i in range(nz + 1):

            phi = pi*i/nz
            z = cos(phi)
            xy = sin(phi)
            nh = int(nxy * xy + tol)

            if nh < 1: nh = 1

            for j in range(nh):

                psi = 2*pi*j/nh
                x = cos(psi)*xy
                y = sin(psi)*xy

                if nu >= points: return surfacePoints

                nu += 1
                surfacePoints.append((x , y , z ))

    return surfacePoints

def surfaceView(distribution):
    #plot the molecular VdW surface
    
    import pylab as p
    import mpl_toolkits.mplot3d.axes3d as p3

    fig=p.figure()
    ax = p3.Axes3D(fig)
    x_s=[];y_s=[]; z_s=[]

    for i in range(len(distribution)):
        x_s.append( distribution[i][0]); y_s.append( distribution[i][1]); z_s.append( distribution[i][2])

    ax.scatter3D( asarray( x_s), asarray( y_s), asarray( z_s))   

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.view_init(azim=60, elev=60)
    p.show()

def respData(molAtom, molBasis, molData, options):
    #restrained electrostatic potentials

    vdwRadius = [1.20, \
                1.20, 1.37, 1.45, 1.45, 1.50, 1.50, 1.40, 1.35, 
                1.30, 1.57, 1.36, 1.24, 1.17, 1.80, 1.75, 1.70 ]

    if 'points' not in options:    options['points']    = ['density', 1.0]
    if 'shell' not in options :    options['shell']     = [1, 0.0, 1.0]
    if 'file' not in options  :    options['file']      = ['w', 'esp.npz', 'clear']
    if 'constrain' not in options: options['constrain'] = []  
    if 'radii' not in options:     options['radii']     = []
    if 'sphere' not in options:    options['sphere']    = 'con'
    if 'view' not in options:      options['view']      = False

    #work in Angstroms
    charge = [molData['charge']]
    nAtoms = len(molAtom)
    if options['radii'] != []:
        for i in options['radii'] :
            vdwRadius[getNumber(i[0])-1] = i[1]

    #geometry and Van der Waals scaled to bohr
    geo = zeros((nAtoms,3))
    radii = zeros(nAtoms)

    for i in range(nAtoms): 
        geo[i,:] = molAtom[i].center[:] * getConstant('bohr->angstrom')
        radii[i] = vdwRadius[molAtom[i].number - 1]

    distribution = []
    esp = []

    #number of points to evaluate if not function of radius
    if options['points'][0] == 'atom':
        nPoints = int(options['points'][1])

    for iShell in range(options['shell'][0]):

        scaleFactor = options['shell'][2] + (iShell * options['shell'][1])

        for i in range(nAtoms): 

            #effective shell radius accounting for scale factor
            shellRadius = radii[i] * scaleFactor

            #density depends on radius - recalculate for scaling
            if options['points'][0] == 'density':
                nPoints = int(4.0 * pi * shellRadius * shellRadius * options['points'][1])

            surface = asarray(equiDistribution(points = nPoints, variant = options['sphere']) )

            #translate to atom center
            for j in range(len(surface)):
                surface[j][:] = surface[j][:] * shellRadius + geo[i,:] 

            #remove points inside other atoms
            for j in range(len(surface)):

                valid = True

                for k in range(nAtoms):

                    if i == k:  continue

                    if (norm(surface[j][:] - geo[k,:])) < (radii[k] * scaleFactor): 
                        valid = False
                        break

                if valid: 
                    distribution.append(surface[j])

    if options['view']: surfaceView(distribution)

    #build [points, atoms] matrix of inverse distances for each point to each atom
    #switch to bohr for electrostatic calculation
    invRadii = zeros((len(distribution), nAtoms))
    for p in range(len(distribution)):
        for atom in range(nAtoms):
            invRadii[p, atom] = 1.0/norm(distribution[p][:] - geo[atom][:]) 
            invRadii[p, atom] *= getConstant('bohr->angstrom')

    #we need density so do a scf calculation
    e = rhf.scf(molAtom, molBasis, molData,[])

    #compute the electrostatic potentials
    for p in range(len(distribution)):

        x, y, z = distribution[p] / getConstant('bohr->angstrom')
        t = buildElectricField(molAtom, molBasis, rhf.density, asarray([x, y, z]))

        #build return electric components [0-3] and nuclear [4-7], field first then potential
        esp.append(t[7]-t[3])

    if options['file'][0] == 'w':
        savez(options['file'][1], distribution = distribution, esp = esp, inv = invRadii, n = nPoints)


def buildmatrix(invRadii, constraintLength, nAtoms):

    n = nAtoms + 1 + constraintLength
    a = zeros((n,n))

    for i in range(nAtoms):
        for j in range(nAtoms):
            for k in range(invRadii.shape[0]):
                a[i,j] += invRadii[k,i] * invRadii[k,j]

    a[:nAtoms, nAtoms] = 1.0
    a[nAtoms, :nAtoms] = 1.0

    return a

def buildvector(invRadii, esp, constraintLength, nAtoms, charge):

    n = nAtoms + 1 + constraintLength
    b = zeros(n)
    for i in range(nAtoms):
        for j in range(invRadii.shape[0]):
            b[i] += esp[j] * invRadii[j,i]

    b[nAtoms] = charge

    return b


def doRestraint(q, a, parameters, z):
    #apply hyperbolic restraints

    A = a.copy()

    for atom in range(len(z)):

        if not parameters['h'] or z[atom] != 1 :
            A[atom, atom] = a[atom, atom] + (parameters['a'] / sqrt(q[atom]*q[atom] + parameters['b']*parameters['b']))

    return A

def carbonGroups(molAtom):
    #get all groups with hydrogen attached to carbon

    groups = []
    for atom in range(len(molAtom)):

        if molAtom[atom].number == 6:
            group = []
            #get connections
            row = bondMatrix(molAtom)[atom,:]
            for i in range(row.shape[0]):
                if (row[i] == 1) and (molAtom[i].number == 1): group.append(i+1)

            groups.append([atom+1, group])

    return groups

def stageTwoConstraints(options, groups, nAtoms, q):
    #translate groups into constraints

    if 'a' in options['stage two']: options['restrain']['a'] = options['stage two']['a']
    else:  options['restrain']['a'] = 0.001
    if 'b' in options['stage two']: options['restrain']['b'] = options['stage two']['b']

    #build new constraints
    options['constrain'] = []
    atomlist = list(range(1, nAtoms+1))

    #for each CH grouping constrain hydrogens to be equal [0, [-n,m,..]]
    for group in groups:
        options['constrain'].append([0, group[1]])
        for i in group[1]: atomlist.remove(i)
        atomlist.remove(group[0])
        options['constrain'][-1][1][0] *= -1

        #other atoms take restained charge
        for i in atomlist:
            options['constrain'].append([q[i-1],[i]])

    return options

def doRestraintIteration(q, a, b, parameters, z):
    #main loop for restraints

    Q = q.copy()
    nAtoms = len(z)
    iteration = 0

    while iteration < parameters['cycles'] :

        iteration += 1
        A = doRestraint(q, a, parameters, z)

        q = solve(A, b)
        
        if sqrt( max(pow(q[:nAtoms] - Q[:nAtoms],2))) < parameters['tol'] : return q[:nAtoms], iteration

        Q = q.copy()

    return -1, iteration

def processConstraints(option):
    #expand equal charge constraints

    if option == [] : return [], []

    chargeGroup = []
    indexGroup = []

    for i in option:
        if i[1][0] > 0:
            chargeGroup.append(i[0])
            indexGroup.append(i[1])
        else:
            for j in range(len(i[1])-1):
                chargeGroup.append(0)
                group = []
                group.append(-abs(i[1][j]))
                group.append((i[1][j+1]))
                indexGroup.append(group)

    return chargeGroup, indexGroup

def rESPsolve(molAtom, nAtoms, charge, options, invRadii, esp):
    #do the constraint and restraint solve

    #process constraints
    conCharge, conIndex = processConstraints(options['constrain'])

    constraintLength = len(conCharge)

    #construct a-matrix with total charge constraints
    a = buildmatrix(invRadii, constraintLength, nAtoms)
    b = buildvector(invRadii, esp, constraintLength, nAtoms, charge)

    #add applied constraints
    for i in range(constraintLength):
        b[nAtoms +1+i] = conCharge[i]

        for j in conIndex[i] :
            if j > 0:
                a[nAtoms+1+i, j-1] = 1
                a[j-1, nAtoms+1+i] = 1
            else:
                a[nAtoms+1+i, -j-1] = -1
                a[-j-1, nAtoms+1+i] = -1

    #remove zero rows or columns from a
    a = a[~all(a == 0, axis=1)]
    a = a[~all(a == 0, axis=0)]

    #solve
    try:
        q = solve(a,b)
    except:
        print('solve failure in rESP')
        exit('solve')

    constrainedCharges = q[:nAtoms]

    #atoms types
    z = zeros(nAtoms)
    for atom in range(nAtoms):
        z[atom] = molAtom[atom].number

    if 'restrain' in options:
        q, nCycles = doRestraintIteration(q, a, b, options['restrain'], z)
        if q[0] == -1: 
            print ('restraint iteration failure in rESP')
            exit('iteration')

        restrainedCharges = q[:nAtoms]

        rms = deltaRMS(esp, restrainedCharges, invRadii)

    else: rms = deltaRMS(esp, constrainedCharges, invRadii)

    return constrainedCharges, restrainedCharges, nCycles, rms

def restrainedESP(molAtom, molBasis, molData, options):

    #get molecular objects and options - passed values

    nAtoms = len(molAtom)
    charge = molData['charge']
    
    if options['file'][0] == 'w':

        respData(molAtom, molBasis, molData, options )
        options['file'][0] = 'r'


    if options['file'][0] == 'r':
        #get data from npz file
        data = load(options['file'][1])
        distribution = data['distribution']
        esp = data['esp']
        invRadii = data['inv']
        nPoints = data['n']

        #restraint options (set defaults if necessary)
        if 'restrain' not in options:  options['restrain']  = {}
        else:
            if 'a' not in options['restrain']:      options['restrain']['a'] = 5e-4
            if 'b' not in options['restrain']:      options['restrain']['b'] = 0.1
            if 'h' not in options['restrain']:      options['restrain']['h'] = True
            if 'tol' not in options['restrain']:    options['restrain']['tol'] = 1e-6
            if 'cycles' not in options['restrain']: options['restrain']['cycles'] = 30

        #solve the system (stage one)
        constrainedCharges, restrainedCharges, nCycles, rms = rESPsolve(molAtom, nAtoms, charge, options, invRadii, esp)

        stagetwo = 'stage two' in options

        writerESP(molData, options, constrainedCharges, restrainedCharges, [nPoints, nCycles, rms], '1')

        #solve for stage two if selected
        if stagetwo:

            groups = carbonGroups(molAtom)
            if groups == []: exit('no stage two')

            options = stageTwoConstraints(options, groups, nAtoms, restrainedCharges)

            constrainedCharges, restrainedCharges, nCycles, rms = rESPsolve(molAtom, nAtoms, charge, options, invRadii, esp)
            writerESP(molData, options, constrainedCharges, restrainedCharges, [nCycles, rms], '2')

    #html Output file exit and clear data file
    post()
    if options['file'][2] == 'clear': os.remove(options['file'][1])

    return constrainedCharges, restrainedCharges

