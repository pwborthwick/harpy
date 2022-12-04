from __future__ import division
#import system modules
from scipy.linalg import fractional_matrix_power as fractPow
import sys

#import class modules
from atom import atom, nuclearRepulsion, zMatrix, getConstant, atomList
from basis import checkBasis, buildBasis, electronCount
#import user modules
from view import geometry, orbitals, pre, preSCF, SCF, post, postSCF
from integral import buildOverlap, buildKinetic, buildCoulomb, buildEri, \
                     buildHamiltonian, buildFock, buildDensity
from diis import diis_f
from post import charges, energyPartition, buildDipole, polarizabilities, buildQuadrupole, hyperPolarizabilities, \
                 bondOrder, buildMp2Dipole
from mp import mollerPlesset, orbitalOptimisedMP2, mp2LaplaceTransform
from mbpt import mbptEvaluateMPn
from ci import cis, ciSpinAdaptedSingles, ciSpinAdaptedTriples, randomPhaseApproximation
from cc.scc import ccsd, lccd, ccd, cc2, lccsd
from force import forces
from ep import electronPropagator2, electronPropagator2spin, electronPropagator3spin, koopmanAGFcorrection
from fci import fci, cisd, ciss
from cc.fcc import fastCoupledCluster
from cc.cctn import coupledClusterTriplesVariations
import numpy as np


def rmsDensity(Da, Db):
    #compute rms difference of last two cycles of density matrix
    n = Da.shape[0]
    sum = 0.0

    for i in range(0, n):
        for j in range(0, n):
            sum += pow((Da[i,j] - Db[i,j]),2)

    return np.sqrt(sum)

def mol(show, file = 'project.hpf', molecule='', method='rhf'):
    #read input file

    coordinates = []
    cartesian = False

    #loop over input file
    with open(file,'r') as data:
        f = data.read().split('\n')

        #commands, options and check flags
        valid   = {'name':[False,'string'],'basis':[False,'string'],'diis':[False,'on','off'],'engine':[False,'aello','native'], \
                   'gauge':[False,'origin','nuclear','mass'],'guess':[False,'core','gwh'],'charge':[False,'int'],'multiplicity':[False,'int'], \
                   'cycles':[False,'int'],'capacity':[False,'init'],'tolerance':[False,'float'],'uhfmix':[False, 'float'],'post':[False,'string'],'cogus':[False,'string'], \
                   'matrix':[False,'c','z'],'units':[False,'bohr','angstrom'],'no_mint_density':[False]}
        values  = {'name':'','basis':'sto-3g','diis':'on','engine':'aello','gauge':'origin','guess':'core','charge':0,'multiplicity':1, \
                   'cycles':100,'capacity':8,'tolerance':1e-8,'uhfmix':0.0,'post':'','cogus':'{}','matrix':'','units':'bohr','no_mint_density':False}

        for nline, line in enumerate(f):
            
            line = line.strip()
            
            #find requested molecule
            if molecule != '':
                if not ('name=' + molecule) == line:
                    continue
                else:
                    molecule = ''

            if line.strip() == 'end': break
            
            if (len(line.split()) != 0) and (line.split()[0][0] == '#'): continue

            if '=' in line:

                #string commands
                if any(element in line for element in ['diis','engine','gauge','guess', 'matrix','units']):

                    data = line.split('=')
                    key = data[0]
                    value = data[1]
                    if value in valid[key]:
                        values[key] = value
                        valid[key][0] = True   
                        if value == 'c': cartesian = True

                #integer commands
                if any(element in line for element in ['charge', 'multiplicity','cycles','capacity']):

                    data = line.split('=')
                    key = data[0]
                    if data[1].replace('-','',1).replace('+','',1).isnumeric():
                        values[key] = int(data[1])
                        valid[key][0] = True

                #float command
                if any(element in line for element in ['tolerance', 'uhfmix']):

                    data = line.split('=')
                    key = data[0]
                    if data[1].replace('e','',1).replace('-','',1).replace('.','').isnumeric():
                        values[key] = float(data[1])
                        valid[key][0] = True

                if any(element in line for element in['name','basis','post','cogus']):

                    data = line.split('=')
                    key = data[0]
                    values[key] = data[1]
                    valid[key][0] = True

            elif 'no_mint_density' in line:
                data = line.split('=')
                key = data[0]
                values[key] = True
                valid[key][0] = True
            else:

                #process atom definitions - 'cartesian' flags cartesian or z-matrix
                data = line.split()
                if (len(data) != 0):
                    if cartesian:
                        coordinates.append([data[0], int(data[1]), float(data[2]), float(data[3]), float(data[4])])             
                if not cartesian:
                    coordinates.append(line)

    #check options and defaults [no default for matrix - critical error]
    for i in valid:
        if i[0] == 'False':
            exit('critical input error')

    #construct atom object array
    molAtom = []

    #convert to bohr if necessary
    conversion = 1.0
    if values['units'] == 'angstrom':
        conversion = 1.0/getConstant('bohr->angstrom')

    if cartesian:
        for i in range(len(coordinates)):
            molAtom.append(atom(coordinates[i][0], coordinates[i][1], [coordinates[i][2]*conversion, \
                                                                       coordinates[i][3]*conversion, \
                                                                       coordinates[i][4]*conversion]))
    else:
        z = zMatrix(coordinates)
        for i in range(len(z)):
            molAtom.append(atom(z[i][0], int(z[i][1]), [float(z[i][2])*conversion, \
                                                        float(z[i][3])*conversion, \
                                                        float(z[i][4])*conversion]))

    #check to see if valid basis
    basisName = values['basis']
    if not checkBasis(basisName,molAtom):
        exit('basis not in library')

    #get charge
    charge = values['charge'] ; multiplicity = values['multiplicity']

    #build molecular basis
    molBasis = buildBasis(molAtom, basisName)

    #write output if requested
    name = values['name']
    if show != []:
        pre(name, method)
        if 'geometry' in show:
            geometry(molAtom)
        #output orbital and basis information
        if 'orbitals' in show:
            orbitals(molAtom, charge, multiplicity, basisName, molBasis, method)

    #return atom object array, basis object array and [name,basis,diis,integral engine,gauge origin, 
    # initial hamiltonian guess,charge,multiplicity,cycles,tolerance,,units]

    return molAtom, molBasis, values

def rebuildCenters(molAtom, molBasis, geo):
    #a change of geometry means atom.center and  basis.center need updating
    for atom in range(0, len(molAtom)):
        molAtom[atom].center = geo[atom, :]

        for basis in range(0, len(molBasis)):
            if molBasis[basis].atom == atom:
                molBasis[basis].center = geo[atom, :]

    return molAtom, molBasis

def scf(molAtom, molBasis, run, show, cptDensity = None):

    global gaugeOrigin 

    #options
    name, basisName, diisStatus, integralEngine, gaugeOrigin, hamiltonianGuess, charge, multiplicity, \
          iterations, diisCapacity, convergence, uhfmix, p, cogus, matrix, units, no_mint_density = run.values()
    cogus = cogus.replace('{','').replace('}','')

    if gaugeOrigin == 'nuclear': gaugeOrigin = 'nuclear charge'

    #check closed shell
    if (electronCount(molAtom, charge) % 2) != 0:
        exit('open shell molecule')

    #globals so molecular dynamics can be run seperately
    global C, fock, density, S, ERI, coreH, SCFenergy, e
    
    #pre-SCF matrices - S{overlap}  T{1e kinetic/exchange/resonance}  V{1e Coulomb} ERI(2e electron repulsion)
    if integralEngine == 'native':
        S = buildOverlap(molBasis)
        T = buildKinetic(molBasis)
        V = buildCoulomb(molAtom, molBasis)
        ERI = buildEri(molBasis)
    elif integralEngine == 'aello':
        from aello import aello
        S, T, V, ERI = aello(molAtom, molBasis)

    #orthogonalising matrix X
    X = fractPow(S, -0.5)

    #initial core guess
    coreH, _ = buildHamiltonian(hamiltonianGuess,S,T,V)
    e, orthogonalC = np.linalg.eigh(np.dot(X.T, np.dot(coreH, X)))

    C = np.dot(X, orthogonalC)

    occupiedOrbitals = electronCount(molAtom, charge)//2
    basisFunctions = len(molBasis)

    #check-point density - use if there
    if (type(cptDensity) != np.ndarray) or (cptDensity.shape[0] != S.shape[0]) or not no_mint_density:
        density = buildDensity(basisFunctions, occupiedOrbitals, C)
    else:
        density = cptDensity

    #define storage for diis
    if diisStatus == 'on':
        diis = diis_f(diisCapacity)


    #the SCF loop
    for cycle in range(0 , iterations):

        #build initial fock matrix as core Hamiltonian
        fock, G = buildFock(coreH, ERI, density, integralEngine)

        #do diis if selected
        if (cycle != 0) and (diisStatus == 'on'):
            fock = diis.build(fock, density, S, X)

        #compute SCF energy
        energy = 0.0
        for i in range(0, basisFunctions):
            for k in range(0, basisFunctions):
                energy += density[i,k] * (fock[i,k] + coreH[i,k])

        #orthogonalise Fock
        orthogonalFock = np.dot(X.T , np.dot(fock , X))

        #diagonalise Fock
        e, orthogonalC = np.linalg.eigh(orthogonalFock)

        #transform eigenvalues back to non-orthogonal AO basis
        C = np.dot(X,orthogonalC)

        #build density matrix
        density = buildDensity(basisFunctions, occupiedOrbitals, C)

        #convergence control
        if cycle != 0:
            deltaEnergy = np.abs(preEnergy - energy)
            deltaDensity = rmsDensity(preDensity, density)
            if 'SCF' in show:
                SCF(energy, deltaEnergy, deltaDensity, cycle, diisStatus, iterations, convergence)
            if (deltaEnergy < convergence) and (deltaDensity < convergence) :
                break
                
        preEnergy = energy
        preDensity = density

        #output pre-SCF details
        if cycle==0 and ('preSCF' in show):
            preSCF(S, T, V, ERI, X, orthogonalFock, density, energy, hamiltonianGuess)

    #if failed to converge exit with messages
    if (cycle + 1) == (iterations):
        print('SCF failed to converge in ' + str(cycle + 1) + ' iterations')
        if diisStatus == 'off': print('Try diis = \'on\'')

        post(False)
        sys.exit('convergence failure')

    #final eigensolution of final Fock matrix are orbital energies (e) and MO coefficients(C)
    SCFenergy = energy + nuclearRepulsion(molAtom)

    if 'SCF' in show:
        postSCF([SCFenergy, cycle, C, e, density],'eigen')

    if 'postSCF' in show:

        if 'ch' in p: charges(density ,S, molAtom, molBasis)
        if 'bo' in p: bondOrder(density, S, molAtom, molBasis)
        if 'en' in p: energyPartition(energy, nuclearRepulsion(molAtom), density, T, V, G, e, \
                        electronCount(molAtom,charge))
        if 'di' in p: 
                      buildDipole(molAtom, molBasis, density, gaugeOrigin, integralEngine)
                      buildMp2Dipole(molAtom, molBasis, gaugeOrigin, C, e, ERI, occupiedOrbitals)
        if 'qu' in p: buildQuadrupole(molAtom, molBasis, density, gaugeOrigin)

        if 'mp' in p: mollerPlesset(molAtom, charge, molBasis, C, e, fock, ERI, SCFenergy)
        if 'om' in p: orbitalOptimisedMP2(C, coreH, e, molBasis, nuclearRepulsion(molAtom), ERI, electronCount(molAtom,charge))
        if 'ml' in p: mp2LaplaceTransform(molBasis, C, ERI, e, occupiedOrbitals, SCFenergy)
        if 'mb' in p: mbptEvaluateMPn(molAtom, molBasis, run, SCFenergy)

        if 'ci' in p: cis(molAtom,charge,molBasis,C,fock,ERI)
        if 'ss' in p: ciSpinAdaptedSingles(molAtom,charge,molBasis,C,fock,ERI)
        if 'st' in p: ciSpinAdaptedTriples(molAtom,charge,molBasis,C,fock,ERI)
        if 'rp' in p: randomPhaseApproximation(molAtom,charge,molBasis,C,fock,ERI,type='block')
        if 'fc' in p: fci(molAtom, molBasis, charge, C, ERI, coreH)
        if 'sd' in p: cisd(molAtom, molBasis, charge, C, ERI, coreH)
        if 'cs' in p: ciss(molAtom, molBasis, charge, C, ERI, coreH)

        if 'ct' in p: ccsd(molAtom, C, charge, fock, ERI, iterations, convergence, SCFenergy, diisStatus)
        if 'c2' in p: cc2(molAtom, C, charge, fock, ERI, iterations, convergence, SCFenergy, diisStatus)
        if 'lc' in p: lccd(molAtom, C, charge, fock, ERI, iterations, convergence, SCFenergy)
        if 'cd' in p: ccd(molAtom, C, charge, fock, ERI, iterations, convergence, SCFenergy)
        if 'ld' in p: lccsd(molAtom, C, charge, fock, ERI, iterations, convergence, diisStatus, SCFenergy)
        
        if 'fo' in p: forces(molAtom, molBasis, density, fock, integralEngine, 'both')

        if 'ep' in p: electronPropagator2(molBasis ,C ,ERI ,e , occupiedOrbitals ,2 ,4 )
        if 'e2' in p: electronPropagator2spin(molBasis, C, ERI, e, occupiedOrbitals, 5)
        if 'e3' in p: electronPropagator3spin(molBasis, C, ERI, e, occupiedOrbitals, 5)
        if 'ko' in p: koopmanAGFcorrection(molBasis ,C ,ERI, e, occupiedOrbitals, 5)

        if 'po' in p: polarizabilities(molAtom, molBasis, C, fock, ERI, occupiedOrbitals, gaugeOrigin)
        if 'hy' in p: hyperPolarizabilities(molAtom, molBasis, C, ERI, e, fock, occupiedOrbitals, gaugeOrigin)

        #symbolic generated code
        if cogus != '':
            from cogus import symbolicGeneratedCoupledCluster

            repertoire = {'d':'ccd','sd':'ccsd','sd(t)':'ccsd_t','2':'cc2','ld':'lccd','lsd':'lccsd','3':'cc3', 'sdt':'ccsdt'}            
            cogusList = cogus.split(',')
            results = []

            for cogusItem in cogusList:
                name, cluster, perturbation = symbolicGeneratedCoupledCluster(repertoire[cogusItem], fock, ERI, C, e, \
                                        [charge, nuclearRepulsion(molAtom), electronCount(molAtom,charge)])
                results.append([('CC'+cogusItem.upper()).replace('(t)','(T)').replace('CCL','LCC'), cluster, perturbation])
            
            postSCF(results, 'cogus')

        #process any fast coupled-cluster
        if '+' in p:
            repertoire = {'+d':'ccd','+s':'ccsd','+t':'ccsd(t)','+2':'cc2','+l':'lccd','+L':'lccsd','+^':'lambda', '+q':'qcisd'}
            data = {'cycle_limit': 50, 'precision':1e-10, 'electrons': electronCount(molAtom,charge)}
            for key in repertoire.keys():
                method = repertoire[key]
                if key in p: fastCoupledCluster(method, fock, e, C, ERI, nuclearRepulsion(molAtom), data)

        #process triples and triples approximations
        if '-' in p:
            repertoire = {'-T':'ccsdt','-t':'ccsd(t)','-1a':'ccsdt-1a','-1b':'ccsdt-1b','-2':'ccsdt-2','-3':'ccsdt-3','-4':'ccsdt-4'}
            for key in repertoire.keys():
                method = repertoire[key]
                if key in p: coupledClusterTriplesVariations(method, fock, ERI, C, e, 
                    [charge, nuclearRepulsion(molAtom), electronCount(molAtom,charge)],
                    [True, 50, 1e-10, True])

    #save molecular integrals to zip archive
    if 'mints' in show:
        np.savez_compressed('../mints/' + name + '-' + basisName + '-mints.npz', s=S, k=T, j=V, i=ERI, \
                         f=fock, d=density, c=C, e=e, E=SCFenergy, m=[charge, name, basisName, nuclearRepulsion(molAtom)], \
                         a=atomList(molAtom))

    #clean up outfile
    if show != []: post()

    return SCFenergy

