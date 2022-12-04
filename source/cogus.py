from __future__ import division
import numpy as np

internalLambdaCode ="""
#declarations
import numpy as np

def spatialTospin(eriMO, nbf):
    #openfermion style transformation

    import numpy as np

    spin = np.zeros((2*nbf, 2*nbf, 2*nbf, 2*nbf))

    eriMO = eriMO.transpose(0,2,3,1)
    for p in range(nbf):
        for q in range(nbf):
            for r in range(nbf):
                for s in range(nbf):
                    #anti-spin
                    spin[2*p, 2*q+1, 2*r+1, 2*s], spin[2*p+1, 2*q, 2*r, 2*s+1] = [eriMO[p,q,r,s]] * 2
                    #syn-spin
                    spin[2*p, 2*q, 2*r, 2*s], spin[2*p+1, 2*q+1, 2*r+1, 2*s+1] = [eriMO[p,q,r,s]] * 2

    return spin

def gMOspin(e, c, eri, nbf):
    #construct MO spin eri

    import numpy as np

    def iEri(i,j,k,l):
        #index into the four-index eri integrals
        p = max(i*(i+1)/2 + j, j*(j+1)/2 + i)
        q = max(k*(k+1)/2 + l, l*(l+1)/2 + k)
        return  int(max(p*(p+1)/2 + q, q*(q+1)/2 + p))

    #get 4 index eri and spinblock to spin basis
    g = np.zeros((nbf,nbf,nbf,nbf))
    for i in range(nbf):
        for j in range(nbf):
            for k in range(nbf):
                for l in range(nbf):
                    g[i,j,k,l] = eri[iEri(i,j,k,l)]
 
    #eri to MO
    g = np.einsum('pQRS,pP->PQRS', np.einsum('pqRS,qQ->pQRS', np.einsum('pqrS,rR->pqRS', np.einsum('pqrs,sS->pqrS', \
    g, c, optimize=True), c, optimize=True), c, optimize=True), c, optimize=True)

    return g, np.kron(e, np.ones(2))

iterations = 50
tolerance = 1e-10

charge, nuclearRepulsion, electrons = scfData

#orbital occupations
spinOrbitals = (fock.shape[0]) * 2
nsocc = electrons
nsvir = spinOrbitals - nsocc

#get one electron operators
h1 = np.dot(c.T, np.dot(hcore, c))
hcSpin = np.kron(h1, np.eye(2))
nbf = h1.shape[0]

#get fock in MO spin basis
cSpin = np.kron(c, np.eye(2))
fock = np.dot(cSpin.T, np.dot(np.kron(fock, np.eye(2)), cSpin))

#get two-electron repulsion integrals in MO basis
eriMO, eps = gMOspin(e, c, eri, spinOrbitals//2)
eriMOspin = spatialTospin(eriMO, spinOrbitals//2)
g = (np.einsum('ijkl', eriMOspin) - np.einsum('ijlk', eriMOspin)).transpose(0, 1, 3, 2)

#slices
n = np.newaxis
o = slice(None,nsocc)
v = slice(nsocc, None)

#D tensors
d_ai = 1.0 / (-eps[v, n] + eps[n, o])
d_abij = 1.0 / (-eps[v, n, n, n] - eps[n, v, n, n] + eps[n, n, o, n] + eps[n, n, n, o])
d_abcijk = 1.0 / (- eps[ v, n, n, n, n, n] - eps[n, v, n, n, n, n] - eps[n, n, v, n, n, n]
                   + eps[n, n, n, o, n, n] + eps[n, n, n, n, o, n] + eps[n, n, n, n, n, o] )

#HF energy
HFenergy = 1.0 * np.einsum('ii', fock[o, o]) -0.5 * np.einsum('ijij', g[o, o, o, o])

#amplitude initialisation
ts = np.zeros((nsvir, nsocc))
td = np.zeros((nsvir,nsvir,nsocc,nsocc))

#get initial cluster energy
lastCycleEnergy =  cc_energy(fock, g, o, v, t1=ts, t2=td, t3=None )

#dummies
triples = None

#iterations
for cycle in range(iterations):

    #update amplitudes
    singles = cc_singles(fock, g, o, v, t1=ts, t2=td, t3=None) * d_ai + ts 
    doubles = cc_doubles(fock, g, o, v, t1=ts, t2=td, t3=None) * d_abij + td

    #recalculate energy
    cycleEnergy = cc_energy(fock, g, o, v, t1=singles, t2=doubles, t3=None)
    deltaEnergy = np.abs(lastCycleEnergy - cycleEnergy)

    #convergence test
    if deltaEnergy < tolerance:

        cycleEnergy =  cc_energy(fock, g, o, v, t1=singles, t2=doubles, t3=None) - HFenergy
        ts = singles
        td = doubles
        break
    else:
        ts = singles
        td = doubles
        lastCycleEnergy = cycleEnergy
else:
    print("Did not converge")
    exit('cc failed')

#for lagrange amplitudes l are transposes of t
d_ia   = d_ai.transpose(1,0)
d_ijab = d_abij.transpose(2,3,0,1)

#initial values for lagrange amplitudes
ls = ts.transpose(1,0)
ld = td.transpose(2,3,0,1)

lastCycleEnergy = cc_lambda_lagrangian_energy(fock, g, o, v, t1=ts, t2=td, l1=ls, l2=ld)

#iterations
for cycle in range(iterations):

    singlesResidual = cc_lambda_singles(fock, g, o, v, ts, td, ls, ld)
    doublesResidual = cc_lambda_doubles(fock, g, o, v, ts, td, ls, ld)

    lambdaResidual = np.linalg.norm(singlesResidual) + np.linalg.norm(doublesResidual)

    singles = singlesResidual * d_ia + ls
    doubles = doublesResidual * d_ijab + ld

    lambdaCycleEnergy = cc_lambda_lagrangian_energy(fock, g, o, v, ts, td, singles, doubles)
    pseudoEnergy = 0.25 * np.einsum('jiab,jiab', g[o, o, v, v], ld)

    energyDelta = np.abs(lastCycleEnergy - lambdaCycleEnergy)

    if energyDelta < 1e-10 and lambdaResidual < 1e-10:
        ls = singles
        ld = doubles
        lambdaCycleEnergy =  cc_lambda_lagrangian_energy(fock, g, o, v, ts, td, ls, ld) - HFenergy
        break
    else:
        ls = singles
        ld = doubles
        lastCycleEnergy = lambdaCycleEnergy
else:
    print("Did not converge")
    exit('lambda failed')

#response density matrices
opdm = cc_oprdm(o, v, ts, td, ls, ld)
tpdm = cc_tprdm(o, v, ts, td, ls, ld)

rdmEnergy = np.einsum('ij,ij', hcSpin, opdm) + 0.25 * np.einsum('ijlk,ijlk',tpdm, g)- HFenergy
"""

internalSpinClusterCode = """
#declarations
import numpy as np

g = eri
eps = e

iterations = 50
tolerance = 1e-10

charge, nuclearRepulsion, electrons = scfData

#orbital occupations
spinOrbitals = (fock.shape[0])
nsocc = electrons
nsvir = spinOrbitals - nsocc

#slices
n = np.newaxis
o = slice(None,nsocc)
v = slice(nsocc, None)

#D tensors
eps = np.kron(e, np.ones(2))

d_ai = 1.0 / (-eps[v, n] + eps[n, o])
d_abij = 1.0 / (-eps[v, n, n, n] - eps[n, v, n, n] + eps[n, n, o, n] + eps[n, n, n, o])
d_abcijk = 1.0 / (- eps[ v, n, n, n, n, n] - eps[n, v, n, n, n, n] - eps[n, n, v, n, n, n]
                   + eps[n, n, n, o, n, n] + eps[n, n, n, n, o, n] + eps[n, n, n, n, n, o] )

#HF energy
HFenergy = 1.0 * np.einsum('ii', fock[o, o]) -0.5 * np.einsum('ijij', g[o, o, o, o])

#amplitude initialisation
ts = np.zeros((nsvir, nsocc))
td = np.zeros((nsvir,nsvir,nsocc,nsocc))
tt = np.zeros((nsvir,nsvir,nsvir,nsocc,nsocc,nsocc))

#reciprocal D tensors
fock_d_ai = np.reciprocal(d_ai)
fock_d_abij = np.reciprocal(d_abij)
fock_d_abcijk = np.reciprocal(d_abcijk)

#get initial cluster energy
lastCycleEnergy =  cc_energy(fock, g, o, v, t1=ts, t2=td, t3=tt )

#dummies

if not 'S' in level: singles = None
if not 'D' in level: doubles = None
if not (('T' in level) or ('t' in level)): triples = None

#iterations
for cycle in range(iterations):

    #update amplitudes
    if 'S' in level: singles = cc_singles(fock, g, o, v, t1=ts, t2=td, t3=tt) * d_ai + ts 
    if 'D' in level: doubles = cc_doubles(fock, g, o, v, t1=ts, t2=td, t3=tt) * d_abij + td
    if (('T' in level) or ('t' in level)) : triples = cc_triples(fock, g, o, v, t1=ts, t2=td, t3=tt) * d_abcijk + tt

    #recalculate energy
    cycleEnergy = cc_energy(fock, g, o, v, t1=singles, t2=doubles, t3=triples )
    deltaEnergy = np.abs(lastCycleEnergy - cycleEnergy)

    #convergence test
    if deltaEnergy < tolerance:

        cycleEnergy =  cc_energy(fock, g, o, v, t1=singles, t2=doubles, t3=triples ) - HFenergy
        break
    else:
        ts = singles
        td = doubles
        tt = triples
        lastCycleEnergy = cycleEnergy
else:
    print("Did not converge")
    exit('cc failed')

perturbationEnergy = 0.0
if 't' in level: 
    perturbativeTriples = cc_triples(fock, g, o, v, t1=singles, t2=doubles, t3=triples)
    triples = perturbativeTriples + fock_d_abcijk * triples
    triples = triples * d_abcijk 
    l1, l2 = [singles.transpose(1,0) ,doubles.transpose(2,3,0,1)]

    perturbationEnergy = cc_perturbation_energy(fock, g, o, v, l1, l2, triples)

"""
internalClusterCode = """
#declarations
import numpy as np

def gMOspin(e, c, eri, nbf):
    #construct MO spin eri

    import numpy as np

    def iEri(i,j,k,l):
        #index into the four-index eri integrals
        p = max(i*(i+1)/2 + j, j*(j+1)/2 + i)
        q = max(k*(k+1)/2 + l, l*(l+1)/2 + k)
        return  int(max(p*(p+1)/2 + q, q*(q+1)/2 + p))

    #get 4 index eri and spinblock to spin basis
    g = np.zeros((nbf,nbf,nbf,nbf))
    for i in range(nbf):
        for j in range(nbf):
            for k in range(nbf):
                for l in range(nbf):
                    g[i,j,k,l] = eri[iEri(i,j,k,l)]
 
    spinBlock = np.kron(np.eye(2), np.kron(np.eye(2), g).T)
    g = spinBlock.transpose(0,2,1,3) - spinBlock.transpose(0,2,3,1)

    #prepare orbital energies
    eps = np.concatenate((e,e), axis=0)
    C = np.block([
                 [c, np.zeros_like(c)],
                 [np.zeros_like(c), c]])
    C =C[:, eps.argsort()]
    eps = np.sort(eps)

    #eri to MO
    g = np.einsum('pQRS,pP->PQRS', np.einsum('pqRS,qQ->pQRS', np.einsum('pqrS,rR->pqRS', np.einsum('pqrs,sS->pqrS', \
    g, C, optimize=True), C, optimize=True), C, optimize=True), C, optimize=True)

    return g


iterations = 50
tolerance = 1e-10

charge, nuclearRepulsion, electrons = scfData

#orbital occupations
spinOrbitals = (fock.shape[0]) * 2
nsocc = electrons
nsvir = spinOrbitals - nsocc

#get fock in MO spin basis
cSpin = np.kron(c, np.eye(2))
fock = np.dot(cSpin.T, np.dot(np.kron(fock, np.eye(2)), cSpin))

#get two-electron repulsion integrals in MO basis
g = gMOspin(e, c, eri, spinOrbitals//2)

#slices
n = np.newaxis
o = slice(None,nsocc)
v = slice(nsocc, None)

#D tensors
eps = np.kron(e, np.ones(2))

d_ai = 1.0 / (-eps[v, n] + eps[n, o])
d_abij = 1.0 / (-eps[v, n, n, n] - eps[n, v, n, n] + eps[n, n, o, n] + eps[n, n, n, o])
d_abcijk = 1.0 / (- eps[ v, n, n, n, n, n] - eps[n, v, n, n, n, n] - eps[n, n, v, n, n, n]
                   + eps[n, n, n, o, n, n] + eps[n, n, n, n, o, n] + eps[n, n, n, n, n, o] )

#HF energy
HFenergy = 1.0 * np.einsum('ii', fock[o, o]) -0.5 * np.einsum('ijij', g[o, o, o, o])

#amplitude initialisation
ts = np.zeros((nsvir, nsocc))
td = np.zeros((nsvir,nsvir,nsocc,nsocc))
tt = np.zeros((nsvir,nsvir,nsvir,nsocc,nsocc,nsocc))

#reciprocal D tensors
fock_d_ai = np.reciprocal(d_ai)
fock_d_abij = np.reciprocal(d_abij)
fock_d_abcijk = np.reciprocal(d_abcijk)

#get initial cluster energy
lastCycleEnergy =  cc_energy(fock, g, o, v, t1=ts, t2=td, t3=tt )

#dummies

if not 'S' in level: singles = None
if not 'D' in level: doubles = None
if not (('T' in level) or ('t' in level)): triples = None

#iterations
for cycle in range(iterations):

    #update amplitudes
    if 'S' in level: singles = cc_singles(fock, g, o, v, t1=ts, t2=td, t3=tt) * d_ai + ts 
    if 'D' in level: doubles = cc_doubles(fock, g, o, v, t1=ts, t2=td, t3=tt) * d_abij + td
    if (('T' in level) or ('t' in level)) : triples = cc_triples(fock, g, o, v, t1=ts, t2=td, t3=tt) * d_abcijk + tt

    #recalculate energy
    cycleEnergy = cc_energy(fock, g, o, v, t1=singles, t2=doubles, t3=triples )
    deltaEnergy = np.abs(lastCycleEnergy - cycleEnergy)

    #convergence test
    if deltaEnergy < tolerance:

        cycleEnergy =  cc_energy(fock, g, o, v, t1=singles, t2=doubles, t3=triples ) - HFenergy
        break
    else:
        ts = singles
        td = doubles
        tt = triples
        lastCycleEnergy = cycleEnergy
else:
    print("Did not converge")
    exit('cc failed')

perturbationEnergy = 0.0
if 't' in level: 
    perturbativeTriples = cc_triples(fock, g, o, v, t1=singles, t2=doubles, t3=triples)
    triples = perturbativeTriples + fock_d_abcijk * triples
    triples = triples * d_abcijk 
    l1, l2 = [singles.transpose(1,0) ,doubles.transpose(2,3,0,1)]

    perturbationEnergy = cc_perturbation_energy(fock, g, o, v, l1, l2, triples)

"""

def symbolicGeneratedCoupledCluster(name, fock, eri, c, e, scfData):
    #execute symbolically generated cluster codes

    if not name in ['ccd','ccsd','ccsdt','ccsd_t','lccd','lccsd','cc2','cc3'] : return 0.0

    levelDictionary = {'ccd':['C','D'], 'ccsd':['C','SD'], 'ccsdt':['C','SDT'], 'ccsd_t':['C', 'SDt'], \
                       'cc2':['A','2'], 'cc3':['A','3'], \
                       'lccd':['L','D'], 'lccsd':['L','SD']}
    
    #get external code
    type, level = levelDictionary[name]
    file = '../codes/' + name + '.py'
    f = open(file, 'r')
    externalCode = f.read()
    cycleEnergy = 0.0

    if type == 'A':
        if level == '2': level = 'SD' 
        if level == '3': level = 'SDT' 

    #exec dictionary
    data = {'level':level, 'fock':fock, 'c':c, 'eri':eri, 'e':e, 'scfData':scfData}
    exec(externalCode + internalClusterCode,{},data)

    #prepare return values
    coupledClusterCorrection = data['cycleEnergy']
    perturbationCorrection = data['perturbationEnergy']
    
    return name, coupledClusterCorrection, perturbationCorrection


def symbolicGeneratedLambda(name, fock, eri, c, e, hcore, scfData):
    #execute symbolically generated lambda equation codes

    if not name in ['ccsd'] : return 0.0

    #get external code
    files = ['../codes/ccsd.py', '../codes/ccsd_lambda.py', '../codes/cc_rdm.py']

    externalCode = """global cc_energy, cc_singles, cc_doubles """

    for f in files:
        input = open(f, 'r')
        externalCode += input.read()

    data = {'fock':fock, 'c':c, 'eri':eri, 'e':e, 'hcore': hcore, 'scfData':scfData}
    exec(externalCode + internalLambdaCode,{},data)

    #prepare return values
    lambdaLagrangeEnergy = data['lambdaCycleEnergy']
    pseudoEnergy         = data['pseudoEnergy']
    rdmEnergy            = data['rdmEnergy']

    return name, lambdaLagrangeEnergy, pseudoEnergy, rdmEnergy

def symbolicAmplitudes(name, fock, eri, c, e, scfData, spinned = False):
        #execute symbolically generated cluster codes for amplitudes

    if not name in ['ccd','ccsd','ccsdt','ccsd_t'] : return 0.0

    levelDictionary = {'ccd':['C','D'], 'ccsd':['C','SD'], 'ccsdt':['C','SDT'], 'ccsd_t':['C', 'SDt']}
    
    #get external code
    type, level = levelDictionary[name]
    file = '../codes/' + name + '.py'
    f = open(file, 'r')
    externalCode = f.read()
    cycleEnergy = 0.0

    #exec dictionary
    data = {'level':level, 'fock':fock, 'c':c, 'eri':eri, 'e':e, 'scfData':scfData}
    if not spinned:
        exec(externalCode + internalClusterCode,{},data)
    else:
        exec(externalCode + internalSpinClusterCode,{},data)

    #prepare return values
    singlesAmplitudes = data['ts']
    doublesAmplitudes = data['td']
    triplesAmplitudes = data['tt']
    
    return name, singlesAmplitudes, doublesAmplitudes, triplesAmplitudes
