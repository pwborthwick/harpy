from __future__ import division
from scipy.linalg import fractional_matrix_power as fractPow
from numpy.linalg import solve, eigvals, norm
from numpy import zeros, dot, trace, eye, mean, asarray, sqrt, zeros_like, round, einsum, argsort
from view import postSCF
from mp import mp2UnrelaxedDensity
from atom import gaugeCenter, getConstant
from integral import mu, buildEriMO, q, electricField, iEri, expandEri

def charges(D, S, atoms, bases):
    #currently Mulliken -m , Lowdin - l

    #Mulliken charges
    mulliken(D, S, atoms, bases)

    #Lowdin charges
    lowdin(D, S, atoms, bases)

def mulliken(D, S, atoms, bases):
    #compute a mulliken population analysis
    n = D.shape[0]
    orbitalPopulation = zeros(n)

    #orbital populations
    for i in range(0, n):
        for j in range(0, n):
            orbitalPopulation[i] += 2.0 * D[i,j] * S[i,j]

    #gross atomic population - orbital reduced to atoms
    m = len(atoms)
    grossAtomicPopulation = zeros(m)
    for i in range(0, n):
        grossAtomicPopulation[bases[i].atom] += orbitalPopulation[i]

    #write results to output file
    postSCF([orbitalPopulation, bases, grossAtomicPopulation, atoms], 'mulliken')

    return grossAtomicPopulation

def lowdin(D, S, atoms, bases):
    #compute the lowdin population analysis
    n = D.shape[0]
    m = len(atoms)
    grossAtomicPopulation = zeros(m)

    U = fractPow(S, 0.5)

    #gross atomic population - orbital reduced to atoms
    for i in range(0, n):
        grossAtomicPopulation[bases[i].atom] += 2.0 * dot(dot(U, D) ,U)[i,i]


    #write results to output file
    postSCF([grossAtomicPopulation, atoms], 'lowdin')

    return grossAtomicPopulation

def bondOrder(D, S, atoms, basis):
    #compute Mayer bond orders

    beta_density = alpha_density = D
    alpha_mayer = dot(alpha_density, S) ; beta_mayer = dot(beta_density, S)

    nBasis = D.shape[0]
    nAtoms = len(atoms)

    bond_orders =  zeros((nAtoms, nAtoms))
    valence     =  zeros(nAtoms)

    for mu in range(nBasis):
        for nu in range(mu):

            atom_mu = basis[mu].atom
            atom_nu = basis[nu].atom

            if atom_mu == atom_nu: continue

            alpha = alpha_mayer[mu,nu] * alpha_mayer[nu,mu]
            beta  = beta_mayer[mu,nu]  * beta_mayer[nu,mu]

            bond_orders[atom_mu, atom_nu] += 2 * (alpha + beta)
            bond_orders[atom_nu, atom_mu] += 2 * (alpha + beta)

    for i in range(nAtoms):
        for j in range(nAtoms):

             valence[i] += bond_orders[i,j]

    postSCF([atoms, round(bond_orders,4), round(valence,2)],'bonds')
    

def energyPartition(E, N, D, T, V, G, e, en):
    #energy partitioning 
    partition = {'e' : [E, 'electronic', 't+v + g'], 'n' : [N, 'nuclear', 'n'], 'k' : [0.0, 'kinetic', 't'], 'j' : [0.0, 'coulomb', 'v'],  \
               '1' : [0.0, 'one electron', 't+v'], '2' : [0.0, 'two electron', 'g'] ,'t' : [0.0, 'total', 't+v + 2g + n'] ,                \
               'p' : [0.0, ' total potential', 'v + g + n'], 'tk' : [0.0, ' total kinetic', 't'], 'v' : [0.0, 'viral ratio', '(v+g+n)/t'], \
               'b' : [0.0, 'bare hamiltonian', 't+v'] , 'e~' : [0.0, 'orbital', '2&#949;']}
    n = D.shape[0]

    k = v = g =  0
    for i in range(0, n):
        for j in range(0, n):
            d = 2*D[i,j]
            k += d*T[i,j]
            v += d*V[i,j]
            g += d*G[i,j]       

    partition['k'][0] = k
    partition['j'][0] = v
    partition['1'][0] = v+k
    partition['2'][0] = g*0.5
    partition['t'][0] = k + v + g*0.5 + partition['n'][0]
    partition['p'][0] = v + g*0.5 + partition['n'][0]
    partition['tk'][0] = k
    partition['v'][0] = partition['p'][0] / k
    partition['b'][0] = k+v
    #orbital energies doubly occupied
    sum = 0.0
    for i in range(0, int(en/2)):
        sum += e[i]
    partition['e~'][0] = 2*sum

    postSCF(partition, 'energy')

def dipoleComponent(atoms, bases, type, gauge):
    #compute dipole matrices
    n = len(bases)
    dipole = zeros((n,n))

    gaugeOrigin = gaugeCenter(atoms, gauge)

    for i in range(0, n):
        for j in range(i, -1, -1):
            dipole[i,j] = mu(bases[i], bases[j], gaugeOrigin, type)
            if i != j:
                dipole[j,i] = dipole[i,j]

    return dipole


def buildDipole(atoms, bases, density, gauge, engine = 'aello'):
    #compute the components of the dipole moment

    dipoles = zeros(4)

    if engine == 'aello':
        import aello
        dipoles[:3] = aello.aello(atoms, bases, 'dipole', density, gauge)

        dipoles[3] = dipoles[0]*dipoles[0] + dipoles[1]*dipoles[1] + dipoles[2]*dipoles[2]

    elif engine == 'native':
        gaugeOrigin = gaugeCenter(atoms, gauge)
        cartesians = ('x', 'y', 'z')

        for dim in range(0, 3):
            for i in range(0, len(atoms)):
                #nuclear and gauge displacement
                dipoles[dim] += atoms[i].number * (atoms[i].center[dim] - gaugeOrigin[dim])

            #electronic
            dipoles[dim] += -2.0 * trace(dot(density, dipoleComponent(atoms, bases, cartesians[dim], gauge)))
            dipoles[3] += dipoles[dim]*dipoles[dim]

    #dipoles has [0,1,2] = electron dipole, [3]=square of resultant 
    postSCF(dipoles, 'dipole')
    
    return dipoles
    
def buildMp2Dipole(atoms, bases, gauge, c, e, eri, nOccupied):
    #compute the mp2 level dipole from the unrelaxed mp2 density matrix

    #get electric and nuclear dipole components
    hfDipoleElectric = asarray([dipoleComponent(atoms, bases, type ,gauge) for type in ['x','y','z']])
    charges = [a.number for a in atoms]
    centers = [a.center for a in atoms]
    hfDipoleNuclear = einsum('i,ix->x', charges, centers, optimize=True)

    #ao to mo basis
    mu_mo = einsum('rp,xrs,sq->xpq', c, hfDipoleElectric, c, optimize=True)

    #get unrelaxed mp2 density
    nbf = len(bases)
    mp2Density = mp2UnrelaxedDensity(c, e, eri, nbf, nOccupied)

    mp2DipoleElectric = einsum('xij,ij->x', mu_mo, mp2Density, optimize=True)
    mp2Dipole =  list(hfDipoleNuclear - mp2DipoleElectric)

    mp2Dipole.append(norm(mp2Dipole)**2)

    postSCF(mp2Dipole, 'mp2-dipole')

    return mp2Dipole

def quadrupoleComponent(atoms, bases, direction, gauge):
    #compute quadrupole matrices
    n = len(bases)
    qupole = zeros((n,n))

    gaugeOrigin = gaugeCenter(atoms, gauge)
    for i in range(0, n):
        for j in range(i, -1, -1):
            qupole[i,j] = -q(bases[i], bases[j], gaugeOrigin, direction)
            if i != j:
                qupole[j,i] = qupole[i,j]

    return qupole

def buildQuadrupole(atoms, bases, density, gauge):
    #build the quadrupole


    quadrupoleMoments = zeros(14)
    cartesians = {0 : ['xx',0,0], 1 : ['yy',1,1], 2 : ['zz',2,2], 3 : ['xy',0,1], 4 : ['yz',1,2], 5 : ['zx',2,0]}

    for axis in range(len(cartesians)):

        #nuclear
        for atom in range(len(atoms)):
            quadrupoleMoments[axis] += atoms[atom].center[cartesians[axis][1]]*atoms[atom].center[cartesians[axis][2]]*atoms[atom].number

        quadrupoleMoments[axis] += 2.0 * trace(dot(density, quadrupoleComponent(atoms, bases, cartesians[axis][0], gauge)))

    #quadrupoles has [xx,yy,zz,xy,yz,zx]
    quadrupoleMoments = quadrupoleMoments * getConstant('bohr->angstrom') * getConstant('au->debye')

    #traceless
    traceless = (quadrupoleMoments[0]+quadrupoleMoments[1]+quadrupoleMoments[2])/3.0
    for i in range(3):
        quadrupoleMoments[6+i] = quadrupoleMoments[i] - traceless

    #eigenvalues -construct traceless tensor
    quad = zeros((3,3))
    quad[0,1] = quadrupoleMoments[3]
    quad[0,2] = quadrupoleMoments[5]
    quad[1,2] = quadrupoleMoments[4]
    quad = quad + quad.T
    for i in range(3):
        quad[i,i] = quadrupoleMoments[i+6]

    q = eigvals(quad)
    q = q[argsort(q)][::-1]

    for i in range(3):
        quadrupoleMoments[9+i] = q[i]
    
    #amplitude and asymmetry
    quadrupoleMoments[12] = sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2])
    quadrupoleMoments[13] = (q[1]-q[2])/q[0]

    #[0-5 are quadrupole, 6-8 traceless diagonals, 9-11 eigenvalues of traceless, 12 amplitude, 13 asymmetry]
    postSCF(quadrupoleMoments, 'quad')

    return quadrupoleMoments


def polarizabilities(atoms, bases, c, f, ERI, nOccupied, gauge):

    nBasis = c.shape[0]
    nVirtual = nBasis - nOccupied
    nRotation = nOccupied * nVirtual

    #fock to molecular basis
    f = dot(c.T, f).dot(c)

    #2-electron repulsion integrals to molecular basis tensor
    eri = buildEriMO(c, ERI)
    eriMO = expandEri(eri, nBasis)

    unit = eye(nBasis)
    #construct orbital rotation matrices
    alpha = zeros((nBasis, nBasis, nBasis, nBasis))
    for i in range(0, nBasis):
        for j in range(0, nBasis):
            for a in range(0, nBasis):
                for b in range(0, nBasis):
                    alpha[i,a,j,b] += unit[i,j] * f[a,b] - unit[a,b] * f[i,j]  \
                                   + 2.0 * eriMO[a,i,j,b] - eriMO[j,i,a,b]

    beta = zeros((nBasis, nBasis, nBasis, nBasis))
    for i in range(0, nBasis):
        for j in range(0, nBasis):
            for a in range(0, nBasis):
                for b in range(0, nBasis):
                    beta[i,a,j,b] += 2.0 * eriMO[a,i,j,b] - eriMO[a,j,b,i]

    #electronic Hessian
    hessian = alpha + beta

    #transform Hessian to square form 
    h = zeros((nRotation, nRotation))
    for i in range(0, nOccupied):
        for a in range(nOccupied, nBasis):
            for j in range(0, nOccupied):
                for b in range(nOccupied, nBasis):
                    h[i*(nVirtual)+(a-nOccupied),j*(nVirtual)+(b-nOccupied)] = hessian[i,a,j,b]

    #get dipole tensors
    dipoleTensor = []
    cartesianDict = {0:'x', 1:'y', 2:'z'}
    for direction in range(0, 3):
        dipoleTensor.append(-2.0 * dipoleComponent(atoms, bases, cartesianDict[direction], gauge))

    #responses
    responseTensor = []
    for p in range(0, len(dipoleTensor)):
        dipoleTensor[p] = dot(c.T, dipoleTensor[p]).dot(c)
        i = 0
        ravelTensor= zeros(nOccupied*nVirtual)
        for r in range(0, nOccupied):
            for s in range(0, nVirtual):
                ravelTensor[i] = dipoleTensor[p][r,s+nOccupied]
                i += 1
        responseTensor.append(ravelTensor)

    responses = []
    for perturbation in responseTensor:
        responses.append(solve(h, -perturbation))

    #polarizabilities
    polarizabilities = zeros((3, 3))
    for i in range(3):
        for j in range(3):
            polarizabilities[i, j] = -dot(responseTensor[i], responses[j])

    principalComponents = eigvals(polarizabilities)
    isotropicPolarizability = mean(principalComponents)

    postSCF([principalComponents, isotropicPolarizability], 'po')

    return principalComponents, isotropicPolarizability, asarray(responses), asarray(dipoleTensor)*0.5

def electricFieldNuclear(atoms, gauge):
    #compute the nuclear contribution to electric field

    gaugeOrigin = gaugeCenter(atoms, gauge)
    n = len(atoms)

    coordinates = zeros((n,3))
    field = zeros(4)

    for atom in range(n):
        for axis in range(3):
            coordinates[atom, axis] = atoms[atom].center[axis] - gaugeOrigin[axis]

        r2 = coordinates[atom,0]*coordinates[atom,0] + coordinates[atom,1]*coordinates[atom,1] + \
             coordinates[atom,2]*coordinates[atom,2]
        r = sqrt(r2)

        if r < 1e-10: continue

        #field
        field[0] += atoms[atom].number * coordinates[atom,0] / (r2 * r)
        field[1] += atoms[atom].number * coordinates[atom,1] / (r2 * r)
        field[2] += atoms[atom].number * coordinates[atom,2] / (r2 * r)
        #potential
        field[3] += atoms[atom].number / r

    return field

def buildElectricField(atoms, bases, density, gauge):
    #compute expectation values for electric field components and electric potential

    electrics = zeros(8)

    cartesians = ('x', 'y', 'z', 'p')
    for dim in range(0, 4):
                
        electrics[dim] = 2.0 * trace(dot(density, electricField(atoms, bases, cartesians[dim], gauge)))

    #get nuclear contributions
    electrics[4:] = electricFieldNuclear(atoms, gauge)

    return electrics

def hyperPolarizabilities(atoms, bases, c, eri, e, f, nOccupied, gauge):

    #get responses and dipole tensors
    _, _, responses , dipoles_mo = polarizabilities(atoms, bases, c, f, eri, nOccupied, gauge)

    #number of components
    nComponents = responses.shape[0]

    nOrbitals = c.shape[0]
    nVirtual = nOrbitals - nOccupied

    #reshape responses (3, vo) -> (3,o+v,o+v) compatible with dipoles
    u = zeros_like(dipoles_mo)
    uv = zeros((nComponents, nOccupied, nVirtual))
    for comp in range(nComponents):
        k = 0
        for p in range(nOccupied):
            for q in range(nVirtual):
                u[comp, p, nOccupied+q] = 0.5 * responses[comp, k]
                uv[comp, p, q] = responses[comp, k]
                u[comp, q + nOccupied, p] = -0.5 * responses[comp, k]
                k += 1

    #get occupied and virtual c matrices
    co = zeros((nOrbitals, nOccupied))
    co = c[:, :nOccupied]
    cv = zeros((nOrbitals, nVirtual))
    cv = c[:,nOccupied:]

    #define matrices needed for fock build
    r = zeros((nOrbitals, nOccupied))
    V = zeros_like(c)
    T = zeros_like(c)
    G = zeros_like(u)

    for comp in range(nComponents):

        #build right density component from responses
        r = dot(uv[comp], cv.T).T
        #build density
        d = dot(co, r.T)
        
        #build new V and T with response density
        for i in range(0, nOrbitals):
            for m in range(0, nOrbitals):
                V[i,m] = 0.0
                T[i,m] = 0.0
                for k in range(0, nOrbitals):
                    for l in range(0, nOrbitals):
                        V[i,m] += 0.5 * d[k,l] * eri[iEri(i,m,k,l)] 
                        T[i,m] += 0.5 * d[k,l] * eri[iEri(i,k,m,l)]

        #build response Fock
        F = (c.T).dot(4 * V - T.T - T).dot(c)

        #add dipole field
        G[comp,...] = F + dipoles_mo[comp]

    #epsilon
    epsilon = G.copy()

    #static case 
    omega = 0.0

    #left multiply by orbital energies
    eUl = zeros((nOrbitals, nOrbitals))
    eUr = zeros((nOrbitals, nOrbitals))
    for comp in range(nComponents):
        for p in range(nOrbitals):
            for q in range(nOrbitals):
                eUl[p,q] = (e[p] + omega) * u[comp,p,q]
    #right multiply by orbital energies
        for p in range(nOrbitals):
            for q in range(nOrbitals):
                eUr[p,q] = u[comp,p,q] * e[q]
    
        epsilon[comp, :,:] += (eUl - eUr)

    #use Kleinman symmetry to reduce calculations
    syma = [0,1,2,0,0,1]
    symb = [0,1,2,1,2,2]

    #define tensor for final results
    hyperpolarizability = zeros((6, 3))

    '''
    hyperpolarizability tensor is 
    [xxx xxy xxz]
    [yyx yyy yyz]
    [zzx zzy zzz]
    [xyx xyy xyz]
    [xzx xzy xzz]
    [yzx yzy yzz]
    '''
    for r in range(6):
        b = syma[r]
        c = symb[r]
        for a in range(3):

            traceLeft = 2.0 * trace(dot(u[a,:,:], dot(G[b,:,:], u[c,:,:]))[:nOccupied, :nOccupied]) +  \
                        2.0 * trace(dot(u[a,:,:], dot(G[c,:,:], u[b,:,:]))[:nOccupied, :nOccupied]) +  \
                        2.0 * trace(dot(u[c,:,:], dot(G[a,:,:], u[b,:,:]))[:nOccupied, :nOccupied])

            traceRight = trace(dot(u[c,:,:], dot(u[b,:,:], epsilon[a,:,:]))[:nOccupied, :nOccupied]) + \
                         trace(dot(u[b,:,:], dot(u[c,:,:], epsilon[a,:,:]))[:nOccupied, :nOccupied]) + \
                         trace(dot(u[c,:,:], dot(u[a,:,:], epsilon[b,:,:]))[:nOccupied, :nOccupied]) + \
                         trace(dot(u[a,:,:], dot(u[c,:,:], epsilon[b,:,:]))[:nOccupied, :nOccupied]) + \
                         trace(dot(u[b,:,:], dot(u[a,:,:], epsilon[c,:,:]))[:nOccupied, :nOccupied]) + \
                         trace(dot(u[a,:,:], dot(u[b,:,:], epsilon[c,:,:]))[:nOccupied, :nOccupied])

            hyperpolarizability[r, a] = -2.0 * (traceLeft - traceRight)

    #calculate defined propeties
    beta = zeros(3)
    #beta[i] = ixx+iyy+izz 
    beta[0] = (hyperpolarizability[0,0]+hyperpolarizability[1,0]+hyperpolarizability[2,0])
    beta[1] = (hyperpolarizability[0,1]+hyperpolarizability[1,1]+hyperpolarizability[2,1])
    beta[2] = (hyperpolarizability[1,2]+hyperpolarizability[1,2]+hyperpolarizability[2,2])

    #magnitude
    amplitude = sqrt(beta[0]*beta[0] + beta[1]*beta[1] + beta[2]*beta[2])

    #parallel and perpendicular components wrt z axis
    parallel = 0.2 * beta[2]
    perpendicular = 0.6 * beta[2]

    postSCF([beta, amplitude, parallel, perpendicular], 'hyper')

    return [beta, amplitude, parallel, perpendicular], hyperpolarizability
