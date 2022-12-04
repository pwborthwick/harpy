from __future__ import division
from scipy.linalg import expm, norm
from numpy import dot, conjugate, zeros, pi, zeros_like, sin, array, hstack, trace, kron
from numpy import arange, where, all, sqrt, sum, log, max, min, linspace, exp, eye
import rhf
from scipy.linalg import fractional_matrix_power as fractPow
from post import buildDipole, dipoleComponent
from integral import iEri, buildAngular, buildNabla, buildFockMOspin
from atom import nuclearRepulsion, getConstant
from aello import tdhfFock
from numpy.linalg import solve
from ci import randomPhaseApproximation, ciDegeneracy

def tdhf(do, run, shape, units ):
    #time-dependent Hartree-Fock by Magnus expansion

    #arguments
    timeIncrement, iterations, direction  = run

    #pulse definition
    if len(shape) == 2:
        pulseType, fieldStrength = shape
        rho = 0.1
    elif len(shape) == 3:
        pulseType, fieldStrength, rho = shape

    #initial molecule properties
    molAtom, molBasis, molData = rhf.mol([])

    SCFenergy = rhf.scf(molAtom, molBasis, molData, [])
    S = rhf.S
    density = rhf.density
    ERI = rhf.ERI
    fock = rhf.fock
    coreH = rhf.coreH

    reset = False

    #transform AO <-> AO orthogonal
    X = fractPow(S, -0.5)
    U = fractPow(S,  0.5)

    #initial AO orthogonal matrices
    orthoDensity = dot(U, dot(density, U.T))
    orthoFock = dot(X.T , dot(fock , X))

    h = -1j * timeIncrement

    #|---------------------------------subroutines----------------------------------|

    def initialise():
        #ground state initialiation

        rhf.scf(molAtom, molBasis, molData, [])

        S = rhf.S
        density = rhf.density
        ERI = rhf.ERI
        fock = rhf.fock
        coreH = rhf.coreH

    def pulse(t, type = 'gaussian'):

        if type == 'kick':
           if t == 0: return 1.0
           else: return 0.0
        elif type == 'gaussian':
            return exp(-(t**2)/ (2 * rho * rho))

    def commutator(a, b):
        #commutator bracket of a,b

        return dot(a,b) - dot(b,a)

    def orthoField(t):
        #add dipole field to fock - in orthogonal basis

        axis = ['x','y','z'].index(direction)

        profile = pulse(t, type=pulseType) * fieldStrength

        dp = dipoleComponent(molAtom, molBasis, direction, 'nuclear charge')

        fockCorrection = profile * dp 

        return dot(X.T , dot(fockCorrection , X))

    def updateFock(density, engine = 'aello'):
        #updating Fock complex version
        n = density.shape[0]
        fock = zeros((n,n), dtype=complex)

        if engine == 'aello':
            fock = tdhfFock(n, density, coreH.astype('complex'), ERI.astype('complex'))

        elif engine == 'native':
            G = zeros((n,n), dtype=complex)

            for i in range(0, n):
                for m in range(0, n):
                    G[i,m] = 0.0
                    for k in range(0, n):
                        for l in range(0, n):
                            G[i,m] += density[k,l] * ( 2* ERI[iEri(i,m,k,l)].astype('complex') - ERI[iEri(i,k,m,l)].astype('complex'))
                    fock[i,m] = coreH[i,m].astype('complex') + G[i,m]

        return fock

    def updateState(u, cycleDensity):
        #propagate time U(t) -> U(t+timeIncrement)

        orthoDensity = dot(u, dot(cycleDensity, conjugate(u).T))

        #build fock in non-orthogonal ao basis
        density = dot(X, dot(orthoDensity, X.T))
        fock = updateFock(density)

        #orthogonalize for next step
        orthoFock = dot(X.T, dot(fock, X))

        return orthoFock, orthoDensity

    def updateEnergy(molAtom, molBasis, fock, density):
        #compute SCF energy

        energy = 0.0
        for i in range(0, len(molBasis)):
            for k in range(0, len(molBasis)):
                energy += density[i,k] * (fock[i,k] + coreH[i,k])

        return energy + nuclearRepulsion(molAtom)

    def magnus2(h, orthoFock, orthoDensity, density):
        #second order Magnus expansion

        time = []
        HFenergy = []
        dipole = []
        field = []

        axis = ['x','y','z'].index(direction)
        nbf = orthoDensity.shape[0]
        energy = SCFenergy

        for cycle in range(iterations):

            k = zeros((2,nbf,nbf)).astype('complex')

            dipole.append(buildDipole(molAtom, molBasis, density.real,'nuclear charge', 'aello')[axis]) 

            time.append(cycle * timeIncrement)
            HFenergy.append(energy)
            field.append(pulse(cycle * timeIncrement, type=pulseType))

            cycleDensity = orthoDensity.copy()

            #equation (13)
            k[0] = h * (orthoFock + orthoField(cycle * timeIncrement))
            l = k[0]
            u = expm(l)

            orthoFock, orthoDensity = updateState(u, cycleDensity)
            k[1] = h * (orthoFock + orthoField((cycle+1) * timeIncrement))
            l = 0.5*(k[0] + k[1])
            u = expm(l)
            orthoFock, orthoDensity = updateState(u, cycleDensity)

            #unorthogonalise for energy calculation
            fock = dot(U.T, dot(orthoFock, U))
            density = dot(X, dot(orthoDensity, X.T))
            energy = updateEnergy(molAtom, molBasis, fock, density)
            
        return time, HFenergy, dipole, field

    def magnus4(h, orthoFock, orthoDensity, density):
        #second order Magnus expansion

        #reset plotted quantities and return to ground state
        time = []
        HFenergy = []
        dipole = []
        field = []

        axis = ['x','y','z'].index(direction)
        nbf = orthoDensity.shape[0]
        energy = SCFenergy

        for cycle in range(iterations):

            k = zeros((6,nbf,nbf)).astype('complex')
            q = zeros_like(k).astype('complex')

            dipole.append(buildDipole(molAtom, molBasis, density.real,'nuclear charge', 'aello')[axis])  

            time.append(cycle * timeIncrement)
            HFenergy.append(energy)
            field.append(pulse(cycle * timeIncrement, type=pulseType))

            cycleDensity = orthoDensity.copy()

            #equation (14) - with reordering of terms, [] is line in (14), <> line in 15
            k[0] = h * (orthoFock + orthoField(cycle * timeIncrement))
            q[0] = k[0]
            l = k[0] * 0.5
            u = expm(l)                                                                                  #[2]
            orthoFock, orthoDensity = updateState(u, cycleDensity)

            k[1] = h * (orthoFock + orthoField((cycle+0.5) * timeIncrement))
            q[1] = k[1] - k[0]                                                                           #[4]
            l = 0.5 * q[0] + 0.25 * q[1]                                                                 #[3]
            u = expm(l)
            orthoFock, orthoDensity = updateState(u, cycleDensity)

            k[2] = h * (orthoFock + orthoField((cycle+0.5) * timeIncrement))
            q[2] = k[2] - k[1]                                                                           #[2]
            l = q[0] + q[1]                                                 
            u = expm(l)
            orthoFock, orthoDensity = updateState(u, cycleDensity)

            k[3] = h * (orthoFock + orthoField((cycle+1.0) * timeIncrement))
            q[3] = k[3] -2.0 * k[1] + k[0]                                                               #[6]
            l = 0.5 * q[0] + 0.25 * q[1] + q[2]/3.0 - q[3]/24.0 - commutator(q[0], q[1])/48.0            #<1>
            u = expm(l)
            orthoFock, orthoDensity = updateState(u, cycleDensity)

            k[4] = h * (orthoFock + orthoField((cycle+0.5) * timeIncrement))
            q[4] = k[4] - k[1]
            l = q[0] + q[1] + 2.0 * q[2]/3.0 + q[3]/6.0 - commutator(q[0], q[1]) /6.0                    #[7]
            u = expm(l)
            orthoFock, orthoDensity = updateState(u, cycleDensity)

            k[5] = h * (orthoFock + orthoField((cycle+1.0) * timeIncrement))
            q[5] = k[5] - 2.0 * k[1] + k[0]                                                              #<3>
            l = q[0] + q[1] + 2.0 * q[4]/3.0 + q[5]/6.0 - commutator(q[0], q[1]-q[2]+q[4]+0.5*q[5])/6.0  #<4>
            u = expm(l)
            orthoFock, orthoDensity = updateState(u, cycleDensity)

            #unorthogonalise for energy calculation
            fock = dot(U.T, dot(orthoFock, U))
            density = dot(X, dot(orthoDensity, X.T))
            energy = updateEnergy(molAtom, molBasis, fock, density)
                
        return time, HFenergy, dipole, field

    def tdhfFile(ft,fe, fd, ff, fs):
        #write time and dipole to file

        from numpy import savez
        fileName = molData['name'] + '-' + molData['basis'] + '-tdhf.npz'
        savez(fileName, p =[fs, units[0], units[1], pulseType, fieldStrength, timeIncrement, iterations, direction], \
                        t = ft,e = fe, d = fd, f= ff)

    #|-------------------------------------main-------------------------------------|

    if not 'noplot' in do:
        import matplotlib.pyplot as plt
        plt.grid()
        plt.title(molData['name'] + ' (' + molData['basis'] + ')')
        plt.xlabel('time (' + units[0] + ')')
        plt.ylabel(direction + '-dipole (' + units[1] + ')')

    #call Magnus 2nd order
    if '2' in do:
        if reset: initialise()
        reset = not reset

        t, e, d, f = magnus2(h, orthoFock, orthoDensity, density)
        if units[0] == 'fs': t = array(t) * getConstant('atu->femtosecond')
        if units[1] == 'debye': d = array(d) * getConstant('au->debye')
        if not 'noplot' in do: plt.plot(t,d,'r',label='Magnus2', linewidth=0.1)

        if 'file[2]' in do: tdhfFile(t, e, d, f, '2')

    #call Magnus 4th order
    if '4' in do:
        if reset: initialise()
        reset = not reset

        t, e, d, f = magnus4(h, orthoFock, orthoDensity, density)
        if units[0] == 'fs': t = array(t) * getConstant('atu->femtosecond')
        if units[1] == 'debye': d = array(d) * getConstant('au->debye')
        if not 'noplot' in do: plt.plot(t,d,'k',label='Magnus4', linewidth=0.2)

        if 'file[4]' in do: tdhfFile(t, e, d, f, '4')

    if not 'noplot' in do:
        plt.legend(loc=1)
        plt.show()

    return array(t), array(e), array(d), array(f)

#reference for Pade: Accelerated Broadband Spectra Using Transition Dipole Decomposition
#and Pade Approximants - Bruner, LaMaster and Lopata

def pade(a, b, w):
	#Compute Pade approximant via extended Euclidean algorithm for 
	#polynomial greatest common divisor - ref equations (28)(29)

	sp = 0.0
	sq = 0.0
	n = len(a)

	#evaluate power series
	for i in range(n):
		sp += a[i] * pow(w, n-1-i)
	for i in range(n):
		sq += b[i] * pow(w, n-1-i)

	return sp / sq


def toeplitz(type, col, row = 0):
	#Toeplitz manipulations - 'make' and 'lower triangular'

	n = len(col)
	toeplitz = zeros((n,n))

	if all(row) == 0: row = col

	if type == 'make':
		for i in range(n):
			for j in range(n-i):
				toeplitz[j, i+j] = row[i]
				toeplitz[i+j, j] = col[i]

	if type == 'Ltri':
		for i in range(1,n):
			for j in range(n-i):
				toeplitz[i+j, j] = col[i]

	return toeplitz

def solveSpectrum(x, y, options):
	#solve for spectrum, ref equations (30)-(35)

	dampFactor, maxPoints, elimit, eticks = options

	#zero at t=0, y[0] = 0 !
	y -= y[0]

	#apply damping - Lorentzian shape
	step = x[1] - x[0]
	damping = exp(-(step*arange(len(y)))/float(dampFactor))
	y *= damping

	#diagonal Pade scheme
	n = len(y)//2
	if n > maxPoints: n = maxPoints

	#generate vector and limit points
	X = -y[n+1:2*n]

	#compute Toeplitz matrix [n-1, n-1] and solve 
	A = toeplitz('make', y[n:2*n-1], y[n:1:-1])
	try:
		b = solve(A,X)
	except:
		exit('singular matrix - no field')

	#[1, [n-1]] -> [n] column vector
	b = hstack((1.0,b))

	#v[n]*toeplitz[n,n] a strictly lower triangular matrix
	a = dot(toeplitz('Ltri',y[0:n]),b)

	#frequency range
	frequency = arange(0.0 ,elimit ,eticks)

	w = exp(-1j*frequency*step)

	#Pade approximant via extended Euclidean algorithm
	fw = pade(a, b, w)

	return fw, frequency 

def getSpectrum(time, dipole, options):
 	#solve from spectra

	spd , frequency = solveSpectrum(time, dipole, options[:4])

	#get return spectrum type real, imaginary or absolute
	type = options[4]
	if type == 'r':
		omega = spd.real
	elif type == 'i':
		omega = spd.imag
	elif type == 'a':
		omega = abs(spd)

	#absorption formula
	field = options[5]
	spectrum = (4.0*pi*getConstant('alpha')*frequency*(omega))/field

	return spectrum, frequency

def getPeaks(spectrum, frequency, tolerance):
	#find the peaks in the spectrum aove tolerance

	from scipy.signal import argrelmax as pks

	extrema = pks(abs(spectrum))

    #apply tolerance
	idx = where((abs(spectrum[extrema]) >= tolerance))
	jdx = extrema[0][idx[0]] 

	nPeaks = len(jdx)
	peaks = zeros(nPeaks)
	for i in range(nPeaks):
		peaks[i] = frequency[jdx][i]

	return peaks

def TDHFproperties(molAtom, molBasis, charge, c, fock, ERI, method, TDHFdata):
    #compute a dictionary of TD properties

    def integral(type, molAtom, molBasis, co, cv):
        #compute the required property integral

        a = zeros((3, nBasis, nBasis))
        axes = ['x', 'y', 'z']

        if type == 'dipole':
        #electric transition dipole in length gauge
            for i, axis in enumerate(axes):
                a[i] = -dipoleComponent(molAtom, molBasis, axis, 'nuclear charge')

        elif type == 'nabla':
        #electric transition dipole in velocity gauge
            for i, axis in enumerate(axes):
                a[i] = buildNabla(molAtom, molBasis, axis)

        elif type == 'angular':
        #magnetic transition dipole in length gauge
            for i, axis in enumerate(axes):
                a[i] = buildAngular(molAtom, molBasis, axis, 'origin')

        aMO = []
        for i in range(3):
            aMO.append(kron(dot(co.T, dot(a[i], cv)), eye(2)))

        return array(aMO)

    #only tamm-dancoff at moment
    if method != 'tamm-dancoff': return {}

    [nOccupied, nVirtual, roots] = TDHFdata

    #get excited states
    e, v = randomPhaseApproximation(molAtom, charge, molBasis, c, fock, ERI, method)

    #get shaping matrices
    co = c[:,:nOccupied]
    cv = c[:,nOccupied:]

    #get singlets
    ciMultiplicity = ciDegeneracy(e)
    i = 0
    for j, mult in enumerate(ciMultiplicity): 
        if 's' in mult: 
            v[:,i] = v[:, where(e == mult[0])[0][0]]
            e[i] = mult[0]

            i +=1

    properties = []
    responses = {'excitation energy' : 0, 'electric transition dipole (length gauge)' : [], 'oscillator strength (length gauge)' : 0, \
                 'electric transition dipole (length gauge)' : [], 'oscillator strength (length gauge)' : 0 ,\
                 'magnetic transition dipole (length gauge)' : [], 'rotatory strength (length gauge)' : 0, 'rotatory strength (velocity gauge)' : 0 }

    nBasis = c.shape[0]

    muMO =  integral('dipole', molAtom, molBasis, co, cv)
    nabMO = integral('nabla', molAtom, molBasis, co, cv)
    angMO = integral('angular', molAtom, molBasis, co, cv)

    #get properties for each root
    for root in range(roots):

        responses['excitation energy'] = e[root]
        
        alpha = v[:, root].reshape(2*nOccupied, 2*nVirtual)

        #electric transition dipole (length)
        tdm = []
        for i in range(3):
            tdm.append(sum(alpha * muMO[i]))

        responses['electric transition dipole (length gauge)'] = tdm

        #oscillator strength (length)
        responses['oscillator strength (length gauge)'] = 2 * e[root] * sum(array(tdm)**2)/3

        #electric transition dipole (velocity)
        tdm = []
        for i in range(3):
            tdm.append(sum(alpha * nabMO[i]))
        responses['electric transition dipole (velocity gauge)'] = tdm

        #oscillator strength (length)
        responses['oscillator strength (velocity gauge)'] = 2 / (3 * e[root]) * sum(array(tdm)**2)

        #magnetic transition dipole (velocity)
        tdm = []
        for i in range(3):
            tdm.append(0.5 * sum(alpha * angMO[i]))
        responses['magnetic transition dipole (length gauge)'] = tdm

        #rotatory strengths
        responses['rotatory strength (length gauge)'] = sum(array(responses['electric transition dipole (length gauge)']) * \
                                                          array(responses['magnetic transition dipole (length gauge)']))        

        responses['rotatory strength (velocity gauge)'] = -sum(array(responses['electric transition dipole (velocity gauge)']) * \
                                                            array(responses['magnetic transition dipole (length gauge)']) ) /  \
        													responses['excitation energy']

        properties.append(responses.copy())

    return properties

def TDHFspectrum(spectrum):
	#compute the opa or ecd convolutions and bars

	def profile(p, d = None, mode = 'wave'):
		#compute the waveform profile

		if spectrum['units'] == 'nm':
			g = lambda x : (x * x * spectrum['gamma'] * getConstant('Eh')) / (getConstant('planck') * getConstant('c') * 1e7)
		elif spectrum['units'] == 'eV':
			g = lambda x: spectrum['gamma'] * getConstant('hartree->eV')

		if spectrum['shape'] == 'gaussian':

			if mode == 'max': return 2.0 / (g(p) * sqrt(2.0 * pi))

			factor =    2.0 / (g(p) * sqrt(2.0 * pi))
			exponent = -2.0 * ((d - p) / g(p))**2

			return factor * exp(exponent)

		elif spectrum['shape'] == 'lorentzian':

			if mode == 'max': return 2.0 / (pi * g(p))

			factor = 1.0 / pi

			return factor * g(p) * 0.5 / ((d - p)**2 + (g(p) * 0.5)**2)

	def prefactor():
		#calculate the prefactor for the residue expression


		units = {'au to Coulomb cm' : getConstant('e') * getConstant('bohr->cm') * 100, \
		         'au to Joules inverse Tesla' : 2.0 * getConstant('bohr magneton') * 100 }

		if spectrum['type'] == 'opa': 
			conversion = units['au to Coulomb cm'] * units['au to Coulomb cm']
			return 8.0 * pow(pi, 3.0) * getConstant('avogadro') * conversion / \
		           (3.0 * 1000 * log(10) * (4.0 * pi * getConstant('electric constant')) * \
		           (getConstant('c')/100) * getConstant('planck')) 

		if spectrum['type'] == 'ecd': 
			conversion = units['au to Coulomb cm'] * units['au to Joules inverse Tesla']
			return 32.0 * pow(pi, 3.0) * getConstant('avogadro')  * conversion/ \
		           (3.0 * 1000 * log(10) * (4.0 * pi * getConstant('electric constant')) * \
		           pow(getConstant('c')/100, 2.0) * getConstant('planck')) 

	#set defaults
	defaults = [['type','opa'],['shape','gaussian'],['units','nm'],['points',5000],['gamma',0.02]]
	for default in defaults:
		spectrum.setdefault(default[0],default[1])

	#get poles and residues
	poles = array([sub['excitation energy'] for sub in spectrum['data']])
	if spectrum['type'] == 'opa':
		residues = array([norm(sub['electric transition dipole (length gauge)'])**2 for sub in spectrum['data']])
		transform = lambda x: x*x
	elif spectrum['type'] == 'ecd':
		residues = array([sub['rotatory strength (length gauge)'] for sub in spectrum['data']])
		transform = lambda x: x

	#units - 'c' in cm, m->nm
	units = spectrum['units']
	if units == 'nm': 
		unitFactor = getConstant('c') * getConstant('planck') * 1e7 / getConstant('Eh')
		poles = unitFactor/poles
	if units == 'eV': 
		unitFactor = getConstant('hartree->eV')
		poles *= unitFactor

	#pole axis
	margin = (max(poles) - min(poles)) * 0.2
	x = linspace(min(poles) - margin, max(poles) + margin, spectrum['points'])

	#get prefactors
	multiplier = prefactor()

	#residues axis
	y = multiplier * x  * sum([transform(r) * profile(p, d=x, mode='wave') for p, r in zip(poles, residues)], axis=0)

	#bars
	bars = multiplier * array([p * transform(r) * profile(p, mode='max') for p, r in zip(poles, residues)])
	
	return {'x' : x, 'y' : y, 'poles' : poles, 'bars' : bars}

def td_ccsd(molAtom, molBasis, ts, td, ls, ld, ims, fockMOspin, eriMOspin, run, metrics):
	#time-dependent coupled-cluster singles and doubles Nascimento-2016 (see md file)

	interval, cycles, polarization = run
	spinOrbitals, nElectrons = metrics

	#get spin MO dipole
	from post import dipoleComponent

	mu = dipoleComponent(molAtom, molBasis, polarization, 'origin')
	muMOspin = buildFockMOspin(spinOrbitals, rhf.C, mu)

	#we have full spin dimensioned arrays rather than sliced o,v
	dipoleTrace = trace(muMOspin[:nElectrons, :nElectrons])

	#right dipole function                                                               #[18]
	rightHandFunction = []
	rightHandFunction.append(dipoleTrace)
	rightHandFunction.append(muMOspin)
	rightHandFunction.append(zeros_like(ld))

	#left dipole function
	leftHandFunction = []
	lhf = dipoleTrace
	for i in range(0, nElectrons):
		for a in range(nElectrons, spinOrbitals):
			lhf += muMOspin[i,a] * ls[i,a]
	leftHandFunction.append(lhf)

	lhf = muMOspin + dipoleTrace *ls
	for i in range(0, nElectrons):
		for a in range(nElectrons, spinOrbitals):
			for e in range(nElectrons, spinOrbitals):
				lhf[i,a] += muMOspin[e,a] * ls[i,e]
			for m in range(0, nElectrons):
				lhf[i,a] -= muMOspin[i,m] * ls[m,a]
				for e in range(nElectrons, spinOrbitals):
					lhf[i,a] += ld[i,m,a,e] * muMOspin[e,m]
	leftHandFunction.append(lhf)

	lhf = dipoleTrace * ld
	for i in range(0, nElectrons):
		for a in range(nElectrons, spinOrbitals):
			for j in range(0, nElectrons):
				for b in range(nElectrons, spinOrbitals):
					lhf[i,j,a,b] += ls[i,a] * muMOspin[j,b] - ls[i,b] * muMOspin[j,a] - \
					                ls[j,a] * muMOspin[i,b] + ls[j,b] * muMOspin[i,a]
					for e in range(nElectrons, spinOrbitals):
					 	lhf[i,j,a,b] += ld[i,j,e,b] * muMOspin[e,a] - ld[i,j,e,a] * muMOspin[e,b] 
					for m in range(0, nElectrons):
					 	lhf[i,j,a,b] += -muMOspin[i,m] * ld[m,j,a,b] + muMOspin[j,m] * ld[m,i,a,b]
	leftHandFunction.append(lhf)

	#get explicit lambda intermediates - put back diagonals
	fae, fmi, fme, wmnij, wabef, wmbej, wmnie, wamef, wmbij, wabei = ims
	fae += fockMOspin[:, :]
	fmi += fockMOspin[:, :]

	def dtRightDipoleFunction0(mr1, mr2):
		#time derivative of rightHandFunction[0]                                         #[24]

		dt = 0.0
		for i in range(0, nElectrons):
			for a in range(nElectrons, spinOrbitals):
				dt += mr1[i,a] * fae[i,a]
				for j in range(0, nElectrons):
					for b in range(nElectrons, spinOrbitals):
						dt += 0.25 * mr2[i,j,a,b] * eriMOspin[i,j,a,b]
		return dt * -1j

	def dtRightDipoleFunction1(mr1, mr2):
		#time derivative of rightHandFunction[1]                                          #[25]
		#x is offset because dipole function is [o,v] not [s,s]

		dt = zeros_like(mr1, dtype=complex)
		for i in range(0, nElectrons):
			for a in range(nElectrons, spinOrbitals):
				for b in range(nElectrons, spinOrbitals):
					dt[i,a] += mr1[i,b] * fae[a,b]
					for j in range(0, nElectrons):
						dt[i,a] += mr1[j,b] * wmbej[j,a,b,i]
						dt[i,a] += mr2[i,j,a,b] * fme[j,b]
						for k in range(0, nElectrons):
							dt[i,a] -= 0.5 * wmnie[j,k,i,b] * mr2[j,k,a,b]
						for c in range(nElectrons, spinOrbitals):
							dt[i,a] += 0.5 * mr2[i,j,b,c] * wamef[a,j,b,c]
				for j in range(0, nElectrons):
					dt[i,a] -= fmi[j,i] * mr1[j,a]
		return dt * -1j

	def dtRightDipoleFunction2(mr1, mr2):
		#time derivative of rightHandFunction[2]                                         #[26]+ccsd terms

		#intermediates
		ia = zeros_like(mr1)
		ic = zeros_like(mr1)
		for m in range(0, nElectrons):
			for n in range(0, nElectrons):
				for i in range(0, nElectrons):
					for e in range(nElectrons, spinOrbitals):
						ia[i,m] += mr1[n,e] * wmnie[m,n,i,e]
						for f in range(nElectrons, spinOrbitals):
							ic[i,m] += 0.5 * mr2[i,n,e,f] * eriMOspin[m,n,e,f]

		ib = zeros_like(mr1)
		id = zeros_like(mr1)
		for a in range(nElectrons, spinOrbitals):
			for m in range(0, nElectrons):
				for e in range(nElectrons, spinOrbitals):
					for f in range(nElectrons, spinOrbitals):
						ib[a,e] -= mr1[m,f] * wamef[a,m,e,f]
						for n in range(0, nElectrons):
							id[a,e] += 0.5 * mr2[m,n,a,f] * eriMOspin[m,n,e,f]

		dt = zeros_like(mr2, dtype=complex)
		for i in range(0, nElectrons):
			for j in range(0, nElectrons): 
				for a in range(nElectrons, spinOrbitals):
					for b in range(nElectrons, spinOrbitals):
						for m in range(0, nElectrons):
							dt[i,j,a,b] += mr1[m,b] * wmbij[m,a,i,j] - mr1[m,a] * wmbij[m,b,i,j]
							dt[i,j,a,b] += mr2[j,m,a,b] * fmi[m,i] - mr2[i,m,a,b] * fmi[m,j]
							dt[i,j,a,b] += ic[i,m] * td[a,b,j,m] - ic[j,m] * td[a,b,i,m]
							dt[i,j,a,b] += ia[i,m] * td[a,b,j,m] - ia[j,m] * td[a,b,i,m]
							for n in range(0, nElectrons):
								dt[i,j,a,b] += 0.5 * mr2[m,n,a,b] * wmnij[m,n,i,j]
						for e in range(nElectrons, spinOrbitals):
							dt[i,j,a,b] += -mr1[j,e] * wabei[a,b,e,i] + mr1[i,e] * wabei[a,b,e,j]
							dt[i,j,a,b] += mr2[i,j,a,e] * fae[b,e] - mr2[i,j,b,e] * fae[a,e]
							dt[i,j,a,b] += ib[a,e] * td[b,e,i,j] - ib[b,e] * td[a,e,i,j]
							dt[i,j,a,b] += id[a,e] * td[b,e,i,j] - id[b,e] * td[a,e,i,j]
							for m in range(0, nElectrons):
								dt[i,j,a,b] += mr2[m,j,a,e] * wmbej[m,b,e,i] - mr2[m,j,b,e] * wmbej[m,a,e,i] - \
								               mr2[m,i,a,e] * wmbej[m,b,e,j] + mr2[m,i,b,e] * wmbej[m,a,e,j]          
							for f in range(nElectrons, spinOrbitals):
								dt[i,j,a,b] += 0.5 * mr2[i,j,e,f] * wabef[a,b,e,f]

		return dt * -1j


	M = []
	for i in range(3):
		M.append(rightHandFunction[i] + 1j * 0.0)
	dt = [dtRightDipoleFunction0, dtRightDipoleFunction1, dtRightDipoleFunction2]

	#list of results
	time = []
	dipoles = []

	#begin time propogation using Runge-Kutta(4th order)
	for cycle in range(cycles):

		t = cycle * interval

		#step 1
		k = [[], [], [], []]
		for i in range(3):
			k[0].append(dt[i](M[1], M[2]))

		next = []
		for i in range(3):
			next.append(M[i] + 0.5 * interval * k[0][i])

		#steps 2,3,4
		for j in range(1,4):
			f = 0.5
			for i in range(3):
				k[j].append(dt[i](next[1], next[2]))
			if j == 3 : break
			if j == 2 : f = 1.0
			for i in range(3):
				next[i] = M[i] + f * interval * k[j][i]

		#re-compute dipole function at time + interval
		for j in range(3):
			M[j] += (interval/6) * (k[0][j] + 2.0 * k[1][j] + 2.0 * k[2][j] + k[3][j])

		#compute autocorrelation function for this step                                  [13]
		autoCorrelation = leftHandFunction[0] * M[0]
		for i in range(0, nElectrons):
			for a in range(nElectrons, spinOrbitals):
				autoCorrelation += leftHandFunction[1][i,a] * M[1][i,a]
				for j in range(0, nElectrons):
					for b in range(nElectrons, spinOrbitals):
						autoCorrelation += 0.25 * leftHandFunction[2][i,j,a,b] * M[2][i,j,a,b]
		
		time.append(t)
		dipoles.append(autoCorrelation)

	return time, dipoles
