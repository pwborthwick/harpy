from __future__ import division
import numpy as np
from atom import getConstant
from scipy.special import genlaguerre, sph_harm, assoc_laguerre
import math
import matplotlib.pyplot as py
import matplotlib.colors as pcolors
from matplotlib import cm

#reduced bohr radius
a = 1.0

def transform(mode, i, j, k):
    #coordinate system transforms

    if mode == 'cartesian->spherical':
        x, y, z = i, j, k
        r = np.sqrt(x**2 + y**2 + z**2)
        with np.errstate(divide='ignore', invalid='ignore'):
            theta = np.arccos(z/r)
        phi = np.arctan2(y,x)

        return r, theta, phi

    elif mode == 'spherical->cartesian':
        r, theta, phi = i, j, k
        x, y, z = [r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta)]

        return x, y, z

def angularSolution(m, l, theta, phi):
    #angular hydrogenic solution

    return sph_harm(m, l, phi, theta)

def angularSolutionPlot(ax, m, l, parameters):
    #plot the spherical harmonic Y[ml]

    #construct a theta, phi grid
    theta_extent = np.linspace(0, np.pi, parameters['points'])
    phi_extent   = np.linspace(0, 2*np.pi, parameters['points'])
    (theta, phi) = np.meshgrid(theta_extent, phi_extent)

    point = np.array(transform('spherical->cartesian', 1.0, theta, phi))

    Y = angularSolution(abs(m), l, theta, phi)
    if m < 0:   Y = np.sqrt(2) * (-1)**m * Y.imag
    elif m > 0: Y = np.sqrt(2) * (-1)**m * Y.real

    x, y, z = np.abs(Y) * point

    color_map = py.cm.ScalarMappable(cmap = py.get_cmap(parameters['color_map']))
    color_map.set_clim(-0.5, 0.5)

    surface = ax.plot_surface(x, y, z, facecolors = color_map.to_rgba(Y.real), rstride = 2, cstride = 2, alpha = parameters['alpha'])

    py.title(r'$Y_{{{},{}}}$'.format(l, m))

    extent = parameters['extent']
    ax.plot([0,0], extent, [0,0], c='0.5', lw=1, zorder=10)
    ax.plot([0,0], [0,0], extent, c='0.5', lw=1, zorder=10)
    ax.plot(extent, [0,0], [0,0], c='0.5', lw=1, zorder=10)
    ax.set_xlim(extent)
    ax.set_ylim(extent)
    ax.set_zlim(extent)
    ax.axis(parameters['axes'])

    if parameters['bar'] == 'on': py.colorbar(color_map, shrink=0.5, aspect=30, orientation='horizontal')

    return

def angularSolutionPlotProjection(m, l, parameters= {'points':70,'extent':[-0.5,0.5], 'alpha':0.3, 'color_map':'coolwarm', \
                                                     'axes':'off','levels':20}):

    #construct a theta, phi grid
    theta_extent = np.linspace(0, np.pi, parameters['points'])
    phi_extent   = np.linspace(0, 2*np.pi, parameters['points'])
    (theta, phi) = np.meshgrid(theta_extent, phi_extent)

    point = np.array(transform('spherical->cartesian', 1.0, theta, phi))

    Y = angularSolution(abs(m), l, theta, phi)
    if m < 0:   Y = np.sqrt(2) * (-1)**m * Y.imag
    elif m > 0: Y = np.sqrt(2) * (-1)**m * Y.real

    x, y, z = np.abs(Y) * point

    fig = py.figure(figsize=(5,5))
    ax = fig.add_subplot(111, projection='3d') 

    color_map = py.cm.ScalarMappable(cmap = py.get_cmap(parameters['color_map']))
    color_map.set_clim(-0.5, 0.5)
    ax.plot_surface(x, y, z, facecolors = color_map.to_rgba(Y.real), alpha=parameters['alpha'])

    cset = ax.contour(x, y, z, parameters['levels'], zdir='z', offset=-1, cmap='summer', alpha=parameters['alpha']*0.5)
    cset = ax.contour(x, y, z, parameters['levels'], zdir='y', offset= 1, cmap='winter', alpha=parameters['alpha']*0.5)
    cset = ax.contour(x, y, z, parameters['levels'], zdir='x', offset=-1, cmap='autumn', alpha=parameters['alpha']*0.5)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.axis(parameters['axes'])

    py.title(r'$Y_{{{},{}}}$'.format(l, m))

    py.show()

def angularSolutionPlotSingle(m, l, parameters = {'points':70,'extent':[-0.5,0.5],'color_map':'coolwarm', \
                                      'bar':'on','axes':'off','alpha':0.8}):

    fig = py.figure(figsize = py.figaspect(1.0))
    ax = fig.add_subplot(projection='3d')

    angularSolutionPlot(ax, m, l, parameters)

    py.show()

    return

def angularSolutionPlotFamily(l_maximum, parameters = {'points':70,'extent':[-0.5,0.5],'color_map':'coolwarm', \
                                                 'bar':'off','axes':'off','alpha':1.0}):
    #plot a family of spherical harmonics
    import  matplotlib.gridspec as gridspec

    figsize_size, DPI = 500, 100
    figsize_inches = figsize_size/DPI

    fig = py.figure(figsize=(figsize_inches, figsize_inches), dpi=DPI)
    spec = gridspec.GridSpec(ncols = 2*l_maximum+1, nrows = l_maximum+1, figure = fig)

    for l in range(l_maximum + 1):
        for m in range(-l, l+1):
            ax = fig.add_subplot(spec[l, m + l_maximum], projection='3d')
            angularSolutionPlot(ax, m, l, parameters)

    py.tight_layout()
    py.show()

    return

def angularVerify():
    #compare analytic solution with sympy

    from random import randint
    l = randint(0, 4)
    m, theta, phi = [randint(-l, l), randint(0, 5000) * np.pi * 1e-4, randint(0, 20000) * np.pi * 1e-4]

    analytic = sph_harm(m, l, phi, theta)

    from sympy import Ynm, evalf
    values = {'l':l, 'm':m, 'theta':theta, 'phi':phi}
    symbolic = Ynm(l, m, theta, phi).evalf(subs=values)

    return ((analytic-complex(symbolic)) < (1e-15 + 1e-15j))

def radialSolution(n, l, r):
	#radial hydrogenic wavefunction

    rho = (2*r/(n*a))
    L = genlaguerre(n-l-1, 2*l+1)(rho)
    normalization = np.sqrt(((2/(n*a))**3.0)*math.factorial(n-l-1)/(2*n*math.factorial(n+l)))
    psi = (rho**l) * np.exp(-rho/2) * L * normalization
        
    return psi

def radialVerify():
    #compare analytic solution with sympy

    from random import randint
    n = randint(1, 5)
    l, r = [randint(0, n-1), randint(0, 20000) * 1e-3]

    from sympy.physics.hydrogen import R_nl
    analytic = radialSolution(n, l, r)
    symbolic = R_nl(n, l, r, 1)

    return ((analytic - symbolic) < 1e-15)

def radialSolutionType(n, l, r, psi_type = 'radial distribution'):
	#return either radial distribution, radial probability density or radial probability distribution

    psi = radialSolution(n, l, r)

    if   psi_type == 'radial distribution'     : return psi
    elif psi_type == 'probability density'     : return psi * psi
    elif psi_type == 'probability distribution': return (4.0*np.pi*r*r) * (psi*psi)
    else:
        exit(psi_type, ' not implemented')

def radialSolutionPlot(n, l, psi_type = 'radial distribution', psi_normal = False,\
                       parameters = {'points':100, 'size':[7,5], 'extent':[20,0.2], 'equal':False}):
	#plot either for given n range of l, of for given l range of n

    subshell = ['s', 'p', 'd', 'f', 'g']

    yLabel = {'radial distribution':'$\psi(r)$', 'probability density':'$\psi(r)^2$',\
              'probability distribution':'$4\pi{r}^2\psi(r)$'}

    points = parameters['points']

    f = py.figure() 
    f.set_figheight(parameters['size'][1])
    f.set_figwidth(parameters['size'][0])
    ax = py.subplot()
    ax.grid()
    py.title(psi_type + ' for hydrogenic $\Psi$')

    r = np.linspace(0, parameters['extent'][0], points)

	#plot all the shells and subshell in lists
    for i in range(len(n)):

        if l[i] >= n[i]: continue
        psi = radialSolutionType(n[i], l[i], r, psi_type)
        if psi_normal: psi /= np.max(psi)
        py.plot(r, psi, label= str(n[i]) + subshell[l[i]])


    py.ylim(parameters['extent'][1])
    if parameters['equal']: ax.set_box_aspect(1)
    py.xlabel('r($[a]$)')
    ax.legend(loc='upper right')
    py.ylabel(yLabel[psi_type])

    py.show()

    return

def wavefunction(n, l, m, r, theta, phi, grid):
    #evaluate the radial and angular components from full psi

    if grid == 'cartesian':
        #if cartesian transform to r, theta, phi
        r, theta, phi = transform('cartesian->spherical', r, theta, phi)

    #get radial solution
    radial = radialSolution(n, l, r)
    
    #get angular solution
    angular = angularSolution(abs(m), l, theta, phi)
    
    return radial * angular

def wavefunctionContour(n, l, m, parameters={'points':100, 'extent':[-20, 20], 'color_map':'coolwarm', \
                                             'plane':'xy', 'elevation':0.0, 'contour': False}):
    #plot contour map of wavefunction

    extent = parameters['extent']
    points = parameters['points']

    step = (extent[1] - extent[0])/points
    grid = np.linspace(extent[0], extent[1], points)

    x, y, z = np.meshgrid(grid, grid, grid) 
    r, theta, phi = transform('cartesian->spherical', x, y, z)
    Y = angularSolution(abs(m), l, theta, phi)
    if m < 0:   Y = np.sqrt(2) * (-1)**m * Y.imag
    elif m > 0: Y = np.sqrt(2) * (-1)**m * Y.real

    data = np.nan_to_num(Y * radialSolution(n, l, r))
    data = abs(data)**2

    fig, ax = py.subplots()

    slice = max(min(parameters['elevation'], extent[1]) , extent[0])
    slice =  (-extent[0] + parameters['elevation'])

    switch = {'xy':(2,0,1),'yz':(1,2,0),'zx':(0,1,2)}
    data = data.transpose(switch[parameters['plane']])

    im = py.imshow(data[int((-extent[0]+parameters['elevation'])/step),:,:], vmin= 0.0, vmax = np.max(data), \
                                                extent = (extent + extent), cmap = parameters['color_map'])

    py.colorbar(shrink=0.5, aspect=30, orientation = 'horizontal')
    ax.set_title('hydrogen orbital ' + parameters['plane'] + '-slice at displacement ' + str(parameters['elevation']) +  \
                                                    '\nn=' + str(n) + ', l=' + str(l) + ', m=' + str(m), fontsize=10)
    py.xlabel(parameters['plane'][0])
    py.ylabel(parameters['plane'][1])

    if parameters['contour']:
        levels = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.1]
        cs = ax.contour(data[int((-extent[0]+parameters['elevation'])/step),:,:] , levels, origin='lower', cmap='gray', extend='both',
                    linewidths=0.2, extent=(extent + extent))

    py.show()

    return

def wavefunctionVerify():
    #compare analytic solution with sympy

    from random import randint
    n = randint(1, 5)
    l, r, theta, phi = [randint(0, n-1), randint(0, 20000) * 1e-3, randint(0, 5000) * np.pi * 1e-4, randint(0, 20000) * np.pi * 1e-4]
    m = randint(-l, l)

    from sympy.physics.hydrogen import Psi_nlm
    analytic = wavefunction(n, l, m, r, theta, phi, 'spherical polar')
    values = {'n':n,'l':l, 'm':m,'r':r, 'theta':theta, 'phi':phi}
    symbolic = Psi_nlm(n, l, m, r, phi, theta, 1).evalf(subs=values)

    return ((analytic-complex(symbolic)) < (1e-15 + 1e-15j))

def numerov(g, u_zero, du, dh):
    #Numerov method for integrating ode

    #array for wavefunction values
    u = np.zeros_like(g)    
    u[0] = u_zero
    u[1] = u[0] + dh*du
    
    h = (dh*dh)/12.
    f = [u[0]*(1.0 - h*g[0]), u[1]*(1.0 - h*g[1]), 0.0]

    #current cycle values
    cycle_value = [u[1], g[1]]

    for i in range(2, g.size):

        f[2] = 2*f[1] - f[0] + dh*dh * cycle_value[0] * cycle_value[1]
        cycle_value[1] = g[i] 

        cycle_value[0] = f[2]/(1.0 - h* cycle_value[1])
        u[i]=cycle_value[0]

        f[0] = f[1]
        f[1] = f[2]

    return u

def numerovBoundEnergy(r, l, n_range, energy_range):

    from scipy.optimize import brentq

    def shootingMethod(E, r, l):
        #boundary value by 'shooting' method
        from scipy.integrate import simps

        g = l*(l+1.)/r**2 - 2./r - E
        u = numerov(g, 0.0,-1e-8 ,-r[-2] + r[-1])

        #normalize wavefunction
        u /= np.sqrt(abs(simps(u**2,x = r)))
        u /= r**l

        #interpolate back to zero
        f_zero = u[-1] + (u[-2] - u[-1])*(0.0-r[-1])/(r[-2]-r[-1])

        return f_zero

    n = l
    bound_energy = []
    u_zero = shootingMethod(energy_range[0],r,l)

    for i in range(1, len(energy_range)):
        u = shootingMethod(energy_range[i],r,l)
        if u_zero * u  < 0:
            e = brentq(shootingMethod, energy_range[i-1], energy_range[i], xtol=1e-16, args=(r,l))
            bound_energy.append((n+1, l, e))

            n += 1

            if len(bound_energy) >= n_range: break

        u_zero = u
    
    return bound_energy

def numerovHydrogenicEnergies(parameters = {'points':2000, 'extent':[1e-8, 100], 'n_range': 5}):
    #compute the bound energies using Numerov

    energy_range = -1.2/(np.arange(1,20,0.2)**2.0)

    r = np.linspace(parameters['extent'][0],parameters['extent'][1],parameters['points'])[::-1]

    n_range = parameters['n_range']

    bound_states = []
    for l in range(n_range):
        bound_states += numerovBoundEnergy(r, l, n_range-l, energy_range)

    bound_states.sort(key = lambda x: x[2])  

    return bound_states

def numerovEnergyVerify(boundEnergies):
    #check bound energies against theoretical values

    valid = []
    for n,l,e in boundEnergies:
        theoretical = -1.0/(n * n)
        valid.append((e - theoretical) <= 1e-4)

    return np.all(valid), valid

def numerovHydrogenicRadialDensity(parameters = {'points':2000, 'extent':[1e-8, 100], 'n_range': 5, \
                                                       'radius':[0,15],'occupied': [1, True, False]}):
    #numerov wavefunction densities

    from scipy.integrate import simps
    r = np.linspace(parameters['extent'][0],parameters['extent'][1],parameters['points'])[::-1]

    #get the bound energies for the states
    bound_states = numerovHydrogenicEnergies(parameters)

    electrons = 0
    total_rho = np.zeros_like(r)
    Z = parameters['occupied'][0]
    states = []

    ax = py.subplot()
    subshell = ['s', 'p', 'd', 'f', 'g']

    for (n, l, e) in bound_states:

        #solve Numerov for bound energies
        g = l*(l+1.)/r**2 - 2./r - e
        u = numerov(g, 0.0,-1e-8 ,-r[-2]+r[-1])
        u /= np.sqrt(abs(simps(u*u, x = r)))

        occupation = 2*(2*l + 1)

        #fermi level at atomic number
        if (electrons + occupation) <= Z:
            fermi =1.0
        else:
            fermi = (Z - electrons)/float(occupation)

        #compute current orbital contribution to total
        if parameters['occupied'][1] : 
            rho = u*u * fermi * occupation/(4*np.pi*r*r)
        else:
            rho = u*u / (4*np.pi*r*r)

        total_rho += rho

        #new atomic occupation 
        electrons += occupation

        states.append((n,l,e,fermi))

        py.plot(r, rho*(4*np.pi*r*r), label= str(n) + subshell[l])

        if electrons >= Z: break

    py.xlim(parameters['radius'])
    py.xlabel('r($[a]$)')
    ax.legend(loc='upper right')
    ax.grid()
    py.ylabel(r'$\rho(\psi)$')
    py.title('Hydrogenic Wavefuntions (z=' + str(Z) + ')')
    py.show()

    if parameters['occupied'][2]:
        py.plot(r, total_rho*(4*np.pi*r*r))
        py.xlabel('r($[a]$)')
        ax.grid()
        py.ylabel(r'$\rho(\psi)$')
        py.title('charge density')
        py.xlim(parameters['radius'])
        py.show()

    return states

def numerovRadialVerify():

    #compare Numerov and sympy wavefunction profiles

    quantum_numbers = [[1,0],[2,0],[2,1],[3,0],[3,1],[3,2]]
    parameters = {'points':1000, 'extent':[1e-8, 100], 'z': 1}

    #sample points
    r_points = np.linspace(parameters['extent'][0],parameters['extent'][1],parameters['points'])[::-1]

    #compute analytic solution
    from sympy import lambdify
    from sympy.abc import r, z
    from sympy.physics.hydrogen import R_nl

    u_symbolic = [lambdify((r, z), r * R_nl(n, l, r, z), 'numpy')(r_points, 1) for n, l in quantum_numbers]

    #compute Numerov solution
    u_numerov = []
    i = 0
    for n, l in quantum_numbers:
        e = -parameters['z']/(n*n)
        g = l*(l+1.)/r_points**2 - 2./r_points - e

        #correct start from above or below y=0
        asymptote = [1, -1, 1, 1, -1, 1]

        u = numerov(g, 0.0,-1e-8 , asymptote[i]*(-r_points[-2] + r_points[-1]))
        i += 1

        from scipy.integrate import simps
        u /= np.sqrt(abs(simps(u**2,x = r_points)))
        u_numerov.append(u)

    #plot wavefunctions
    fig = py.figure(figsize=(8, 4), dpi=100)
    for i in range(len(quantum_numbers)):

        ax = py.subplot(2, 3, i+1)

        ax.plot(r_points, u_symbolic[i], lw=0.6, color = 'black', label = r'sympy')
        ax.plot(r_points, u_numerov[i], markersize = 4, lw=0, color = 'orange', marker = 'o', markevery=0.1, label = r'numerov')

        ax.set_xlim(-2, 16 + i * 8)
        ax.set_xlabel(r'$r$[$a$]', labelpad=5)
        ax.set_ylabel(r'$u_{{ {}{} }}(r)$'.format(*quantum_numbers[i]), labelpad=5)
        py.xticks(fontsize=8)
        py.yticks(fontsize=8)
        ax.legend(loc='best', fontsize='x-small', ncol=1)

    fig.suptitle('Numerov-Actual Comparison')
    py.tight_layout()
    py.show()

    return

def finiteDifferenceRadial(parameters = {'l': 0, 'points': 4000, 'extent':[0, 40], 'levels':25, 'show':5}):
    #matrix finite difference solution

    #parameters
    l = parameters['l']

    #constants
    hbar = getConstant('planck')/(2.0*np.pi)
    e = getConstant('e')
    epsilon_zero = getConstant('electric constant')
    electron_mass = 9.1093837015e-31

    from scipy import sparse

    def potentialMatrix(r):
        #1/r schroedinger term

        v = e*e/(4.0*np.pi*epsilon_zero*r)

        return sparse.diags(v)

    def angularMatrix(r, l):
        #1/r*r angular momentum term

        a = l*(l+1)/(r*r)

        return sparse.diags(a)

    def kineticMatrix(r):
        #Laplace matrix kinetic term

        step = r[1] - r[0]
        mainDiagonal = (-2.0/(step*step))*np.ones(len(r))
        offDiagonal =  ( 1.0/(step*step))*np.ones(len(r)-1)

        return sparse.diags([mainDiagonal, offDiagonal, offDiagonal], (0, -1, 1))

    r = np.linspace(parameters['extent'][1]*1.0e-10, parameters['extent'][0], parameters['points'], endpoint=False)

    #construct Hamiltonian
    h =  (-hbar*hbar/(2.0*electron_mass))*(kineticMatrix(r) - angularMatrix(r, l)) - potentialMatrix(r)

    from scipy.sparse.linalg import eigs
    e_levels = parameters['levels']
    E, c = eigs(h, k=e_levels, which='SM')

    #sort the eigensolution
    eigenvectors = np.array([x for _, x in sorted(zip(E, c.T), key=lambda pair: pair[0])])
    eigenvalues = np.sort(E).real/e

    #plot radial density
    radial_density = [pow(np.absolute(eigenvectors[i,:]),2) for i in range(e_levels)]

    rydberg = getConstant('hartree->eV')/2.0
    legend =  ['{: >5.2f} eV [{: >5.2f}]'.format(eigenvalues[i], -rydberg/pow(i+1+l,2)) for i in range(parameters['show'])]
    for i in range(parameters['show']):
        py.plot(r*1.0e10, radial_density[i], label=legend[i])

    py.xlabel('r ($\\mathrm{\AA}$)')
    py.ylabel('$\psi(r)^2$')
    py.xlim([0, 20])
    py.grid()
    py.legend()
    py.title('radial wavefunctions for l=' + str(l))
    py.show()

def hydrogenSpectralLines(type):
    #compute and plot Hydrogen spectrum series

    lineDictionary = {}

    wavenumberFactor = getConstant('eV[-1]->nm')
    energyFactor     = getConstant('rydberg->eV')

    def series(n):

        baseEnergy = boundEnergies[n][2] * energyFactor
        lines = []
        for i in boundEnergies[(n+1):]:
            energy = i[2] * energyFactor - baseEnergy
            lines.append([energy , wavenumberFactor/energy, str(i[0])+'->'+str(n+1)])

        return lines

    boundEnergies = numerovHydrogenicEnergies({'points':4000, 'extent':[1e-8, 200], 'n_range': 8})
    boundEnergies = [i for i in boundEnergies if i[1] == 0]

    if 'lyman'   in type: lineDictionary['lyman']   = series(0)
    if 'balmer'  in type: lineDictionary['balmer']  = series(1)
    if 'paschen' in type: lineDictionary['paschen'] = series(2)

    return lineDictionary

def monteCarloHydrogen(n, l, m, view='xy', plot=True):
    '''a Monte-Carlo approach to hydrogenic orbitals see 
       'A smooth path to plot hydrogen atom via Monte Carlo Method' - 
       Pedro Henrique Fernandes Lobo, Everaldo Arashiro, Alcides Castro e Silva, Carlos Felipe and Saraiva Pinheiro
    '''
    import random
    
    def unlinspace(l, u, delta, spacing=1.1):
        #custom linspace

        return [l + (i/(delta-1))**spacing*(u-l) for i in range(delta)]

    def angularDensity(t, l, m):
        #get the angular density

        wavefunction = sph_harm(m, l, 0.0, t)

        return (wavefunction * wavefunction).real

    def radialDensity(r, n, l):
        #get the radial wavefunction density

        factor   = np.sqrt(((2.0/n)**3.0) * math.factorial(n-l-1) / (2.0 * n * math.factorial(n+l)))
        laguerre = assoc_laguerre(2.0*r/n, n-l-1, 2.0*l+1)

        wavefunction = factor * np.exp(-r/n) * ((2.0*r/n)**l) * laguerre

        return (wavefunction * wavefunction).real

    def get_wavefunction_maximum(L, n, l, m, nr, nt, plot=True):
        #get the maximum value of the wavefunction
        
        #maximum radial is half diagonal of cube
        maxL = np.linalg.norm(L) * 0.5

        #recalculate radial and angular increments
        dr, dt = maxL/nr, math.pi/nt

        x = [] ; y = []; z = []

        max_value = 0.0
        for i in range(nr):
            r = i * dr

            for j in range(nt):
                t = j * dt

                angular_density = angularDensity(t, l, m)
                radial_density  = radialDensity(r, n, l)

                density = angular_density * radial_density
                x.append(r) ; y.append(t) ; z.append(density)

                if density > max_value: max_value = density

        if plot:
            fig = py.figure(figsize=(4,4))
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(x, y, z, s=0.002)

            py.title(r'$\Phi_{{{},{},{}}}$'.format(n, l, m))
            for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
                axis.set_ticklabels([])
                axis._axinfo['axisline']['linewidth'] = 1
                axis._axinfo['axisline']['color'] = (0, 0, 0)
                axis._axinfo['grid']['linewidth'] = 0.5
                axis._axinfo['grid']['linestyle'] = "-"
                axis._axinfo['grid']['color'] = (0, 0, 0)
                axis._axinfo['tick']['inward_factor'] = 0.0
                axis._axinfo['tick']['outward_factor'] = 0.0
                axis.set_pane_color((0.95, 0.95, 0.95))
            ax.yaxis.labelpad=-15
            ax.xaxis.labelpad=-15

            ax.set_xlabel(r'$r(a_0)$')
            ax.set_ylabel(r'$\theta$')

            py.show()

        return max_value.real

    #decode view
    view = view.replace('x','0',1).replace('y','1',1).replace('z','2',1)

    #define the view - viewport is dependent on n
    plane = [int(view[0]), int(view[1])]
    L = [0, 0, 0]
    for i in plane:
        L[i] = n*25.0 
    dx, dy, dz = L

    dr, dt, icr = 0.1, 0.1, 1.05

    nr = math.ceil(np.linalg.norm(L)/dr)
    nt = math.ceil(2 * math.pi/dt)

    if plot:
        fig = py.figure(figsize=(4,4))
        ax = fig.add_subplot(111)
        r = np.array(unlinspace(0, np.linalg.norm(L)*0.5, 1000, spacing=4))

        R = radialDensity(r, n, l)
        py.plot(r, np.sqrt(R), '.k', markersize=2)
        ax.set_yticks([])
        ax.set_xlabel(r'$r(a_0)$')
        py.title(r'$\Phi_{{{},{},{}}}$'.format(n, l, m))
        py.show()

    wf_maximum = get_wavefunction_maximum(L, n, l, m, nr, nt, plot)

    cycle, cycles = 1, n*2000

    mc_points = np.zeros((cycles+1,3))
    while cycle < cycles:
        
        #generate random value less than maximum of wavefunction + (icr-1)%
        random_wavefunction = random.random() * wf_maximum * icr

        #generate random point in cube space
        v = [dx * (random.random() - 0.5), dy * (random.random() - 0.5), dz * (random.random() - 0.5)]
        r = np.linalg.norm(v)

        t = math.acos(v[2]/r)

        wavefunction = radialDensity(r, n, l) * angularDensity(t, l, m)

        if random_wavefunction < wavefunction:
            cycle += 1
            mc_points[cycle] = v

    fig = py.figure(figsize=(4,4))
    ax = fig.add_subplot(111)

    py.plot(mc_points[:,plane[0]], mc_points[:,plane[1]], 'ok', markersize=1)
    ax.set_xlabel(r'$x(a_0)$')
    ax.set_ylabel(r'$y(a_0)$')
    py.title(r'$\Phi_{{{},{},{}}}$'.format(n, l, m))
    py.show()

if __name__ == '__main__':

    import sys
    args = ''
    for arg in sys.argv:
        args += arg

    #get svf type
    if 'rp' in args:
        radialSolutionPlot([1,2,3,2,3,3], [0,0,0,1,1,2], 'radial distribution', False, {'points':100,'size':[7,5],'extent':[20,[-0.15,0.2]],'equal':False})
    elif 'rv' in args:
        print('Radial verify returned ', radialVerify())
    elif 'as' in args:
        angularSolutionPlotSingle(0, 3, {'points':100,'extent':[-0.5,0.5],'color_map':'coolwarm','axes':'off','bar':'on','alpha':0.2})
    elif 'av' in args:
        print('Angular verify returned ', angularVerify())
    elif 'ap' in args:
        angularSolutionPlotProjection(1, 3)
    elif 'af' in args:
        angularSolutionPlotFamily(3)   
    elif 'wc' in args:
        wavefunctionContour(3, 2, 0, {'points':80, 'extent':[-30, 30], 'color_map':'gist_yarg', 'plane':'zx', 'elevation':0, 'contour': True})
    elif 'wv' in args:
        print('Wavefunction verify returned ', wavefunctionVerify())
    elif 'ne' in args:
        boundEnergies = numerovHydrogenicEnergies({'points':2000, 'extent':[1e-8, 100], 'n_range': 5})
        print('  n     l      Numerov Energy (E/E\N{SUBSCRIPT ZERO})    Actual')
        print('---------------------------------------------------------------')
        for e in boundEnergies:
            print('  {:1n}     {:1n}            {:<5.4f}            {:<5.4f}'.format(e[0], e[1], e[2], -1.0/(e[0]*e[0])))
    elif 'Nev' in args:
        boundEnergies = numerovHydrogenicEnergies({'points':2000, 'extent':[1e-8, 100], 'n_range': 5})
        overall_validity, valid_list = numerovEnergyVerify(boundEnergies)
        print('Numerov energy verify returned an overall  ',overall_validity)
        print('Numerov energy verify returned individuals ', valid_list)
    elif 'NRv' in args:
        numerovRadialVerify()
    elif 'Nw' in args:
        numerovHydrogenicRadialDensity({'points':2000, 'extent':[1e-8, 100], 'n_range': 5, \
                                        'radius':[0,25],'occupied': [28, True, True]})
    elif 'fr' in args:
        finiteDifferenceRadial()
    elif 'se' in args:
        lines = hydrogenSpectralLines(['lyman', 'balmer', 'paschen'])
        print('transition     energy (eV)   wavelength (nm)')

        print('Lyman series\n------------')
        for i in lines['lyman'][:5]:
            print('    {:<6s}     {:>8.3f}         {:>6.1f}'.format(i[2], i[0], i[1]))
        print('Balmer series\n-------------')
        for i in lines['balmer'][:5]:
            print('    {:<6s}     {:>8.3f}         {:>6.0f}'.format(i[2], i[0], i[1]))
        print('Paschen series\n--------------')
        for i in lines['paschen'][:5]:
            print('    {:<6s}     {:>8.3f}         {:>6.0f}'.format(i[2], i[0], i[1]))
    elif 'mc' in args:
        monteCarloHydrogen(6, 3, 1, view='yz', plot=True)
    else:
        print('key not recognized')
