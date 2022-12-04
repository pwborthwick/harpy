from __future__ import division
from atom import getMass, getConstant
from force import forces
import numpy as np
import matplotlib.pyplot as plt
from math import acos
from scipy import signal
import rhf
import time

def mdBornOppenheimer(timeIncrement, iterations, integrator = 'velocity-verlet', out=['plot','file']):

    #constants relative atomic masses to atomic units and atomic time units to (femto) seconds
    em2amu =  getConstant('em2->amu')
    toFemtoSeconds =getConstant('atu->femtosecond')

    #initial scf calculation
    molAtom, molBasis, molData = rhf.mol([])
    totalEnergy = rhf.scf(molAtom, molBasis, molData , [])

    #titles
    name, basisName = [molData['name'], molData['basis']]

    #initial forces
    force = []
    vis = forces(molAtom, molBasis, rhf.density, rhf.fock)
    force.append(vis)

    #initial velocities - start at rest
    velocity = np.zeros_like(vis)

    #initial geometry
    geo = np.zeros_like(vis)
    for a in range(vis.shape[0]):
        geo[a,:] = molAtom[a].center[:]

    #output set-up
    if 'file' in out:
        #output to file - number of atoms, number of iterations
        #file will be of length 1 + number of iterations * number of atoms
        f = open('md.hdf','w')
        f.write(name + ' ' + basisName + ' ' + integrator + '\n')
        f.write("%2d" % vis.shape[0] + ' ' + "%4d" % iterations + ' ' + "%4d" % timeIncrement + '\n')
        f.write("%.2f" % 0 + ' ' + "%12.8f" % totalEnergy + ' ' + "%8.3f" % 0 + '\n')
    if 'plot' in out:
        energy = []

    #initial state
    for a in range(vis.shape[0]):
        if 'file' in out:
            f.write('{0:10.6f} {1:10.6f} {2:10.6f} {3:10.6f} {4:10.6f} {5:10.6f} {6:10.6f} {7:10.6f} {8:10.6f} \n'.format(geo[a,0], geo[a,1],geo[a,2], \
                    velocity[a,0], velocity[a,1], velocity[a,2], \
                    force[0][a,0], force[0][a,1], force[0][a,2]))
    if 'plot' in out:
        energy.append(totalEnergy)


    t = time.time()
    for cycle in range(1, iterations):

        if (integrator == 'velocity-verlet') or ((integrator in ['beeman','adams-moulton']) and (cycle == 1)):
            #update position
            for a in range(vis.shape[0]):
                for cart in range(3):
                    geo[a, cart] += velocity[a,cart] * timeIncrement + 0.5 * timeIncrement * timeIncrement * force[-1][a,cart] / (getMass(molAtom, a) * em2amu)

            #run scf for new position
            molAtom, molBasis = rhf.rebuildCenters(molAtom, molBasis, geo)
            totalEnergy = rhf.scf(molAtom, molBasis, molData , [], rhf.density)

            #get new forces
            vis = forces(molAtom, molBasis, rhf.density, rhf.fock)
            force.append(vis)

            #update velocity
            for a in range(vis.shape[0]):
                for cart in range(3):
                    velocity[a,cart] += 0.5 * timeIncrement * (force[-2][a,cart] + force[-1][a,cart]) /  (getMass(molAtom, a) * em2amu)

        elif integrator in ['beeman','adams-moulton']:
            #update position
            for a in range(vis.shape[0]):
                for cart in range(3):
                    geo[a, cart] += velocity[a, cart] * timeIncrement + timeIncrement* timeIncrement * (4.0 * force[-1][a, cart] - force[-2][a, cart] ) \
                                  / (getMass(molAtom, a) * em2amu * 6.0)

            #run scf for new position
            molAtom, molBasis = rhf.rebuildCenters(molAtom, molBasis, geo)
            totalEnergy = rhf.scf(molAtom, molBasis, molData , [], rhf.density)

            #get new forces
            vis = forces(molAtom, molBasis, rhf.density, rhf.fock)
            force.append(vis)

            #update velocity
            if integrator == 'beeman':
                for a in range(vis.shape[0]):
                    for cart in range(3):
                        velocity[a, cart] +=  timeIncrement * (2.0 * force[-1][a, cart] + 5.0 * force[-2][a, cart] - force[-3][a, cart]) \
                                            / (getMass(molAtom, a) * em2amu * 6.0)
            elif integrator == 'adams-moulton':
                for a in range(vis.shape[0]):
                    for cart in range(3):
                        velocity[a, cart] +=  timeIncrement * (5.0 * force[-1][a, cart] + 8.0 * force[-2][a, cart] - force[-3][a, cart]) \
                                           / (getMass(molAtom, a) * em2amu * 12.0)


        elapsed = (cycle + 1) * timeIncrement

        if 'file' in out:
        #write output file
            f.write("%.2f" % elapsed + ' ' + "%12.8f" % totalEnergy + ' ' + "%8.3f" % (time.time() - t) + '\n')
            for a in range(vis.shape[0]):
                f.write('{0:10.6f} {1:10.6f} {2:10.6f} {3:10.6f} {4:10.6f} {5:10.6f} {6:10.6f} {7:10.6f} {8:10.6f} \n'.format(geo[a,0], geo[a,1],geo[a,2], \
                        velocity[a,0], velocity[a,1], velocity[a,2], \
                        force[-1][a,0], force[-1][a,1], force[-1][a,2]))
        if 'plot' in out:
            energy.append(totalEnergy)

    if 'file' in out:
        f.close()
    if 'plot' in out:
        plt.title('[' + integrator + '][' + name + '][' + basisName + ']')
        plt.xlabel('t')
        plt.ylabel('E')
        plt.plot( np.arange(0, timeIncrement*iterations, timeIncrement)*toFemtoSeconds, energy)
        plt.grid(True)
        plt.show()

def mdVisualise(type):

    toFemtoSeconds = getConstant('atu->femtosecond')

    #get type and attribute either eg 'bond{0:2}' or 'angle{1:0:2}''
    if type.find('{') == 0:
        type = type + '{'
    attribute = type[type.find('{'):]
    type = type[:type.find('{')]
    if type == 'bond':
        l = int(attribute[1:attribute.find(':')])
        r = int(attribute[attribute.find(':')+1:-1])
    if type == 'angle':
        i = attribute.find(':')
        l = int(attribute[1:i])
        m = int(attribute[i+1:attribute.find(':',i+1)])
        r = int(attribute[attribute.find(':',i+1)+1:-1])


    file = open('md.hdf', 'r')

    #time, force, velocity and energy
    t = []
    E = []

    #read molecule, basis and integrator
    data = file.readline().split()
    if len(data) != 3:
        exit('data format error')

    title = '[' + data[0] + '][' + data[1] + '][' + data[2] + ']'

    #read nAtoms and number of steps
    data = file.readline().split()
    atoms = int(data[0])
    steps = int(data[1])

    #coordinates and molecular attribute
    q = []
    Q = []

    #loop over file lines
    for step in range(steps):

        #time and energy
        data = file.readline().split()
        t.append(float(data[0])*toFemtoSeconds)
        E.append(float(data[1]))

        #coordinates
        for a in range(atoms):
            data = file.readline().split()
            q.append([float(data[0]), float(data[1]), float(data[2])])

        if type == 'bond':  
            Q.append(np.linalg.norm(np.array(q[-atoms+l])-np.array(q[-atoms+r])))
        if type == 'angle':
            va = np.array(q[-atoms+l])-np.array(q[-atoms+m])
            vb = np.array(q[-atoms+m])-np.array(q[-atoms+r])
            Q.append(acos(np.dot(va,vb)/(np.linalg.norm(va)*np.linalg.norm(vb))) * getConstant('radian->degree'))

    #plot energy and attribute
    plt.figure()
    plt.subplot(211)
    plt.xlabel('t (fs)')
    plt.ylabel('E (Eh)')
    plt.grid(True)
    plt.title(title)
    plt.plot(t,E)

    if type == 'bond':
        plt.subplot(212)
        plt.xlabel('t (fs)')
        plt.ylabel('r (bohr)')
        plt.grid(True)
        plt.plot(t,Q)

    if type == 'angle':
        plt.subplot(212)
        plt.xlabel('t (fs)')
        plt.ylabel('angle')
        plt.grid(True)
        plt.plot(t,Q)

    plt.show()

def velocityAutocorrelation(options):
    #get intensity spectrum from velocity autocorrelation function

    def BOMDfileRead(input):
        #reads file for BOMD run parameters and geometry

        with open(input, 'r') as f:

            AVFdata = []
            count = 0
            for line in f:

                data = line.strip().replace('   ', ' ').replace('  ', ' ').split( ' ')
                if count == 0:
                    title = data[0] + '/' + data[1] + ' [' + data[2] + ']'
                elif count == 1:
                    nAtoms = int(data[0])
                    steps = int(data[1])
                    timeStep = float(data[2])
                elif len(data) == 9:
                    AVFdata.append(data[:3])

                count += 1

            return np.array(AVFdata, dtype = float).reshape(steps, nAtoms, 3), [timeStep, steps, nAtoms, title]

    def getVibrationMode(options, geometry):
        #get the length or angular displacement required

        #get stretch
        if options['mode'] == 's':
            return np.linalg.norm(geometry[:, options['atoms'][0], :] - geometry[:, options['atoms'][1], :], axis=1)
        #get bend
        if options['mode'] == 'b':
            u = geometry[:,options['atoms'][0],:] - geometry[:,options['atoms'][1],:]
            v = geometry[:,options['atoms'][2],:] - geometry[:,options['atoms'][1],:]
            theta = np.arccos((u*v).sum(axis=1)/(np.linalg.norm(v, axis=1)*np.linalg.norm(u, axis=1))) * getConstant('radian->degree')
            return theta
        #get dihedral
        if options['mode'] == 'd':
            #plane 1 defined by u,v and plane 2 by v,w
            u = geometry[:,options['atoms'][1],:] - geometry[:,options['atoms'][0],:]
            v = geometry[:,options['atoms'][2],:] - geometry[:,options['atoms'][1],:]
            w = geometry[:,options['atoms'][3],:] - geometry[:,options['atoms'][2],:]
            #normals to planes
            normalToPlaneUV = np.cross(u,v)
            normalToPlaneVW = np.cross(v,w)
            #dot product of normals is cos.|a|.|b|
            dotProduct = (normalToPlaneUV * normalToPlaneVW).sum(axis=1)
            #get normal norms
            normNormalToPlaneUV = np.linalg.norm(normalToPlaneUV,axis=1)
            normNormalToPlaneVW = np.linalg.norm(normalToPlaneVW,axis=1)
        
            phi = np.arccos(dotProduct / (normNormalToPlaneUV * normNormalToPlaneVW))

            return phi * getConstant('radian->degree')

    def getTimeDerivatives(vibrations, dt):
        #get the derivatives of the displacements

        dy = np.gradient(vibrations)
        
        return np.divide(dy, dt)

    def getAutoCorrelation(f):
        #compute the velocity auto-correlation function

        fUnbiased = f - np.mean(f)
        fNormed = np.sum(fUnbiased * fUnbiased)
        
        autocorrelation = signal.fftconvolve(f, f[::-1], mode='full')[len(f)-1:] / fNormed

        return autocorrelation

    def getViewport(options, dataLength):
        #define a window

        data = options['window'].split(',')
        if len(data) == 1:
            return signal.get_window(data[0], dataLength, False)
        else:
            return signal.get_window((data[0], float(data[1])), dataLength, False)

    def getPadding(dataLength):
        #get next power of 2 beyond data length

        if (dataLength and not (dataLength & (dataLength - 1))) : return dataLength
        n = 1
        while (n < dataLength):
            n <<= 1

        return n

    def autocorrelationFFT(ACF, window):
        #perform a discrete fast Fourier Transform analysis of autocorrelation function

        dataLength = len(ACF)
        #unbiased units: data/mean   
        w = window / (sum(window) / dataLength)
        
        #convolve data with window
        convolvedSignal = ACF * w

        #get zero padding
        padding = getPadding(dataLength)

        #do the FFT
        FFT = np.fft.fft(convolvedSignal, padding) / dataLength

        #return square of absolute intensity
        return np.absolute(FFT) * np.absolute(FFT)

    #retrieve the displacement vectors from BOMD file
    displacements, bomdParameters = BOMDfileRead(options['file'])
    timeStep, steps, atoms, title = bomdParameters
    timeStep *= getConstant('atu->femtosecond') * 1e-15

    #compute the stretches
    vibrationMode = getVibrationMode(options, displacements)

    #get the derivatives of the mode displacements (velocities)
    derivatives = getTimeDerivatives(vibrationMode, timeStep)

    #compute the auto-correlation function
    vACF = getAutoCorrelation(derivatives)

    #define a viewport
    window = getViewport(options, steps)

    #Wiener–Khinchin theorem: FFT of autocorrelation is power spectrum
    spectrum = autocorrelationFFT(vACF, window)

    #get frequency (omega) and intensity
    spectrumLength = len(spectrum)

    frequency = np.fft.fftfreq(spectrumLength, timeStep * getConstant('c') )[0:int(spectrumLength * 0.5)]
    intensity = spectrum[0:int(spectrumLength * 0.5)]

    #visualisation
    nPlot = len(options['plot'])
    import matplotlib.pyplot as plt

    derivatives = derivatives * timeStep
    xtick = np.arange(0, steps+1, int(steps * 0.25)) * timeStep * 1e15
    xlabel = []
    for x in xtick:
        xlabel.append(str(int(x)))

    plotCount = 0
    if 'derivative' in options['plot']:
        plotCount += 1
        plt.subplot(nPlot,1,plotCount)
        plt.title(title, fontsize='small')

        dataPoints = np.arange(len(derivatives))
        plt.plot(dataPoints, derivatives, color="red", linewidth=1.5 )
        plt.axis([0, len(derivatives) ,      
                  1.1 * np.min(derivatives),
                  1.1 * np.max(derivatives)])
        #converts steps to femtoseconds for x axis
        plt.xticks(np.arange(0, steps+1, int(steps * 0.25)), xlabel, fontsize = 'x-small')
        plt.yticks(np.linspace(np.min(derivatives),np.max(derivatives),4), fontsize = 'x-small')

        plt.xlabel("Time (fs)", fontsize = 'x-small')
        if options['mode'] == 's': plt.ylabel("Derivative of stretch (bohr)", fontsize = 'x-small')
        if options['mode'] == 'b': plt.ylabel("Derivative of bend (degree)", fontsize = 'x-small')
        if options['mode'] == 'd': plt.ylabel("Derivative of dihedral (degree)", {fontsize : 'x-small'})

    if 'ACf' in options['plot']:
        plotCount += 1
        plt.subplot(nPlot,1,plotCount)
        dataPoints = np.arange(len(vACF))
        plt.plot(dataPoints, vACF, color='red', linewidth=1.5)
        plt.axis([0, len(vACF),
                  1.1 * np.min(vACF),
                  1.1 * np.max(vACF)])
        plt.xticks(np.arange(0, steps+1, int(steps * 0.25)), xlabel, fontsize = 'x-small')
        plt.yticks(np.linspace(np.min(vACF),np.max(vACF),4), fontsize = 'x-small')
        plt.xlabel("Time (fs)", fontsize = 'x-small')
        plt.ylabel("velocity ACF (au)", fontsize = 'x-small')
    elif 'ACp' in options['plot']:
        plotCount += 1
        plt.subplot(nPlot,1,plotCount)
        plt.acorr(vACF, usevlines=True, normed=True, maxlags=len(dataPoints)//4, color='red')


    if 'spectrum' in options['plot']:
        plotCount += 1
        plt.subplot(nPlot,1,plotCount)
        plt.plot(frequency, intensity, color="black", linewidth=1.5)
        minIntensity = np.min(intensity)
        maxIntensity = np.max(intensity)
        plt.axis([0, len(frequency)//2,
                  1.1 * minIntensity,
                  1.1 * maxIntensity])
        if options['mode'] == 's': 
            attribute = 'stretch ' + str(options['atoms'][0]) + '-' + str(options['atoms'][1])
        elif options['mode'] == 'b':
            attribute = 'bend ' + str(options['atoms'][1]) + '-' + str(options['atoms'][0]) + '-' + str(options['atoms'][2])
        elif options['mode'] == 'd':
            attribute = 'dihedral ' + str(options['atoms'][1]) + '-' + str(options['atoms'][0]) + '-' + str(options['atoms'][2]) + \
                                '-' + str(options['atoms'][2])
        plt.text(len(frequency)/100,maxIntensity - maxIntensity/6, attribute, fontsize = 'x-small' )
        plt.grid()
        plt.xticks(np.linspace(0, len(frequency)//2, 10), fontsize = 'x-small')
        plt.yticks(np.around(np.linspace(np.min(intensity),np.max(intensity),4), 3), fontsize = 'x-small')
        plt.xlabel("wavenumber (cm$^{-1}$)", fontsize = 'x-small')
        plt.ylabel("intensity (au)", fontsize = 'x-small')
        plt.subplots_adjust(hspace = 0.5)

    plt.show()

    return frequency, intensity
