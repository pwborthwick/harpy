# Born-Oppenheimer Molecular Dynamics

The module describes the time development of a molecule using Born-Oppenheimer molecular dynamics. 

1.	**mdBornOppenheimer(timeIncrement, iterations, integrator = 'velocity-verlet', out=['plot','file']):**
	
	parameters - *timeIncrement* is the basic step of time in atomic time units, *iterations* are the number of timeIncrement steps to be made, *integrator* is the method used to propogate the dynamics, default is the *velocity-verlet* method, *out* is a list which can be 'plot'|'file'. If 'plot' is specified a diagram of energy against time is displayed, if 'file' is specified the results are written to a file (local directory) for post-processing. Note that the RHF density is passed to the next cycle. Don't push these algorithms too far or the errors will accumulate. The output file has the following format

		h2o sto-3g velocity-verlet
	    3   200   10
		0.00 -74.94207995    0.000
		  0.000000  -0.143226   0.000000   0.000000   0.000000   0.000000   0.000000   0.097441   0.000000 
		  1.638037   1.136549  -0.000000   0.000000   0.000000   0.000000  -0.086300  -0.048721   0.000000 
		 -1.638037   1.136549  -0.000000   0.000000   0.000000   0.000000   0.086300  -0.048721   0.000000 
		10.00 -74.94221751    3.293
		  0.000000  -0.143184   0.000000   0.000000   0.000017   0.000000   0.000000   0.097192   0.000000 
		  1.637450   1.136217   0.000000  -0.000235  -0.000132   0.000000  -0.086111  -0.048596   0.000000 
		 -1.637450   1.136217   0.000000   0.000235  -0.000132   0.000000   0.086111  -0.048596   0.000000 

That is\
line 1 {'molecule name'} {'basis name'} {'integrator'}\
line 2 {number of atoms} {number of steps} {step size}\
followed by {number of steps} blocks of...\
&nbsp; &nbsp;       {time elapsed in simulation} {energy at this step} {real time elapsed}\
and {number of atoms} lines of triples (x,y,z)\
&nbsp; &nbsp;      {position} {velocity} {force}\


2.	**mdVisualise(type)**

	parameters - *type* is either 'bond{*n*:*m*}' or 'angle{*l*:*n*:*m*}', where *l, m, n* are atom designators. Will use the saved file from mdBornOppenheimer to plot the energy against time and either the bond or angle specified. If type = '' then just the energy will be plotted. Internally all units are atomic units, mdVisualise plots time as femtoseconds.

	Details of the velocity-verlet can be found [here](https://en.wikipedia.org/wiki/Verlet_integration). It is the most common of the symplectic integrators. 
	The Beeman algorithm can be found [here](https://en.wikipedia.org/wiki/Beeman%27s_algorithm).

Dynamics can be run as (set molecule data in project.hpf) then
```python
	from bomd import mdBornOppenheimer, mdVisualise
         
	mdBornOppenheimer(5, 200, 'velocity-verlet', ['plot', 'file'])
	mdVisualise('bond{0:1}')

```
**Example 1. Hydrogen Molecule**

These are the results for the energy variation and bond-length changes in the sto-3g basis. ![](/media/h2-sto-3g.png).

You can see the dynamics over a period of 20 femtoseconds. The period of the bond vibration is twice that of the energy vibration. The minimum of the energy occurs when the bond length is about 1.35 bohr, and extrema of the bond length occur when the energy is a maximum. The frequency of bond vibration is \~5472 cm<sup>-1</sup>.  The results in different bases are

| basis           |  frequency (cm<sup>-1</sup>)  |
|-----------------|-------------------|
| sto-3g          |      5472         |
| 3-21g           |    4659          |
| 6-31g           |    4643          |
| dz              |    4643         |
| cc-pvdz         |     4597          |
| expt.           |     4342        | [](chem,perdue.edu/gchelp/vibs/h2/html)

**Example 2. Water Molecule**

These are the results for the water molecule in sto-3g basis. ![](/media/h2o-sto-3g.png). 


The molecular vibrations are a symmetric stretch with a bend. At minimum energy the bond length is about 1.9 bohr and again maxima of the energy correspond to the extrema of the bond length. The profile of curves for the other OH bond are identical indicating a symmetric stretch. The frequency of the stretch is 4116 cm<sup>-1</sup>. The bend is depicted below ![](/media/h2o-sto-3g-a.png)

The frequency of the bend is 2206 cm<sup>-1</sup>. The frequency of the bend is about half that of the stretch so the energy profile is not a simple oscillation.  The results for different bases are

| basis           |  stretch  |  bend  |
|-----------------|-------------------|--------------|
|    sto-3g   |         4116       |          2206  |
|   3-21g    |      3727              |   1802        |
|   6-31     |     3778                 |     1692          |
|  dz        |     3830                |      1671            |
| cc-pvdz    |     3884            |          1839          |
| expt.   |     3657    |      1595    |



Methods currently available are the **velocity-verlet**

r(t + *dt*) = r(t)*dt* + a(t)*dt*<sup>2</sup>/2
v(t + *dt*) = v(t) + [a(t) + a(t+*dt*)] *dt*/2

**Beeman** algorithm

r(t + *dt*) = r(t)*dt* + v(t)*dt* + 2a(t)*dt*<sup>2</sup>/3 - a(t-*dt*)*dt*<sup>2</sup>/6
v(t + *dt*) = v(*dt*) + (2a(t+*dt*) + 5a(t) - a(t-*dt*))*dt*/6

and **adams-moulton** algorithm

r(t + *dt*) = r(t)*dt* + v(t)*dt* + 2a(t)*dt*<sup>2</sup>/3 - a(t-*dt*)*dt*<sup>2</sup>/6
v(t + *dt*) = v(*dt*) + (5a(t+*dt*) + 8a(t) - a(t-*dt*))*dt*/12

- - -
## Velocity Autocorrelation Function
An autocorrelation function is the similarity of a function with a time lagged version of itself. Details can be found [here](https://en.wikipedia.org/wiki/Autocorrelation). It's useful to us because of the Wiener–Khinchin theorem which states that the autocorrelation function has a spectral decomposition given by the power spectrum. This means by calculating the autocorrelation function and then doing a discrete FFT on it will give us an intensity spectrum. The routine **velocityAutocorrelation** works as follows
+ A BOMD calculation is performed to produce an output file containing atom displacements. Remember the BOMD uses atomic time and length measurements, as a rough guide to convert atu to femtoseconds divide by 40. The more points you can get the better, I used ammonia with 10000 steps and a step of 10 atu's as a test (overnight run!).
+ Having got the displacements from the BOMD (**BOMDfileRead**) calculate the gradients of the displacements (np.gradient) and divide by the time step (**getTimeDerivatives**). This gives us a rate of change with respect to time ie a velocity, hence this is a 'velocity' autocorrelation.
+ Now compute the autocorrelation of the derivatives (**getAutoCorrelation**). To do this we convolve the autocorrelation with a reversed copy of itself using signal.fftconvolve..
+ Define a windowing function (**getViewPort**), this can be any defined in scipy.signal.window.
+ Perform a discrete fast Fourier transform (np.fft.fft) after first convolving autocorrelation function with the window and applying zero padding (**getPadding**).
+ Get the ftrequency from np.fft.fftfreq and plot frequency v intensity (square of fft).

This is performed by

3.  **velocityAutocorrelation(options)**

    parameters - *options* is a dictionary with keys 'mode', 'atoms', 'window', 'file', 'plot'. 'mode' can be one of 's' (stretch), 'b' (bend) or 'd' (dihedral), 'atoms' is a list containing integers representing the atoms defining the stretch, bend, or dihedral. 'window' is a string and can be one of 'boxcar', 'triang', 'blackman', 'hamming', 'hann', 'bartlett', 'flattop', 'parzen', 'bohman', 'blackmanharris', 'nuttall', 'barthann', 'cosine', 'exponential', 'tukey', 'taylor', 'kaiser'\*, 'gaussian'\*, 'chebwin'\*. All except the last 3 (\*) require a string eg 'hann', the last three require an additional parameter so 'gaussian, 500', where the extra parameter is for 'gaussian' the standard deviation, for 'chebwin' it's the attenuation in dB and for kaiser it's the shape parameter &beta;. 'file' is a string specifying where the input data file resides and 'plot' and be any of 'derivative', either 'ACp' or 'ACf' and 'spectrum'. 'ACp' will plot the autocorrelation (pyplot.acorr) whereas 'ACf' plots the actual autocorrelation function. Returns wavenumber (cm<sup>-1</sup>) and intensity (arbitary units).

   Routines contained within velocityAutocorrelation are

4.  **BOMDfileRead(input)**

    parameters - *input* is a string specifying the location of the BOMD data. Returns array of displacements and list containing time step, number of steps, number of atoms and title of BOMD file.

5.  **getVibrationMode(options, geometry)**

    parameters - *options* are the options defining the run and *geometry* is the array of displacements. Returns array of either displacements (stretch) or angles (bend and dihedral). 

6.  **getTimeDerivatives(vibrations, dt)**

    parameters - *vibrations* are the linear or angular displacements and *dt* is the time step. Returns time derivatives.

7.  **getAutoCorrelation(f)**

    parameters - *f* are the velocities ie time derivatives of displacements. Returns autocorrelation function values.

8.  **getViewport(options, dataLength)**

    parameters - *options* are the options defining the run and *dataLength* is the length of the autocorrelation data. Returns pointer to a window function.

9.  **getPadding(dataLength)**

    parameters - *dataLength* is the length of the autocorrelation data. Returns next power of 2 beyond end of autocorrelation data.

10. **autocorrelationFFT(ACF, window)**

    parameters - *ACF* are the autocorrelation function values and *window* is the windowing function. Returns square of result of FFT.

This is an example of it's use
```python
mdBornOppenheimer(10, 10000, 'velocity-verlet', ['plot', 'file'])

velocityAutocorrelation({ 'mode' : 's' , 'atoms' : [0,1],  'window' : 'gaussian,500', \
                          'file' : 'md.hdf', 'plot' : ['derivative', 'ACf', 'spectrum']})  
```
Running the above for an ammonia molecule in STO-3G gives 

![ammonia N-H bond stretch](/media/nh3-vacf-bond.png)
![ammonia H-N-H bend](/media/nh3-vacf-bend.png)

We see a stretch at ~3800 cm<sup>-1</sup> and a bend at ~1400 cm<sup>-1</sup>. This compares with theoretical HF/STO-3G values from NIST given [here](https://cccbdb.nist.gov/vibs3x.asp?method=1&basis=20) of 3833 cm<sup>-1</sup> and 1412 cm<sup>-1</sup>. 

Using *getPeaks* from tdhf module gives values of 3838 and 1406 so good agreement with NIST.
