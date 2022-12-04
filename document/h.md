## Hydrogenic Wavefunctions
These routines explore various aspects of the hydrogen solution of the Schrodinger equation. They're designed to give pointers to various ways of solving the hydrogen and visualizing the solutions.

1.  **transform(mode, i, j, k)**

    parameters - *mode* is either 'cartesian->spherical' or 'spherical->cartesian', *i* is either an x-coordinate or an r-coordinate, *j* is either a y-coordinate or a &theta;-coordinate and *k* is either a z-coordinate or a &phi;-coordinate depending on the *mode*. The routine performs a coordinate transformation according to the *mode* and returns the transformed coordinates.

2.  **angularSolution(m, l, theta, phi)**

    parameters - *m* is the magnetic quantum number, *l* is the angular quantum number, *theta* is the polar coordinate and *phi* the azimuthal coordinate. Calls routine scipy.special.sph_harm and returns the spherical harmonic Y<sup>m</sup><sub>l</sub>(&theta;,&phi;).

3.  **angularSolutionPlot(ax, m, l, parameters)**

    parameters - *ax* are the matplotlib axes as defined eg from 'fig.add_subplot', *m* is the magnetic quantum number, *l* is the angular quantum number, and *parameters* is a directory of values defining the plot. The *parameters* directory is defined as follows - parameters = {'points':70,'extent':[-0.5,0.5],'color_map':'coolwarm', 'bar':'on','axes':'off','alpha':0.8} where 'points' are the number of data points in the (&theta;, &phi;)- grid, 'extent' is extent of the radial values to display, 'color_map' is the matplotlib color mapping to use, 'bar' determines whether to draw a heat bar, 'axes' controls whether to draw the coordinate axes or not and 'alpha' is the transparancy factor. Uses matplotlib.plot_surface to display the spherical harmonic in the color map requested. Optionally displays a colorbar. Does not 'show' the plot.

4.  **angularSolutionPlotSingle(m, l, parameters = {'points':70,'extent':[-0.5,0.5],'color_map':'coolwarm','bar':'on','axes':'off','alpha':0.8})**

    parameters - *m* is the magnetic quantum number, *l* is the angular quantum number and and *parameters* is a directory of values defining the plot. This routine displays a spherical harmonic defined by *m* and *l* according to the display conditions in *parameters*. An example for Y</sub>3,0</sub> is
    
    ![image](https://user-images.githubusercontent.com/73105740/136229523-1f09d93e-3834-497a-a2e5-5bbbd8db491e.png)


5.  **angularSolutionPlotFamily(l_maximum, parameters = {'points':70,'extent':[-0.5,0.5],'color_map':'coolwarm','bar':'off','axes':'off','alpha':1.0})**

    parameters - *l_maximum* is the highest value of the angular quantum number to display. *parameters* is a directory of values defining the plot. This routine will display all spherical harmonics for l=0, l=1, ... l=l_maximum where -l &le; m &ge; l. An example for l_maximum = 3 is
    
    ![image](https://user-images.githubusercontent.com/73105740/136230126-100046a3-7ee5-4435-ae2f-dc43eb5c22b3.png)


6.  **angularVerify()**

    parameters - none. This generates random values of l, m, &theta; and &phi;, and then calculates the (scipy) value from angularSolution routine comparing it with the (sympy) value from the **Ynm** function. Returns True or False.

7.  **radialSolution(n, l, r)**

    parameters - *n* is the principal quantum number, *l* is the angular quantum number (0 &ge; l &le; n) and *r* is the radial distance (in units of the Bohr radius). Calls scipy.special.genlaguerre to return the normalised radial wavefunction value at r.

8.  **radialVerify()**

    parameters - none. This generates random values of n, l and r , and then calculates the (scipy) value from radialSolution routine comparing it with the (sympy) value from the **R_nl** function. Returns True or False.

9.  **radialSolutionType(n, l, r, psi_type = 'radial distribution')**

    parameters - *n* is the principal quantum number, *l* is the angular quantum number (0 &ge; l &le; n), *r* is the radial distance (in units of the Bohr radius) and *psi_type* is the type of wavefunction to be plotted. *psi_type* can be one of 'radial distribution' (&psi;), 'probability density' (|&psi;|<sup>2</sup>) and 'probability distribution' (4&pi;r<sup>2</sup>|&psi;|<sup>2</sup>). Returns the value of the *psi_type*.

10. **radialSolutionPlot(n, l, psi_type = 'radial distribution', psi_normal = False, parameters = {'points':100, 'size':[7,5], 'extent':[20,0.2], 'equal':False})**

    parameters - *n* is a list of principal quantum numbers to plot, *l* is a list of angular quantum numbers corresponding to the list of principal quantum numbers (0 &ge; l &le; n) to plot, *r* is the radial distance (in units of the Bohr radius) and *psi_type* is the type of wavefunction to be plotted. *psi_type* can be one of 'radial distribution' (&psi;), 'probability density' (|&psi;|<sup>2</sup>) and 'probability distribution' (4&pi;r<sup>2</sup>|&psi;|<sup>2</sup>). *psi_normal* determines if the wavefunction should be normalised. *parameters* is a dictionary of values defining the plot appearance, these values are 'points' - the number of data points to calculate, 'size' - the width and height of the plot in inches, 'extent' - the horizontal and vertical ranges and 'equal' - forces a square aspect ratio. Plots the radial distribution. This is the plot for\
    radialSolutionPlot([1,2,3,2,3,3], [0,0,0,1,1,2], 'radial distribution', False, {'points':100,'size':[7,5],'extent':[20,[-0.15,0.2]],'equal':False})
    
![image](https://user-images.githubusercontent.com/73105740/136762722-b95f78b7-3f3a-4a40-9898-f2aa2119c2c1.png)

11. **wavefunction(n, l, m, r, theta, phi, grid)**

    parameters - *n* is a principal quantum number to plot, *l* is an angular quantum number, *m* is the magnetic quantum number, *r*, *theta* and *phi* are the coordinates of a point in spherical polars or cartesian (*grid*='cartesian'). Calculates the product of radial and angular solutions and returns value.

12. **wavefunctionContour(n, l, m, parameters={'points':100, 'extent':[-20, 20], 'color_map':'coolwarm','plane':'xy', 'elevation':0.0, 'contour': False})**

    parameters - *n* is a principal quantum number to plot, *l* is an angular quantum number, *m* is the magnetic quantum number and *parameters* is a dictionary of values defining the plot appearance, these values are 'points' - the number of data points to calculate, 'extent' - the horizontal and vertical ranges, 'color_map' - is the matplotlib name of the color theme to use, 'plane' - specifies which plane to view the contour in, 'elevation - is the height of the slice in 'plane' and 'contour' - specifies if the contour lines should be plotted. Shows a contour plot of the wavefunction. Examples for\
    wavefunctionContour(3, 2, 0, {'points':80, 'extent':[-30, 30], 'color_map':'gist_yarg', 'plane':'zx', 'elevation':0, 'contour': True})\
    wavefunctionContour(3, 2, 0, {'points':80, 'extent':[-30, 30], 'color_map':'gray', 'plane':'zx', 'elevation':0, 'contour': False})
![image](https://user-images.githubusercontent.com/73105740/136765639-5d801e98-995e-4d87-8760-6ba1925fe9a2.png)

13. **wavefunctionVerify()**

    parameters - None. This generates random values of *n*, *l*, *m* and *r*, *theta*, *phi* and uses these to compare wavefunction values generated by the routine wavefuction and the (sympy) routine Psi_nlm. Returns True or False.

14. **numerov(g, u_zero, du, dh)**

    parameters - *g* are the values (at each r) of the expression in second-order differential equation u(r)<sup>''</sup> = g(r)u(r) + u(r), *u_zero* is the initial value of u ie u(0), *du* is the grid step in u and *dh* is the grid step between sucessive r values. This routine returns the value of u at each value of r computed using the [Numerov algorithm](https://en.wikipedia.org/wiki/Numerov%27s_method).

### Energy Levels
In order to determine the bounded energy levels we must solve a boundary value problem. The technique we use is called the 'shooting method' because of it's use in artillery trajectories. Here we fire at the target and if our shot falls to the left of the target we aim a bit more to the right and similarly if it falls to the right we aim more to the left. For our problem we know that the values of the wavefuction at zero at the origin and at infinity. Our strategy is to (under) guess an energy and calculate the wavefunction at r=0, if the value is not zero we increase our guess and recalculate until the value at r=0 is zero. The algorithm is full is
+ Make an initial guess at the energy (𝐸<sub>1</sub>) that is less than than the absolute minimum energy (-1.1).
+ Compute the value of the wavefunction for 𝐸𝑛 and values of the quantum numbers for the wavefunction considered.
+ Increment guess 𝐸<sub>𝑛</sub> as 𝐸<sub>𝑛+1</sub> and re-compute the value of the wavefunction. We do this by using a sequence like −1.2/l<sup>2</sup>,−1.2/(1+𝛿𝐸)<sup>2</sup>,...,
+ If 𝑢<sub>0</sub>∗𝑢<sub>𝑛+1</sub> > 0 then go to previous step.
+ Here energy is in \[E<sub>n</sub>, E<sub>n+1</sub>] so apply a root finding method to find where zero occurs.

15. **shootingMethod(E, r, l)**

    parameters - *E* is an energy, *r* is an array of radial values and *l* is an angular quantum number. Returns value past zero crossing.

16. **numerovBoundEnergy(r, l, n_range, energy_range)**

    parameters - *r* is an array of radial values to evaluate the bound energy at,  *l* is an angular quantum number, *n_range* is a number specifying the maximum principal quantum number to evaluate and *energy_range* is an array of energy values at which to search for solutions. Calls *shootingMethod* routine and from scipy.optimize import brentq which is a root-finding routine. Returns the list of bound energies (*n*, *l*, energy).

17. **numerovHydrogenicEnergies(parameters = {'points':2000, 'extent':[1e-8, 100], 'n_range': 5})**

    parameters - *parameters* is a dictionary of values defining the problem, these values are 'points' - number of radial sample points, 'extent' - radial range values and 'n_range' - is a number specifying the maximum principal quantum number to evaluate. Calls numerovBoundEnergy to compute the bound energy values. Returns a sorted list of bound energy values sorted by energies. For the parameters\
    numerovHydrogenicEnergies({'points':2000, 'extent':[1e-8, 100], 'n_range': 5})\
    The result is
```
    [(1, 0, -0.9999221089559599),   (2, 1, -0.2500000156117056),   (2, 0, -0.24999019020652957), 
     (3, 1, -0.11111111678091953),  (3, 2, -0.11111111114690334),  (3, 0, -0.11110820082299608), 
     (4, 1, -0.06250000255860046),  (4, 2, -0.0625000000252697),   (4, 3, -0.0625000000008716), 
     (4, 0, -0.062498771694931364), (5, 4, -0.039999998545078866), (5, 3, -0.0399999917769602), 
     (5, 2, -0.039999976852458416), (5, 1, -0.03999995861210411),  (5, 0, -0.039999313973705056)]
```

18. **numerovEnergyVerify(boundEnergies)**

    parameters - *boundEnergies* are the energies computed by the above routine. Compares the computed *boundEnergies* with the theoretical values of -1.0/n<sup>2</sup>. Returns overall validity and a list of individual validity tests.The validity criterion is 1e-4.

19. **numerovHydrogenicRadialDensity(parameters = {'points':2000, 'extent':[1e-8, 100], 'n_range': 5, 'radius':[0,15],'occupied': [1, True, False]})**

    parameters - *parameters* is a dictionary of values defining the problem, these values are 'points' - number of radial sample points, 'extent' - radial range values, 'n_range' - is a number specifying the maximum principal quantum number to evaluate, 'radius' - is the range of radial values for xlim and 'occupied' - is a list, the first entry is the atomic number of the hydrogenic type atom to plot and the second entry is a boolean which controls whether the orbitals should be considered as occupied. The third entry is a boolean which if True will plot the overall electron density in the radial direction. Plots the hydrogenic radial wavefunctions and returns a list of plotted states - each element of list is a 4-tuple containing (*n*, *l*, *e*, fermi level). As an example\
numerovHydrogenicRadialDensity({'points':2000, 'extent':[1e-8, 100], 'n_range': 5, 'radius':[0,25],'occupied': [28, True, False]})\
    This plots the occupied hydrogenic wavefunctions for an atom with 28 electrons (Ni).
    ![image](https://user-images.githubusercontent.com/73105740/136944573-ea821ba8-52e1-4a71-bb96-d672908124b4.png)
    
Changing 'occupied'->[28, True, False] gives the total electron density

![image](https://user-images.githubusercontent.com/73105740/136949551-e9cf391e-f154-4389-957e-ce24138fcb5d.png)

The returned list is
```
[(1, 0, -0.9999221089559599,  1.0), (2, 1, -0.2500000156117056,  1.0), (2, 0, -0.24999019020652957, 1.0), 
 (3, 1, -0.11111111678091953, 1.0), (3, 2, -0.11111111114690334, 1.0), (3, 0, -0.11110820082299608, 1.0)]
```

20. **numerovRadialVerify(n, l)**

    parameters - *n* is a principal quantum number, *l* is an angular quantum number. Routine plots Numerov wavefunction profile and scipy calculated profile for comparison. No return value, displays the plotted profiles for Numerov and Sympy radial wavefunctions for (n,l) = (1,0),(2,0),(2,1),(3,0),(3,1),(3,2). The resulting plot is
    
![image](https://user-images.githubusercontent.com/73105740/137095203-360f13eb-9ac5-48c2-a3e8-072a8a0c4bda.png)

21. **finiteDifferenceRadial(parameters = {'l': 0, 'points': 4000, 'extent':[0, 40], 'levels':25, 'show':5})**

    parameters - *parameters* is a directory of values defining the problem, these values are 'l' - quantum angular number, 'points' - are the number of data sample points, 'extent' - is a list denoting the range of the radial extent in nm, 'levels' - the number of eigenvalues to calculate and 'show' - the number of states to plot. This routine uses a three-point stencil finite difference method. The discrete radial grid is generated as an equidistant mesh between 'extent' in 'points' steps. The 'extent' should be sufficient to include contributing parts of the wavefunction. The default [0, 40] is sufficient for first 5 energy levels to an accuracy of two decimal places. Increasing 'points' can improve the accuracy up to a point. The 'show' levels are plotted and the legend shows the computed energy level in eV, included in [] after the computed value is the theoretical value of 1/(n+l)<sup>2</sup> Ry. The plot is r and probability density. The default parameters produce the following plot
    
![image](https://user-images.githubusercontent.com/73105740/137509411-06d0765b-327d-4843-884b-82b4d0b940f7.png)

22. **hydrogenSpectralLines(type)**

    parameter - type is list of line types to be calculated, can be any or all of 'lyman', 'balmer' , 'paschen'. Routine computes the spectral lines for the specified series and returns a dictionary with the line names as keys. Each key entry is a list [energy (eV), wavelength (nm), transition as 'n->m']. The returned list for 'balmer' as an example is 
```
    [[10.203347934756827, 121.5132237896726, '2->1'], [12.092934418025163, 102.52612460644374, '3->1'], 
    [12.754299650697403, 97.20970456674237, '4->1'], [13.060419709416564, 94.93122951524093, '5->1'], 
    [13.226707972305539, 93.73773909547381, '6->1']]
```
23. **angularSolutionPlotProjection(m, l,  parameters= {'points':70,'extent':[-0.5,0.5], 'alpha':0.3, 'color_map':'coolwarm', 'axes':'off','levels':20})**

    parameters - *m* is the magnetic quantum number, *l* is the angular quantum number and and *parameters* is a directory of values defining the plot. This routine displays a spherical harmonic defined by *m* and *l* according to the display conditions in *parameters* including contour projections on the plane surfaces. An example for Y</sub>3,1</sub> is
![Screenshot from 2022-08-28 10-35-26](https://user-images.githubusercontent.com/73105740/187163257-ee8e453c-badc-4239-92b6-6b76edf1ee09.png)

24. **monteCarloHydrogen(n, l, m, view='xy', plot=True)**

    parameters - *n* is the principal quantum number, *l* is the angular quantum number, *m* is the magnetic quantum number, *view* is the plane to display, *plot* is a boolean controlling the auxilliary plots. This routine uses Monte-Carlo techniques to plot a hydrogen orbital. If *plot* is true auxilliary plots of radial wavefunction and density distribution are plotted. The results for n,l,m = 6,3,1 are 
    ![image](https://user-images.githubusercontent.com/73105740/187165179-f5c98308-e956-4ac2-b46b-a8f72544e30c.png)
and 
 ![image](https://user-images.githubusercontent.com/73105740/187165497-a3d635ae-7c39-4270-8946-b4511237a5c4.png)


These routines can be run using the default parameters supplied from the command line as 'python h.py -\<key>'. The available keys are\
    'rp' - radialSolutionPlot (10)\
    'rv' - radialVerify (8)\
    'as' - angularSolutionPlotSingle (4)\
    'af' - angularSolutionPlotFamily (5)\
    'av' - angularVerify (6)\
    'wc' - wavefunctionContour (12)\
    'wv' - wavefunctionVerify (13)\
    'ne' - numerovHydrogenicEnergies (17)\
    'Nev'- numerovEnergyVerify (18)\
    'NRv'- numerovRadialVerify (20)\
    'Nw' - numerovHydrogenicRadialDensity (19)\
    'fr' - finiteDifferenceRadial (21)\
    'se' - hydrogenSpectralLines (22)\
    'ap' - angularSolutionPlotProjection (23)\
    'mc' - Monte-Carlo wavefunction
