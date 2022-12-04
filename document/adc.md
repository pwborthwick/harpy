### Algebraic Diagrammatic construction (2)

This is a class illustrating ADC at the second-order level. Included is a Davidson solver which is essentially the one written by Ollie Backhouse used in the ADC folder of psi4Numpy. The EE equations are taken from ['Development and Implementation of Theoretical Methods for the Description of Electronically Core-Excited States' by Jan Wenzel](https://archiv.ub.uni-heidelberg.de/volltextserver/20514/1/Jan_Wenzel_Thesis.pdf) and [Development and Application of Hermitian Methods for Molecular Properties and Excited Electronic States by Manuel Hodecker](https://core.ac.uk/download/pdf/322693292.pdf). The IP and EA are discussed in [Reduced-cost second-order algebraic-diagrammatic construction method for excitation energies and transition moments, D Mester,P R. Nagy and M Kállay](https://core.ac.uk/download/pdf/154885128.pdf). 

This is an example of the code to run an ADC computation for electron excitation and the associated output for water in sto-3g. The solve parameter is [multiply roots by this factor for number of guess vectors, number of sub-space vectors per root]. 

    import rhf
    molAtom, molBasis, molData = rhf.mol([])
    e_scf = rhf.scf(molAtom, molBasis, molData, [])

    from basis import electronCount

    charge, electrons = [molData['charge'], electronCount(molAtom, molData['charge'])]

    adc = ADC('ee', rhf, electrons, roots=6, solve=[2, 10])

    adc_a = adc_analyse(adc)
    adc_a.summary()
    adc_a.detail(0)
    adc_a.detail(3)
    adc_a.detail(4)

and the output with the print threshold set at >0.1. This is a spin-treatment and we print, for the summary, only the first of a multiplicity of roots. However the details can ask for all roots so as we have a multiplicity of 3,1,2(of 3) to get the unique details we must ask for roots 0,3,4.

```
moller-plesset(2) energy summary
rhf total energy         -74.9420799540
mp2 correlation energy    -0.0491496367
mp2 corrected energy     -74.9912295907

  n   m         energy          1h-1p:-----------------------> 2h-2p:---->
              Hr       eV       norm       i  a                 norm 
--------------------------------------------------------------------------
  1   3   0.285509  7.769103    0.9847   [ 5, 6 ] 0.7017        0.0153
  2   1   0.343916  9.358436    0.9784   [ 5, 6 ] 0.6994        0.0216
  3   3   0.363627  9.894796    0.9949   [ 4, 6 ] 0.6683        0.0051

-->state (root)  0
   polarization type = EE        energy = 0.285509  eV
   multiplicity =  3
block  type           excitation
--------------------------------------
1h-1p  α->α      [ 5, 6 ]      0.7017

-->state (root)  3
   polarization type = EE        energy = 0.343916  eV
   multiplicity =  1
block  type           excitation
--------------------------------------
1h-1p  α->α      [ 5, 6 ]      0.6994

-->state (root)  4
   polarization type = EE        energy = 0.363627  eV
   multiplicity =  3
block  type           excitation
--------------------------------------
1h-1p  α->α      [ 4, 6 ]      0.6683
1h-1p  α->α      [ 3, 7 ]      0.2170
```

We can compare with [ADCC](https://adc-connect.org/v0.15.13/index.html) output
```
+--------------------------------------------------------------+
| adc2                                        any ,  converged |
+--------------------------------------------------------------+
|  #        excitation energy     osc str    |v1|^2    |v2|^2  |
|          (au)           (eV)                                 |
|  0     0.2855092      7.769102   0.0000    0.9847   0.01528  |
|  1     0.3439185        9.3585   0.0021    0.9784   0.02158  |
|  2     0.3636277      9.894815   0.0000    0.9948  0.005164  |
+--------------------------------------------------------------+
+---------------------------------------------------+
| State   0 ,     0.2855092 au,      7.769102 eV    |
+---------------------------------------------------+
| HOMO          -> LUMO           b ->b      +0.702 |
| HOMO          -> LUMO           a ->a      -0.702 |

+---------------------------------------------------+
| State   1 ,     0.3439185 au,        9.3585 eV    |
+---------------------------------------------------+
| HOMO          -> LUMO           a ->a      -0.699 |
| HOMO          -> LUMO           b ->b      -0.699 |

+---------------------------------------------------+
| State   2 ,     0.3636277 au,      9.894815 eV    |
+---------------------------------------------------+
| HOMO-1        -> LUMO           a ->a      -0.668 |
| HOMO-1        -> LUMO           b ->b      +0.668 |
| HOMO-2        -> LUMO+1         b ->b      -0.217 |
| HOMO-2        -> LUMO+1         a ->a      +0.217 |

```

We can compare the IP and EA results with pySCF, for water sto-3g IP we get 

```
moller-plesset(2) energy summary
rhf total energy         -74.9420799540
mp2 correlation energy    -0.0491496367
mp2 corrected energy     -74.9912295907

  n   m         energy          1h:-----------------------> 2h-1p:------------------------>
              Hr       eV       norm       i                   norm       i  j  a         
-------------------------------------------------------------------------------------------
  1   2   0.272050  7.402858    0.9125   [ 5 ] 0.9552        0.0875   [ 3, 5, 7 ] 0.1541
  2   2   0.377743 10.278914    0.9415   [ 4 ] 0.9691        0.0585   [ 4, 4, 6 ] 0.1338
  3   2   0.536346 14.594707    0.9719   [ 3 ] 0.9858        0.0281

-->state (root)  0
   polarization type = IP        energy = 0.272050  eV
   multiplicity =  2
block  type           excitation
--------------------------------------
1h     α->α          [ 5 ]      0.9552
2h-1p  αα->ββ     [ 3, 5, 7 ]   0.1359
2h-1p  αα->ββ     [ 4, 5, 6 ]   0.1322
2h-1p  αβ->αβ     [ 3, 5, 7 ]   0.1541
2h-1p  αβ->αβ     [ 4, 5, 6 ]   0.1369

```
from pySCF 
```
adc(2) root 0  |  Energy (Eh) =   0.2720500188  |  Energy (eV) =   7.40286188  |  Spec factors = 1.82390838  |  conv = True
adc(2) root 1  |  Energy (Eh) =   0.3777431275  |  Energy (eV) =  10.27891934  |  Spec factors = 1.86951116  |  conv = True
adc(2) root 2  |  Energy (Eh) =   0.5363456124  |  Energy (eV) =  14.59471500  |  Spec factors = 1.92335906  |  conv = True

adc(2) | root 0 | norm(1h)  = 0.9125 | norm(2h1p) = 0.0875 

1h block: 
     i     U(i)
------------------
     5    0.9552

2h1p(alpha|beta|alpha) block: 
     i     j     a     U(i,j,a)
-------------------------------
     3     5     7      0.1541
     4     5     6      0.1369

2h1p(beta|beta|beta) block: 
     i     j     a     U(i,j,a)
-------------------------------
     3     5     7      0.1359
     5     3     7     -0.1359
     5     4     6     -0.1322
     4     5     6      0.1322
```

There is a hf_reference class which is just a holder for rhf, molAtom and molBasis objects to have the ground state quantities all in one place. The mp2_property class has functions to calculate the mp2 level density correction, both relaxed and unrelaxed and to calculate the dipole moment in ground state, unrelaxed and relaxed levels.
```
mp_prop = mp2_properties(hf_reference(rhf, molAtom, molBasis), adc)
dipoles = mp_prop.dipoles()

print()
caption = ['hf  reference dipole', 'mp2 unrelaxed dipole', 'mp2 relaxed dipole']
for i, mu in enumerate(['hf', 'mpu', 'mpr']):
    x , y, z = dipoles[mu] * getConstant('au->debye')
    print('{:<20s}    x= {:<8.4f}   y= {:<8.4f}   z= {:<8.4f}  D'.format(caption[i], x, y, z))
```
will return for H<sub>2</sub>O in 3-21g basis
```
hf  reference dipole    x= 0.0000     y= 2.4137     z= -0.0000   D
mp2 unrelaxed dipole    x= 0.0000     y= 2.3782     z= -0.0000   D
mp2 relaxed dipole      x= 0.0000     y= 2.2881     z= -0.0000   D
```
The adc_analyse class has a method transition_density which can be used to compute transition properties (for electron excitation). This is an example of it's use to get oscillator strengths, further examples of calculating transition properties are to be found in the adc(1) code
```
    adc = ADC('ee', rhf, electrons, roots=20, solve=[2, 10])
    adc_a = adc_analyse(adc)

    root = 14
    dm = adc_a.transition_density(root, mp_prop.mp2_density(type='relaxed'))

    from post import dipoleComponent
    mu_component = [dipoleComponent(mp_prop.hf.atoms, mp_prop.hf.basis, x, 'origin') for x in ['x','y','z']]
    charges      = [a.number for a in mp_prop.hf.atoms]
    centers      = [a.center for a in mp_prop.hf.atoms]

    #ao->mo
    mu_mo = np.kron(np.einsum('rp,xrs,sq->xpq', mp_prop.hf.rhf.C, mu_component, mp_prop.hf.rhf.C, optimize=True), np.eye(2))

    #get transition dipole moment
    tdm = np.einsum('ia,xia->x', dm, mu_mo, optimize=True)

    #oscillator strength
    os = (2/3) * adc.eig_energy[root] * np.einsum('x,x->', tdm, tdm, optimize=True)
    print('electric length gauge oscillator strength = {:<8.6f}  for excitation {:<8.6f} eV'.
                             format(os , adc.eig_energy[root]*getConstant('hartree->eV')))

```
which returns
```
electric length gauge oscillator strength = 0.062968  for excitation 13.719527 eV

```
The ADC code is in a sepaerate directory 'adc', to run use 'python adc/adc.py' from 'harpy/source/' directory.

### Algebraic Diagrammatic Construction (1)

The module adc(1) contains the class first_order_adc. This is an implementation of ADC at the first order level which is formally equivalent to CIS. The code includes examples of calculating transition properties in both CIS and ADC formalisms. Transition dipoles are computed for electric in length and velcity gauges and for magnetic in length gauge. Oscillator strengths and cross-sections are also computed. Typical usage would be

+ do general ADC calculation with 0 roots to initialise the class ie
```
adc = ADC('ee', rhf, electrons, roots=0, solve=[2, 10])
adc.roots = 20
```
+ create a ground state reference class instance
```
hf = hf_reference(rhf, molAtom, molBasis)
```
+ run the ADC(1) class. Note you use either the ADC moludes Davidson solver or a Python direct solver (solver='eigh')
```
adc_1 = first_order_adc(adc, hf, solver='davidson')
```
The results are written to a cache property which can either be accessed as \'adc_1.cache[\'kw\']\' or by \'energy, v, converged = adc_1.get()\'. The cache has keywords\
+ 'e' for energies (au).
+ 'u' for eigenvectors returned as an [nocc\*nvir, roots] array.
+ 'c' is a boolean flag indicating the converged status of the diagonalisation (True = converged OK)

You can further use
```
adc_1.get_transition_properties()
```
This will give the follow cache options
+ 'dipole:electric:length:CIS'
+ 'oscillator:electric:length:CIS'
+ 'dipole:electric:length:ADC'
+ 'oscillator:electric:length:ADC'
+ 'dipole:electric:velocity:CIS'
+ 'oscillator:electric:velocity:CIS'
+ 'dipole:electric:velocity:ADC'
+ 'oscillator:electric:velocity:ADC'
+ 'dipole:magnetic:length:CIS'
+ 'dipole:magnetic:length:ADC'
+ 'cross-section:CIS'
+ 'cross-section:ADC'

where appropriate all units are atomic. Examples of obtaining output are given in the main section of the module, as an example for water in STO-3G using the above code we get
```
ADC(1) Excited States
-----------------------------------------------------------------------------------------------
root                energy                      excitation             osc.  (CIS)       (ADC)
-----------------------------------------------------------------------------------------------
  1   [3]    0.287256    7.816620         0.7071      HOMO -> LUMO    
  2   [3]    0.344425    9.372282         0.6283    HOMO-1 -> LUMO    
  3   [1]    0.356462    9.699819         0.7071      HOMO -> LUMO           0.0023     0.0022   
  4   [3]    0.365989    9.959068         0.7071      HOMO -> LUMO+1  
  5   [3]    0.394514   10.735267         0.4179    HOMO-1 -> LUMO+1  
  6   [1]    0.416072   11.321889         0.7071      HOMO -> LUMO+1  
  7   [1]    0.505628   13.758846         0.6202    HOMO-1 -> LUMO           0.0649     0.0485   
  8   [3]    0.514290   13.994543         0.4093    HOMO-2 -> LUMO    
  9   [1]    0.555192   15.107540         0.4911    HOMO-1 -> LUMO+1         0.0155     0.0144 
```
and for a designated state
```
Excited state  7      root number  14       energy  0.505628
------------------------------------ ------------------------------------------------------------------
   type       gauge                    CIS                                       ADC             
                              dipole           oscillator             dipole           oscillator
------------------------------------------------------------------------------------------------------
 electric    length  [ 0.0000  0.4389  0.0000]   0.0649      [ 0.0000  0.3795  0.0000]   0.0485
 electric   velocity [ 0.0000  0.2735 -0.0000]   0.0252      [-0.0000 -0.3015  0.0000]   0.0306
 magnetic    length  [ 0.0000  0.0000 -0.0000]               [-0.0000 -0.0000  0.0000] 

Cross-sections        CIS        0.009352   ADC    0.006991
```
We can plot the oscillator strengths (or cross-sections or dipole norms) either as bars or with a broadening, an example is given in the module which will produce eg

![image](https://user-images.githubusercontent.com/73105740/164206868-c4e9af64-6dd5-46a8-b626-3f90490c71c3.png)
