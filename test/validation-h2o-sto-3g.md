## Validation test on water using STO-3G basis

The program validation-h2o-sto-3g.py executes an evaluation of most of the codes contained within the **harpy** system. It can be run with parameters 'Q' (quick) or 'F' (full). The 'Q' option will run in about 8 minutes while the 'F' option is best run overnight as the eom routines take a lot of time. The output shows
+ elapsed time of the validation-h2o-sto-3g run
+ time of the particular process being run
+ description of the process being tested
+ -if process produces multiples values the particular one displayed is given in  \[&nbsp;]-
+ value returned by process
+ boolean, True is passed test, False is failed test
+ precision of test eg 6 means test passes if difference between computed and reference values is better than 1e-6.
+ the code of comparison source. **P4**-Psi4, **PN**-psi4numpy, **GA**-Gaussian, **CW**-Crawford Projects, **HA**-harpy (internal check), **MM**-McMurchie-Davidson, **PS**-pyscf, **AD**-ADCC

**sto-3g should be the first entry in project.hpf**
Most tests are done to 1e-6 precision. The molecule geometry is defined within the routine and does not read the project.hpf file (but some called routines do). The expected output for a full run is
```
=====================================================================================================
                                         Water Test in STO-3G                        mode[ F ]
-----------------------------------------------------------------------------------------------------
  Elapsed       Step                            Process                         Value    Pass   Check
-----------------------------------------------------------------------------------------------------
00:00:00.2   00:00:00.2   SCF energy                                        -74.94207995 True   6  CW 
00:00:00.2   00:00:00.0   Mulliken oxygen charge                             -0.25314612 True   6  CW 
00:00:00.2   00:00:00.0   Dipole Moment resultant                             0.60352134 True   6  CW 
00:00:00.6   00:00:00.4   Quadrupole Moment amplitude                         1.33769187 True   6  HA 
00:00:00.7   00:00:00.1   Polarizabilities                    [isotropic]     3.68471970 True   6  HA 
00:00:00.9   00:00:00.2   Hyperpolarizabilities                     [0,1]    -9.34242926 True   6  HA 
=====================================================================================================
00:00:01.0   00:00:00.0   Moller-Plesset (2)                                 -0.04914964 True   6  CW 
00:00:01.0   00:00:00.0   Spin-Component Scaled MP (2)                       -0.05628751 True   6  PN 
00:00:49.7   00:00:48.8   Orbital-Optimised MP (2)                           -0.04939137 True   6  HA 
00:00:49.8   00:00:00.1   Laplace Transform MP (2)                           -0.04914581 True   6  PN 
=====================================================================================================
00:00:50.6   00:00:00.8   CIS                                         [0]     0.28725552 True   6  CW 
00:00:51.4   00:00:00.8   Random Phase Approximation                  [0]     0.28516373 True   6  CW 
00:00:51.4   00:00:00.0   Spin-Adapted CI singles                     [0]     0.35646178 True   6  CW 
00:00:51.5   00:00:00.0   Spin-Adapted CI triples                     [0]     0.28725552 True   6  CW 
=====================================================================================================
00:01:46.0   00:00:54.5   Coupled-Cluster Singles and Doubles                -0.07068009 True   6  CW 
00:01:57.9   00:00:11.9   CCSD Perturbative Triples                          -0.00009988 True   6  CW 
00:21:44.3   00:19:46.5   Coupled-Cluster Doubles                            -0.07015049 True   6  PN 
00:22:12.3   00:00:28.0   Linear Coupled-Cluster Doubles                     -0.07192916 True   6  P4 
00:22:25.0   00:00:12.7   Linear Coupled-Cluster Singles and Doubles         -0.07257659 True   6  HA 
00:24:45.3   00:02:20.3   Coupled-Cluster (2)                                -0.04939911 True   6  P4 
00:25:23.4   00:00:38.1   Coupled-Cluster Singles and Doubles-Λ              -0.06888820 True   6  P4 
=====================================================================================================
00:25:23.5   00:00:00.0   Electron Propagator (2)                     [0]   -10.34032984 True   4  HA 
00:25:24.5   00:00:01.0   Spin-Adapted Electron Propagator (2)        [0]  -541.75575877 True   4  HA 
00:25:46.8   00:00:22.3   Spin-Adapted Electron Propagator (3)        [0]  -543.63064680 True   4  PN 
00:25:47.7   00:00:00.9   Koopman Green Function Correction           [0]   -12.66439467 True   4  HA 
=====================================================================================================
00:25:56.7   00:00:09.0   Full Configuration Interaction              [0]    -0.07090027 True   6  P4 
00:25:57.6   00:00:00.9   Determinant CIS                             [0]     0.28725552 True   6  MM 
00:25:59.4   00:00:01.8   Determinant CISD                            [0]    -0.06914307 True   6  MM 
=====================================================================================================
00:26:11.8   00:00:03.2   Forces (ocypete - cython)                 [0,1]     0.09744138 True   6  MM 
00:34:54.8   00:08:42.9   Forces (native  - python)                 [0,1]     0.09744138 True   6  MM 
=====================================================================================================
00:35:58.8   00:01:04.0   Restrained Electrostatic Potential          [0]    -0.47703059 True   6  PN 
=====================================================================================================
07:31:49.1   06:55:50.3   EOM-CCSD                                    [0]     7.49020351 True   4  GA 
10:01:08.5   02:29:19.5   EOM-MBPT (2)                                [0]     7.12218877 True   4  GA 
=====================================================================================================
10:01:10.0   00:00:01.5   TDHF transition properties         [excitation]     0.35646178 True   6  P4 
10:01:10.0   00:00:00.0   TDHF OPA spectrum                           [x]    57.87004982 True   6  P4 
=====================================================================================================
10:01:10.3   00:00:00.3   cogus - ccd                                        -0.07015049 True   8  PN 
10:01:11.1   00:00:00.8   cogus - ccsd                                       -0.07068009 True   8  P4 
10:01:29.6   00:00:18.5   cogus - ccsdt                                      -0.07081281 True   8  P4 
10:01:31.5   00:00:01.9   cogus - ccsd(t)                                    -0.00009988 True   8  P4 
10:01:31.8   00:00:00.3   cogus - cc2                                        -0.04939914 True   8  P4 
10:01:39.4   00:00:07.5   cogus - cc3                                        -0.07077803 True   8  P4 
10:01:39.5   00:00:00.1   cogus - lccd                                       -0.07192916 True   8  HA 
10:01:39.7   00:00:00.2   cogus - lccsd                                      -0.07257659 True   8  HA 
10:01:39.9   00:00:02.8   cogus - Λ                                          -0.07068009 True   8  HA =====================================================================================================
10:01:49.9   00:00:10.1   Many-Bodied Perturbation Theory             [2]    -0.04914964 True   6  P4 
10:01:49.9   00:00:10.1   Many-Bodied Perturbation Theory             [3]    -0.01418782 True   6  P4 
10:01:49.9   00:00:10.1   Many-Bodied Perturbation Theory             [4]    -0.00469027 True   6  HA 
=====================================================================================================
10:01:50.1   00:00:00.2   uhf energy                                        -74.94207995 True   6  PS 
10:01:50.5   00:00:00.5   uhf energy cation                                 -74.48785027 True   6  PS 

```
The fast EOM has been added and will produce a line 
```
00:10:09.0   00:03:33.2   EOM-CCSD  (fast)                            [0]     7.49014856 True   4  GA 
```
