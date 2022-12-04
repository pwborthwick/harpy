### Coupled-Cluster Triples and Variations

The theoretical background to this can be found in *documents/CCSDT-n.ipynb*. This covers CCSDT-1a, CCSDT-1b, CCSDT-n for n=2,3,4. The code can be accessed through a *post* command in the harpy input as eg *post={-1a}*. Valid identifiers are -1a, -1b, -2, -3, -4, -T (full CCSDT) and -t (CCSD(T). The code will output to the console rather than the HTML output file. The source file contains a DIIS class specially modified for triples, the routine *cluster_triples_n_diagrams* which contains the extra cluster diagrams needed for the triples variations and *coupledClusterTriplesVariations* which is the main routine. 

The process is as follows: from harpy take the 2-electron repulsion integrals, Fock matrix, mo coefficients (C) and the molecular orbital energies. We convert the eri and Fock to spin versions and use these to pass to the COGUS ccsd(t) code. From the COGUS code we get back (for each cycle) the $T_1$ and $T_2$ amplitudes. These amplitudes are then on each cycle augmented with the extra diagrams needed for the method being computed. For the straight triples the plain COGUS code is used unmodified. The results have been tested by using CC_ADE to explicitly construct the exponential terms involved in the ansatz for each method and also against results given in 'Coupled-Cluster Studies in Computational Chemistry' a Master of Science thesis by Ole T. B. Norli. The results from this source have themselves been verified against J. Noga & R.J. Bartlett Chemical Physics Letters Vol. 134, issue 2, 20 Feb 1987. The molecule used is $H_2 O$ in a DZ basis the geometry is in Bohr units

| atom |  x  |   y  |  z  |
|------|-----|------|-----|
| O    | 0.000000  | 0.000000 | -0.009000 |
| H    |  1.515263 | 0.000000 | -1.058898 |
| H    |  -1.515263 | 0.000000 | -1.058898 |

The results for references are calculated to 1-e7 and for harpy 1e-8, units are Hartrees

|  method |    reference   |   harpy  |
|---------|----------------|----------|
| CCSDT-1a| -0.147577 | -0.1475768  |
| CCSDT-1b| -0.147580 | -0.1475803  |
| CCSDT-2 | -0.147459 | -0.1474589  |
| CCSDT-3 | -0.147450 | -0.1474501  |
| CCSDT-4 | -0.147613 | -0.1476129
| CCSDT   |  0.147594 | -0.1475938  |
| CCSD    |           | -0.1462381  |
|      (T)|           | -0.0012159  |

