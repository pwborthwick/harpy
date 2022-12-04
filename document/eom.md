# EOM-CCSD
The motivation for this is Josh Goings program *pyqchem* on Github. The following draws from that program and documents Josh kindly supplied.
The paper [Simplified methods for equation-of-motion coupled-cluster excited state calculations - Steven R. Gwaltney, Marcel Nooijen, Rodney J. Bartlett](https://notendur.hi.is/agust/rannsoknir/papers/cpl248-189-96.pdf) gives the following equations for the partitioning of the EOM-CCSD hamiltonian. The Hamiltonian is not Hermitian and these are equations for right hand eigenvectors.

**H<sub>SS</sub>**\
F<sub>ae</sub>C<sup>e</sup><sub>i</sub> - F<sub>mi</sub>C<sup>a</sup><sub>m</sub> + W<sub>amie</sub>C<sup>e</sup><sub>m</sub> , *which re-indexed {e->c,m->k} is*\
F<sub>ac</sub>C<sup>c</sup><sub>i</sub> - F<sub>ki</sub>C<sup>a</sup><sub>k</sub> + W<sub>akic</sub>C<sup>c</sup><sub>k</sub> &nbsp;&nbsp;, *uniform vectors*\
F<sub>ac</sub>C<sup>c</sup><sub>k</sub>&delta;<sub>ik</sub> - F<sub>ki</sub>C<sup>c</sup><sub>k</sub>&delta;<sub>ac</sub>+ W<sub>akic</sub>C<sup>c</sup><sub>k</sub>
+ H<sub>SS</sub><sup>**a**</sup><sub>**i**</sub> = \[F<sub>**a**c</sub> &delta;<sub>**i**k</sub> -  F<sub>k**i**</sub> &delta;<sub>**a**c</sub> + W<sub>**a**k**i**c</sub>] C<sup>c</sup><sub>k</sub>

**H<sub>SD</sub>**\
F<sub>me</sub>C<sup>ae</sup><sub>im</sub> + 0.5W<sub>amef</sub>C<sup>ef</sup><sub>im</sub> - 0.5W<sub>mnie</sub>C<sup>ae</sup><sub>mn</sub> , *which re-indexed {m->l,e->d,f->c,n->k} is*\
F<sub>ld</sub>C<sup>ad</sup><sub>il</sub> + 0.5W<sub>aldc</sub>C<sup>dc</sup><sub>il</sub> - 0.5W<sub>lkid</sub>C<sup>ad</sup><sub>lk</sub> &nbsp;&nbsp;, uniform vectors\
F<sub>ld</sub>C<sup>cd</sup><sub>kl</sub>&delta;<sub>ac</sub>&delta;<sub>ik</sub> + 0.5W<sub>alcd</sub>C<sup>cd</sup><sub>kl</sub>&delta;<sub>ik</sub> **<sup>1</sup>**- 0.5W<sub>klid</sub>C<sup>cd</sup><sub>kl</sub>&delta;<sub>ac</sub> **<sup>2</sup>**\
**<sup>1</sup>** *repeated indices c and d interchanged.*\
**<sup>2</sup>** *repeated indices k and l interchanged.*
+ H<sub>SD</sub><sup>**a**</sup><sub>**i**</sub> = \[F<sub>ld</sub> &delta;<sub>**i**k</sub>&delta;<sub>**a**c</sub> + 0.5  W<sub>**a**lcd</sub> &delta;<sub>**i**k</sub> - 0.5 W<sub>lk**i**d</sub> &delta;<sub>**a**c</sub>] C<sup>cd</sup><sub>kl</sub>

**H<sub>DS</sub><sup>**\
 P(ab) W<sub>maij</sub>C<sup>b</sup><sub>m</sub> + P(ij) W<sub>abej</sub>C<sup>e</sup><sub>i</sub> + P(ab) W<sub>bmfe</sub>t<sup>af</sup><sub> ij</sub>C<sup>e</sup><sub>m</sub> -  P(ij) W<sub>nmje</sub>t<sup>ab</sup><sub>in</sub>C<sup>e</sup><sub>m</sub> , *which re-indexed {m->k,e->c,f->e,n->m}*\
 P(ab) W<sub>kaij</sub>C<sup>b</sup><sub>k</sub> + P(ij) W<sub>abcj</sub>C<sup>c</sup><sub>i</sub> + P(ab) W<sub>bkec</sub>t<sup>ae</sup><sub>ij</sub>C<sup>c</sup><sub>k</sub> -  P(ij) W<sub>mkjc</sub>t<sup>ab</sup><sub>im</sub>C<sup>c</sup><sub>k</sub> &nbsp;&nbsp;, *uniform vectors*\
P(ab) W<sub>kaij</sub>C<sup>c</sup><sub>k</sub> &delta;<sub>bc</sub> + P(ij) W<sub>abcj</sub>C<sup>c</sup><sub>k</sub>&delta;<sub>ik</sub> + P(ab) W<sub>bkec</sub>t<sup>ae</sup><sub>ij</sub>C<sup>c</sup><sub>k</sub> -  P(ij) W<sub>mkjc</sub>t<sup>ab</sup><sub>im</sub>C<sup>c</sup><sub>k</sub> 
+ H<sub>DS</sub><sup>**ab**</sup><sub>**ij**</sub> = \[*P*(ab) W<sub>k**aij**</sub> &delta;<sub>**b**c</sub> + *P*(ij) W<sub>**ab**c**j**</sub> &delta;<sub>**i**k</sub> + *P*(ab) W<sub>**b**kec</sub> t<sup>**a**e</sup><sub>**ij**</sub> - *P*(ij) W<sub>mk**j**c</sub> t<sup>**ab**</sup><sub>**i**m]</sub>] C<sup>c</sup><sub>k</sub> 

**H<sub>DD</sub>**\
P(ab) F<sub>be</sub>C<sup>ae</sup><sub>ij</sub> - P(ij) F<sub>mj</sub>C<sup>ab</sup><sub>im</sub> + 0.5W<sub>abef</sub>C<sup>ef</sup><sub> ij</sub> + 0.5W<sub>mnij</sub>C<sup>ab</sup><sub>mn</sub> + P(ab)P(ij) W<sub>bmje</sub>t<sup>ae</sup><sub>im</sub>C<sup>ae</sup><sub>im</sub> - 0.5P(ab) W<sub>nmfe</sub>t<sup>fb</sup><sub>ij</sub>C<sup>ea</sup><sub>mn</sub> - 0.5P(ij) W<sub>nmfe</sub>t<sup>ab</sup><sub>jn</sub>C<sup>fe</sup><sub>im</sub>, *which re-indexed {e->c,m->k,n->m/l,f->e/d}*\
P(ab) F<sub>bc</sub>C<sup>ac</sup><sub>ij</sub> - P(ij) F<sub>kj</sub>C<sup>ab</sup><sub>ik</sub> + 0.5W<sub>abcd</sub>C<sup>cd</sup><sub>ij</sub> + 0.5W<sub>klij</sub>C<sup>ab</sup><sub>kl</sub> + P(ab)P(ij) W<sub>bkjc</sub>C<sup>ac</sup><sub>ik</sub> - 0.5P(ab) W<sub>lkec</sub>t<sup>eb</sup><sub>ij</sub>C<sup>ca</sup><sub>kl</sub> + 0.5P(ij) W<sub>mkdc</sub>t<sup>ab</sup><sub>jm</sub>C<sup>dc</sup><sub>ik</sub> &nbsp;&nbsp;, *uniform vectors*\
P(ab) F<sub>bc</sub>C<sup>cd</sup><sub>kl</sub>&delta;<sub>ad</sub>&delta;<sub>il</sub>&delta;<sub>jk</sub>- P(ij) F<sub>kj</sub>C<sup>cd</sup><sub>kl</sub>&delta;<sub>bc</sub>&delta;<sub>ad</sub>&delta;<sub>il</sub> + 0.5W<sub>abcd</sub>C<sup>cd</sup><sub>kl</sub>&delta;<sub>ik</sub>&delta;<sub>jl</sub>+ 0.5W<sub>klij</sub>C<sup>cd</sup><sub>kl</sub>&delta;<sub>ac</sub>&delta;<sub>bd</sub> + P(ab)P(ij) W<sub>bkjc</sub>C<sup>cd</sup><sub>kl</sub>&delta;<sub>ad</sub>&delta;<sub>il</sub> - 0.5P(ab) W<sub>lkec</sub>t<sup>eb</sup><sub>ij</sub>C<sup>cd</sup><sub>kl</sub>&delta;<sub>ad</sub> + 0.5P(ij) W<sub>mkdc</sub>t<sup>ab</sup><sub>jl</sub>C<sup>cd</sup><sub>kl</sub>&delta;<sub>il</sub>
+ H<sub>DD</sub><sup>**ab**</sup><sub>**ij**</sub> = \[*P*(ab) F<sub>**b**c</sub> &delta;<sub>**j**k</sub> &delta;<sub>**i**l</sub> &delta;<sub>**a**d</sub> - F<sub>k**j**</sub></sub> &delta;<sub>**a**d</sub> &delta;<sub>**i**l</sub> &delta;<sub>**b**c</sub> + 0.5W<sub>**ab**cd</sub> &delta;<sub>**i**k</sub> &delta;<sub>**j**l</sub> + 0.5W<sub>kl**ij**</sub> &delta;<sub>**a**c</sub > &delta;<sub>**b**d</sub> + *P*(ij)*P*(ab) W<sub>**a**k**i**c</sub> &delta;<sub>**j**l</sub> &delta;<sub>**b**d</sub> - 0.5W<sub>lkec</sub>t<sup>e**b**</sup><sub>**ij**</sub> &delta;<sub>**a**d</sub> + 0.5W<sub>mkdc</sub>t<sup>**ab**</sup><sub>**j**m</sub> &delta;<sub>**i**l</sub>] C<sup>cd</sup><sub>kl</sub>

*(Einstein summation implied on repeated indices)*
- - -
Note g<sub>abcd</sub> = <ab||cd> = -<ba||cd> = -<ab||dc> = <ba||dc> \
&tau;<sup>ab</sup><sub>ij</sub> = t<sup>ab</sup><sub>ij</sub> + 2 t<sup>a</sup><sub>i</sub><sup>b</sup><sub>j</sub> \
&tau;<sup>ab</sup><sub>ij</sub> = -&tau;<sup>ba</sup><sub>ij</sub> = -&tau;<sup>ab</sup><sub>ji</sub> \
P(ij) = f(ij) - f(ji)\
For EOM-CCSD t<sup>a</sup><sub>i</sub> and t<sup>ab</sup><sub>ij</sub> are the converged single and double amplitudes from a CCSD calculation.
- - - -
#### Intermediates
F<sub>me</sub> = F<sub>(ov)</sub> = *f*<sub>me</sub> + t<sup>f</sup><sub>n</sub> g<sub>mnef</sub> \
F<sub>mi</sub> = F<sub>(oo)</sub> = *f*<sub>mi</sub> + t<sup>e</sup><sub>i</sub>*f*<sub>me</sub> + t<sup>e</sup><sub>n</sub> g<sub>mnie</sub> + 0.5 &tau;<sup>ef</sup><sub>in</sub> g<sub>mnef</sub> \
F<sub>ae</sub> = F<sub>(vv)</sub> = *f*<sub>ae</sub> - t<sup>a</sup><sub>m</sub>*f*<sub>me</sub> + t<sup>f</sup><sub>m</sub> g<sub>amef</sub> - 0.5 &tau;<sup>af</sup><sub>mn</sub> g<sub>mnef</sub> \
\
W<sub>mnij</sub> = W<sub>(oooo)</sub> = g<sub>mnij</sub> + P(ij) t<sup>e</sup><sub>j</sub> g<sub>mnie</sub> + 0.5&tau;<sup>ef</sup><sub> ij</sub> g<sub>mnef</sub> \
W<sub>abef</sub> = W<sub>(vvvv)</sub> = g<sub>abef</sub> - P(ab) t<sup>b</sup><sub>m</sub> g<sub>amef</sub> + 0.5&tau;<sup>ab</sup><sub>mn</sub> g<sub>mnef</sub> \
W<sub>amef</sub> = W<sub>(vovv)</sub> = g<sub>amef</sub> - t<sup>a</sup><sub>n</sub> g<sub>nmef</sub> \
W<sub>mnie</sub> = W<sub>(ooov)</sub> = g<sub>mnie</sub> + t<sup>f</sup><sub> i</sub> g<sub>mnfe</sub> \
W<sub>mbej</sub> = W<sub>(ovvo)</sub> = g<sub>mbej</sub> + t<sup>f</sup><sub>j</sub> g<sub>mbef</sub> - t<sup>b</sup><sub>n</sub> g<sub>mnej</sub> - (t<sup>fb</sup><sub>jn</sub> + t<sup>f</sup><sub>j</sub>t<sup>b</sup><sub>n</sub>) g<sub>nmfe</sub> \
W<sub>mbje</sub> = W<sub>(ovov)</sub> = g<sub>mbje</sub> + t<sup>f</sup><sub>j</sub> g<sub>bmef</sub> - t<sup>b</sup><sub>n</sub> g<sub>mnje</sub> - (t<sup>fb</sup><sub>jn</sub> + t<sup>f</sup><sub>j</sub>t<sup>b</sup><sub>n</sub>) g<sub>nmef</sub> \
W<sub>abei</sub> = W<sub>(vvvo)</sub> = g<sub>abei</sub> - F<sub>me</sub>t<sup>ab</sup><sub>mi</sub> + t<sup>f</sup><sub> i</sub>W<sub>abef</sub> + 0.5&tau;<sup>ab</sup><sub>mn</sub> g<sub>mnei</sub> - P(ab) t<sup>af</sup><sub>mi</sub> g<sub>mbef</sub> - P(ab) t<sup>a</sup><sub>m</sub> {g<sub>mbei</sub> - t<sup>bf</sup><sub>ni</sub> g<sub>mnef</sub>} \
W<sub>mbij</sub> = W<sub>(ovoo)</sub> = g<sub>mbij</sub> - F<sub>me</sub>t<sup>be</sup><sub>ij</sub> - t<sup>b</sup><sub>n</sub>W<sub>mnij</sub> + 0.5&tau;<sup>ef</sup><sub> ij</sub> g<sub>mbef</sub> + P(ij) t<sup>be</sup><sub>jn</sub> g<sub>mnie</sub> + P(ij) t<sup>e</sup><sub>i</sub> {g<sub>mbej</sub> - t<sup>bf</sup><sub>nj</sub> g<sub>mnef</sub>} 
- - - -
#### H<sub>SS</sub> 
\[F<sub>**a**c</sub> &delta;<sub>**i**k</sub> -  F<sub>k**i**</sub> &delta;<sub>**a**c</sub> + W<sub>**a**k**i**c</sub>] r<sup>ck</sup> \
\
Equations for terms taken from  [J. Chem. Phys. 98, 7029 (1993); https://doi.org/10.1063/1.46474698, 7029© 1993 American Institute of Physics.The equation of motion coupled-clustermethod. A systematic biorthogonal approach to molecular excitation energies, transition probabilities, and excited state properties](https://www.theochem.ru.nl/files/local/jcp-98-7029-1993.pdf) 

+ +F<sub>**a**c</sub> = *f*<sub>**a**c</sub> &delta;<sub>**i**k</sub> - t<sup>**a**</sup><sub>m</sub> *f*<sub>mc</sub>&delta;<sub>**i**k</sub> + t<sup>e</sup> </sub>m</sub> g<sub>m**a**ec</sub> &delta;<sub>**i**k</sub> - 0.5 t<sup>e**a**</sup><sub>nm</sub> g<sub>mnce</sub> &delta;<sub>**i**k</sub> - t<sup>e</sup><sub>n</sub> t<sup>**a**</sup><sub>m</sub> g<sub>mnce</sub> &delta;<sub>**i**k</sub>

    + [1]  +*f*<sub>**a**c</sub> &delta;<sub>**i**k</sub> &nbsp;&nbsp;<sup>1</sup>
    + [2]  -t<sub>m</sub><sup>**a**</sup> *f*<sub>mc</sub> &delta;<sub>**i**k</sub> &nbsp;&nbsp;<sup>9</sup>
    + [3]  +t<sup>e</sup><sub>m</sub> g<sub>m**a**ec</sub> &delta;<sub>**i**k</sub> &nbsp;&nbsp;<sup>4</sup>
    + [4]  -0.5 t<sup>e**a**</sup><sub>nm</sub> g<sub>mnce</sub> &delta;<sub>**i**k</sub> &nbsp;&nbsp;<sup>10</sup>
    + [5]  -t<sup>e</sup><sub>n</sub> t<sup>**a**</sup><sub>m</sub> g<sub>mnce</sub> &delta;<sub>**i**k</sub>  &nbsp;&nbsp;<sup>15</sup>
   
+ -F<sub>k**i**</sub> = -(*f*<sub>k**i**</sub> &delta;<sub>**a**c</sub> + t<sup>e</sup><sub>**i**</sub> *f*<sub>ke</sub> &delta;<sub>**a**c</sub> + t<sup>e</sup><sub>m</sub> g<sub>km**i**e</sub> &delta;<sub>**a**c</sub> + 0.5 t<sup>ef</sup><sub>**i**m</sub> g<sub>kmef</sub> &delta;<sub>**a**c</sub> + t<sup>e</sup><sub>**i**</sub> t<sup>f</sup><sub>m</sub> g<sub>kmef</sub> &delta;<sub>**a**c</sub> )

    + [6]  -*f*<sub>k**i**</sub> &delta;<sub>**a**c</sub> &nbsp;&nbsp;<sup>2</sup>
    + [7]  -t<sup>e</sup><sub>**i**</sub> *f*<sub>ke</sub> &delta;<sub>**a**c</sub>  &nbsp;&nbsp;<sup>8</sup>
    + [8]  -t<sup>e</sup><sub>m</sub> g<sub>km**i**e</sub> &delta;<sub>**a**c</sub>  &nbsp;&nbsp;<sup>5</sup>
    + [9]  -0.5 t<sup>ef</sup><sub>**i**m</sub> g<sub>kmef</sub> &delta;<sub>**a**c</sub>  &nbsp;&nbsp;<sup>11</sup>
    + [10] -t<sup>e</sup><sub>**i**</sub> t<sup>f</sup><sub>m</sub> g<sub>kmef</sub> &delta;<sub>**a**c</sub>  &nbsp;&nbsp;<sup>14</sup>

+ +W<sub>**a**k**i**c</sub> = g<sub>**a**k**i**c</sub> + t<sup>e</sup><sub>**i**</sub> g<sub>**a**kec</sub> - t<sup>**a**</sup><sub>m</sub> g<sub>mk**i**c</sub> - (t<sup>e**a**</sup><sub>**i**m</sub> + t<sup>e</sup><sub>**i**</sub> t<sup>**a**</sup><sub>m</sub>) g<sub>mkec</sub>

    + [11] +g<sub>**a**k**i**c</sub>  &nbsp;&nbsp;<sup>3</sup>
    + [12] +t<sup>e</sup><sub>**i**</sub> g<sub>**a**kec</sub>  &nbsp;&nbsp;<sup>6</sup>
    + [13] -t<sup>**a**</sup><sub>m</sub>  g<sub>mk**i**c</sub>  &nbsp;&nbsp;<sup>7</sup>
    + [14] -t<sup>e**a**</sup><sub>**i**m</sub>g<sub>mkec</sub>  &nbsp;&nbsp;<sup>12</sup>
    + [15] -t<sup>e</sup><sub>**i**</sub> t<sup>**a**</sup><sub>m</sub> g<sub>mkec</sub>  &nbsp;&nbsp;<sup>13</sup>
 
- - -
#### H<sub>SD</sub>
\[F<sub>ld</sub> &delta;<sub>**i**k</sub>&delta;<sub>**a**c</sub> + 0.5  W<sub>**a**lcd</sub> &delta;<sub>**i**k</sub> - 0.5 W<sub>kl**i**d</sub> &delta;<sub>**a**c</sub>] r<sup>lkcd</sup>

+ +F<sub>ld</sub> = f<sub>ld</sub> &delta;<sub>**i**k</sub>&delta;<sub>**a**c</sub> + t<sup>e</sup><sub>m</sub> g<sub>lmde</sub> &delta;<sub>**i**k</sub>&delta;<sub>**a**c</sub>

    + [16] +f<sub>ld</sub> &delta;<sub>**i**k</sub>&delta;<sub>**a**c</sub>  &nbsp;&nbsp;<sup>16</sup> 
    + [17] +t<sup>e</sup><sub>m</sub> g<sub>lmde</sub> &delta;<sub>**i**k</sub>&delta;<sub>**a**c</sub>  &nbsp;&nbsp;<sup>21</sup>
    
+  +0.5 W<sub>**a**lcd</sub> = 0.5 g<sub>**a**lcd</sub>  &delta;<sub>**i**k</sub> - 0.5 t<sup>**a**</sup><sub>m</sub>g<sub>mlcd</sub> &delta;<sub>**i**k</sub>

    + [18] +0.5 g<sub>**a**lcd</sub> &delta;<sub>**i**k</sub> &nbsp;&nbsp;<sup>17</sup>
    + [19] -0.5 t<sup>**a**</sup><sub>m</sub> g<sub>mlcd</sub> &delta;<sub>**i**k</sub> &nbsp;&nbsp;<sup>20</sup>

+ -0.5 W<sub>kl**i**d</sub> = -0.5 (g<sub>kl**i**d</sub> </sub>&delta;<sub>**a**d</sub> + t<sup>e</sup><sub>**i**</sub> g<sub>kled</sub></sub> &delta;<sub>**a**c</sub>)

    + [20] -0.5 g<sub>kl**i**d</sub> </sub>&delta;<sub>**a**c</sub> &nbsp;&nbsp;<sup>18</sup>
    + [21] -0.5 t<sup>e</sup><sub>**i**</sub> g<sub>kled</sub></sub> &delta;<sub>**a**c</sub> &nbsp;&nbsp;<sup>19</sup>
    
*There is disagreement between reference [2] and [Coupled-cluster calculations of nuclear magnetic resonance chemical shifts](www2.chemia.uj.edu.pl/~migda/Literatura/pdf/JCP03561.pdf) we have taken reference [3] which agrees with coding in psi4numpy/pyscf. Reference [2] has g<sub>kild</sub> + t<sup>e</sup><sub>k</sub>g<sub>kied</sub> and reference [3] g<sub>lkid</sub> + t<sup>e</sup><sub>i</sub>g<sub>lked</sub>*

- - -
#### H<sub>DS</sub>
 \[*P*(ab) W<sub>k**aij**</sub> &delta;<sub>**b**c</sub> + *P*(ij) W<sub>**ab**c**j**</sub> &delta;<sub>**i**k</sub> + *P*(ab) W<sub>**b**kec</sub> t<sup>**a**e</sup><sub>**ij**</sub> - *P*(ij) W<sub>mk**j**c</sub> t<sup>**ab**</sup><sub>**i**m</sub>] r<sup>kc</sup> 
 
+ +*P*(ab) {W<sub>k**aij**</sub>} = *P*(ab) {g<sub>k**aij**</sub> &delta;<sub>**b**c</sub> + *P*(ij) t<sup>**a**e</sup><sub>m**j**</sub> g<sub>km**i**e</sub> &delta;<sub>**b**c</sub> + 0.5&tau;<sup>ef</sup><sub>**ij**</sub> g<sub>k**a**ef</sub> &delta;<sub>**b**c</sub> - t<sup>**a**</sup><sub>m</sub> W<sub>km**ij**</sub> &delta;<sub>**b**c</sub> +*P*(ij) t<sup>e</sup><sub>**i**</sub> (g<sub>k**a**e**j**</sub> - t<sup>**a**f</sup><sub>m**j**</sub> g<sub>kmef</sub>) &delta;<sub>**b**c</sub> -t<sup>ae</sup><sub>**ij**</sub> F<sub>ke</sub> &delta;<sub>**b**c</sub>}

    + [22] +*P*(ab) {g<sub>k**aij**</sub> &delta;<sub>**b**c</sub>}  &nbsp;&nbsp;<sup>23</sup>
    + [23] +*P*(ab)*P*(ij) t<sup>e**a**</sup><sub>m**j**</sub> g<sub>km**i**e</sub> &delta;<sub>**b**c</sub> &nbsp;&nbsp;<sup>31</sup>
    + [--] +*P*(ab) 0.5&tau;<sup>ef</sup><sub>**ij**</sub> g<sub>k**a**ef</sub> &delta;<sub>**b**c</sub> 
        + [24] +*P*(ab) 0.5t<sup>ef</sup><sub>**ij**</sub> g<sub>k**a**ef</sub> &delta;<sub>**b**c</sub> &nbsp;&nbsp;<sup>32</sup>
        + [25] +*P*(ab) t<sup>e</sup><sub>**i**</sub> t<sup>f</sup><sub>**j**</sub> g<sub>k**a**ef</sub> &delta;<sub>**b**c</sub> &nbsp;&nbsp;<sup>36</sup>
    + [--] -*P*(ab) t<sup>**a**</sup><sub>m</sub>W<sub>km**ij**</sub> &delta;<sub>**b**c</sub>
        + W<sub>km**ij**</sub> =  g<sub>km**ij**</sub> + *P*(ij) t<sup>e</sup><sub>**j**</sub> g<sub>km**i**e</sub> + 0.5&tau;<sup>ef</sup><sub>**ij**</sub> g<sub>kmef</sub> 
            + [26] -*P*(ab) t<sup>**a**</sup><sub>m</sub> g<sub>km**ij**</sub> &delta;<sub>**b**c</sub>  &nbsp;&nbsp;<sup>25</sup>
            + [27] +*P*(ab) {t<sup>**a**</sup><sub>m</sub> *P*(ij) t<sup>e</sup><sub>**j**</sub> g<sub>kme**i**</sub> &delta;<sub>**b**c</sub>} &nbsp;&nbsp;<sup>38</sup>
            + [--] -*P*(ab) 0.5t<sup>**a**</sup><sub>m</sub> &tau;<sup>ef</sup><sub>**ij**</sub> g<sub>kmef</sub> &delta;<sub>**b**c</sub>
                + [28] -*P*(ab) 0.5t<sup>**a**</sup><sub>m</sub> t<sup>ef</sup><sub>**ij**</sub> g<sub>kmef</sub> &delta;<sub>**b**c</sub>  &nbsp;&nbsp;<sup>43</sup>
                + [29] -*P*(ab) t<sup>**a**</sup><sub>m</sub> t<sup>e</sup><sub>**i**</sub> t<sup>f</sup><sub>**j**</sub> g<sub>kmef</sub> &delta;<sub>**b**c</sub> &nbsp;&nbsp;<sup>50</sup>
    + [--] +*P*(ab) {*P*(ij) t<sup>e</sup><sub>**i**</sub> (g<sub>k**a**e**j**</sub> - t<sup>**a**f</sup><sub>m**j**</sub> g<sub>kmef</sub>) &delta;<sub>**b**c</sub>}
        + [30] +*P*(ab)*P*(ij) t<sup>e</sup><sub>**i**</sub> g<sub>k**a**ej</sub> &delta;<sub>**b**c</sub> &nbsp;&nbsp;<sup>27</sup>
        + [31] -*P*(ab)*P*(ij) t<sup>e</sup><sub>**i**</sub> t<sup>**a**f</sup><sub>m**j**</sub>g<sub>kmef</sub> &delta;<sub>**b**c</sub> &nbsp;&nbsp;<sup>39</sup>
    + [--] -*P*(ab) +t<sup>**a**e</sup><sub>**ij**</sub> F<sub>ke</sub> &delta;<sub>**b**c</sub> = -*P*(ab) {t<sup>**a**e</sup><sub>**ij**</sub> *f*<sub>ke</sub> &delta;<sub>**b**c</sub> + t<sup>**a**e</sup><sub>**ij**</sub> t<sup>f</sup><sub>m</sub> g<sub>kmef</sub> &delta;<sub>**b**c</sub>}
        + [32] -*P*(ab) t<sup>**a**e</sup><sub>**ij**</sub> *f*<sub>ke</sub> &delta;<sub>**b**c</sub>  &nbsp;&nbsp;<sup>29</sup>
        + [48] -*P*(ab) t<sup>**a**e</sup><sub>**ij**</sub> t<sup>f</sup><sub>m</sub> g<sub>kmef</sub> &delta;<sub>**b**c</sub>  &nbsp;&nbsp;<sup>49</sup>
                                     
+ *P*(ij) {W<sub>**ab**c**j**</sub>} = *P*(ij) {g<sub>**ab**c**j**</sub> &delta;<sub>**i**k</sub>  - *P*(ab) g<sub>m**b**cf</sub> t<sup>**a**f</sup><sub>m**j**</sub> &delta;<sub>**i**k</sub>  + 0.5&tau;<sup>**ab**</sup><sub>mn</sub> g<sub>mnc**j**</sub> &delta;<sub>**i**k</sub> + t<sup>e</sup><sub>**j**</sub> W<sub>**ab**ce</sub> &delta;<sub>**i**k</sub>  - *P*(ab) t<sup>**a**</sup><sub>m</sub> (g<sub>m**b**c**j**</sub> &delta;<sub>**i**k</sub>  - t<sup>**b**e</sup><sub>n**j**</sub> g<sub>mnce</sub>) &delta;<sub>**i**k</sub>  - F<sub>mc</sub> t<sup>**ab**</sup><sub>m**j**</sub> &delta;<sub>**i**k</sub>}

    + [33] +*P*(ij) g<sub>**ab**c**j**</sub> &delta;<sub>**i**k</sub> &nbsp;&nbsp;<sup>22</sup>
    + [34] -*P*(ij)*P*(ab) g<sub>**b**mce</sub> t<sup>e**a**</sup><sub>mj</sub> &delta;<sub>**i**k</sub>   &nbsp;&nbsp;<sup>30</sup>
    + [--] +*P*(ij) 0.5&tau;<sup>**ab**</sup><sub>mn</sub> g<sub>mnc**j**</sub> &delta;<sub>**i**k</sub>
        + [35] +*P*(ij) 0.5t<sup>**ab**</sup><sub>mn</sub> g<sub>mnc**j**</sub> &delta;<sub>**i**k</sub>  &nbsp;&nbsp;<sup>33</sup>
        + [36] +*P*(ij) t<sup>**a**</sup><sub>m</sub> t<sup>**b**</sup><sub>n</sub> g<sub>mnc**j**</sub> &delta;<sub>**i**k</sub>  &nbsp;&nbsp;<sup>37</sup>                           
    + [--] *P*(ij) W<sub>**ab**ce</sub> = *P*(ij) {(g<sub>**ab**ce</sub> - *P*(ab) t<sup>**b**</sup><sub>m</sub> g<sub>**a**mce</sub> + 0.5&tau;<sup>**ab**</sup><sub>mn</sub> g<sub>mnce</sub>) &delta;<sub>**i**k</sub>}
        + [37] +*P*(ij) t<sup>e</sup><sub>**j**</sub> g<sub>**ab**ce</sub> &delta;<sub>**i**k</sub>  &nbsp;&nbsp;<sup>24</sup>
        + [38] *P*(ij)*P*(ab) t<sup>e</sup><sub>**j**</sub> t<sup>**b**</sup><sub>m</sub> g<sub>m**a**ce</sub> &delta;<sub>**i**k</sub>  &nbsp;&nbsp;<sup>40*</sup>
        + [--] *P*(ij) 0.5&tau;<sup>**ab**</sup><sub>mn</sub> g<sub>mnce</sub> &delta;<sub>**i**k</sub>
            + [39] *P*(ij) 0.5t<sup>e</sup><sub>**j**</sub> t<sup>**ab**</sup><sub>mn</sub> g<sub>mnce</sub> &delta;<sub>**i**k</sub> &nbsp;&nbsp;<sup>42</sup>
            + [40] *P*(ij) t<sup>e</sup><sub>**j**</sub> t<sup>**a**</sup><sub>m</sub> t<sup>**b**</sup><sub>n</sub> g<sub>mnce</sub> &delta;<sub>**i**k</sub> &nbsp;&nbsp;<sup>41</sup>
    + [41] -*P*(ij)*P*(ab) t<sup>**a**</sup><sub>m</sub> g<sub>m**b**c**j**</sub> &delta;<sub>**i**k</sub>  &nbsp;&nbsp;<sup>26</sup>
    + [42] -*P*(ij)*P*(ab) t<sup>**a**</sup><sub>m</sub> t<sup>e**b**</sup><sub>n**j**</sub> g<sub>mnce</sub> &delta;<sub>**i**k</sub>  &nbsp;&nbsp;<sup>44</sup>
    + [--] -*P*(ij) F<sub>mc</sub> t<sup>**ab**</sup><sub>m**j**</sub> &delta;<sub>**i**k</sub> = *P*(ij) {-t<sup>**ab**</sup><sub>m**j**</sub> *f*<sub>mc</sub> &delta;<sub>**i**k</sub> - t<sup>**ab**</sup><sub>m**j**</sub> t<sup>e</sup><sub>n</sub> g<sub>mnce</sub> &delta;<sub>**i**k</sub>}
        + [47] -*P*(ij) t<sup>**ab**</sup><sub>m**j**</sub> *f*<sub>mc</sub> &delta;<sub>**i**k</sub>  &nbsp;&nbsp;<sup>28</sup>
        + [49] -*P*(ij) t<sup>**ab**</sup><sub>m**j**</sub> t<sup>e</sup><sub>n</sub> g<sub>mnce</sub> &delta;<sub>**i**k</sub> &nbsp;&nbsp;<sup>47</sup>
   
+ *P*(ab) W<sub>**b**kec</sub> t<sup>**a**e</sup><sub>**ij**</sub> = *P*(ab) {t<sup>**a**e</sup><sub>**ij**</sub>(g<sub>**b**kec</sub> - t<sup>**b**</sup><sub>m</sub> g<sub>mkec</sub>)}

    + [43] +*P*(ab) t<sup>**a**e</sup><sub>**ij**</sub> g<sub>**b**kec</sub>  &nbsp;&nbsp;<sup>34</sup>
    + [44] -*P*(ab) t<sup>**a**e</sup><sub>**ij**</sub> t<sup>**b**</sup><sub>m</sub> g<sub>mkec</sub>  &nbsp;&nbsp;<sup>48</sup>
    
+ -*P*(ij) W<sub>mk**j**c</sub> t<sup>**ab**</sup><sub>**i**m</sub> =  -*P*(ij){ t<sup>**ab**</sup><sub>**i**m</sub>(g<sub>mk**j**c</sub> - t<sup>e</sup><sub>**j**</sub> g<sub>mkec</sub>)}
    + [45] -*P*(ij) t<sup>**ab**</sup><sub>**i**m</sub> g<sub>mk**j**c</sub> &nbsp;&nbsp;<sup>35</sup>
    + [46] -*P*(ij) t<sup>**ab**</sup><sub>**i**m</sub> t<sup>e</sup><sub>**j**</sub> g<sub>mkec</sub>  &nbsp;&nbsp;<sup>46</sup>                  

- - -
#### H<sub>DD</sub>
 \[*P*(ab) F<sub>**b**c</sub> &delta;<sub>**j**k</sub> &delta;<sub>**i**l</sub> &delta;<sub>**a**d</sub> - F<sub>k**j**</sub></sub> &delta;<sub>**a**d</sub> &delta;<sub>**i**l</sub> &delta;<sub>**b**c</sub> + 0.5W<sub>**ab**cd</sub> &delta;<sub>**i**k</sub> &delta;<sub>**j**l</sub> + 0.5W<sub>kl**ij**</sub> &delta;<sub>**a**c</sub > &delta;<sub>**b**d</sub> + *P*(ij)*P*(ab) W<sub>**a**k**i**c</sub> &delta;<sub>**j**l</sub> &delta;<sub>**b**d</sub> - 0.5W<sub>lkec</sub>t<sup>e**b**</sup><sub>**ij**</sub> &delta;<sub>**a**d</sub> + 0.5W<sub>mkdc</sub>t<sup>**ab**</sup><sub>**j**m</sub> &delta;<sub>**i**l</sub>] r<sup>lkcd</sup>

+ +*P*(ab) F<sub>**b**c</sub> = *P*(ab) {(*f*<sub>**b**c</sub> - *f*<sub>mc</sub> t<sup>**b**</sup><sub>m</sub> + t<sup>e</sup><sub>m</sub> g<sub>m**b**ec</sub> - 0.5 &tau;<sup>**b**e</sup><sub>mn</sub>g<sub>mnce</sub>) &delta;<sub>**j**k</sub>&delta;<sub>**i**l</sub>&delta;<sub>**a**d</sub>}
    + [50] +*P*(ab) *f*<sub>**b**c</sub> &delta;<sub>**j**k</sub>&delta;<sub>**i**l</sub>&delta;<sub>**a**d</sub>  &nbsp;&nbsp;<sup>52</sup>  
    + [51] -*P*(ab) *f*<sub>mc</sub> t<sup>**b**</sup><sub>m</sub> &delta;<sub>**j**k</sub>&delta;<sub>**i**l</sub>&delta;<sub>**a**d</sub>  &nbsp;&nbsp;<sup>58</sup>
    + [52] +*P*(ab) t<sup>e</sup><sub>m</sub>g<sub>m**b**ec</sub> &delta;<sub>**j**k</sub>&delta;<sub>**i**l</sub>&delta;<sub>**a**d</sub>  &nbsp;&nbsp;<sup>63</sup>
    + [53] -*P*(ab) 0.5t<sup>e**b**</sup><sub>mn</sub> g<sub>mnec</sub> &delta;<sub>**j**k</sub>&delta;<sub>**i**l</sub>&delta;<sub>**a**d</sub>  &nbsp;&nbsp;<sup>71</sup>
    + [54] -*P*(ab) t<sup>e</sup><sub>m</sub> t<sup>**b**</sup><sub>n</sub> g<sub>mnec</sub>&delta;<sub>**j**k</sub>&delta;<sub>**i**l</sub>&delta;<sub>**a**d</sub>  &nbsp;&nbsp;<sup>76</sup>
    
+ -*P*(ij) F<sub>k**j**</sub> = *P*(ij) {(*f*<sub>k**j**</sub> + *f*<sub>ke</sub> t<sup>e</sup><sub>**j**</sub> + t<sup>e</sup><sub>m</sub> g<sub>km**j**e</sub> + 0.5&tau;<sup>ef</sup><sub>**j**m</sub> g<sub>kmef</sub> ) &delta;<sub>**a**d</sub>&delta;<sub>**i**l</sub>&delta;<sub>**b**c</sub>}
    + [55] -*P*(ij) *f*<sub>k**j**</sub></sub> &delta;<sub>**a**d</sub>&delta;<sub>**i**l</sub>&delta;<sub>**b**c</sub>  &nbsp;&nbsp;<sup>53</sup>
    + [56] -*P*(ij) *f*<sub>ke</sub> t<sup>e</sup><sub>**j**</sub> &delta;<sub>**a**d</sub>&delta;<sub>**i**l</sub>&delta;<sub>**b**c</sub> &nbsp;&nbsp;<sup>57</sup>
    + [57] -*P*(ij) t<sup>e</sup><sub>m</sub> g<sub>km**j**e</sub>  &delta;<sub>**a**d</sub>&delta;<sub>**i**l</sub>&delta;<sub>**b**c</sub>  &nbsp;&nbsp;<sup>64</sup>
    + [58] +*P*(ij) 0.5t<sup>fe</sup><sub>**j**m</sub> g<sub>kmef</sub> &delta;<sub>**a**d</sub>&delta;<sub>**i**l</sub>&delta;<sub>**b**c</sub>  &nbsp;&nbsp;<sup>69</sup>
    + [59] +*P*(ij) t<sup>f</sup><sub>**j**</sub> t<sup>e</sup><sub>m</sub> g<sub>kmef</sub> &delta;<sub>**a**d</sub>&delta;<sub>**i**l</sub>&delta;<sub>**b**c</sub>  &nbsp;&nbsp;<sup>75</sup>
    
+ +0.5W<sub>**ab**cd</sub> = 0.5 (g<sub>**ab**cd</sub> - *P*(ab) t<sup>**b**</sup><sub>m</sub> g<sub>**a**mcd</sub> + 0.5&tau;<sup>**ab**</sup><sub>mn</sub> g<sub>mncd</sub>) &delta;<sub>**i**k</sub>&delta;<sub>**j**l</sub>
    + [60] +0.5g<sub>**ab**cd</sub> &delta;<sub>**i**k</sub>&delta;<sub>**j**l</sub>  &nbsp;&nbsp;<sup>54</sup>
    + [61] -P(ab) 0.5t<sup>**b**</sup><sub>m</sub> g<sub>**a**mcd</sub> &delta;<sub>**i**k</sub>&delta;<sub>**j**l</sub>  &nbsp;&nbsp;<sup>61</sup>
    + [62] +0.25t<sup>**ab**</sup><sub>mn</sub> g<sub>mncd</sub> &delta;<sub>**i**k</sub>&delta;<sub>**j**l</sub>  &nbsp;&nbsp;<sup>65</sup>
    + [63] +0.5t<sup>**a**</sup><sub>m</sub> t<sup>**b**</sup><sub>n</sub> g<sub>mncd</sub> &delta;<sub>**i**k</sub>&delta;<sub>**j**l</sub>  &nbsp;&nbsp;<sup>73</sup>
    
+ +0.5W<sub>kl**ij**</sub> = 0.5 (g<sub>kl**ij**</sub> + *P*(ij) t<sup>e</sup><sub>**j**</sub> g<sub>kl**i**e</sub> + 0.5&tau;<sup>ef</sup><sub>**ij**</sub> g<sub>klef</sub>) &delta;<sub>**a**c</sub>&delta;<sub>**b**d</sub> 
    + [64] +0.5g<sub>kl**ij**</sub> &delta;<sub>**a**c</sub>&delta;<sub>**b**d</sub>  &nbsp;&nbsp;<sup>55</sup>
    + [65] +P(ij) 0.5t<sup>e</sup><sub>**j**</sub> g<sub>kl**i**e</sub> &delta;<sub>**a**c</sub>&delta;<sub>**b**d</sub>  &nbsp;&nbsp;<sup>62</sup>
    + [66] +0.25t<sup>ef</sup><sub>**ij**</sub> g<sub>klef</sub> &delta;<sub>**a**c</sub>&delta;<sub>**b**d</sub>  &nbsp;&nbsp;<sup>66</sup>
    + [67] +0.5t<sup>e</sup><sub>**i**</sub> t<sup>f</sup><sub>**j**</sub> g<sub>klef</sub> &delta;<sub>**a**c</sub>&delta;<sub>**b**d</sub>  &nbsp;&nbsp;<sup>72</sup>
    
+ +*P*(ij)*P*(ab) W<sub>**a**k**i**c</sub> = *P*(ij)*P*(ab) {(g<sub>**a**k**i**c</sub> + t<sup>e</sup><sub>**i**</sub> g<sub>**a**kec</sub> - t<sup>**a**</sup><sub>m</sub> g<sub>mk**i**c</sub> - (t<sup>e**a**</sup><sub>**i**m</sub> + t<sup>e</sup><sub>**i**</sub> t<sup>**a**</sup><sub>m</sub>) g<sub>mkec</sub>) &delta;<sub>**j**l</sub>&delta;<sub>d**b<**</sub>}
    + [68] +*P*(ij)*P*(ab) g<sub>**a**k**i**c</sub>  &delta;<sub>**j**l</sub>&delta;<sub>d**b**</sub>   &nbsp;&nbsp;<sup>56</sup>
    + [69] +*P*(ij)*P*(ab) t<sup>e</sup><sub>**i**</sub> g<sub>**a**kec</sub> &delta;<sub>**j**l</sub>&delta;<sub>d**b**</sub>  &nbsp;&nbsp;<sup>59</sup>  
    + [70] -*P*(ij)*P*(ab) t<sup>**a**</sup><sub>m</sub>g<sub>mk**i**c</sub> &delta;<sub>**j**l</sub>&delta;<sub>d**b**</sub>  &nbsp;&nbsp;<sup>60</sup>
    + [71] -*P*(ij)*P*(ab) t<sup>e**a**</sup><sub>im</sub> g<sub>mkec</sub> &delta;<sub>**j**l</sub>&delta;<sub>d**b**</sub> &nbsp;&nbsp;<sup>67</sup>
    + [72] -*P*(ij)*P*(ab) t<sup>e</sup><sub>**i**</sub> t<sup>**a**</sup><sub>m</sub> g<sub>mkec</sub> &delta;<sub>**j**l</sub>&delta;<sub>d**b**</sub> &nbsp;&nbsp;<sup>74</sup>
  
+ -0.5W<sub>lkec</sub> t<sup>e**b**</sup><sub>**ij**</sub> = -0.5t<sup>e**b**</sup><sub>**ij**</sub> g<sub>lkec</sub> &delta;<sub>**a**d</sub>
    + [73] -0.5t<sup>e**b**</sup><sub>**ij**</sub>g<sub>lkec</sub> &delta;<sub>**a**d</sub>  &nbsp;&nbsp;<sup>70</sup>

+ +0.5W<sub>mkdc</sub> t<sup>**ab**</sup><sub>**j**m</sub> = +0.5W<sub>mkdc</sub >t<sup>**ab**</sup><sub>**j**m</sub> &delta;<sub>**i**l</sub>
    + [74] -0.5W<sub>kmcd</sub> t<sup>**ab**</sup><sub>m**j**</sub> &delta;<sub>**i**l</sub>  &nbsp;&nbsp;<sup>68</sup>
- - -
- - -
 # EOM-MBPT(2)
 This is an approximation to full EOM-CCSD obtained by only retaining terms through second-order. This means\
 t<sup>a</sup></sub>i</sub> = **0**\
 t<sup>ab</sup><sub>ij</sub> = g<sub>abij</sub>/(*f*<sub>ii</sub>+*f*<sub>jj</sub>-*f*<sub>aa</sub>-*f*<sub>bb</sub>)\
 *f*<sub>ai</sub> = **0**
 
 A good reference is ['Assessment of low-scaling approximations to the equation of motion coupled-cluster singles and doubles equations' - Joshua J Goings, Marco Caricato, Michael J Frisch and Xiaosong Li](http://dx.doi.org/10.1063/1.4898709)
 #### H<sub>SS</sub> 
\[F<sub>**a**c</sub> &delta;<sub>**i**k</sub> - F<sub>k**i**</sub> &delta;<sub>**a**c</sub> + W<sub>**a**k**i**c</sub>] r<sup>ck</sup> \
+ +F<sub>**a**c</sub> = *f*<sub>**a**c</sub> &delta;<sub>**i**k</sub> - 0.5 t<sup>e**a**</sup><sub>nm</sub> g<sub>mnce</sub> &delta;<sub>**i**k</sub> 

    + [1]  +*f*<sub>**a**c</sub> &delta;<sub>**i**k</sub> &nbsp;&nbsp;<sup>1</sup>
    + [4]  -0.5 t<sup>e**a**</sup><sub>nm</sub> g<sub>mnce</sub> &delta;<sub>**i**k</sub> &nbsp;&nbsp;<sup>10</sup>

+ -F<sub>k**i**</sub> = -\(*f*<sub>k**i**</sub> &delta;<sub>**a**c</sub>  + 0.5 t<sup>ef</sup><sub>**i**m</sub> g<sub>kmef</sub> &delta;<sub>**a**c</sub> )

    + [6]  -*f*<sub>k**i**</sub> &delta;<sub>**a**c</sub> &nbsp;&nbsp;<sup>2</sup>
    + [9]  -0.5 t<sup>ef</sup><sub>**i**m</sub> g<sub>kmef</sub> &delta;<sub>**a**c</sub>  &nbsp;&nbsp;<sup>11</sup>

+ +W<sub>**a**k**i**c</sub> = g<sub>**a**k**i**c</sub> - t<sup>e**a**</sup><sub>**i**m</sub> g<sub>mkec</sub>

    + [11] +g<sub>**a**k**i**c</sub>  &nbsp;&nbsp;<sup>3</sup>
    + [14] -t<sup>e**a**</sup><sub>**i**m</sub>g<sub>mkec</sub>  &nbsp;&nbsp;<sup>12</sup>
 
- - -
#### H<sub>SD</sub>
\[F<sub>ld</sub> &delta;<sub>**i**k</sub>&delta;<sub>**a**c</sub> + 0.5  W<sub>**a**lcd</sub> &delta;<sub>**i**k</sub> - 0.5 W<sub>kl**i**d</sub> &delta;<sub>**a**c</sub>] r<sup>lkcd</sup>

+ +F<sub>ld</sub> = 0 
    
+  +0.5 W<sub>**a**lcd</sub> = 0.5 g<sub>**a**lcd</sub>  &delta;<sub>**i**k</sub> 

    + [18] +0.5 g<sub>**a**lcd</sub> &delta;<sub>**i**k</sub> &nbsp;&nbsp;<sup>17</sup>

+ -0.5 W<sub>kl**i**d</sub> = -0.5 g<sub>kl**i**d</sub> </sub>&delta;<sub>**a**d</sub>

    + [20] -0.5 g<sub>kl**i**d</sub> </sub>&delta;<sub>**a**c</sub> &nbsp;&nbsp;<sup>18</sup>
- - -
#### H<sub>DS</sub>
 \[*P*(ab) W<sub>k**aij**</sub> &delta;<sub>**b**c</sub> + *P*(ij) W<sub>**ab**c**j**</sub> &delta;<sub>**i**k</sub> + *P*(ab) W<sub>**b**kec</sub> t<sup>**a**e</sup><sub>**ij**</sub> - *P*(ij) W<sub>mk**j**c</sub> t<sup>**ab**</sup><sub>**i**m]</sub>] r<sup>kc</sup> 
 
+ +*P*(ab) {W<sub>k**aij**</sub>} = *P*(ab) {g<sub>k**aij**</sub> &delta;<sub>**b**c</sub> + *P*(ij) t<sup>**a**e</sup><sub>m**j**</sub> g<sub>km**i**e</sub> &delta;<sub>**b**c</sub> + 0.5&tau;<sup>ef</sup><sub>**ij**</sub> g<sub>k**a**ef</sub> &delta;<sub>**b**c</sub>}

    + [22] +*P*(ab) {g<sub>k**aij**</sub> &delta;<sub>**b**c</sub>}  &nbsp;&nbsp;<sup>23</sup>
    + [23] +*P*(ab)*P*(ij) t<sup>e**a**</sup><sub>m**j**</sub> g<sub>km**i**e</sub> &delta;<sub>**b**c</sub> &nbsp;&nbsp;<sup>31</sup>
    + [--] +*P*(ab) 0.5&tau;<sup>ef</sup><sub>**ij**</sub> g<sub>k**a**ef</sub> &delta;<sub>**b**c</sub> 
        + [24] +*P*(ab) 0.5t<sup>ef</sup><sub>**ij**</sub> g<sub>k**a**ef</sub> &delta;<sub>**b**c</sub> &nbsp;&nbsp;<sup>32</sup>
                                     
+ *P*(ij) {W<sub>**ab**c**j**</sub>} = *P*(ij) {g<sub>**ab**c**j**</sub> &delta;<sub>**i**k</sub>  - *P*(ab) g<sub>m**b**cf</sub> t<sup>**a**f</sup><sub>m**j**</sub> &delta;<sub>**i**k</sub>  + 0.5&tau;<sup>**ab**</sup><sub>mn</sub> g<sub>mnc**j**</sub> &delta;<sub>**i**k</sub> 

    + [33] +*P*(ij) g<sub>**ab**c**j**</sub> &delta;<sub>**i**k</sub> &nbsp;&nbsp;<sup>22</sup>
    + [34] -*P*(ij)*P*(ab) g<sub>**b**mce</sub> t<sup>e**a**</sup><sub>mj</sub> &delta;<sub>**i**k</sub>   &nbsp;&nbsp;<sup>30</sup>
    + [--] +*P*(ij) 0.5&tau;<sup>**ab**</sup><sub>mn</sub> g<sub>mnc**j**</sub> &delta;<sub>**i**k</sub>
        + [35] +*P*(ij) 0.5t<sup>**ab**</sup><sub>mn</sub> g<sub>mnc**j**</sub> &delta;<sub>**i**k</sub>  &nbsp;&nbsp;<sup>33</sup>
           
+ *P*(ab) W<sub>**b**kec</sub> t<sup>**a**e</sup><sub>**ij**</sub> = *P*(ab) {t<sup>**a**e</sup><sub>**ij**</sub>(g<sub>**b**kec</sub> - t<sup>**b**</sup><sub>m</sub> g<sub>mkec</sub>)}

    + [43] +*P*(ab) t<sup>**a**e</sup><sub>**ij**</sub> g<sub>**b**kec</sub>  &nbsp;&nbsp;<sup>34</sup>
    
+ -*P*(ij) W<sub>mk**j**c</sub> t<sup>**ab**</sup><sub>**i**m</sub> =  -*P*(ij){ t<sup>**ab**</sup><sub>**i**m</sub>(g<sub>mk**j**c</sub> - t<sup>e</sup><sub>**j**</sub> g<sub>mkec</sub>)}
    + [45] -*P*(ij) t<sup>**ab**</sup><sub>**i**m</sub> g<sub>mk**j**c</sub> &nbsp;&nbsp;<sup>35</sup>
- - -
#### H<sub>DD</sub>
 \[*P*(ab) F<sub>**b**c</sub> &delta;<sub>**j**k</sub> &delta;<sub>**i**l</sub> &delta;<sub>**a**d</sub> - F<sub>k**j**</sub></sub> &delta;<sub>**a**d</sub> &delta;<sub>**i**l</sub> &delta;<sub>**b**c</sub> + 0.5W<sub>**ab**cd</sub> &delta;<sub>**i**k</sub> &delta;<sub>**j**l</sub> + 0.5W<sub>kl**ij**</sub> &delta;<sub>**a**c</sub > &delta;<sub>**b**d</sub> + *P*(ij)*P*(ab) W<sub>**a**k**i**c</sub> &delta;<sub>**j**l</sub> &delta;<sub>**b**d</sub> - 0.5W<sub>lkec</sub>t<sup>e**b**</sup><sub>**ij**</sub> &delta;<sub>**a**d</sub> + 0.5W<sub>mkdc</sub>t<sup>**ab**</sup><sub>**j**m</sub> &delta;<sub>**i**l</sub>] r<sup>lkcd</sup>

+ +*P*(ab) F<sub>**b**c</sub> = *P*(ab) {(*f*<sub>**b**c</sub> - 0.5 &tau;<sup>**b**e</sup><sub>mn</sub>g<sub>mnce</sub>) &delta;<sub>**j**k</sub>&delta;<sub>**i**l</sub>&delta;<sub>**a**d</sub>}
    + [50] +*P*(ab) *f*<sub>**b**c</sub> &delta;<sub>**j**k</sub>&delta;<sub>**i**l</sub>&delta;<sub>**a**d</sub>  &nbsp;&nbsp;<sup>52</sup>  
    + [53] -*P*(ab) 0.5t<sup>e**b**</sup><sub>mn</sub> g<sub>mnec</sub> &delta;<sub>**j**k</sub>&delta;<sub>**i**l</sub>&delta;<sub>**a**d</sub>  &nbsp;&nbsp;<sup>71</sup>
    
+ -*P*(ij) F<sub>k**j**</sub> = *P*(ij) {(*f*<sub>k**j**</sub> + 0.5&tau;<sup>ef</sup><sub>**j**m</sub> g<sub>kmef</sub> ) &delta;<sub>**a**d</sub>&delta;<sub>**i**l</sub>&delta;<sub>**b**c</sub>}
    + [55] -*P*(ij) *f*<sub>k**j**</sub></sub> &delta;<sub>**a**d</sub>&delta;<sub>**i**l</sub>&delta;<sub>**b**c</sub>  &nbsp;&nbsp;<sup>53</sup>
    + [58] +*P*(ij) 0.5t<sup>fe</sup><sub>**j**m</sub> g<sub>kmef</sub> &delta;<sub>**a**d</sub>&delta;<sub>**i**l</sub>&delta;<sub>**b**c</sub>  &nbsp;&nbsp;<sup>69</sup>
    
+ +0.5W<sub>**ab**cd</sub> = 0.5 (g<sub>**ab**cd</sub> + 0.5&tau;<sup>**ab**</sup><sub>mn</sub> g<sub>mncd</sub>) &delta;<sub>**i**k</sub>&delta;<sub>**j**l</sub>
    + [60] +0.5g<sub>**ab**cd</sub> &delta;<sub>**i**k</sub>&delta;<sub>**j**l</sub>  &nbsp;&nbsp;<sup>54</sup>
    + [62] +0.25t<sup>**ab**</sup><sub>mn</sub> g<sub>mncd</sub> &delta;<sub>**i**k</sub>&delta;<sub>**j**l</sub>  &nbsp;&nbsp;<sup>65</sup>
    
+ +0.5W<sub>kl**ij**</sub> = 0.5 (g<sub>kl**ij**</sub> + 0.5&tau;<sup>ef</sup><sub>**ij**</sub> g<sub>klef</sub>) &delta;<sub>**a**c</sub>&delta;<sub>**b**d</sub> 
    + [64] +0.5g<sub>kl**ij**</sub> &delta;<sub>**a**c</sub>&delta;<sub>**b**d</sub>  &nbsp;&nbsp;<sup>55</sup>
    + [66] +0.25t<sup>ef</sup><sub>**ij**</sub> g<sub>klef</sub> &delta;<sub>**a**c</sub>&delta;<sub>**b**d</sub>  &nbsp;&nbsp;<sup>66</sup>
    
+ +*P*(ij)*P*(ab) W<sub>**a**k**i**c</sub> = *P*(ij)*P*(ab) {(g<sub>**a**k**i**c</sub> - t<sup>e**a**</sup><sub>**i**m</sub> g<sub>mkec</sub>) &delta;<sub>**j**l</sub>&delta;<sub>d**b**</sub>}
    + [68] +*P*(ij)*P*(ab) g<sub>**a**k**i**c</sub>  &delta;<sub>**j**l</sub>&delta;<sub>d**b**</sub>   &nbsp;&nbsp;<sup>56</sup>
    + [71] -*P*(ij)*P*(ab) t<sup>e**a**</sup><sub>im</sub> g<sub>mkec</sub> &delta;<sub>**j**l</sub>&delta;<sub>d**b**</sub> &nbsp;&nbsp;<sup>67</sup>
  
+ -0.5W<sub>lkec</sub> t<sup>e**b**</sup><sub>**ij**</sub> = -0.5t<sup>e**b**</sup><sub>**ij**</sub> g<sub>lkec</sub> &delta;<sub>**a**d</sub>
    + [73] -0.5t<sup>e**b**</sup><sub>**ij**</sub>g<sub>lkec</sub> &delta;<sub>**a**d</sub>  &nbsp;&nbsp;<sup>70</sup>

+ +0.5W<sub>mkdc</sub> t<sup>**ab**</sup><sub>**j**m</sub> = +0.5W<sub>mkdc</sub >t<sup>**ab**</sup><sub>**j**m</sub> &delta;<sub>**i**l</sub>
    + [74] -0.5g<sub>kmcd</sub> t<sup>**ab**</sup><sub>m**j**</sub> &delta;<sub>**i**l</sub>  &nbsp;&nbsp;<sup>68</sup>

 - - -
 P-EOM
 
 In the partitioned scheme the H<sub>DD</sub>(2) block is replaced with H<sub>DD</sub>(0). That is 
 #### H<sub>DD</sub>
 \[*P*(ab) F<sub>**b**c</sub> &delta;<sub>**j**k</sub> &delta;<sub>**i**l</sub> &delta;<sub>**a**d</sub> - F<sub>k**j**</sub></sub> &delta;<sub>**a**d</sub> &delta;<sub>**i**l</sub> &delta;<sub>**b**c</sub>]\
     + [50] +*P*(ab) *f*<sub>**b**c</sub> &delta;<sub>**j**k</sub>&delta;<sub>**i**l</sub>&delta;<sub>**a**d</sub>  &nbsp;&nbsp;<sup>52</sup>  
     + [55] -*P*(ij) *f*<sub>k**j**</sub></sub> &delta;<sub>**a**d</sub>&delta;<sub>**i**l</sub>&delta;<sub>**b**c</sub>  &nbsp;&nbsp;<sup>53</sup>
 
 - - -
 
 1. **maximumAmplitudes(t, top = 5, mode = 0)**

    parameters -*t* is an array of the amplitudes and the maximum *top* amplitude are returned. If *mode* is 0 order of amplitudes is diminishing absolute values, if *mode* is +1 order of amplitudes is diminishing positive amplitudes and if *mode* = -1 order is increasing negative amplitudes (ie -3,-2,-1). Returns a list of the 'top' amplitudes in the format eg '-0.356412 (0, 4)'.


2.  **eomccsd(fockMOspin, eriMOspin, ts, td, nOccupied, nVirtual, spinOrbitals, partitioned = False, dialog = True)**
 
    parameters - *fockMOspin* is the converged Fock matrix in the molecular spin basis, *eriMOspin* are the electron repulsion integrals in the molecular spin basis, *ts* are the converged singles amplitudes from a ccsd computation, *td* are the converged doubles amplitudes from a ccsd computation, *nOccupied* is the number of occupied spin orbitals, *nVirtual* are the number of virtual spin orbitals, *spinOrbitals* are the number of spin orbitals. *partitioned* is a boolean flag, if true the H<sub>DD</sub> block will be set to H<sup>(0)</sup><sub>DD</sub> and *dialog* if true will print the timings to the console. Returns the eigenvalues and right eigenvectors.

3.  **eommbpt2(fockMOspin, eriMOspin, nOccupied, nVirtual, spinOrbitals, partitioned = False, dialog = True)**
 
    parameters - *fockMOspin* is the converged Fock matrix in the molecular spin basis, *eriMOspin* are the electron repulsion integrals in the molecular spin basis, *nOccupied* is the number of occupied spin orbitals, *nVirtual* are the number of virtual spin orbitals, *spinOrbitals* are the number of spin orbitals. *partitioned* is a boolean flag, if true the H<sub>DD</sub> block will be set to H<sup>(0)</sup><sub>DD</sub> and *dialog* if true will print the timings to the console. Returns the eigenvalues and right eigenvectors.

4.  **main**  
    This is an example of an eom calculation. A harpy project file is produced, saved and passed to the scf routine. After the scf has run the file is deleted. The Fock and eri integrals from the scf calculation are then converted to a molecular orbital spin basis and used in a CCSD calculation from which the singles and doubles amplitude are retained. The 5 most significant amplitudes (values and indices) are determined from the routine *maximumAmplitudes*. The eom-ccsd/mbpt(2) routines are run. After each routine has run the eigenvalues are sorted and converted to electron volts and passed to *ciDegeneracy* which returns tuples of (energy, degeneracy). Some of the energy values are compared to Gaussian derived values as a check. The SCF and CCSD correction energies, maximum amplitudes, excitation tuples and a range of eV to output are then passed to *postSCF* for printing in the harpy.html file. The output is 

    <p><b>equation of motion</b></p>
    <p style='margin-left:20px;font-size:10px;'>coupled-cluster singles and doubles calculation
    <br><table><tr><td>scf energy&nbsp;&nbsp;&nbsp;</td><td>-1.1229402577</td></tr>
    <tr><td>ccsd energy</td><td>-0.0248728759</td></tr>
    <tr><td>total energy</td><td>-1.1478131337</td></tr></table>
    <br>
  <p style='margin-left:20px;font-size:10px;'>most significant amplitudes
    <br><table>
            <tr><td><b>t<sup>a</sup><sub>i</sub></b></td></tr>
            <tr><td>-0.005758 (4, 0)</td></tr>
            <tr><td>-0.005758 (5, 1)</td></tr>
            <tr><td><b>t<sup>ab</sup><sub>ij</sub></b></td></tr>
            <tr><td>0.084054 (2, 3, 1, 0)</td></tr>
            <tr><td>0.084054 (3, 2, 0, 1)</td></tr>
            <tr><td>-0.084054 (3, 2, 1, 0)</td></tr>
            <tr><td>-0.084054 (2, 3, 0, 1)</td></tr>
            <tr><td>-0.047829 (5, 4, 1, 0)</td></tr>
    </table>
    <table><caption>eom-ccsd</caption>
        <td>10.852658 (t)</td>
        <td>15.898413 (s)</td>
        <td>26.471214 (t)</td>
        <td>30.521616 (s)</td>
        <td>31.881407 (s)</td>
        </tr>
        <tr>
        <td>40.401967 (t)</td>
        <td>41.140804 (s)</td>
        <td>43.232123 (t)</td>
        </tr>
    </table>
    <table><caption>eom-mbpt(2)</caption>
        <td>10.657194 (t)</td>
        <td>15.708727 (s)</td>
        <td>26.265493 (t)</td>
        <td>30.222336 (s)</td>
        <td>31.678520 (s)</td>
        </tr>
        <tr>
        <td>40.207311 (t)</td>
        <td>40.912816 (s)</td>
        <td>43.016807 (t)</td>
        </tr>
    </table>

__Fast EOM_CCSD__
There is a routine which codes an einsum version of EOM-CCSD using intermediates in _cc.fcc_.
