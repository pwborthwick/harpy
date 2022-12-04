
'''s - spin orbitals, o - occupied spin orbitals, v - virtual spin orbitals
t1[v,o] - singles amplitudes, t2[v,v,o,o] - doubles amplitudes
l1[o,v] - singles lambda, l2[o,o,v,v] - doubles lambda
'''

def cc_energy(f, g, o, v, t1=None, t2=None, t3=None):

    '''
        COGUS generated level [2] on 28 Jul 2021   
    '''
    from numpy import einsum, swapaxes


    #  -0.50000 *  <i,j||i,j> 
    T = -0.50000 * einsum('ijij->' ,g[o,o,o,o], optimize=True)

    #  1.0000 *  f(i,a) t1(a,i) 
    T += 1.0000 * einsum('ia,ai->' ,f[o,v] ,t1, optimize=True)

    #  0.2500 *  <i,j||a,b> t2(a,b,i,j) 
    T += 0.2500 * einsum('ijab,abij->' ,g[o,o,v,v] ,t2, optimize=True)

    #  -0.50000 *  <i,j||a,b> t1(a,j) t1(b,i) 
    T += -0.50000 * einsum('ijab,aj,bi->' ,g[o,o,v,v] ,t1 ,t1, optimize=True)

    #  1.0000 *  f(i,i) 
    T += 1.0000 * einsum('ii->' ,f[o,o], optimize=True)

    return T

def cc_singles(f, g, o, v, t1=None, t2=None, t3=None):

    '''
        COGUS generated level [2] on 28 Jul 2021   
    '''
    from numpy import einsum, swapaxes


    #  1.0000 *  f(a,b) t1(b,i) 
    T = 1.0000 * einsum('ab,bi->ai' ,f[v,v] ,t1, optimize=True)

    #  1.0000 *  f(j,b) t2(a,b,i,j) 
    T += 1.0000 * einsum('jb,abij->ai' ,f[o,v] ,t2, optimize=True)

    #  1.0000 *  <a,j||i,b> t1(b,j) 
    T += 1.0000 * einsum('ajib,bj->ai' ,g[v,o,o,v] ,t1, optimize=True)

    #  0.5000 *  <a,j||b,c> t2(b,c,i,j) 
    T += 0.5000 * einsum('ajbc,bcij->ai' ,g[v,o,v,v] ,t2, optimize=True)

    #  -1.0000 *  f(j,i) t1(a,j) 
    T += -1.0000 * einsum('ji,aj->ai' ,f[o,o] ,t1, optimize=True)

    #  -0.50000 *  <j,k||i,b> t2(a,b,j,k) 
    T += -0.50000 * einsum('jkib,abjk->ai' ,g[o,o,o,v] ,t2, optimize=True)

    #  1.0000 *  <j,k||i,b> t1(a,k) t1(b,j) 
    T += 1.0000 * einsum('jkib,ak,bj->ai' ,g[o,o,o,v] ,t1 ,t1, optimize=True)

    #  1.0000 *  <j,k||b,c> t1(b,j) t2(a,c,i,k) 
    T += 1.0000 * einsum('jkbc,bj,acik->ai' ,g[o,o,v,v] ,t1 ,t2, optimize=True)

    #  -1.0000 *  f(j,b) t1(a,j) t1(b,i) 
    T += -1.0000 * einsum('jb,aj,bi->ai' ,f[o,v] ,t1 ,t1, optimize=True)

    #  -1.0000 *  <a,j||b,c> t1(b,j) t1(c,i) 
    T += -1.0000 * einsum('ajbc,bj,ci->ai' ,g[v,o,v,v] ,t1 ,t1, optimize=True)

    #  -0.50000 *  <j,k||b,c> t1(a,j) t2(b,c,i,k) 
    T += -0.50000 * einsum('jkbc,aj,bcik->ai' ,g[o,o,v,v] ,t1 ,t2, optimize=True)

    #  -0.50000 *  <j,k||b,c> t1(b,i) t2(a,c,j,k) 
    T += -0.50000 * einsum('jkbc,bi,acjk->ai' ,g[o,o,v,v] ,t1 ,t2, optimize=True)

    #  -1.0000 *  <j,k||b,c> t1(a,k) t1(b,j) t1(c,i) 
    T += -1.0000 * einsum('jkbc,ak,bj,ci->ai' ,g[o,o,v,v] ,t1 ,t1 ,t1, optimize=True)

    #  1.0000 *  f(a,i) 
    T += 1.0000 * einsum('ai->ai' ,f[v,o], optimize=True)

    return T

def cc_doubles(f, g, o, v, t1=None, t2=None, t3=None):

    '''
        COGUS generated level [2] on 28 Jul 2021   
    '''
    from numpy import einsum, swapaxes


    #  1.0000 * P(i,j) f(k,i) t2(a,b,j,k) 
    t = 1.0000 * einsum('ki,abjk->abij' ,f[o,o] ,t2, optimize=True)
    T = t - t.swapaxes(3, 2)

    #  1.0000 * P(i,j) <a,b||i,c> t1(c,j) 
    t = 1.0000 * einsum('abic,cj->abij' ,g[v,v,o,v] ,t1, optimize=True)
    T += t - t.swapaxes(3, 2)

    #  1.0000 *  <a,b||c,d> t1(c,i) t1(d,j) 
    T += 1.0000 * einsum('abcd,ci,dj->abij' ,g[v,v,v,v] ,t1 ,t1, optimize=True)

    #  1.0000 *  <k,l||i,j> t1(a,k) t1(b,l) 
    T += 1.0000 * einsum('klij,ak,bl->abij' ,g[o,o,o,o] ,t1 ,t1, optimize=True)

    #  -1.0000 * P(a,b) f(a,c) t2(b,c,i,j) 
    t = -1.0000 * einsum('ac,bcij->abij' ,f[v,v] ,t2, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  -1.0000 * P(a,b) <a,k||i,j> t1(b,k) 
    t = -1.0000 * einsum('akij,bk->abij' ,g[v,o,o,o] ,t1, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  1.0000 * P(i,j) <k,l||i,c> t1(a,k) t1(b,l) t1(c,j) 
    t = 1.0000 * einsum('klic,ak,bl,cj->abij' ,g[o,o,o,v] ,t1 ,t1 ,t1, optimize=True)
    T += t - t.swapaxes(3, 2)

    #  1.0000 *  <k,l||c,d> t1(a,k) t1(b,l) t1(c,i) t1(d,j) 
    T += 1.0000 * einsum('klcd,ak,bl,ci,dj->abij' ,g[o,o,v,v] ,t1 ,t1 ,t1 ,t1, optimize=True)

    #  -1.0000 * P(a,b)P(i,j) <a,k||i,c> t1(b,k) t1(c,j) 
    t = -1.0000 * einsum('akic,bk,cj->abij' ,g[v,o,o,v] ,t1 ,t1, optimize=True)
    T += t - t.swapaxes(1, 0) - t.swapaxes(3, 2) + t.swapaxes(1, 0).swapaxes(3, 2)

    #  -1.0000 * P(a,b) <a,k||c,d> t1(b,k) t1(c,i) t1(d,j) 
    t = -1.0000 * einsum('akcd,bk,ci,dj->abij' ,g[v,o,v,v] ,t1 ,t1 ,t1, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  1.0000 *  <a,b||i,j> 
    T += 1.0000 * einsum('abij->abij' ,g[v,v,o,o], optimize=True)

    return T

