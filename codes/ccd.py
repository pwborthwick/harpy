
'''s - spin orbitals, o - occupied spin orbitals, v - virtual spin orbitals
t1[v,o] - singles amplitudes, t2[v,v,o,o] - doubles amplitudes
l1[o,v] - singles lambda, l2[o,o,v,v] - doubles lambda
'''

def cc_energy(f, g, o, v, t1=None, t2=None, t3=None):

    '''
        COGUS generated level [D] on 28 Jul 2021   
    '''
    from numpy import einsum, swapaxes


    #  -0.50000 *  <i,j||i,j> 
    T = -0.50000 * einsum('ijij->' ,g[o,o,o,o], optimize=True)

    #  0.2500 *  <i,j||a,b> t2(a,b,i,j) 
    T += 0.2500 * einsum('ijab,abij->' ,g[o,o,v,v] ,t2, optimize=True)

    #  1.0000 *  f(i,i) 
    T += 1.0000 * einsum('ii->' ,f[o,o], optimize=True)

    return T

def cc_doubles(f, g, o, v, t1=None, t2=None, t3=None):

    '''
        COGUS generated level [D] on 28 Jul 2021   
    '''
    from numpy import einsum, swapaxes


    #  0.5000 *  <a,b||c,d> t2(c,d,i,j) 
    T = 0.5000 * einsum('abcd,cdij->abij' ,g[v,v,v,v] ,t2, optimize=True)

    #  0.5000 *  <k,l||i,j> t2(a,b,k,l) 
    T += 0.5000 * einsum('klij,abkl->abij' ,g[o,o,o,o] ,t2, optimize=True)

    #  1.0000 * P(i,j) f(k,i) t2(a,b,j,k) 
    t = 1.0000 * einsum('ki,abjk->abij' ,f[o,o] ,t2, optimize=True)
    T += t - t.swapaxes(3, 2)

    #  -1.0000 * P(a,b) f(a,c) t2(b,c,i,j) 
    t = -1.0000 * einsum('ac,bcij->abij' ,f[v,v] ,t2, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  0.2500 *  <k,l||c,d> t2(a,b,k,l) t2(c,d,i,j) 
    T += 0.2500 * einsum('klcd,abkl,cdij->abij' ,g[o,o,v,v] ,t2 ,t2, optimize=True)

    #  1.0000 * P(a,b)P(i,j) <a,k||i,c> t2(b,c,j,k) 
    t = 1.0000 * einsum('akic,bcjk->abij' ,g[v,o,o,v] ,t2, optimize=True)
    T += t - t.swapaxes(1, 0) - t.swapaxes(3, 2) + t.swapaxes(1, 0).swapaxes(3, 2)

    #  1.0000 * P(i,j) <k,l||c,d> t2(a,c,i,k) t2(b,d,j,l) 
    t = 1.0000 * einsum('klcd,acik,bdjl->abij' ,g[o,o,v,v] ,t2 ,t2, optimize=True)
    T += t - t.swapaxes(3, 2)

    #  0.5000 * P(i,j) <k,l||c,d> t2(a,b,i,l) t2(c,d,j,k) 
    t = 0.5000 * einsum('klcd,abil,cdjk->abij' ,g[o,o,v,v] ,t2 ,t2, optimize=True)
    T += t - t.swapaxes(3, 2)

    #  -0.50000 * P(a,b) <k,l||c,d> t2(a,c,k,l) t2(b,d,i,j) 
    t = -0.50000 * einsum('klcd,ackl,bdij->abij' ,g[o,o,v,v] ,t2 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  1.0000 *  <a,b||i,j> 
    T += 1.0000 * einsum('abij->abij' ,g[v,v,o,o], optimize=True)

    return T

