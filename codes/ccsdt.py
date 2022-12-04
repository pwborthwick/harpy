
'''s - spin orbitals, o - occupied spin orbitals, v - virtual spin orbitals
t1[v,o] - singles amplitudes, t2[v,v,o,o] - doubles amplitudes
l1[o,v] - singles lambda, l2[o,o,v,v] - doubles lambda
'''

def cc_energy(f, g, o, v, t1=None, t2=None, t3=None):

    '''
        COGUS generated level [SDT] on 28 Jul 2021   
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
        COGUS generated level [SDT] on 28 Jul 2021   
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

    #  0.2500 *  <j,k||b,c> t3(a,b,c,i,j,k) 
    T += 0.2500 * einsum('jkbc,abcijk->ai' ,g[o,o,v,v] ,t3, optimize=True)

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
        COGUS generated level [SDT] on 28 Jul 2021   
    '''
    from numpy import einsum, swapaxes


    #  1.0000 *  f(k,c) t3(a,b,c,i,j,k) 
    T = 1.0000 * einsum('kc,abcijk->abij' ,f[o,v] ,t3, optimize=True)

    #  0.5000 *  <a,b||c,d> t2(c,d,i,j) 
    T += 0.5000 * einsum('abcd,cdij->abij' ,g[v,v,v,v] ,t2, optimize=True)

    #  0.5000 *  <k,l||i,j> t2(a,b,k,l) 
    T += 0.5000 * einsum('klij,abkl->abij' ,g[o,o,o,o] ,t2, optimize=True)

    #  1.0000 * P(i,j) f(k,i) t2(a,b,j,k) 
    t = 1.0000 * einsum('ki,abjk->abij' ,f[o,o] ,t2, optimize=True)
    T += t - t.swapaxes(3, 2)

    #  1.0000 * P(i,j) <a,b||i,c> t1(c,j) 
    t = 1.0000 * einsum('abic,cj->abij' ,g[v,v,o,v] ,t1, optimize=True)
    T += t - t.swapaxes(3, 2)

    #  1.0000 *  <a,b||c,d> t1(c,i) t1(d,j) 
    T += 1.0000 * einsum('abcd,ci,dj->abij' ,g[v,v,v,v] ,t1 ,t1, optimize=True)

    #  1.0000 *  <k,l||i,j> t1(a,k) t1(b,l) 
    T += 1.0000 * einsum('klij,ak,bl->abij' ,g[o,o,o,o] ,t1 ,t1, optimize=True)

    #  1.0000 *  <k,l||c,d> t1(c,k) t3(a,b,d,i,j,l) 
    T += 1.0000 * einsum('klcd,ck,abdijl->abij' ,g[o,o,v,v] ,t1 ,t3, optimize=True)

    #  0.5000 * P(i,j) <k,l||i,c> t3(a,b,c,j,k,l) 
    t = 0.5000 * einsum('klic,abcjkl->abij' ,g[o,o,o,v] ,t3, optimize=True)
    T += t - t.swapaxes(3, 2)

    #  -1.0000 * P(a,b) f(a,c) t2(b,c,i,j) 
    t = -1.0000 * einsum('ac,bcij->abij' ,f[v,v] ,t2, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  -1.0000 * P(a,b) <a,k||i,j> t1(b,k) 
    t = -1.0000 * einsum('akij,bk->abij' ,g[v,o,o,o] ,t1, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  -0.50000 * P(a,b) <a,k||c,d> t3(b,c,d,i,j,k) 
    t = -0.50000 * einsum('akcd,bcdijk->abij' ,g[v,o,v,v] ,t3, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  0.2500 *  <k,l||c,d> t2(a,b,k,l) t2(c,d,i,j) 
    T += 0.2500 * einsum('klcd,abkl,cdij->abij' ,g[o,o,v,v] ,t2 ,t2, optimize=True)

    #  1.0000 * P(a,b) f(k,c) t1(a,k) t2(b,c,i,j) 
    t = 1.0000 * einsum('kc,ak,bcij->abij' ,f[o,v] ,t1 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  1.0000 * P(i,j) f(k,c) t1(c,i) t2(a,b,j,k) 
    t = 1.0000 * einsum('kc,ci,abjk->abij' ,f[o,v] ,t1 ,t2, optimize=True)
    T += t - t.swapaxes(3, 2)

    #  1.0000 * P(a,b)P(i,j) <a,k||i,c> t2(b,c,j,k) 
    t = 1.0000 * einsum('akic,bcjk->abij' ,g[v,o,o,v] ,t2, optimize=True)
    T += t - t.swapaxes(1, 0) - t.swapaxes(3, 2) + t.swapaxes(1, 0).swapaxes(3, 2)

    #  1.0000 * P(a,b) <a,k||c,d> t1(c,k) t2(b,d,i,j) 
    t = 1.0000 * einsum('akcd,ck,bdij->abij' ,g[v,o,v,v] ,t1 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  1.0000 * P(i,j) <k,l||c,d> t2(a,c,i,k) t2(b,d,j,l) 
    t = 1.0000 * einsum('klcd,acik,bdjl->abij' ,g[o,o,v,v] ,t2 ,t2, optimize=True)
    T += t - t.swapaxes(3, 2)

    #  0.5000 * P(i,j) <k,l||i,c> t1(c,j) t2(a,b,k,l) 
    t = 0.5000 * einsum('klic,cj,abkl->abij' ,g[o,o,o,v] ,t1 ,t2, optimize=True)
    T += t - t.swapaxes(3, 2)

    #  0.5000 *  <k,l||c,d> t1(a,k) t1(b,l) t2(c,d,i,j) 
    T += 0.5000 * einsum('klcd,ak,bl,cdij->abij' ,g[o,o,v,v] ,t1 ,t1 ,t2, optimize=True)

    #  0.5000 * P(a,b) <k,l||c,d> t1(a,k) t3(b,c,d,i,j,l) 
    t = 0.5000 * einsum('klcd,ak,bcdijl->abij' ,g[o,o,v,v] ,t1 ,t3, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  0.5000 *  <k,l||c,d> t1(c,i) t1(d,j) t2(a,b,k,l) 
    T += 0.5000 * einsum('klcd,ci,dj,abkl->abij' ,g[o,o,v,v] ,t1 ,t1 ,t2, optimize=True)

    #  0.5000 * P(i,j) <k,l||c,d> t1(c,i) t3(a,b,d,j,k,l) 
    t = 0.5000 * einsum('klcd,ci,abdjkl->abij' ,g[o,o,v,v] ,t1 ,t3, optimize=True)
    T += t - t.swapaxes(3, 2)

    #  0.5000 * P(i,j) <k,l||c,d> t2(a,b,i,l) t2(c,d,j,k) 
    t = 0.5000 * einsum('klcd,abil,cdjk->abij' ,g[o,o,v,v] ,t2 ,t2, optimize=True)
    T += t - t.swapaxes(3, 2)

    #  -1.0000 * P(i,j) <k,l||i,c> t1(c,k) t2(a,b,j,l) 
    t = -1.0000 * einsum('klic,ck,abjl->abij' ,g[o,o,o,v] ,t1 ,t2, optimize=True)
    T += t - t.swapaxes(3, 2)

    #  -0.50000 * P(a,b) <a,k||c,d> t1(b,k) t2(c,d,i,j) 
    t = -0.50000 * einsum('akcd,bk,cdij->abij' ,g[v,o,v,v] ,t1 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  -0.50000 * P(a,b) <k,l||c,d> t2(a,c,k,l) t2(b,d,i,j) 
    t = -0.50000 * einsum('klcd,ackl,bdij->abij' ,g[o,o,v,v] ,t2 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  1.0000 * P(a,b)P(i,j) <a,k||c,d> t1(c,i) t2(b,d,j,k) 
    t = 1.0000 * einsum('akcd,ci,bdjk->abij' ,g[v,o,v,v] ,t1 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0) - t.swapaxes(3, 2) + t.swapaxes(1, 0).swapaxes(3, 2)

    #  1.0000 * P(i,j) <k,l||i,c> t1(a,k) t1(b,l) t1(c,j) 
    t = 1.0000 * einsum('klic,ak,bl,cj->abij' ,g[o,o,o,v] ,t1 ,t1 ,t1, optimize=True)
    T += t - t.swapaxes(3, 2)

    #  1.0000 *  <k,l||c,d> t1(a,k) t1(b,l) t1(c,i) t1(d,j) 
    T += 1.0000 * einsum('klcd,ak,bl,ci,dj->abij' ,g[o,o,v,v] ,t1 ,t1 ,t1 ,t1, optimize=True)

    #  1.0000 * P(a,b) <k,l||c,d> t1(a,l) t1(c,k) t2(b,d,i,j) 
    t = 1.0000 * einsum('klcd,al,ck,bdij->abij' ,g[o,o,v,v] ,t1 ,t1 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  1.0000 * P(i,j) <k,l||c,d> t1(c,k) t1(d,i) t2(a,b,j,l) 
    t = 1.0000 * einsum('klcd,ck,di,abjl->abij' ,g[o,o,v,v] ,t1 ,t1 ,t2, optimize=True)
    T += t - t.swapaxes(3, 2)

    #  -1.0000 * P(a,b)P(i,j) <a,k||i,c> t1(b,k) t1(c,j) 
    t = -1.0000 * einsum('akic,bk,cj->abij' ,g[v,o,o,v] ,t1 ,t1, optimize=True)
    T += t - t.swapaxes(1, 0) - t.swapaxes(3, 2) + t.swapaxes(1, 0).swapaxes(3, 2)

    #  -1.0000 * P(a,b) <a,k||c,d> t1(b,k) t1(c,i) t1(d,j) 
    t = -1.0000 * einsum('akcd,bk,ci,dj->abij' ,g[v,o,v,v] ,t1 ,t1 ,t1, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  -1.0000 * P(a,b)P(i,j) <k,l||i,c> t1(a,k) t2(b,c,j,l) 
    t = -1.0000 * einsum('klic,ak,bcjl->abij' ,g[o,o,o,v] ,t1 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0) - t.swapaxes(3, 2) + t.swapaxes(1, 0).swapaxes(3, 2)

    #  -1.0000 * P(a,b)P(i,j) <k,l||c,d> t1(a,k) t1(c,i) t2(b,d,j,l) 
    t = -1.0000 * einsum('klcd,ak,ci,bdjl->abij' ,g[o,o,v,v] ,t1 ,t1 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0) - t.swapaxes(3, 2) + t.swapaxes(1, 0).swapaxes(3, 2)

    #  1.0000 *  <a,b||i,j> 
    T += 1.0000 * einsum('abij->abij' ,g[v,v,o,o], optimize=True)

    return T

def cc_triples(f, g, o, v, t1=None, t2=None, t3=None):

    '''
        COGUS generated level [SDT] on 28 Jul 2021   
    '''
    from numpy import einsum, swapaxes


    #  1.0000 *  f(c,d) t3(a,b,d,i,j,k) 
    T = 1.0000 * einsum('cd,abdijk->abcijk' ,f[v,v] ,t3, optimize=True)

    #  1.0000 *  <c,l||i,j> t2(a,b,k,l) 
    T += 1.0000 * einsum('clij,abkl->abcijk' ,g[v,o,o,o] ,t2, optimize=True)

    #  1.0000 *  <c,l||k,d> t3(a,b,d,i,j,l) 
    T += 1.0000 * einsum('clkd,abdijl->abcijk' ,g[v,o,o,v] ,t3, optimize=True)

    #  0.5000 *  <a,b||d,e> t3(c,d,e,i,j,k) 
    T += 0.5000 * einsum('abde,cdeijk->abcijk' ,g[v,v,v,v] ,t3, optimize=True)

    #  0.5000 *  <l,m||i,j> t3(a,b,c,k,l,m) 
    T += 0.5000 * einsum('lmij,abcklm->abcijk' ,g[o,o,o,o] ,t3, optimize=True)

    #  -1.0000 *  f(l,k) t3(a,b,c,i,j,l) 
    T += -1.0000 * einsum('lk,abcijl->abcijk' ,f[o,o] ,t3, optimize=True)

    #  -1.0000 *  <a,b||k,d> t2(c,d,i,j) 
    T += -1.0000 * einsum('abkd,cdij->abcijk' ,g[v,v,o,v] ,t2, optimize=True)

    #  1.0000 * P(a,b) f(a,d) t3(b,c,d,i,j,k) 
    t = 1.0000 * einsum('ad,bcdijk->abcijk' ,f[v,v] ,t3, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  1.0000 *  f(l,d) t2(a,b,k,l) t2(c,d,i,j) 
    T += 1.0000 * einsum('ld,abkl,cdij->abcijk' ,f[o,v] ,t2 ,t2, optimize=True)

    #  1.0000 * P(a,b) <a,c||k,d> t2(b,d,i,j) 
    t = 1.0000 * einsum('ackd,bdij->abcijk' ,g[v,v,o,v] ,t2, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  1.0000 * P(a,b) <a,l||i,j> t2(b,c,k,l) 
    t = 1.0000 * einsum('alij,bckl->abcijk' ,g[v,o,o,o] ,t2, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  1.0000 * P(a,b) <a,l||k,d> t3(b,c,d,i,j,l) 
    t = 1.0000 * einsum('alkd,bcdijl->abcijk' ,g[v,o,o,v] ,t3, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  1.0000 * P(i,j) <c,l||i,d> t3(a,b,d,j,k,l) 
    t = 1.0000 * einsum('clid,abdjkl->abcijk' ,g[v,o,o,v] ,t3, optimize=True)
    T += t - t.swapaxes(4, 3)

    #  1.0000 *  <c,l||d,e> t1(d,k) t3(a,b,e,i,j,l) 
    T += 1.0000 * einsum('clde,dk,abeijl->abcijk' ,g[v,o,v,v] ,t1 ,t3, optimize=True)

    #  1.0000 *  <l,m||k,d> t1(d,l) t3(a,b,c,i,j,m) 
    T += 1.0000 * einsum('lmkd,dl,abcijm->abcijk' ,g[o,o,o,v] ,t1 ,t3, optimize=True)

    #  1.0000 *  <l,m||d,e> t2(c,d,k,l) t3(a,b,e,i,j,m) 
    T += 1.0000 * einsum('lmde,cdkl,abeijm->abcijk' ,g[o,o,v,v] ,t2 ,t3, optimize=True)

    #  0.5000 *  <c,l||d,e> t2(a,b,k,l) t2(d,e,i,j) 
    T += 0.5000 * einsum('clde,abkl,deij->abcijk' ,g[v,o,v,v] ,t2 ,t2, optimize=True)

    #  0.5000 *  <l,m||d,e> t2(a,b,k,l) t3(c,d,e,i,j,m) 
    T += 0.5000 * einsum('lmde,abkl,cdeijm->abcijk' ,g[o,o,v,v] ,t2 ,t3, optimize=True)

    #  0.5000 *  <l,m||d,e> t2(c,d,i,j) t3(a,b,e,k,l,m) 
    T += 0.5000 * einsum('lmde,cdij,abeklm->abcijk' ,g[o,o,v,v] ,t2 ,t3, optimize=True)

    #  0.5000 *  <l,m||d,e> t2(c,d,l,m) t3(a,b,e,i,j,k) 
    T += 0.5000 * einsum('lmde,cdlm,abeijk->abcijk' ,g[o,o,v,v] ,t2 ,t3, optimize=True)

    #  0.5000 *  <l,m||d,e> t2(d,e,k,l) t3(a,b,c,i,j,m) 
    T += 0.5000 * einsum('lmde,dekl,abcijm->abcijk' ,g[o,o,v,v] ,t2 ,t3, optimize=True)

    #  -1.0000 * P(i,j) f(l,i) t3(a,b,c,j,k,l) 
    t = -1.0000 * einsum('li,abcjkl->abcijk' ,f[o,o] ,t3, optimize=True)
    T += t - t.swapaxes(4, 3)

    #  -1.0000 *  f(l,d) t1(c,l) t3(a,b,d,i,j,k) 
    T += -1.0000 * einsum('ld,cl,abdijk->abcijk' ,f[o,v] ,t1 ,t3, optimize=True)

    #  -1.0000 *  f(l,d) t1(d,k) t3(a,b,c,i,j,l) 
    T += -1.0000 * einsum('ld,dk,abcijl->abcijk' ,f[o,v] ,t1 ,t3, optimize=True)

    #  -1.0000 * P(i,j) <a,b||i,d> t2(c,d,j,k) 
    t = -1.0000 * einsum('abid,cdjk->abcijk' ,g[v,v,o,v] ,t2, optimize=True)
    T += t - t.swapaxes(4, 3)

    #  -1.0000 *  <a,b||d,e> t1(d,k) t2(c,e,i,j) 
    T += -1.0000 * einsum('abde,dk,ceij->abcijk' ,g[v,v,v,v] ,t1 ,t2, optimize=True)

    #  -1.0000 * P(i,j) <c,l||i,k> t2(a,b,j,l) 
    t = -1.0000 * einsum('clik,abjl->abcijk' ,g[v,o,o,o] ,t2, optimize=True)
    T += t - t.swapaxes(4, 3)

    #  -1.0000 *  <c,l||d,e> t1(d,l) t3(a,b,e,i,j,k) 
    T += -1.0000 * einsum('clde,dl,abeijk->abcijk' ,g[v,o,v,v] ,t1 ,t3, optimize=True)

    #  -1.0000 *  <l,m||i,j> t1(c,l) t2(a,b,k,m) 
    T += -1.0000 * einsum('lmij,cl,abkm->abcijk' ,g[o,o,o,o] ,t1 ,t2, optimize=True)

    #  -1.0000 *  <l,m||k,d> t1(c,l) t3(a,b,d,i,j,m) 
    T += -1.0000 * einsum('lmkd,cl,abdijm->abcijk' ,g[o,o,o,v] ,t1 ,t3, optimize=True)

    #  -0.50000 * P(a,b) <a,c||d,e> t3(b,d,e,i,j,k) 
    t = -0.50000 * einsum('acde,bdeijk->abcijk' ,g[v,v,v,v] ,t3, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  -0.50000 * P(i,j) <l,m||i,k> t3(a,b,c,j,l,m) 
    t = -0.50000 * einsum('lmik,abcjlm->abcijk' ,g[o,o,o,o] ,t3, optimize=True)
    T += t - t.swapaxes(4, 3)

    #  -0.50000 *  <l,m||k,d> t2(a,b,l,m) t2(c,d,i,j) 
    T += -0.50000 * einsum('lmkd,ablm,cdij->abcijk' ,g[o,o,o,v] ,t2 ,t2, optimize=True)

    #  0.2500 *  <l,m||d,e> t2(a,b,l,m) t3(c,d,e,i,j,k) 
    T += 0.2500 * einsum('lmde,ablm,cdeijk->abcijk' ,g[o,o,v,v] ,t2 ,t3, optimize=True)

    #  0.2500 *  <l,m||d,e> t2(d,e,i,j) t3(a,b,c,k,l,m) 
    T += 0.2500 * einsum('lmde,deij,abcklm->abcijk' ,g[o,o,v,v] ,t2 ,t3, optimize=True)

    #  1.0000 * P(a,b) f(l,d) t2(a,d,i,j) t2(b,c,k,l) 
    t = 1.0000 * einsum('ld,adij,bckl->abcijk' ,f[o,v] ,t2 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  1.0000 * P(a,b)P(i,j) <a,c||i,d> t2(b,d,j,k) 
    t = 1.0000 * einsum('acid,bdjk->abcijk' ,g[v,v,o,v] ,t2, optimize=True)
    T += t - t.swapaxes(1, 0) - t.swapaxes(4, 3) + t.swapaxes(1, 0).swapaxes(4, 3)

    #  1.0000 * P(a,b) <a,c||d,e> t1(d,k) t2(b,e,i,j) 
    t = 1.0000 * einsum('acde,dk,beij->abcijk' ,g[v,v,v,v] ,t1 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  1.0000 * P(a,b)P(i,j) <a,l||i,d> t3(b,c,d,j,k,l) 
    t = 1.0000 * einsum('alid,bcdjkl->abcijk' ,g[v,o,o,v] ,t3, optimize=True)
    T += t - t.swapaxes(1, 0) - t.swapaxes(4, 3) + t.swapaxes(1, 0).swapaxes(4, 3)

    #  1.0000 * P(a,b) <a,l||k,d> t1(b,l) t2(c,d,i,j) 
    t = 1.0000 * einsum('alkd,bl,cdij->abcijk' ,g[v,o,o,v] ,t1 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  1.0000 * P(a,b) <a,l||d,e> t1(d,k) t3(b,c,e,i,j,l) 
    t = 1.0000 * einsum('alde,dk,bceijl->abcijk' ,g[v,o,v,v] ,t1 ,t3, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  1.0000 * P(i,j) <c,l||i,d> t1(d,j) t2(a,b,k,l) 
    t = 1.0000 * einsum('clid,dj,abkl->abcijk' ,g[v,o,o,v] ,t1 ,t2, optimize=True)
    T += t - t.swapaxes(4, 3)

    #  1.0000 * P(a,b) <c,l||k,d> t1(a,l) t2(b,d,i,j) 
    t = 1.0000 * einsum('clkd,al,bdij->abcijk' ,g[v,o,o,v] ,t1 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  1.0000 * P(i,j) <c,l||k,d> t1(d,i) t2(a,b,j,l) 
    t = 1.0000 * einsum('clkd,di,abjl->abcijk' ,g[v,o,o,v] ,t1 ,t2, optimize=True)
    T += t - t.swapaxes(4, 3)

    #  1.0000 *  <c,l||d,e> t1(d,i) t1(e,j) t2(a,b,k,l) 
    T += 1.0000 * einsum('clde,di,ej,abkl->abcijk' ,g[v,o,v,v] ,t1 ,t1 ,t2, optimize=True)

    #  1.0000 * P(i,j) <c,l||d,e> t1(d,i) t3(a,b,e,j,k,l) 
    t = 1.0000 * einsum('clde,di,abejkl->abcijk' ,g[v,o,v,v] ,t1 ,t3, optimize=True)
    T += t - t.swapaxes(4, 3)

    #  1.0000 * P(i,j) <c,l||d,e> t2(a,d,i,k) t2(b,e,j,l) 
    t = 1.0000 * einsum('clde,adik,bejl->abcijk' ,g[v,o,v,v] ,t2 ,t2, optimize=True)
    T += t - t.swapaxes(4, 3)

    #  1.0000 * P(i,j) <l,m||i,k> t1(c,l) t2(a,b,j,m) 
    t = 1.0000 * einsum('lmik,cl,abjm->abcijk' ,g[o,o,o,o] ,t1 ,t2, optimize=True)
    T += t - t.swapaxes(4, 3)

    #  1.0000 * P(i,j) <l,m||i,d> t1(d,l) t3(a,b,c,j,k,m) 
    t = 1.0000 * einsum('lmid,dl,abcjkm->abcijk' ,g[o,o,o,v] ,t1 ,t3, optimize=True)
    T += t - t.swapaxes(4, 3)

    #  1.0000 * P(i,j) <l,m||i,d> t2(a,b,j,l) t2(c,d,k,m) 
    t = 1.0000 * einsum('lmid,abjl,cdkm->abcijk' ,g[o,o,o,v] ,t2 ,t2, optimize=True)
    T += t - t.swapaxes(4, 3)

    #  1.0000 * P(i,j) <l,m||k,d> t2(a,d,i,l) t2(b,c,j,m) 
    t = 1.0000 * einsum('lmkd,adil,bcjm->abcijk' ,g[o,o,o,v] ,t2 ,t2, optimize=True)
    T += t - t.swapaxes(4, 3)

    #  1.0000 *  <l,m||d,e> t1(d,l) t2(a,b,k,m) t2(c,e,i,j) 
    T += 1.0000 * einsum('lmde,dl,abkm,ceij->abcijk' ,g[o,o,v,v] ,t1 ,t2 ,t2, optimize=True)

    #  1.0000 * P(a,b) <l,m||d,e> t2(a,d,k,l) t3(b,c,e,i,j,m) 
    t = 1.0000 * einsum('lmde,adkl,bceijm->abcijk' ,g[o,o,v,v] ,t2 ,t3, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  1.0000 * P(i,j) <l,m||d,e> t2(c,d,i,l) t3(a,b,e,j,k,m) 
    t = 1.0000 * einsum('lmde,cdil,abejkm->abcijk' ,g[o,o,v,v] ,t2 ,t3, optimize=True)
    T += t - t.swapaxes(4, 3)

    #  0.5000 * P(a,b) <a,l||d,e> t1(c,l) t3(b,d,e,i,j,k) 
    t = 0.5000 * einsum('alde,cl,bdeijk->abcijk' ,g[v,o,v,v] ,t1 ,t3, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  0.5000 * P(a,b) <a,l||d,e> t2(b,c,k,l) t2(d,e,i,j) 
    t = 0.5000 * einsum('alde,bckl,deij->abcijk' ,g[v,o,v,v] ,t2 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  0.5000 * P(i,j) <c,l||d,e> t2(a,b,i,l) t2(d,e,j,k) 
    t = 0.5000 * einsum('clde,abil,dejk->abcijk' ,g[v,o,v,v] ,t2 ,t2, optimize=True)
    T += t - t.swapaxes(4, 3)

    #  0.5000 * P(i,j) <l,m||i,d> t1(d,j) t3(a,b,c,k,l,m) 
    t = 0.5000 * einsum('lmid,dj,abcklm->abcijk' ,g[o,o,o,v] ,t1 ,t3, optimize=True)
    T += t - t.swapaxes(4, 3)

    #  0.5000 * P(i,j) <l,m||k,d> t1(d,i) t3(a,b,c,j,l,m) 
    t = 0.5000 * einsum('lmkd,di,abcjlm->abcijk' ,g[o,o,o,v] ,t1 ,t3, optimize=True)
    T += t - t.swapaxes(4, 3)

    #  0.5000 *  <l,m||d,e> t1(a,l) t1(b,m) t3(c,d,e,i,j,k) 
    T += 0.5000 * einsum('lmde,al,bm,cdeijk->abcijk' ,g[o,o,v,v] ,t1 ,t1 ,t3, optimize=True)

    #  0.5000 *  <l,m||d,e> t1(d,i) t1(e,j) t3(a,b,c,k,l,m) 
    T += 0.5000 * einsum('lmde,di,ej,abcklm->abcijk' ,g[o,o,v,v] ,t1 ,t1 ,t3, optimize=True)

    #  0.5000 * P(i,j) <l,m||d,e> t2(a,b,i,l) t3(c,d,e,j,k,m) 
    t = 0.5000 * einsum('lmde,abil,cdejkm->abcijk' ,g[o,o,v,v] ,t2 ,t3, optimize=True)
    T += t - t.swapaxes(4, 3)

    #  0.5000 * P(a,b) <l,m||d,e> t2(a,d,i,j) t3(b,c,e,k,l,m) 
    t = 0.5000 * einsum('lmde,adij,bceklm->abcijk' ,g[o,o,v,v] ,t2 ,t3, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  0.5000 * P(a,b) <l,m||d,e> t2(a,d,l,m) t3(b,c,e,i,j,k) 
    t = 0.5000 * einsum('lmde,adlm,bceijk->abcijk' ,g[o,o,v,v] ,t2 ,t3, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  0.5000 * P(i,j) <l,m||d,e> t2(d,e,i,l) t3(a,b,c,j,k,m) 
    t = 0.5000 * einsum('lmde,deil,abcjkm->abcijk' ,g[o,o,v,v] ,t2 ,t3, optimize=True)
    T += t - t.swapaxes(4, 3)

    #  -1.0000 * P(a,b) f(l,d) t1(a,l) t3(b,c,d,i,j,k) 
    t = -1.0000 * einsum('ld,al,bcdijk->abcijk' ,f[o,v] ,t1 ,t3, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  -1.0000 * P(i,j) f(l,d) t1(d,i) t3(a,b,c,j,k,l) 
    t = -1.0000 * einsum('ld,di,abcjkl->abcijk' ,f[o,v] ,t1 ,t3, optimize=True)
    T += t - t.swapaxes(4, 3)

    #  -1.0000 * P(i,j) f(l,d) t2(a,d,i,k) t2(b,c,j,l) 
    t = -1.0000 * einsum('ld,adik,bcjl->abcijk' ,f[o,v] ,t2 ,t2, optimize=True)
    T += t - t.swapaxes(4, 3)

    #  -1.0000 * P(i,j) <a,b||d,e> t1(d,i) t2(c,e,j,k) 
    t = -1.0000 * einsum('abde,di,cejk->abcijk' ,g[v,v,v,v] ,t1 ,t2, optimize=True)
    T += t - t.swapaxes(4, 3)

    #  -1.0000 * P(a,b)P(i,j) <a,l||i,k> t2(b,c,j,l) 
    t = -1.0000 * einsum('alik,bcjl->abcijk' ,g[v,o,o,o] ,t2, optimize=True)
    T += t - t.swapaxes(1, 0) - t.swapaxes(4, 3) + t.swapaxes(1, 0).swapaxes(4, 3)

    #  -1.0000 * P(a,b) <a,l||k,d> t1(c,l) t2(b,d,i,j) 
    t = -1.0000 * einsum('alkd,cl,bdij->abcijk' ,g[v,o,o,v] ,t1 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  -1.0000 * P(a,b) <a,l||d,e> t1(d,l) t3(b,c,e,i,j,k) 
    t = -1.0000 * einsum('alde,dl,bceijk->abcijk' ,g[v,o,v,v] ,t1 ,t3, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  -1.0000 * P(a,b) <a,l||d,e> t2(b,d,i,j) t2(c,e,k,l) 
    t = -1.0000 * einsum('alde,bdij,cekl->abcijk' ,g[v,o,v,v] ,t2 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  -1.0000 * P(a,b) <a,l||d,e> t2(b,d,k,l) t2(c,e,i,j) 
    t = -1.0000 * einsum('alde,bdkl,ceij->abcijk' ,g[v,o,v,v] ,t2 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  -1.0000 * P(i,j) <c,l||i,d> t1(d,k) t2(a,b,j,l) 
    t = -1.0000 * einsum('clid,dk,abjl->abcijk' ,g[v,o,o,v] ,t1 ,t2, optimize=True)
    T += t - t.swapaxes(4, 3)

    #  -1.0000 * P(a,b) <c,l||d,e> t2(a,d,i,j) t2(b,e,k,l) 
    t = -1.0000 * einsum('clde,adij,bekl->abcijk' ,g[v,o,v,v] ,t2 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  -1.0000 * P(i,j) <c,l||d,e> t2(a,d,i,l) t2(b,e,j,k) 
    t = -1.0000 * einsum('clde,adil,bejk->abcijk' ,g[v,o,v,v] ,t2 ,t2, optimize=True)
    T += t - t.swapaxes(4, 3)

    #  -1.0000 * P(a,b) <l,m||i,j> t1(a,l) t2(b,c,k,m) 
    t = -1.0000 * einsum('lmij,al,bckm->abcijk' ,g[o,o,o,o] ,t1 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  -1.0000 * P(i,j) <l,m||i,d> t1(c,l) t3(a,b,d,j,k,m) 
    t = -1.0000 * einsum('lmid,cl,abdjkm->abcijk' ,g[o,o,o,v] ,t1 ,t3, optimize=True)
    T += t - t.swapaxes(4, 3)

    #  -1.0000 * P(i,j) <l,m||i,d> t2(a,b,k,l) t2(c,d,j,m) 
    t = -1.0000 * einsum('lmid,abkl,cdjm->abcijk' ,g[o,o,o,v] ,t2 ,t2, optimize=True)
    T += t - t.swapaxes(4, 3)

    #  -1.0000 *  <l,m||k,d> t1(a,l) t1(b,m) t2(c,d,i,j) 
    T += -1.0000 * einsum('lmkd,al,bm,cdij->abcijk' ,g[o,o,o,v] ,t1 ,t1 ,t2, optimize=True)

    #  -1.0000 * P(a,b) <l,m||k,d> t1(a,l) t3(b,c,d,i,j,m) 
    t = -1.0000 * einsum('lmkd,al,bcdijm->abcijk' ,g[o,o,o,v] ,t1 ,t3, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  -1.0000 *  <l,m||d,e> t1(c,l) t1(d,k) t3(a,b,e,i,j,m) 
    T += -1.0000 * einsum('lmde,cl,dk,abeijm->abcijk' ,g[o,o,v,v] ,t1 ,t1 ,t3, optimize=True)

    #  -1.0000 *  <l,m||d,e> t1(c,m) t1(d,l) t3(a,b,e,i,j,k) 
    T += -1.0000 * einsum('lmde,cm,dl,abeijk->abcijk' ,g[o,o,v,v] ,t1 ,t1 ,t3, optimize=True)

    #  -1.0000 *  <l,m||d,e> t1(d,l) t1(e,k) t3(a,b,c,i,j,m) 
    T += -1.0000 * einsum('lmde,dl,ek,abcijm->abcijk' ,g[o,o,v,v] ,t1 ,t1 ,t3, optimize=True)

    #  -0.50000 * P(a,b) <a,l||d,e> t1(b,l) t3(c,d,e,i,j,k) 
    t = -0.50000 * einsum('alde,bl,cdeijk->abcijk' ,g[v,o,v,v] ,t1 ,t3, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  -0.50000 * P(a,b) <c,l||d,e> t1(a,l) t3(b,d,e,i,j,k) 
    t = -0.50000 * einsum('clde,al,bdeijk->abcijk' ,g[v,o,v,v] ,t1 ,t3, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  -0.50000 * P(i,j) <l,m||i,d> t1(d,k) t3(a,b,c,j,l,m) 
    t = -0.50000 * einsum('lmid,dk,abcjlm->abcijk' ,g[o,o,o,v] ,t1 ,t3, optimize=True)
    T += t - t.swapaxes(4, 3)

    #  -0.50000 * P(i,j) <l,m||i,d> t2(a,b,l,m) t2(c,d,j,k) 
    t = -0.50000 * einsum('lmid,ablm,cdjk->abcijk' ,g[o,o,o,v] ,t2 ,t2, optimize=True)
    T += t - t.swapaxes(4, 3)

    #  -0.50000 * P(a,b) <l,m||k,d> t2(a,d,i,j) t2(b,c,l,m) 
    t = -0.50000 * einsum('lmkd,adij,bclm->abcijk' ,g[o,o,o,v] ,t2 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  -0.50000 *  <l,m||d,e> t1(c,l) t2(a,b,k,m) t2(d,e,i,j) 
    T += -0.50000 * einsum('lmde,cl,abkm,deij->abcijk' ,g[o,o,v,v] ,t1 ,t2 ,t2, optimize=True)

    #  -0.50000 *  <l,m||d,e> t1(d,k) t2(a,b,l,m) t2(c,e,i,j) 
    T += -0.50000 * einsum('lmde,dk,ablm,ceij->abcijk' ,g[o,o,v,v] ,t1 ,t2 ,t2, optimize=True)

    #  -0.50000 * P(a,b) <l,m||d,e> t2(a,c,k,l) t3(b,d,e,i,j,m) 
    t = -0.50000 * einsum('lmde,ackl,bdeijm->abcijk' ,g[o,o,v,v] ,t2 ,t3, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  -0.50000 * P(i,j) <l,m||d,e> t2(c,d,i,k) t3(a,b,e,j,l,m) 
    t = -0.50000 * einsum('lmde,cdik,abejlm->abcijk' ,g[o,o,v,v] ,t2 ,t3, optimize=True)
    T += t - t.swapaxes(4, 3)

    #  -0.25000 * P(a,b) <l,m||d,e> t2(a,c,l,m) t3(b,d,e,i,j,k) 
    t = -0.25000 * einsum('lmde,aclm,bdeijk->abcijk' ,g[o,o,v,v] ,t2 ,t3, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  -0.25000 * P(i,j) <l,m||d,e> t2(d,e,i,k) t3(a,b,c,j,l,m) 
    t = -0.25000 * einsum('lmde,deik,abcjlm->abcijk' ,g[o,o,v,v] ,t2 ,t3, optimize=True)
    T += t - t.swapaxes(4, 3)

    #  1.0000 * P(b,c)P(i,j) f(l,d) t2(a,b,i,l) t2(c,d,j,k) 
    t = 1.0000 * einsum('ld,abil,cdjk->abcijk' ,f[o,v] ,t2 ,t2, optimize=True)
    T += t - t.swapaxes(2, 1) - t.swapaxes(4, 3) + t.swapaxes(2, 1).swapaxes(4, 3)

    #  1.0000 * P(a,b)P(i,j) <a,c||d,e> t1(d,i) t2(b,e,j,k) 
    t = 1.0000 * einsum('acde,di,bejk->abcijk' ,g[v,v,v,v] ,t1 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0) - t.swapaxes(4, 3) + t.swapaxes(1, 0).swapaxes(4, 3)

    #  1.0000 * P(a,b)P(i,j) <a,l||i,d> t1(b,l) t2(c,d,j,k) 
    t = 1.0000 * einsum('alid,bl,cdjk->abcijk' ,g[v,o,o,v] ,t1 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0) - t.swapaxes(4, 3) + t.swapaxes(1, 0).swapaxes(4, 3)

    #  1.0000 * P(a,b)P(i,j) <a,l||i,d> t1(d,j) t2(b,c,k,l) 
    t = 1.0000 * einsum('alid,dj,bckl->abcijk' ,g[v,o,o,v] ,t1 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0) - t.swapaxes(4, 3) + t.swapaxes(1, 0).swapaxes(4, 3)

    #  1.0000 * P(a,b)P(i,j) <a,l||k,d> t1(d,i) t2(b,c,j,l) 
    t = 1.0000 * einsum('alkd,di,bcjl->abcijk' ,g[v,o,o,v] ,t1 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0) - t.swapaxes(4, 3) + t.swapaxes(1, 0).swapaxes(4, 3)

    #  1.0000 * P(a,b) <a,l||d,e> t1(b,l) t1(d,k) t2(c,e,i,j) 
    t = 1.0000 * einsum('alde,bl,dk,ceij->abcijk' ,g[v,o,v,v] ,t1 ,t1 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  1.0000 * P(a,b) <a,l||d,e> t1(d,i) t1(e,j) t2(b,c,k,l) 
    t = 1.0000 * einsum('alde,di,ej,bckl->abcijk' ,g[v,o,v,v] ,t1 ,t1 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  1.0000 * P(a,b)P(i,j) <a,l||d,e> t1(d,i) t3(b,c,e,j,k,l) 
    t = 1.0000 * einsum('alde,di,bcejkl->abcijk' ,g[v,o,v,v] ,t1 ,t3, optimize=True)
    T += t - t.swapaxes(1, 0) - t.swapaxes(4, 3) + t.swapaxes(1, 0).swapaxes(4, 3)

    #  1.0000 * P(a,b)P(i,j) <a,l||d,e> t2(b,d,i,k) t2(c,e,j,l) 
    t = 1.0000 * einsum('alde,bdik,cejl->abcijk' ,g[v,o,v,v] ,t2 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0) - t.swapaxes(4, 3) + t.swapaxes(1, 0).swapaxes(4, 3)

    #  1.0000 * P(a,b)P(i,j) <c,l||i,d> t1(a,l) t2(b,d,j,k) 
    t = 1.0000 * einsum('clid,al,bdjk->abcijk' ,g[v,o,o,v] ,t1 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0) - t.swapaxes(4, 3) + t.swapaxes(1, 0).swapaxes(4, 3)

    #  1.0000 * P(a,b) <c,l||d,e> t1(a,l) t1(d,k) t2(b,e,i,j) 
    t = 1.0000 * einsum('clde,al,dk,beij->abcijk' ,g[v,o,v,v] ,t1 ,t1 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  1.0000 * P(a,b)P(i,j) <l,m||i,k> t1(a,l) t2(b,c,j,m) 
    t = 1.0000 * einsum('lmik,al,bcjm->abcijk' ,g[o,o,o,o] ,t1 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0) - t.swapaxes(4, 3) + t.swapaxes(1, 0).swapaxes(4, 3)

    #  1.0000 * P(i,j) <l,m||i,d> t1(c,l) t1(d,k) t2(a,b,j,m) 
    t = 1.0000 * einsum('lmid,cl,dk,abjm->abcijk' ,g[o,o,o,v] ,t1 ,t1 ,t2, optimize=True)
    T += t - t.swapaxes(4, 3)

    #  1.0000 * P(a,b)P(i,j) <l,m||i,d> t2(a,d,j,l) t2(b,c,k,m) 
    t = 1.0000 * einsum('lmid,adjl,bckm->abcijk' ,g[o,o,o,v] ,t2 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0) - t.swapaxes(4, 3) + t.swapaxes(1, 0).swapaxes(4, 3)

    #  1.0000 * P(a,b) <l,m||k,d> t1(a,l) t1(c,m) t2(b,d,i,j) 
    t = 1.0000 * einsum('lmkd,al,cm,bdij->abcijk' ,g[o,o,o,v] ,t1 ,t1 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  1.0000 * P(b,c)P(i,j) <l,m||k,d> t2(a,b,i,l) t2(c,d,j,m) 
    t = 1.0000 * einsum('lmkd,abil,cdjm->abcijk' ,g[o,o,o,v] ,t2 ,t2, optimize=True)
    T += t - t.swapaxes(2, 1) - t.swapaxes(4, 3) + t.swapaxes(2, 1).swapaxes(4, 3)

    #  1.0000 * P(a,b) <l,m||d,e> t1(a,l) t2(b,d,i,j) t2(c,e,k,m) 
    t = 1.0000 * einsum('lmde,al,bdij,cekm->abcijk' ,g[o,o,v,v] ,t1 ,t2 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  1.0000 * P(a,b) <l,m||d,e> t1(a,l) t2(b,d,k,m) t2(c,e,i,j) 
    t = 1.0000 * einsum('lmde,al,bdkm,ceij->abcijk' ,g[o,o,v,v] ,t1 ,t2 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  1.0000 * P(i,j) <l,m||d,e> t1(c,l) t2(a,d,i,m) t2(b,e,j,k) 
    t = 1.0000 * einsum('lmde,cl,adim,bejk->abcijk' ,g[o,o,v,v] ,t1 ,t2 ,t2, optimize=True)
    T += t - t.swapaxes(4, 3)

    #  1.0000 * P(a,b) <l,m||d,e> t1(c,l) t2(a,d,k,m) t2(b,e,i,j) 
    t = 1.0000 * einsum('lmde,cl,adkm,beij->abcijk' ,g[o,o,v,v] ,t1 ,t2 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  1.0000 * P(i,j) <l,m||d,e> t1(d,i) t2(a,b,j,l) t2(c,e,k,m) 
    t = 1.0000 * einsum('lmde,di,abjl,cekm->abcijk' ,g[o,o,v,v] ,t1 ,t2 ,t2, optimize=True)
    T += t - t.swapaxes(4, 3)

    #  1.0000 * P(i,j) <l,m||d,e> t1(d,k) t2(a,e,i,l) t2(b,c,j,m) 
    t = 1.0000 * einsum('lmde,dk,aeil,bcjm->abcijk' ,g[o,o,v,v] ,t1 ,t2 ,t2, optimize=True)
    T += t - t.swapaxes(4, 3)

    #  1.0000 * P(a,b)P(i,j) <l,m||d,e> t2(a,d,i,l) t3(b,c,e,j,k,m) 
    t = 1.0000 * einsum('lmde,adil,bcejkm->abcijk' ,g[o,o,v,v] ,t2 ,t3, optimize=True)
    T += t - t.swapaxes(1, 0) - t.swapaxes(4, 3) + t.swapaxes(1, 0).swapaxes(4, 3)

    #  0.5000 * P(a,b)P(i,j) <a,l||d,e> t2(b,c,i,l) t2(d,e,j,k) 
    t = 0.5000 * einsum('alde,bcil,dejk->abcijk' ,g[v,o,v,v] ,t2 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0) - t.swapaxes(4, 3) + t.swapaxes(1, 0).swapaxes(4, 3)

    #  -1.0000 * P(a,b)P(i,j) <a,l||i,d> t1(c,l) t2(b,d,j,k) 
    t = -1.0000 * einsum('alid,cl,bdjk->abcijk' ,g[v,o,o,v] ,t1 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0) - t.swapaxes(4, 3) + t.swapaxes(1, 0).swapaxes(4, 3)

    #  -1.0000 * P(a,b)P(i,j) <a,l||i,d> t1(d,k) t2(b,c,j,l) 
    t = -1.0000 * einsum('alid,dk,bcjl->abcijk' ,g[v,o,o,v] ,t1 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0) - t.swapaxes(4, 3) + t.swapaxes(1, 0).swapaxes(4, 3)

    #  -1.0000 * P(a,b) <a,l||d,e> t1(c,l) t1(d,k) t2(b,e,i,j) 
    t = -1.0000 * einsum('alde,cl,dk,beij->abcijk' ,g[v,o,v,v] ,t1 ,t1 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  -1.0000 * P(a,b)P(i,j) <a,l||d,e> t2(b,d,i,l) t2(c,e,j,k) 
    t = -1.0000 * einsum('alde,bdil,cejk->abcijk' ,g[v,o,v,v] ,t2 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0) - t.swapaxes(4, 3) + t.swapaxes(1, 0).swapaxes(4, 3)

    #  -1.0000 * P(i,j) <c,l||d,e> t1(d,i) t1(e,k) t2(a,b,j,l) 
    t = -1.0000 * einsum('clde,di,ek,abjl->abcijk' ,g[v,o,v,v] ,t1 ,t1 ,t2, optimize=True)
    T += t - t.swapaxes(4, 3)

    #  -1.0000 * P(i,j) <l,m||i,d> t1(a,l) t1(b,m) t2(c,d,j,k) 
    t = -1.0000 * einsum('lmid,al,bm,cdjk->abcijk' ,g[o,o,o,v] ,t1 ,t1 ,t2, optimize=True)
    T += t - t.swapaxes(4, 3)

    #  -1.0000 * P(a,b)P(i,j) <l,m||i,d> t1(a,l) t3(b,c,d,j,k,m) 
    t = -1.0000 * einsum('lmid,al,bcdjkm->abcijk' ,g[o,o,o,v] ,t1 ,t3, optimize=True)
    T += t - t.swapaxes(1, 0) - t.swapaxes(4, 3) + t.swapaxes(1, 0).swapaxes(4, 3)

    #  -1.0000 * P(i,j) <l,m||i,d> t1(c,l) t1(d,j) t2(a,b,k,m) 
    t = -1.0000 * einsum('lmid,cl,dj,abkm->abcijk' ,g[o,o,o,v] ,t1 ,t1 ,t2, optimize=True)
    T += t - t.swapaxes(4, 3)

    #  -1.0000 * P(a,b)P(i,j) <l,m||i,d> t2(a,d,k,l) t2(b,c,j,m) 
    t = -1.0000 * einsum('lmid,adkl,bcjm->abcijk' ,g[o,o,o,v] ,t2 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0) - t.swapaxes(4, 3) + t.swapaxes(1, 0).swapaxes(4, 3)

    #  -1.0000 * P(i,j) <l,m||k,d> t1(c,l) t1(d,i) t2(a,b,j,m) 
    t = -1.0000 * einsum('lmkd,cl,di,abjm->abcijk' ,g[o,o,o,v] ,t1 ,t1 ,t2, optimize=True)
    T += t - t.swapaxes(4, 3)

    #  -1.0000 *  <l,m||d,e> t1(a,l) t1(b,m) t1(d,k) t2(c,e,i,j) 
    T += -1.0000 * einsum('lmde,al,bm,dk,ceij->abcijk' ,g[o,o,v,v] ,t1 ,t1 ,t1 ,t2, optimize=True)

    #  -1.0000 * P(a,b) <l,m||d,e> t1(a,l) t1(d,k) t3(b,c,e,i,j,m) 
    t = -1.0000 * einsum('lmde,al,dk,bceijm->abcijk' ,g[o,o,v,v] ,t1 ,t1 ,t3, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  -1.0000 * P(a,b) <l,m||d,e> t1(a,m) t1(d,l) t3(b,c,e,i,j,k) 
    t = -1.0000 * einsum('lmde,am,dl,bceijk->abcijk' ,g[o,o,v,v] ,t1 ,t1 ,t3, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  -1.0000 *  <l,m||d,e> t1(c,l) t1(d,i) t1(e,j) t2(a,b,k,m) 
    T += -1.0000 * einsum('lmde,cl,di,ej,abkm->abcijk' ,g[o,o,v,v] ,t1 ,t1 ,t1 ,t2, optimize=True)

    #  -1.0000 * P(i,j) <l,m||d,e> t1(c,l) t1(d,i) t3(a,b,e,j,k,m) 
    t = -1.0000 * einsum('lmde,cl,di,abejkm->abcijk' ,g[o,o,v,v] ,t1 ,t1 ,t3, optimize=True)
    T += t - t.swapaxes(4, 3)

    #  -1.0000 * P(i,j) <l,m||d,e> t1(c,l) t2(a,d,i,k) t2(b,e,j,m) 
    t = -1.0000 * einsum('lmde,cl,adik,bejm->abcijk' ,g[o,o,v,v] ,t1 ,t2 ,t2, optimize=True)
    T += t - t.swapaxes(4, 3)

    #  -1.0000 * P(i,j) <l,m||d,e> t1(d,i) t2(a,b,k,l) t2(c,e,j,m) 
    t = -1.0000 * einsum('lmde,di,abkl,cejm->abcijk' ,g[o,o,v,v] ,t1 ,t2 ,t2, optimize=True)
    T += t - t.swapaxes(4, 3)

    #  -1.0000 * P(i,j) <l,m||d,e> t1(d,l) t1(e,i) t3(a,b,c,j,k,m) 
    t = -1.0000 * einsum('lmde,dl,ei,abcjkm->abcijk' ,g[o,o,v,v] ,t1 ,t1 ,t3, optimize=True)
    T += t - t.swapaxes(4, 3)

    #  -1.0000 * P(a,b) <l,m||d,e> t1(d,l) t2(a,c,k,m) t2(b,e,i,j) 
    t = -1.0000 * einsum('lmde,dl,ackm,beij->abcijk' ,g[o,o,v,v] ,t1 ,t2 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  -1.0000 * P(i,j) <l,m||d,e> t1(d,l) t2(a,e,i,k) t2(b,c,j,m) 
    t = -1.0000 * einsum('lmde,dl,aeik,bcjm->abcijk' ,g[o,o,v,v] ,t1 ,t2 ,t2, optimize=True)
    T += t - t.swapaxes(4, 3)

    #  -0.50000 * P(a,b)P(i,j) <l,m||i,d> t2(a,d,j,k) t2(b,c,l,m) 
    t = -0.50000 * einsum('lmid,adjk,bclm->abcijk' ,g[o,o,o,v] ,t2 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0) - t.swapaxes(4, 3) + t.swapaxes(1, 0).swapaxes(4, 3)

    #  -0.50000 * P(a,b) <l,m||d,e> t1(a,l) t1(c,m) t3(b,d,e,i,j,k) 
    t = -0.50000 * einsum('lmde,al,cm,bdeijk->abcijk' ,g[o,o,v,v] ,t1 ,t1 ,t3, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  -0.50000 * P(a,b) <l,m||d,e> t1(a,l) t2(b,c,k,m) t2(d,e,i,j) 
    t = -0.50000 * einsum('lmde,al,bckm,deij->abcijk' ,g[o,o,v,v] ,t1 ,t2 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  -0.50000 * P(i,j) <l,m||d,e> t1(c,l) t2(a,b,i,m) t2(d,e,j,k) 
    t = -0.50000 * einsum('lmde,cl,abim,dejk->abcijk' ,g[o,o,v,v] ,t1 ,t2 ,t2, optimize=True)
    T += t - t.swapaxes(4, 3)

    #  -0.50000 * P(i,j) <l,m||d,e> t1(d,i) t1(e,k) t3(a,b,c,j,l,m) 
    t = -0.50000 * einsum('lmde,di,ek,abcjlm->abcijk' ,g[o,o,v,v] ,t1 ,t1 ,t3, optimize=True)
    T += t - t.swapaxes(4, 3)

    #  -0.50000 * P(i,j) <l,m||d,e> t1(d,i) t2(a,b,l,m) t2(c,e,j,k) 
    t = -0.50000 * einsum('lmde,di,ablm,cejk->abcijk' ,g[o,o,v,v] ,t1 ,t2 ,t2, optimize=True)
    T += t - t.swapaxes(4, 3)

    #  -0.50000 * P(a,b) <l,m||d,e> t1(d,k) t2(a,e,i,j) t2(b,c,l,m) 
    t = -0.50000 * einsum('lmde,dk,aeij,bclm->abcijk' ,g[o,o,v,v] ,t1 ,t2 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  -0.50000 * P(a,b)P(i,j) <l,m||d,e> t2(a,c,i,l) t3(b,d,e,j,k,m) 
    t = -0.50000 * einsum('lmde,acil,bdejkm->abcijk' ,g[o,o,v,v] ,t2 ,t3, optimize=True)
    T += t - t.swapaxes(1, 0) - t.swapaxes(4, 3) + t.swapaxes(1, 0).swapaxes(4, 3)

    #  -0.50000 * P(a,b)P(i,j) <l,m||d,e> t2(a,d,i,k) t3(b,c,e,j,l,m) 
    t = -0.50000 * einsum('lmde,adik,bcejlm->abcijk' ,g[o,o,v,v] ,t2 ,t3, optimize=True)
    T += t - t.swapaxes(1, 0) - t.swapaxes(4, 3) + t.swapaxes(1, 0).swapaxes(4, 3)

    #  1.0000 * P(a,b)P(i,j) <a,l||d,e> t1(b,l) t1(d,i) t2(c,e,j,k) 
    t = 1.0000 * einsum('alde,bl,di,cejk->abcijk' ,g[v,o,v,v] ,t1 ,t1 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0) - t.swapaxes(4, 3) + t.swapaxes(1, 0).swapaxes(4, 3)

    #  1.0000 * P(a,b)P(i,j) <c,l||d,e> t1(a,l) t1(d,i) t2(b,e,j,k) 
    t = 1.0000 * einsum('clde,al,di,bejk->abcijk' ,g[v,o,v,v] ,t1 ,t1 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0) - t.swapaxes(4, 3) + t.swapaxes(1, 0).swapaxes(4, 3)

    #  1.0000 * P(a,b)P(i,j) <l,m||i,d> t1(a,l) t1(c,m) t2(b,d,j,k) 
    t = 1.0000 * einsum('lmid,al,cm,bdjk->abcijk' ,g[o,o,o,v] ,t1 ,t1 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0) - t.swapaxes(4, 3) + t.swapaxes(1, 0).swapaxes(4, 3)

    #  1.0000 * P(a,b)P(i,j) <l,m||i,d> t1(a,l) t1(d,k) t2(b,c,j,m) 
    t = 1.0000 * einsum('lmid,al,dk,bcjm->abcijk' ,g[o,o,o,v] ,t1 ,t1 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0) - t.swapaxes(4, 3) + t.swapaxes(1, 0).swapaxes(4, 3)

    #  1.0000 * P(a,b) <l,m||d,e> t1(a,l) t1(c,m) t1(d,k) t2(b,e,i,j) 
    t = 1.0000 * einsum('lmde,al,cm,dk,beij->abcijk' ,g[o,o,v,v] ,t1 ,t1 ,t1 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  1.0000 * P(a,b)P(i,j) <l,m||d,e> t1(a,l) t2(b,d,i,m) t2(c,e,j,k) 
    t = 1.0000 * einsum('lmde,al,bdim,cejk->abcijk' ,g[o,o,v,v] ,t1 ,t2 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0) - t.swapaxes(4, 3) + t.swapaxes(1, 0).swapaxes(4, 3)

    #  1.0000 * P(i,j) <l,m||d,e> t1(c,l) t1(d,i) t1(e,k) t2(a,b,j,m) 
    t = 1.0000 * einsum('lmde,cl,di,ek,abjm->abcijk' ,g[o,o,v,v] ,t1 ,t1 ,t1 ,t2, optimize=True)
    T += t - t.swapaxes(4, 3)

    #  1.0000 * P(a,b)P(i,j) <l,m||d,e> t1(d,i) t2(a,e,j,l) t2(b,c,k,m) 
    t = 1.0000 * einsum('lmde,di,aejl,bckm->abcijk' ,g[o,o,v,v] ,t1 ,t2 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0) - t.swapaxes(4, 3) + t.swapaxes(1, 0).swapaxes(4, 3)

    #  1.0000 * P(b,c)P(i,j) <l,m||d,e> t1(d,k) t2(a,b,i,l) t2(c,e,j,m) 
    t = 1.0000 * einsum('lmde,dk,abil,cejm->abcijk' ,g[o,o,v,v] ,t1 ,t2 ,t2, optimize=True)
    T += t - t.swapaxes(2, 1) - t.swapaxes(4, 3) + t.swapaxes(2, 1).swapaxes(4, 3)

    #  1.0000 * P(b,c)P(i,j) <l,m||d,e> t1(d,l) t2(a,b,i,m) t2(c,e,j,k) 
    t = 1.0000 * einsum('lmde,dl,abim,cejk->abcijk' ,g[o,o,v,v] ,t1 ,t2 ,t2, optimize=True)
    T += t - t.swapaxes(2, 1) - t.swapaxes(4, 3) + t.swapaxes(2, 1).swapaxes(4, 3)

    #  -1.0000 * P(a,b)P(i,j) <a,l||d,e> t1(c,l) t1(d,i) t2(b,e,j,k) 
    t = -1.0000 * einsum('alde,cl,di,bejk->abcijk' ,g[v,o,v,v] ,t1 ,t1 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0) - t.swapaxes(4, 3) + t.swapaxes(1, 0).swapaxes(4, 3)

    #  -1.0000 * P(a,b)P(i,j) <a,l||d,e> t1(d,i) t1(e,k) t2(b,c,j,l) 
    t = -1.0000 * einsum('alde,di,ek,bcjl->abcijk' ,g[v,o,v,v] ,t1 ,t1 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0) - t.swapaxes(4, 3) + t.swapaxes(1, 0).swapaxes(4, 3)

    #  -1.0000 * P(a,b)P(i,j) <l,m||i,d> t1(a,l) t1(d,j) t2(b,c,k,m) 
    t = -1.0000 * einsum('lmid,al,dj,bckm->abcijk' ,g[o,o,o,v] ,t1 ,t1 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0) - t.swapaxes(4, 3) + t.swapaxes(1, 0).swapaxes(4, 3)

    #  -1.0000 * P(a,b)P(i,j) <l,m||k,d> t1(a,l) t1(d,i) t2(b,c,j,m) 
    t = -1.0000 * einsum('lmkd,al,di,bcjm->abcijk' ,g[o,o,o,v] ,t1 ,t1 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0) - t.swapaxes(4, 3) + t.swapaxes(1, 0).swapaxes(4, 3)

    #  -1.0000 * P(i,j) <l,m||d,e> t1(a,l) t1(b,m) t1(d,i) t2(c,e,j,k) 
    t = -1.0000 * einsum('lmde,al,bm,di,cejk->abcijk' ,g[o,o,v,v] ,t1 ,t1 ,t1 ,t2, optimize=True)
    T += t - t.swapaxes(4, 3)

    #  -1.0000 * P(a,b) <l,m||d,e> t1(a,l) t1(d,i) t1(e,j) t2(b,c,k,m) 
    t = -1.0000 * einsum('lmde,al,di,ej,bckm->abcijk' ,g[o,o,v,v] ,t1 ,t1 ,t1 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  -1.0000 * P(a,b)P(i,j) <l,m||d,e> t1(a,l) t1(d,i) t3(b,c,e,j,k,m) 
    t = -1.0000 * einsum('lmde,al,di,bcejkm->abcijk' ,g[o,o,v,v] ,t1 ,t1 ,t3, optimize=True)
    T += t - t.swapaxes(1, 0) - t.swapaxes(4, 3) + t.swapaxes(1, 0).swapaxes(4, 3)

    #  -1.0000 * P(a,b)P(i,j) <l,m||d,e> t1(a,l) t2(b,d,i,k) t2(c,e,j,m) 
    t = -1.0000 * einsum('lmde,al,bdik,cejm->abcijk' ,g[o,o,v,v] ,t1 ,t2 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0) - t.swapaxes(4, 3) + t.swapaxes(1, 0).swapaxes(4, 3)

    #  -1.0000 * P(a,b)P(i,j) <l,m||d,e> t1(d,i) t2(a,e,k,l) t2(b,c,j,m) 
    t = -1.0000 * einsum('lmde,di,aekl,bcjm->abcijk' ,g[o,o,v,v] ,t1 ,t2 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0) - t.swapaxes(4, 3) + t.swapaxes(1, 0).swapaxes(4, 3)

    #  -0.50000 * P(a,b)P(i,j) <l,m||d,e> t1(a,l) t2(b,c,i,m) t2(d,e,j,k) 
    t = -0.50000 * einsum('lmde,al,bcim,dejk->abcijk' ,g[o,o,v,v] ,t1 ,t2 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0) - t.swapaxes(4, 3) + t.swapaxes(1, 0).swapaxes(4, 3)

    #  -0.50000 * P(a,b)P(i,j) <l,m||d,e> t1(d,i) t2(a,e,j,k) t2(b,c,l,m) 
    t = -0.50000 * einsum('lmde,di,aejk,bclm->abcijk' ,g[o,o,v,v] ,t1 ,t2 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0) - t.swapaxes(4, 3) + t.swapaxes(1, 0).swapaxes(4, 3)

    #  1.0000 * P(a,b)P(i,j) <l,m||d,e> t1(a,l) t1(c,m) t1(d,i) t2(b,e,j,k) 
    t = 1.0000 * einsum('lmde,al,cm,di,bejk->abcijk' ,g[o,o,v,v] ,t1 ,t1 ,t1 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0) - t.swapaxes(4, 3) + t.swapaxes(1, 0).swapaxes(4, 3)

    #  1.0000 * P(a,b)P(i,j) <l,m||d,e> t1(a,l) t1(d,i) t1(e,k) t2(b,c,j,m) 
    t = 1.0000 * einsum('lmde,al,di,ek,bcjm->abcijk' ,g[o,o,v,v] ,t1 ,t1 ,t1 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0) - t.swapaxes(4, 3) + t.swapaxes(1, 0).swapaxes(4, 3)

    return T

