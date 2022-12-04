
'''s - spin orbitals, o - occupied spin orbitals, v - virtual spin orbitals
t1[v,o] - singles amplitudes, t2[v,v,o,o] - doubles amplitudes
l1[o,v] - singles lambda, l2[o,o,v,v] - doubles lambda
'''


def cc_lambda_lagrangian_energy(f, g, o, v, t1=None, t2=None, l1=None, l2=None):

    '''
        COGUS generated level [SD] on 29 Jul 2021   
    '''

    from numpy import einsum

    T =  cc_energy(f, g, o, v, t1, t2, t3=None)

    T += einsum('ia,ai->', l1, cc_singles(f, g, o, v, t1, t2, t3=None))

    T += einsum('ijab,abij->', l2, cc_doubles(f, g, o, v, t1, t2, t3=None))


    return T

def cc_lambda_singles(f, g, o, v, t1=None, t2=None, l1=None, l2=None):

    '''
        COGUS generated level [SD] on 29 Jul 2021   
    '''
    from numpy import einsum, swapaxes


    #  1.0000 *  f(b,a) l1(i,b) 
    T = 1.0000 * einsum('ba,ib->ia' ,f[v,v] ,l1, optimize=True)

    #  1.0000 *  <i,b||a,j> l1(j,b) 
    T += 1.0000 * einsum('ibaj,jb->ia' ,g[o,v,v,o] ,l1, optimize=True)

    #  1.0000 *  <i,j||a,b> t1(b,j) 
    T += 1.0000 * einsum('ijab,bj->ia' ,g[o,o,v,v] ,t1, optimize=True)

    #  0.5000 *  <b,c||a,j> l2(i,j,b,c) 
    T += 0.5000 * einsum('bcaj,ijbc->ia' ,g[v,v,v,o] ,l2, optimize=True)

    #  -1.0000 *  f(i,j) l1(j,a) 
    T += -1.0000 * einsum('ij,ja->ia' ,f[o,o] ,l1, optimize=True)

    #  -0.50000 *  <i,b||j,k> l2(j,k,a,b) 
    T += -0.50000 * einsum('ibjk,jkab->ia' ,g[o,v,o,o] ,l2, optimize=True)

    #  1.0000 *  <i,b||a,c> l1(j,b) t1(c,j) 
    T += 1.0000 * einsum('ibac,jb,cj->ia' ,g[o,v,v,v] ,l1 ,t1, optimize=True)

    #  1.0000 *  <i,b||c,j> l2(j,k,a,b) t1(c,k) 
    T += 1.0000 * einsum('ibcj,jkab,ck->ia' ,g[o,v,v,o] ,l2 ,t1, optimize=True)

    #  1.0000 *  <i,j||a,b> l1(k,c) t2(b,c,j,k) 
    T += 1.0000 * einsum('ijab,kc,bcjk->ia' ,g[o,o,v,v] ,l1 ,t2, optimize=True)

    #  1.0000 *  <i,k||b,j> l1(j,a) t1(b,k) 
    T += 1.0000 * einsum('ikbj,ja,bk->ia' ,g[o,o,v,o] ,l1 ,t1, optimize=True)

    #  1.0000 *  <i,k||b,j> l2(j,l,a,c) t2(b,c,k,l) 
    T += 1.0000 * einsum('ikbj,jlac,bckl->ia' ,g[o,o,v,o] ,l2 ,t2, optimize=True)

    #  1.0000 *  <b,j||a,c> l1(i,b) t1(c,j) 
    T += 1.0000 * einsum('bjac,ib,cj->ia' ,g[v,o,v,v] ,l1 ,t1, optimize=True)

    #  1.0000 *  <b,j||a,c> l2(i,k,b,d) t2(c,d,j,k) 
    T += 1.0000 * einsum('bjac,ikbd,cdjk->ia' ,g[v,o,v,v] ,l2 ,t2, optimize=True)

    #  0.5000 *  <i,b||a,c> l2(j,k,b,d) t2(c,d,j,k) 
    T += 0.5000 * einsum('ibac,jkbd,cdjk->ia' ,g[o,v,v,v] ,l2 ,t2, optimize=True)

    #  0.5000 *  <i,j||b,c> l1(k,a) t2(b,c,j,k) 
    T += 0.5000 * einsum('ijbc,ka,bcjk->ia' ,g[o,o,v,v] ,l1 ,t2, optimize=True)

    #  0.5000 *  <i,l||j,k> l2(j,k,a,b) t1(b,l) 
    T += 0.5000 * einsum('iljk,jkab,bl->ia' ,g[o,o,o,o] ,l2 ,t1, optimize=True)

    #  0.5000 *  <b,c||a,d> l2(i,j,b,c) t1(d,j) 
    T += 0.5000 * einsum('bcad,ijbc,dj->ia' ,g[v,v,v,v] ,l2 ,t1, optimize=True)

    #  0.5000 *  <j,k||a,b> l1(i,c) t2(b,c,j,k) 
    T += 0.5000 * einsum('jkab,ic,bcjk->ia' ,g[o,o,v,v] ,l1 ,t2, optimize=True)

    #  -1.0000 *  f(i,b) l1(j,a) t1(b,j) 
    T += -1.0000 * einsum('ib,ja,bj->ia' ,f[o,v] ,l1 ,t1, optimize=True)

    #  -1.0000 *  f(j,a) l1(i,b) t1(b,j) 
    T += -1.0000 * einsum('ja,ib,bj->ia' ,f[o,v] ,l1 ,t1, optimize=True)

    #  -1.0000 *  <i,k||a,j> l1(j,b) t1(b,k) 
    T += -1.0000 * einsum('ikaj,jb,bk->ia' ,g[o,o,v,o] ,l1 ,t1, optimize=True)

    #  -1.0000 *  <b,k||a,j> l2(i,j,b,c) t1(c,k) 
    T += -1.0000 * einsum('bkaj,ijbc,ck->ia' ,g[v,o,v,o] ,l2 ,t1, optimize=True)

    #  -0.50000 *  f(i,b) l2(j,k,a,c) t2(b,c,j,k) 
    T += -0.50000 * einsum('ib,jkac,bcjk->ia' ,f[o,v] ,l2 ,t2, optimize=True)

    #  -0.50000 *  f(j,a) l2(i,k,b,c) t2(b,c,j,k) 
    T += -0.50000 * einsum('ja,ikbc,bcjk->ia' ,f[o,v] ,l2 ,t2, optimize=True)

    #  -0.50000 *  <i,k||a,j> l2(j,l,b,c) t2(b,c,k,l) 
    T += -0.50000 * einsum('ikaj,jlbc,bckl->ia' ,g[o,o,v,o] ,l2 ,t2, optimize=True)

    #  -0.25000 *  <i,b||c,d> l2(j,k,a,b) t2(c,d,j,k) 
    T += -0.25000 * einsum('ibcd,jkab,cdjk->ia' ,g[o,v,v,v] ,l2 ,t2, optimize=True)

    #  0.2500 *  <k,l||a,j> l2(i,j,b,c) t2(b,c,k,l) 
    T += 0.2500 * einsum('klaj,ijbc,bckl->ia' ,g[o,o,v,o] ,l2 ,t2, optimize=True)

    #  1.0000 *  <i,j||b,c> l1(k,a) t1(b,j) t1(c,k) 
    T += 1.0000 * einsum('ijbc,ka,bj,ck->ia' ,g[o,o,v,v] ,l1 ,t1 ,t1, optimize=True)

    #  1.0000 *  <j,k||a,b> l1(i,c) t1(b,j) t1(c,k) 
    T += 1.0000 * einsum('jkab,ic,bj,ck->ia' ,g[o,o,v,v] ,l1 ,t1 ,t1, optimize=True)

    #  0.5000 *  <i,b||c,d> l2(j,k,a,b) t1(c,k) t1(d,j) 
    T += 0.5000 * einsum('ibcd,jkab,ck,dj->ia' ,g[o,v,v,v] ,l2 ,t1 ,t1, optimize=True)

    #  0.5000 *  <i,j||b,c> l2(k,l,a,d) t1(b,j) t2(c,d,k,l) 
    T += 0.5000 * einsum('ijbc,klad,bj,cdkl->ia' ,g[o,o,v,v] ,l2 ,t1 ,t2, optimize=True)

    #  0.5000 *  <j,k||a,b> l2(i,l,c,d) t1(b,j) t2(c,d,k,l) 
    T += 0.5000 * einsum('jkab,ilcd,bj,cdkl->ia' ,g[o,o,v,v] ,l2 ,t1 ,t2, optimize=True)

    #  -1.0000 *  <i,j||a,b> l1(k,c) t1(b,k) t1(c,j) 
    T += -1.0000 * einsum('ijab,kc,bk,cj->ia' ,g[o,o,v,v] ,l1 ,t1 ,t1, optimize=True)

    #  -1.0000 *  <i,j||b,c> l2(k,l,a,d) t1(b,k) t2(c,d,j,l) 
    T += -1.0000 * einsum('ijbc,klad,bk,cdjl->ia' ,g[o,o,v,v] ,l2 ,t1 ,t2, optimize=True)

    #  -1.0000 *  <i,k||b,j> l2(j,l,a,c) t1(b,l) t1(c,k) 
    T += -1.0000 * einsum('ikbj,jlac,bl,ck->ia' ,g[o,o,v,o] ,l2 ,t1 ,t1, optimize=True)

    #  -1.0000 *  <b,j||a,c> l2(i,k,b,d) t1(c,k) t1(d,j) 
    T += -1.0000 * einsum('bjac,ikbd,ck,dj->ia' ,g[v,o,v,v] ,l2 ,t1 ,t1, optimize=True)

    #  -1.0000 *  <j,k||a,b> l2(i,l,c,d) t1(c,j) t2(b,d,k,l) 
    T += -1.0000 * einsum('jkab,ilcd,cj,bdkl->ia' ,g[o,o,v,v] ,l2 ,t1 ,t2, optimize=True)

    #  -0.50000 *  <i,j||a,b> l2(k,l,c,d) t1(b,k) t2(c,d,j,l) 
    T += -0.50000 * einsum('ijab,klcd,bk,cdjl->ia' ,g[o,o,v,v] ,l2 ,t1 ,t2, optimize=True)

    #  -0.50000 *  <i,j||a,b> l2(k,l,c,d) t1(c,j) t2(b,d,k,l) 
    T += -0.50000 * einsum('ijab,klcd,cj,bdkl->ia' ,g[o,o,v,v] ,l2 ,t1 ,t2, optimize=True)

    #  -0.50000 *  <k,l||a,j> l2(i,j,b,c) t1(b,l) t1(c,k) 
    T += -0.50000 * einsum('klaj,ijbc,bl,ck->ia' ,g[o,o,v,o] ,l2 ,t1 ,t1, optimize=True)

    #  0.2500 *  <i,j||b,c> l2(k,l,a,d) t1(d,j) t2(b,c,k,l) 
    T += 0.2500 * einsum('ijbc,klad,dj,bckl->ia' ,g[o,o,v,v] ,l2 ,t1 ,t2, optimize=True)

    #  0.2500 *  <j,k||a,b> l2(i,l,c,d) t1(b,l) t2(c,d,j,k) 
    T += 0.2500 * einsum('jkab,ilcd,bl,cdjk->ia' ,g[o,o,v,v] ,l2 ,t1 ,t2, optimize=True)

    #  -0.50000 *  <i,j||b,c> l2(k,l,a,d) t1(b,l) t1(c,k) t1(d,j) 
    T += -0.50000 * einsum('ijbc,klad,bl,ck,dj->ia' ,g[o,o,v,v] ,l2 ,t1 ,t1 ,t1, optimize=True)

    #  -0.50000 *  <j,k||a,b> l2(i,l,c,d) t1(b,l) t1(c,k) t1(d,j) 
    T += -0.50000 * einsum('jkab,ilcd,bl,ck,dj->ia' ,g[o,o,v,v] ,l2 ,t1 ,t1 ,t1, optimize=True)

    #  1.0000 *  f(i,a) 
    T += 1.0000 * einsum('ia->ia' ,f[o,v], optimize=True)

    return T

def cc_lambda_doubles(f, g, o, v, t1=None, t2=None, l1=None, l2=None):

    '''
        COGUS generated level [SD] on 29 Jul 2021   
    '''
    from numpy import einsum, swapaxes


    #  0.5000 *  <i,j||k,l> l2(k,l,a,b) 
    T = 0.5000 * einsum('ijkl,klab->ijab' ,g[o,o,o,o] ,l2, optimize=True)

    #  0.5000 *  <c,d||a,b> l2(i,j,c,d) 
    T += 0.5000 * einsum('cdab,ijcd->ijab' ,g[v,v,v,v] ,l2, optimize=True)

    #  1.0000 * P(i,j) f(i,k) l2(j,k,a,b) 
    t = 1.0000 * einsum('ik,jkab->ijab' ,f[o,o] ,l2, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  1.0000 * P(i,j) <i,c||a,b> l1(j,c) 
    t = 1.0000 * einsum('icab,jc->ijab' ,g[o,v,v,v] ,l1, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  -1.0000 * P(a,b) f(c,a) l2(i,j,b,c) 
    t = -1.0000 * einsum('ca,ijbc->ijab' ,f[v,v] ,l2, optimize=True)
    T += t - t.swapaxes(3, 2)

    #  -1.0000 * P(a,b) <i,j||a,k> l1(k,b) 
    t = -1.0000 * einsum('ijak,kb->ijab' ,g[o,o,v,o] ,l1, optimize=True)
    T += t - t.swapaxes(3, 2)

    #  -1.0000 *  <i,j||c,k> l2(k,l,a,b) t1(c,l) 
    T += -1.0000 * einsum('ijck,klab,cl->ijab' ,g[o,o,v,o] ,l2 ,t1, optimize=True)

    #  -1.0000 *  <c,k||a,b> l2(i,j,c,d) t1(d,k) 
    T += -1.0000 * einsum('ckab,ijcd,dk->ijab' ,g[v,o,v,v] ,l2 ,t1, optimize=True)

    #  0.2500 *  <i,j||c,d> l2(k,l,a,b) t2(c,d,k,l) 
    T += 0.2500 * einsum('ijcd,klab,cdkl->ijab' ,g[o,o,v,v] ,l2 ,t2, optimize=True)

    #  0.2500 *  <k,l||a,b> l2(i,j,c,d) t2(c,d,k,l) 
    T += 0.2500 * einsum('klab,ijcd,cdkl->ijab' ,g[o,o,v,v] ,l2 ,t2, optimize=True)

    #  1.0000 * P(a,b)P(i,j) f(i,a) l1(j,b) 
    t = 1.0000 * einsum('ia,jb->ijab' ,f[o,v] ,l1, optimize=True)
    T += t - t.swapaxes(3, 2) - t.swapaxes(1, 0) + t.swapaxes(3, 2).swapaxes(1, 0)

    #  1.0000 * P(i,j) f(i,c) l2(j,k,a,b) t1(c,k) 
    t = 1.0000 * einsum('ic,jkab,ck->ijab' ,f[o,v] ,l2 ,t1, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  1.0000 * P(a,b) f(k,a) l2(i,j,b,c) t1(c,k) 
    t = 1.0000 * einsum('ka,ijbc,ck->ijab' ,f[o,v] ,l2 ,t1, optimize=True)
    T += t - t.swapaxes(3, 2)

    #  1.0000 * P(a,b)P(i,j) <i,c||a,k> l2(j,k,b,c) 
    t = 1.0000 * einsum('icak,jkbc->ijab' ,g[o,v,v,o] ,l2, optimize=True)
    T += t - t.swapaxes(3, 2) - t.swapaxes(1, 0) + t.swapaxes(3, 2).swapaxes(1, 0)

    #  -1.0000 * P(a,b) <i,j||a,c> l1(k,b) t1(c,k) 
    t = -1.0000 * einsum('ijac,kb,ck->ijab' ,g[o,o,v,v] ,l1 ,t1, optimize=True)
    T += t - t.swapaxes(3, 2)

    #  -1.0000 * P(i,j) <i,k||a,b> l1(j,c) t1(c,k) 
    t = -1.0000 * einsum('ikab,jc,ck->ijab' ,g[o,o,v,v] ,l1 ,t1, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  -1.0000 * P(i,j) <i,l||c,k> l2(j,k,a,b) t1(c,l) 
    t = -1.0000 * einsum('ilck,jkab,cl->ijab' ,g[o,o,v,o] ,l2 ,t1, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  -1.0000 * P(a,b) <c,k||a,d> l2(i,j,b,c) t1(d,k) 
    t = -1.0000 * einsum('ckad,ijbc,dk->ijab' ,g[v,o,v,v] ,l2 ,t1, optimize=True)
    T += t - t.swapaxes(3, 2)

    #  -0.50000 * P(a,b) <i,j||a,c> l2(k,l,b,d) t2(c,d,k,l) 
    t = -0.50000 * einsum('ijac,klbd,cdkl->ijab' ,g[o,o,v,v] ,l2 ,t2, optimize=True)
    T += t - t.swapaxes(3, 2)

    #  -0.50000 *  <i,j||c,d> l2(k,l,a,b) t1(c,l) t1(d,k) 
    T += -0.50000 * einsum('ijcd,klab,cl,dk->ijab' ,g[o,o,v,v] ,l2 ,t1 ,t1, optimize=True)

    #  -0.50000 * P(i,j) <i,k||a,b> l2(j,l,c,d) t2(c,d,k,l) 
    t = -0.50000 * einsum('ikab,jlcd,cdkl->ijab' ,g[o,o,v,v] ,l2 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  -0.50000 * P(i,j) <i,k||c,d> l2(j,l,a,b) t2(c,d,k,l) 
    t = -0.50000 * einsum('ikcd,jlab,cdkl->ijab' ,g[o,o,v,v] ,l2 ,t2, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  -0.50000 *  <k,l||a,b> l2(i,j,c,d) t1(c,l) t1(d,k) 
    T += -0.50000 * einsum('klab,ijcd,cl,dk->ijab' ,g[o,o,v,v] ,l2 ,t1 ,t1, optimize=True)

    #  -0.50000 * P(a,b) <k,l||a,c> l2(i,j,b,d) t2(c,d,k,l) 
    t = -0.50000 * einsum('klac,ijbd,cdkl->ijab' ,g[o,o,v,v] ,l2 ,t2, optimize=True)
    T += t - t.swapaxes(3, 2)

    #  1.0000 * P(a,b)P(i,j) <i,c||a,d> l2(j,k,b,c) t1(d,k) 
    t = 1.0000 * einsum('icad,jkbc,dk->ijab' ,g[o,v,v,v] ,l2 ,t1, optimize=True)
    T += t - t.swapaxes(3, 2) - t.swapaxes(1, 0) + t.swapaxes(3, 2).swapaxes(1, 0)

    #  1.0000 * P(a,b)P(i,j) <i,k||a,c> l1(j,b) t1(c,k) 
    t = 1.0000 * einsum('ikac,jb,ck->ijab' ,g[o,o,v,v] ,l1 ,t1, optimize=True)
    T += t - t.swapaxes(3, 2) - t.swapaxes(1, 0) + t.swapaxes(3, 2).swapaxes(1, 0)

    #  1.0000 * P(a,b)P(i,j) <i,k||a,c> l2(j,l,b,d) t2(c,d,k,l) 
    t = 1.0000 * einsum('ikac,jlbd,cdkl->ijab' ,g[o,o,v,v] ,l2 ,t2, optimize=True)
    T += t - t.swapaxes(3, 2) - t.swapaxes(1, 0) + t.swapaxes(3, 2).swapaxes(1, 0)

    #  -1.0000 * P(i,j) <i,k||c,d> l2(j,l,a,b) t1(c,k) t1(d,l) 
    t = -1.0000 * einsum('ikcd,jlab,ck,dl->ijab' ,g[o,o,v,v] ,l2 ,t1 ,t1, optimize=True)
    T += t - t.swapaxes(1, 0)

    #  -1.0000 * P(a,b)P(i,j) <i,l||a,k> l2(j,k,b,c) t1(c,l) 
    t = -1.0000 * einsum('ilak,jkbc,cl->ijab' ,g[o,o,v,o] ,l2 ,t1, optimize=True)
    T += t - t.swapaxes(3, 2) - t.swapaxes(1, 0) + t.swapaxes(3, 2).swapaxes(1, 0)

    #  -1.0000 * P(a,b) <k,l||a,c> l2(i,j,b,d) t1(c,k) t1(d,l) 
    t = -1.0000 * einsum('klac,ijbd,ck,dl->ijab' ,g[o,o,v,v] ,l2 ,t1 ,t1, optimize=True)
    T += t - t.swapaxes(3, 2)

    #  -1.0000 * P(a,b)P(i,j) <i,k||a,c> l2(j,l,b,d) t1(c,l) t1(d,k) 
    t = -1.0000 * einsum('ikac,jlbd,cl,dk->ijab' ,g[o,o,v,v] ,l2 ,t1 ,t1, optimize=True)
    T += t - t.swapaxes(3, 2) - t.swapaxes(1, 0) + t.swapaxes(3, 2).swapaxes(1, 0)

    #  1.0000 *  <i,j||a,b> 
    T += 1.0000 * einsum('ijab->ijab' ,g[o,o,v,v], optimize=True)

    return T

