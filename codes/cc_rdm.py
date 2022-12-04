
'''s - spin orbitals, o - occupied spin orbitals, v - virtual spin orbitals
t1[v,o] - singles amplitudes, t2[v,v,o,o] - doubles amplitudes
l1[o,v] - singles lambda, l2[o,o,v,v] - doubles lambda
'''

def cc_oprdm(o, v, t1=None, t2=None, l1=None, l2=None):

    '''
        COGUS generated level [CC] on 28 Jul 2021   
    '''
    from numpy import einsum, swapaxes, eye, zeros

    ns = sum(t1.shape)
    d = zeros((ns, ns))
    kd = eye(ns)

    # density matrix block [oo]

    #  -1.0000 *  l1(j,a) t1(a,i) 
    d[o,o] = -1.0000 * einsum('ja,ai->ij' ,l1 ,t1, optimize=True)

    #  -0.50000 *  l2(j,k,a,b) t2(a,b,i,k) 
    d[o,o] += -0.50000 * einsum('jkab,abik->ij' ,l2 ,t2, optimize=True)

    #  1.0000 *  kd[o,o] 
    d[o,o] += 1.0000 * einsum('ij->ij' ,kd[o,o], optimize=True)

    # density matrix block [ov]

    #  1.0000 *  l1(j,b) t2(a,b,i,j) 
    d[o,v] = 1.0000 * einsum('jb,abij->ia' ,l1 ,t2, optimize=True)

    #  -1.0000 *  l1(j,b) t1(a,j) t1(b,i) 
    d[o,v] += -1.0000 * einsum('jb,aj,bi->ia' ,l1 ,t1 ,t1, optimize=True)

    #  -0.50000 *  l2(j,k,b,c) t1(a,j) t2(b,c,i,k) 
    d[o,v] += -0.50000 * einsum('jkbc,aj,bcik->ia' ,l2 ,t1 ,t2, optimize=True)

    #  -0.50000 *  l2(j,k,b,c) t1(b,i) t2(a,c,j,k) 
    d[o,v] += -0.50000 * einsum('jkbc,bi,acjk->ia' ,l2 ,t1 ,t2, optimize=True)

    #  1.0000 *  t1(a,i) 
    d[o,v] += 1.0000 * einsum('ai->ia' ,t1, optimize=True)

    # density matrix block [vo]

    #  1.0000 *  l1(i,a) 
    d[v,o] = 1.0000 * einsum('ia->ai' ,l1, optimize=True)

    # density matrix block [vv]

    #  1.0000 *  l1(i,a) t1(b,i) 
    d[v,v] = 1.0000 * einsum('ia,bi->ab' ,l1 ,t1, optimize=True)

    #  0.5000 *  l2(i,j,a,c) t2(b,c,i,j) 
    d[v,v] += 0.5000 * einsum('ijac,bcij->ab' ,l2 ,t2, optimize=True)

    return d 

def cc_tprdm(o, v, t1=None, t2=None, l1=None, l2=None):

    '''
        COGUS generated level [CC] on 28 Jul 2021   
    '''
    from numpy import einsum, swapaxes, eye, zeros

    ns = sum(t1.shape)
    d = zeros((ns, ns, ns, ns))
    kd = eye(ns)

    # density matrix block [oooo]

    #  0.5000 *  t2(a,b,i,j) l2(k,l,a,b) 
    d[o,o,o,o] = 0.5000 * einsum('abij,klab->ijkl' ,t2 ,l2, optimize=True)

    #  -1.0000 * P(i,j) kd[o,o] kd[o,o] 
    t = -1.0000 * einsum('il,jk->ijkl' ,kd[o,o], kd[o,o], optimize=True)
    d[o,o,o,o] += t - t.swapaxes(1, 0)

    #  0.5000 * P(i,j) t1(a,i) t1(b,j) l2(k,l,a,b) 
    t = 0.5000 * einsum('ai,bj,klab->ijkl' ,t1 ,t1 ,l2, optimize=True)
    d[o,o,o,o] += t - t.swapaxes(1, 0)

    #  -1.0000 * P(i,j)P(k,l) t1(a,i) l1(k,a) kd[o,o] 
    t = -1.0000 * einsum('ai,ka,jl->ijkl' ,t1 ,l1 ,kd[o,o], optimize=True)
    d[o,o,o,o] += t - t.swapaxes(1, 0) - t.swapaxes(3, 2) + t.swapaxes(1, 0).swapaxes(3, 2)

    #  -0.50000 * P(i,j)P(k,l) t2(a,b,i,m) l2(k,m,a,b) kd[o,o] 
    t = -0.50000 * einsum('abim,kmab,jl->ijkl' ,t2 ,l2 ,kd[o,o], optimize=True)
    d[o,o,o,o] += t - t.swapaxes(1, 0) - t.swapaxes(3, 2) + t.swapaxes(1, 0).swapaxes(3, 2)

    # density matrix block [ooov]

    #  1.0000 *  t2(a,b,i,j) l1(k,b) 
    d[o,o,o,v] = 1.0000 * einsum('abij,kb->ijka' ,t2 ,l1, optimize=True)

    #  0.5000 *  t1(a,l) t2(b,c,i,j) l2(k,l,b,c) 
    d[o,o,o,v] += 0.5000 * einsum('al,bcij,klbc->ijka' ,t1 ,t2 ,l2, optimize=True)

    #  -1.0000 * P(i,j) t1(a,i) kd[o,o] 
    t = -1.0000 * einsum('ai,jk->ijka' ,t1 ,kd[o,o], optimize=True)
    d[o,o,o,v] += t - t.swapaxes(1, 0)

    #  1.0000 * P(i,j) t1(a,i) t1(b,j) l1(k,b) 
    t = 1.0000 * einsum('ai,bj,kb->ijka' ,t1 ,t1 ,l1, optimize=True)
    d[o,o,o,v] += t - t.swapaxes(1, 0)

    #  0.5000 * P(i,j) t1(a,i) t2(b,c,j,l) l2(k,l,b,c) 
    t = 0.5000 * einsum('ai,bcjl,klbc->ijka' ,t1 ,t2 ,l2, optimize=True)
    d[o,o,o,v] += t - t.swapaxes(1, 0)

    #  -1.0000 * P(i,j) t1(b,i) t2(a,c,j,l) l2(k,l,b,c) 
    t = -1.0000 * einsum('bi,acjl,klbc->ijka' ,t1 ,t2 ,l2, optimize=True)
    d[o,o,o,v] += t - t.swapaxes(1, 0)

    #  -1.0000 * P(i,j) t2(a,b,i,l) l1(l,b) kd[o,o] 
    t = -1.0000 * einsum('abil,lb,jk->ijka' ,t2 ,l1 ,kd[o,o], optimize=True)
    d[o,o,o,v] += t - t.swapaxes(1, 0)

    #  1.0000 * P(i,j) t1(a,l) t1(b,i) l1(l,b) kd[o,o] 
    t = 1.0000 * einsum('al,bi,lb,jk->ijka' ,t1 ,t1 ,l1 ,kd[o,o], optimize=True)
    d[o,o,o,v] += t - t.swapaxes(1, 0)

    #  0.5000 * P(i,j) t1(a,l) t1(b,i) t1(c,j) l2(k,l,b,c) 
    t = 0.5000 * einsum('al,bi,cj,klbc->ijka' ,t1 ,t1 ,t1 ,l2, optimize=True)
    d[o,o,o,v] += t - t.swapaxes(1, 0)

    #  0.5000 * P(i,j) t1(a,l) t2(b,c,i,m) l2(l,m,b,c) kd[o,o] 
    t = 0.5000 * einsum('al,bcim,lmbc,jk->ijka' ,t1 ,t2 ,l2 ,kd[o,o], optimize=True)
    d[o,o,o,v] += t - t.swapaxes(1, 0)

    #  0.5000 * P(i,j) t1(b,i) t2(a,c,l,m) l2(l,m,b,c) kd[o,o] 
    t = 0.5000 * einsum('bi,aclm,lmbc,jk->ijka' ,t1 ,t2 ,l2 ,kd[o,o], optimize=True)
    d[o,o,o,v] += t - t.swapaxes(1, 0)

    # density matrix block [oovo]
    d[o,o,v,o] -= d[o,o,o,v].swapaxes(3, 2)


    # density matrix block [ovoo]

    #  1.0000 *  t1(b,i) l2(k,l,a,b) 
    d[o,v,o,o] = 1.0000 * einsum('bi,klab->iakl' ,t1 ,l2, optimize=True)

    #  -1.0000 * P(k,l) l1(k,a) kd[o,o] 
    t = -1.0000 * einsum('ka,il->iakl' ,l1 ,kd[o,o], optimize=True)
    d[o,v,o,o] += t - t.swapaxes(3, 2)

    # density matrix block [vooo]
    d[v,o,o,o] -= d[o,v,o,o].swapaxes(1, 0)


    # density matrix block [vvvv]

    #  0.5000 *  t2(c,d,i,j) l2(i,j,a,b) 
    d[v,v,v,v] = 0.5000 * einsum('cdij,ijab->abcd' ,t2 ,l2, optimize=True)

    #  0.5000 * P(a,b) t1(c,i) t1(d,j) l2(i,j,a,b) 
    t = 0.5000 * einsum('ci,dj,ijab->abcd' ,t1 ,t1 ,l2, optimize=True)
    d[v,v,v,v] += t - t.swapaxes(1, 0)

    # density matrix block [vvvo]

    #  -1.0000 *  t1(c,j) l2(i,j,a,b) 
    d[v,v,v,o] = -1.0000 * einsum('cj,ijab->abci' ,t1 ,l2, optimize=True)

    # density matrix block [vvov]
    d[v,v,o,v] -= d[v,v,v,o].swapaxes(3, 2)


    # density matrix block [vovv]

    #  -1.0000 *  t2(c,d,i,j) l1(j,a) 
    d[v,o,v,v] = -1.0000 * einsum('cdij,ja->aicd' ,t2 ,l1, optimize=True)

    #  -0.50000 *  t1(b,i) t2(c,d,j,k) l2(j,k,a,b) 
    d[v,o,v,v] += -0.50000 * einsum('bi,cdjk,jkab->aicd' ,t1 ,t2 ,l2, optimize=True)

    #  1.0000 * P(c,d) t1(c,j) t2(d,b,i,k) l2(j,k,a,b) 
    t = 1.0000 * einsum('cj,dbik,jkab->aicd' ,t1 ,t2 ,l2, optimize=True)
    d[v,o,v,v] += t - t.swapaxes(3, 2)

    #  -1.0000 * P(c,d) t1(c,i) t1(d,j) l1(j,a) 
    t = -1.0000 * einsum('ci,dj,ja->aicd' ,t1 ,t1 ,l1, optimize=True)
    d[v,o,v,v] += t - t.swapaxes(3, 2)

    #  -0.50000 * P(c,d) t1(c,i) t2(d,b,j,k) l2(j,k,a,b) 
    t = -0.50000 * einsum('ci,dbjk,jkab->aicd' ,t1 ,t2 ,l2, optimize=True)
    d[v,o,v,v] += t - t.swapaxes(3, 2)

    #  -0.50000 * P(c,d) t1(c,j) t1(d,k) t1(b,i) l2(j,k,a,b) 
    t = -0.50000 * einsum('cj,dk,bi,jkab->aicd' ,t1 ,t1 ,t1 ,l2, optimize=True)
    d[v,o,v,v] += t - t.swapaxes(3, 2)

    # density matrix block [ovvv]
    d[o,v,v,v] -= d[v,o,v,v].swapaxes(1, 0)


    # density matrix block [oovv]

    #  1.0000 * P(i,j) t1(a,i) t1(b,j) 
    t = 1.0000 * einsum('ai,bj->ijab' ,t1 ,t1, optimize=True)
    d[o,o,v,v] = t - t.swapaxes(1, 0)

    #  0.2500 *  t2(a,b,k,l) t2(c,d,i,j) l2(k,l,c,d) 
    d[o,o,v,v] += 0.2500 * einsum('abkl,cdij,klcd->ijab' ,t2 ,t2 ,l2, optimize=True)

    #  1.0000 * P(a,b) t1(a,k) t2(b,c,i,j) l1(k,c) 
    t = 1.0000 * einsum('ak,bcij,kc->ijab' ,t1 ,t2 ,l1, optimize=True)
    d[o,o,v,v] += t - t.swapaxes(3, 2)

    #  1.0000 * P(i,j) t1(c,i) t2(a,b,j,k) l1(k,c) 
    t = 1.0000 * einsum('ci,abjk,kc->ijab' ,t1 ,t2 ,l1, optimize=True)
    d[o,o,v,v] += t - t.swapaxes(1, 0)

    #  1.0000 * P(i,j) t2(a,c,i,k) t2(b,d,j,l) l2(k,l,c,d) 
    t = 1.0000 * einsum('acik,bdjl,klcd->ijab' ,t2 ,t2 ,l2, optimize=True)
    d[o,o,v,v] += t - t.swapaxes(1, 0)

    #  -0.25000 * P(i,j) t2(a,c,i,j) t2(b,d,k,l) l2(k,l,c,d) 
    t = -0.25000 * einsum('acij,bdkl,klcd->ijab' ,t2 ,t2 ,l2, optimize=True)
    d[o,o,v,v] += t - t.swapaxes(1, 0)

    #  -0.25000 * P(i,j) t2(a,c,k,l) t2(b,d,i,j) l2(k,l,c,d) 
    t = -0.25000 * einsum('ackl,bdij,klcd->ijab' ,t2 ,t2 ,l2, optimize=True)
    d[o,o,v,v] += t - t.swapaxes(1, 0)

    #  1.0000 * P(a,b)P(i,j) t1(a,i) t2(b,c,j,k) l1(k,c) 
    t = 1.0000 * einsum('ai,bcjk,kc->ijab' ,t1 ,t2 ,l1, optimize=True)
    d[o,o,v,v] += t - t.swapaxes(3, 2) - t.swapaxes(1, 0) + t.swapaxes(3, 2).swapaxes(1, 0)

    #  0.2500 * P(i,j) t1(a,k) t1(b,l) t2(c,d,i,j) l2(k,l,c,d) 
    t = 0.2500 * einsum('ak,bl,cdij,klcd->ijab' ,t1 ,t1 ,t2 ,l2, optimize=True)
    d[o,o,v,v] += t - t.swapaxes(1, 0)

    #  0.2500 * P(i,j) t1(c,i) t1(d,j) t2(a,b,k,l) l2(k,l,c,d) 
    t = 0.2500 * einsum('ci,dj,abkl,klcd->ijab' ,t1 ,t1 ,t2 ,l2, optimize=True)
    d[o,o,v,v] += t - t.swapaxes(1, 0)

    #  0.2500 * P(a,b)P(i,j) t2(a,b,i,l) t2(c,d,j,k) l2(k,l,c,d) 
    t = 0.2500 * einsum('abil,cdjk,klcd->ijab' ,t2 ,t2 ,l2, optimize=True)
    d[o,o,v,v] += t - t.swapaxes(3, 2) - t.swapaxes(1, 0) + t.swapaxes(3, 2).swapaxes(1, 0)

    #  -1.0000 * P(a,b)P(i,j) t1(a,i) t1(b,k) t1(c,j) l1(k,c) 
    t = -1.0000 * einsum('ai,bk,cj,kc->ijab' ,t1 ,t1 ,t1 ,l1, optimize=True)
    d[o,o,v,v] += t - t.swapaxes(3, 2) - t.swapaxes(1, 0) + t.swapaxes(3, 2).swapaxes(1, 0)

    #  -1.0000 * P(a,b)P(i,j) t1(a,k) t1(c,i) t2(b,d,j,l) l2(k,l,c,d) 
    t = -1.0000 * einsum('ak,ci,bdjl,klcd->ijab' ,t1 ,t1 ,t2 ,l2, optimize=True)
    d[o,o,v,v] += t - t.swapaxes(3, 2) - t.swapaxes(1, 0) + t.swapaxes(3, 2).swapaxes(1, 0)

    #  -0.50000 * P(a,b)P(i,j) t1(a,i) t1(c,j) t2(b,d,k,l) l2(k,l,c,d) 
    t = -0.50000 * einsum('ai,cj,bdkl,klcd->ijab' ,t1 ,t1 ,t2 ,l2, optimize=True)
    d[o,o,v,v] += t - t.swapaxes(3, 2) - t.swapaxes(1, 0) + t.swapaxes(3, 2).swapaxes(1, 0)

    #  -0.25000 * P(a,b)P(i,j) t1(a,i) t1(b,k) t2(c,d,j,l) l2(k,l,c,d) 
    t = -0.25000 * einsum('ai,bk,cdjl,klcd->ijab' ,t1 ,t1 ,t2 ,l2, optimize=True)
    d[o,o,v,v] += t - t.swapaxes(3, 2) - t.swapaxes(1, 0) + t.swapaxes(3, 2).swapaxes(1, 0)

    #  0.2500 * P(a,b)P(i,j) t1(a,k) t1(b,i) t2(c,d,j,l) l2(k,l,c,d) 
    t = 0.2500 * einsum('ak,bi,cdjl,klcd->ijab' ,t1 ,t1 ,t2 ,l2, optimize=True)
    d[o,o,v,v] += t - t.swapaxes(3, 2) - t.swapaxes(1, 0) + t.swapaxes(3, 2).swapaxes(1, 0)

    #  0.2500 * P(a,b)P(i,j) t1(a,k) t1(b,l) t1(c,i) t1(d,j) l2(k,l,c,d) 
    t = 0.2500 * einsum('ak,bl,ci,dj,klcd->ijab' ,t1 ,t1 ,t1 ,t1 ,l2, optimize=True)
    d[o,o,v,v] += t - t.swapaxes(3, 2) - t.swapaxes(1, 0) + t.swapaxes(3, 2).swapaxes(1, 0)

    #  1.0000 *  t2(a,b,i,j) 
    d[o,o,v,v] += 1.0000 * einsum('abij->ijab' ,t2, optimize=True)

    # density matrix block [vvoo]

    #  1.0000 *  l2(i,j,a,b) 
    d[v,v,o,o] = 1.0000 * einsum('ijab->abij' ,l2, optimize=True)

    # density matrix block [ovov]

    #  -1.0000 *  t1(b,i) l1(j,a) 
    d[o,v,o,v] = -1.0000 * einsum('bi,ja->iajb' ,t1 ,l1, optimize=True)

    #  -1.0000 *  t2(b,d,i,k) l2(j,k,a,d) 
    d[o,v,o,v] += -1.0000 * einsum('bdik,jkad->iajb' ,t2 ,l2, optimize=True)

    #  1.0000 *  t1(b,k) t1(d,i) l2(j,k,a,d) 
    d[o,v,o,v] += 1.0000 * einsum('bk,di,jkad->iajb' ,t1 ,t1 ,l2, optimize=True)

    #  1.0000 *  t1(b,k) l1(k,a) kd[o,o] 
    d[o,v,o,v] += 1.0000 * einsum('bk,ka,ij->iajb' ,t1 ,l1 ,kd[o,o], optimize=True)

    #  0.5000 *  t2(b,d,k,l) l2(k,l,a,d) kd[o,o] 
    d[o,v,o,v] += 0.5000 * einsum('bdkl,klad,ij->iajb' ,t2 ,l2 ,kd[o,o], optimize=True)

    # density matrix block [voov]
    d[v,o,o,v] -= d[o,v,o,v].swapaxes(1, 0)


    # density matrix block [ovvo]

    #  1.0000 *  t1(b,i) l1(j,a) 
    d[o,v,v,o] = 1.0000 * einsum('bi,ja->iabj' ,t1 ,l1, optimize=True)

    #  1.0000 *  t2(b,d,i,k) l2(j,k,a,d) 
    d[o,v,v,o] += 1.0000 * einsum('bdik,jkad->iabj' ,t2 ,l2, optimize=True)

    #  -1.0000 *  t1(b,k) t1(d,i) l2(j,k,a,d) 
    d[o,v,v,o] += -1.0000 * einsum('bk,di,jkad->iabj' ,t1 ,t1 ,l2, optimize=True)

    #  -1.0000 *  t1(b,k) l1(k,a) kd[o,o] 
    d[o,v,v,o] += -1.0000 * einsum('bk,ka,ij->iabj' ,t1 ,l1 ,kd[o,o], optimize=True)

    #  -0.50000 *  t2(b,d,k,l) l2(k,l,a,d) kd[o,o] 
    d[o,v,v,o] += -0.50000 * einsum('bdkl,klad,ij->iabj' ,t2 ,l2 ,kd[o,o], optimize=True)

    # density matrix block [vovo]
    d[v,o,v,o] -= d[o,v,v,o].swapaxes(1, 0)


    return d 

