from __future__ import division
import numpy as np
import scipy as sp
import cc.scc
import rhf
from basis import electronCount
from integral import buildEriMO, buildEriDoubleBar, buildFockMOspin, iEri
from atom import getConstant
import time
import os

def maximumAmplitudes(t, top = 5, mode = 0):
    #return the biggest 'top' amplitudes, mode is '+'|'-'|'abs'
    #for biggest +ve, -ve or absolute value

    #sort and get indexes
    if mode == 0:
        idx = np.unravel_index(np.argsort(np.abs(t), axis=None)[::-1][:top], t.shape)
    else:
        idx = np.unravel_index(np.argsort(t, axis=None)[::-mode][:top], t.shape)

    #convert tuple to list
    ampInformation = []
    for i in range(t.ndim):
        ampInformation.append(np.array(idx)[i,:top])
    ampInformation.append(t[idx][:top])

    #make an information string 'value (index)''
    sInfo = []
    for i in range(top):
        s = str(round(ampInformation[t.ndim][i],6)) + ' ('
        for j in range(t.ndim):
            s += str(ampInformation[j][i]) + ', '

        sInfo.append(s[:-2] + ')')

    return sInfo

def eomccsd(fockMOspin, eriMOspin, ts, td, nOccupied, nVirtual, spinOrbitals, partitioned = False, dialog = True):
    #EOM-CCSD calculation - right eigenvectors

    #get singles-singles block
    nRotate = nOccupied * nVirtual

    start=time.time()

    hss = np.zeros((nRotate, nRotate))
    ia = 0
    for i in range(0,nOccupied):
        for a in range(nOccupied, spinOrbitals):

            kc = 0
            for k in range(nOccupied):
                for c in range(nOccupied, spinOrbitals):
                    
                    hss[ia,kc] += fockMOspin[a,c]*(i==k)                                                                                    #[1]
                    hss[ia,kc] -= fockMOspin[k,i]*(a==c)                                                                                    #[6]    
                    hss[ia,kc] += eriMOspin[a,k,i,c]                                                                                        #[11]
                    for e in range(nOccupied, spinOrbitals):
                        hss[ia,kc] += eriMOspin[a,k,e,c]*ts[e,i]                                                                            #[12]
                        hss[ia,kc] -= (a==c)*fockMOspin[k,e]*ts[e,i]                                                                        #[7]
                    for m in range(nOccupied):
                        hss[ia,kc] -= eriMOspin[m,k,i,c]*ts[a,m]                                                                            #[13]
                        hss[ia,kc] -= (i==k)*fockMOspin[m,c]*ts[a,m]                                                                        #[2]
                        for e in range(nOccupied, spinOrbitals):
                            hss[ia,kc] += eriMOspin[m,a,e,c]*ts[e,m]*(i==k)                                                                 #[3]
                            hss[ia,kc] -= eriMOspin[k,m,i,e]*ts[e,m]*(a==c)                                                                 #[8]
                            hss[ia,kc] -= eriMOspin[m,k,e,c]*td[e,a,i,m]                                                                    #[14]
                            hss[ia,kc] += -eriMOspin[m,k,e,c]*ts[e,i]*ts[a,m]                                                               #[15]
                            for n in range(nOccupied):
                                hss[ia,kc] -= 0.5*(i==k)*eriMOspin[m,n,c,e]*td[e,a,n,m]                                                     #[4]
                            for f in range(nOccupied, spinOrbitals):
                                hss[ia,kc] -= 0.5*(a==c)*eriMOspin[k,m,e,f]*td[e,f,i,m]                                                     #[9]
                                hss[ia,kc] -= (a==c)*eriMOspin[k,m,e,f]*ts[e,i]*ts[f,m]                                                     #[10]
                        for n in range(nOccupied):
                            for f in range(nOccupied, spinOrbitals):
                                hss[ia,kc] -= (i==k)*eriMOspin[m,n,c,f]*ts[a,m]*ts[f,n]                                                     #[5]

                    kc += 1    
            ia += 1

    if dialog: print('finished Hss block in : {:>8.3f} s'.format(time.time()-start))
    start = time.time()

    #get singles-doubles 
    hsd = np.zeros((nRotate, nRotate*nRotate)) 

    ia = 0
    for i in range(nOccupied):
        for a in range(nOccupied, spinOrbitals):

            kcld = 0
            for k in range(nOccupied):
                for c in range(nOccupied, spinOrbitals):
                    for l in range(nOccupied):
                        for d in range(nOccupied, spinOrbitals):

                            hsd[ia,kcld] += (i==k)*(a==c)*fockMOspin[l,d]                                                                   #[16]
                            hsd[ia,kcld] += 0.5*eriMOspin[a,l,c,d]*(i==k)                                                                   #[18]
                            hsd[ia,kcld] -= 0.5*eriMOspin[k,l,i,d]*(a==c)                                                                   #[20]
                            for e in range(nOccupied, spinOrbitals):
                                hsd[ia,kcld] -= 0.5*eriMOspin[k,l,e,d]*ts[e,i]*(a==c)                                                       #[21]
                            for m in range(nOccupied):
                                hsd[ia,kcld] -= 0.5*eriMOspin[m,l,c,d]*ts[a,m]*(i==k)                                                       #[19]
                                for e in range(nOccupied, spinOrbitals):
                                    hsd[ia,kcld] += eriMOspin[m,l,e,d]*ts[e,m]*(i==k)*(a==c)                                                #[17]

                            kcld += 1

            ia += 1

    if dialog: print('finished Hsd block in : {:>8.3f} s'.format(time.time()-start))
    start = time.time()

    #get doubles-singles 
    hds = np.zeros((nRotate*nRotate, nRotate))  

    iajb = 0
    for i in range(nOccupied):
        for a in range(nOccupied, spinOrbitals):
            for j in range(nOccupied):
                for b in range(nOccupied, spinOrbitals):

                    kc = 0
                    for k in range(nOccupied):
                        for c in range(nOccupied, spinOrbitals):

                            hds[iajb,kc] += (i==k)*eriMOspin[a,b,c,j] - (j==k)*eriMOspin[a,b,c,i]                                           #[33]                                                                     
                            hds[iajb,kc] += (b==c)*eriMOspin[k,a,i,j] - (a==c)*eriMOspin[k,b,i,j]                                           #[22]

                            for e in range(nOccupied, spinOrbitals):

                                hds[iajb,kc] += (i==k)*eriMOspin[a,b,c,e]*ts[e,j] - (j==k)*eriMOspin[a,b,c,e]*ts[e,i]                       #[37]

                                hds[iajb,kc] += (b==c)*eriMOspin[k,a,e,j]*ts[e,i] - (b==c)*eriMOspin[k,a,e,i]*ts[e,j] - \
                                                (a==c)*eriMOspin[k,b,e,j]*ts[e,i] + (a==c)*eriMOspin[k,b,e,i]*ts[e,j]                       #[30]

                                hds[iajb,kc] += (b==c)*fockMOspin[k,e]*td[e,a,i,j] - (a==c)*fockMOspin[k,e]*td[e,b,i,j]                     #[32]

                                hds[iajb,kc] += eriMOspin[k,a,c,e]*td[e,b,i,j] -  eriMOspin[k,b,c,e]*td[e,a,i,j]                            #[43]

                                for f in range(nOccupied, spinOrbitals):

                                    hds[iajb,kc] += 0.5*td[e,f,i,j]*( (b==c)*eriMOspin[k,a,e,f] - (a==c)*eriMOspin[k,b,e,f])                #[24]
                                    hds[iajb,kc] += ts[e,i]*ts[f,j]*(-(a==c)*eriMOspin[k,b,e,f] + (b==c)*eriMOspin[k,a,e,f])                #[25]

                            for m in range(nOccupied):

                                hds[iajb,kc] += (a==c)*eriMOspin[k,m,i,j]*ts[b,m] - (b==c)*eriMOspin[k,m,i,j]*ts[a,m]                       #[26]

                                hds[iajb,kc] += (j==k)*eriMOspin[m,b,c,i]*ts[a,m] - (j==k)*eriMOspin[m,a,c,i]*ts[b,m] - \
                                                (i==k)*eriMOspin[m,b,c,j]*ts[a,m] + (i==k)*eriMOspin[m,a,c,j]*ts[b,m]                       #[41]

                                hds[iajb,kc] += (j==k)*fockMOspin[m,c]*td[a,b,m,i] - (i==k)*fockMOspin[m,c]*td[a,b,m,j]                     #[47]

                                hds[iajb,kc] += eriMOspin[k,m,c,j]*td[a,b,m,i] - eriMOspin[k,m,c,i]*td[a,b,m,j]                             #[45]

                                for n in range(nOccupied):
                                    hds[iajb,kc] += 0.5*td[a,b,m,n]*((i==k)*eriMOspin[m,n,c,j] - (j==k)*eriMOspin[m,n,c,i])                 #[35]       
                                    hds[iajb,kc] += ts[a,m]*ts[b,n]*((i==k)*eriMOspin[m,n,c,j] - (j==k)*eriMOspin[m,n,c,i])                 #[36] 

                                for e in range(nOccupied, spinOrbitals):

                                    hds[iajb,kc] += (i==k)*eriMOspin[a,m,c,e]*td[e,b,m,j] - (i==k)*eriMOspin[b,m,c,e]*td[e,a,m,j] - \
                                                    (j==k)*eriMOspin[a,m,c,e]*td[e,b,m,i] + (j==k)*eriMOspin[b,m,c,e]*td[e,a,m,i]           #[34]

                                    hds[iajb,kc] += (b==c)*eriMOspin[k,m,i,e]*td[e,a,m,j] - (b==c)*eriMOspin[k,m,j,e]*td[e,a,m,i] - \
                                                    (a==c)*eriMOspin[k,m,i,e]*td[e,b,m,j] + (a==c)*eriMOspin[k,m,j,e]*td[e,b,m,i]           #[23]

                                    hds[iajb,kc] += (j==k)*ts[e,i]*(ts[a,m]*eriMOspin[m,b,c,e] - ts[b,m]*eriMOspin[m,a,c,e]) - \
                                                    (i==k)*ts[e,j]*(ts[a,m]*eriMOspin[m,b,c,e] - ts[b,m]*eriMOspin[m,a,c,e])                #[38] 

                                    hds[iajb,kc] += ts[e,i]*eriMOspin[k,m,e,j]*(ts[b,m]*(a==c) - ts[a,m]*(b==c)) - \
                                                    ts[e,j]*eriMOspin[k,m,e,i]*(ts[b,m]*(a==c) - ts[a,m]*(b==c))                            #[27]

                                    hds[iajb,kc] += eriMOspin[k,m,c,e]*(ts[e,j]*td[a,b,m,i] - ts[e,i]*td[a,b,m,j])                          #[46]

                                    hds[iajb,kc] += eriMOspin[k,m,c,e]*(ts[b,m]*td[e,a,i,j] - ts[a,m]*td[e,b,i,j])                          #[44]

                                    for n in range(nOccupied):

                                        hds[iajb,kc] += 0.5*td[a,b,m,n]*eriMOspin[m,n,c,e]*((i==k)*ts[e,j] - (j==k)*ts[e,i])                #[39]

                                        hds[iajb,kc] += eriMOspin[m,n,c,e]*ts[a,m]*((j==k)*td[e,b,n,i] - (i==k)*td[e,b,n,j]) - \
                                                        eriMOspin[m,n,c,e]*ts[b,m]*((j==k)*td[e,a,n,i] - (i==k)*td[e,a,n,j])                #[42]
                                            
                                        hds[iajb,kc] += ts[e,m]*eriMOspin[m,n,e,c]*((j==k)*td[a,b,n,i] - (i==k)*td[a,b,n,j])                #[49]

                                        hds[iajb,kc] += ts[a,m]*ts[b,n]*eriMOspin[m,n,e,c]*((j==k)*ts[e,i] - (i==k)*ts[e,j])                #[40]

                                    for f in range(nOccupied, spinOrbitals):

                                        hds[iajb,kc] += 0.5*td[e,f,i,j]*eriMOspin[k,m,e,f]*((a==c)*ts[b,m] - (b==c)*ts[a,m])                #[28]

                                        hds[iajb,kc] += ts[e,i]*eriMOspin[k,m,e,f]*((b==c)*td[f,a,m,j] - (a==c)*td[f,b,m,j]) - \
                                                        ts[e,j]*eriMOspin[k,m,e,f]*((b==c)*td[f,a,m,i] - (a==c)*td[f,b,m,i])                #[31]

                                        hds[iajb,kc] += ts[e,m]*eriMOspin[m,k,e,f]*((b==c)*td[f,a,i,j] - (a==c)*td[f,b,i,j])                #[48]

                                        hds[iajb,kc] += ts[e,i]*ts[f,j]*eriMOspin[m,k,e,f]*((b==c)*ts[a,m] - (a==c)*ts[b,m])                #[29]


                            kc += 1

                    iajb += 1

    if dialog: print('finished Hds block in : {:>8.3f} s'.format(time.time()-start))
    start = time.time()

    #get doubles-doubles 
    hdd = np.zeros((nRotate*nRotate, nRotate*nRotate))  

    if not partitioned:
        iajb = 0
        for i in range(nOccupied):
            for a in range(nOccupied, spinOrbitals):
                for j in range(nOccupied):
                    for b in range(nOccupied, spinOrbitals):

                        kcld = 0
                        for k in range(nOccupied):
                            for c in range(nOccupied, spinOrbitals):
                                for l in range(nOccupied):
                                    for d in range(nOccupied, spinOrbitals):

                                        hdd[iajb,kcld] += (j==k)*(i==l)*(a==d)*fockMOspin[b,c] - \
                                                          (j==k)*(i==l)*(b==d)*fockMOspin[a,c]                                                  #[50]
                                        hdd[iajb,kcld] += (j==l)*(a==d)*(b==c)*fockMOspin[k,i] - \
                                                          (i==l)*(a==d)*(b==c)*fockMOspin[k,j]                                                  #[55]
                                        hdd[iajb,kcld] += 0.5*(i==k)*(j==l)*eriMOspin[a,b,c,d]                                                  #[60]
                                        hdd[iajb,kcld] += 0.5*(a==c)*(b==d)*eriMOspin[k,l,i,j]                                                  #[64]
                                        hdd[iajb,kcld] += (i==l)*(a==d)*eriMOspin[k,b,c,j] - \
                                                          (j==l)*(a==d)*eriMOspin[k,b,c,i] - \
                                                          (i==l)*(b==d)*eriMOspin[k,a,c,j] + \
                                                          (j==l)*(b==d)*eriMOspin[k,a,c,i]                                                      #[68]
                                        for e in range(nOccupied, spinOrbitals):
                                            hdd[iajb,kcld] += (a==c)*(i==l)*(b==d)*fockMOspin[k,e]*ts[e,j] - \
                                                              (a==c)*(j==l)*(b==d)*fockMOspin[k,e]*ts[e,i]                                      #[56]
                                            hdd[iajb,kcld] += (j==l)*(b==d)*eriMOspin[a,k,e,c]*ts[e,i] - \
                                                              (i==l)*(b==d)*eriMOspin[a,k,e,c]*ts[e,j] - \
                                                              (j==l)*(a==d)*eriMOspin[b,k,e,c]*ts[e,i] + \
                                                              (i==l)*(a==d)*eriMOspin[b,k,e,c]*ts[e,j]                                          #[69]
                                            hdd[iajb,kcld] += 0.5*(b==d)*(a==c)*eriMOspin[k,l,e,j]*ts[e,i] - \
                                                              0.5*(b==d)*(a==c)*eriMOspin[k,l,e,i]*ts[e,j]                                      #[65]
                                            hdd[iajb,kcld] += 0.5*(b==d)*eriMOspin[k,l,c,e]*td[e,a,i,j] - \
                                                              0.5*(a==d)*eriMOspin[k,l,c,e]*td[e,b,i,j]                                         #[73] 
                                            for f in range(nOccupied, spinOrbitals):
                                                hdd[iajb,kcld] += 0.25*(a==c)*(b==d)*eriMOspin[k,l,e,f]*td[e,f,i,j]                             #[66]
                                                hdd[iajb,kcld] += 0.5*(a==c)*(b==d)*eriMOspin[k,l,e,f]*ts[e,i]*ts[f,j]                          #[67]
                                        for m in range(nOccupied):
                                            hdd[iajb,kcld] += (i==k)*(j==l)*(a==d)*fockMOspin[m,c]*ts[b,m] - \
                                                              (i==k)*(j==l)*(b==d)*fockMOspin[m,c]*ts[a,m]                                      #[51]
                                            hdd[iajb,kcld] += (i==l)*(b==d)*eriMOspin[m,k,j,c]*ts[a,m] - \
                                                              (j==l)*(b==d)*eriMOspin[m,k,i,c]*ts[a,m] - \
                                                              (i==l)*(a==d)*eriMOspin[m,k,j,c]*ts[b,m] + \
                                                              (j==l)*(a==d)*eriMOspin[m,k,i,c]*ts[b,m]                                          #[70]
                                            hdd[iajb,kcld] += 0.5*(j==l)*(i==k)*eriMOspin[m,a,c,d]*ts[b,m] - \
                                                              0.5*(j==l)*(i==k)*eriMOspin[m,b,c,d]*ts[a,m]                                      #[61]
                                            hdd[iajb,kcld] += 0.5*(j==l)*eriMOspin[k,m,c,d]*td[a,b,m,i] - \
                                                              0.5*(i==l)*eriMOspin[k,m,c,d]*td[a,b,m,j]                                         #[74]
                                            for n in range(nOccupied):
                                                hdd[iajb,kcld] += 0.25*(i==k)*(j==l)*eriMOspin[m,n,c,d]*td[a,b,m,n]                             #[62]
                                                hdd[iajb,kcld] += 0.5*(i==k)*(j==l)*eriMOspin[m,n,c,d]*ts[a,m]*ts[b,n]                          #[63]
                                            for e in range(nOccupied, spinOrbitals):
                                                hdd[iajb,kcld] += (i==k)*(j==l)*(b==d)*eriMOspin[m,a,e,c]*ts[e,m] - \
                                                                  (i==k)*(j==l)*(a==d)*eriMOspin[m,b,e,c]*ts[e,m]                               #[52]
                                                hdd[iajb,kcld] += (a==c)*(i==l)*(b==d)*eriMOspin[m,k,e,j]*ts[e,m] - \
                                                                  (a==c)*(j==l)*(b==d)*eriMOspin[m,k,e,i]*ts[e,m]                               #[57]
                                                hdd[iajb,kcld] += (j==l)*(b==d)*eriMOspin[k,m,c,e]*td[e,a,m,i] - \
                                                                  (i==l)*(b==d)*eriMOspin[k,m,c,e]*td[e,a,m,j] - \
                                                                  (j==l)*(a==d)*eriMOspin[k,m,c,e]*td[e,b,m,i] + \
                                                                  (i==l)*(a==d)*eriMOspin[k,m,c,e]*td[e,b,m,j]                                  #[71]
                                                hdd[iajb,kcld] += (l==j)*(a==d)*eriMOspin[m,k,e,c]*ts[e,i]*ts[b,m] - \
                                                                  (l==j)*(b==d)*eriMOspin[m,k,e,c]*ts[e,i]*ts[a,m] - \
                                                                  (l==i)*(a==d)*eriMOspin[m,k,e,c]*ts[e,j]*ts[b,m] + \
                                                                  (l==i)*(b==d)*eriMOspin[m,k,e,c]*ts[e,j]*ts[a,m]                              #[72]
                                                for n in range(nOccupied):
                                                    hdd[iajb,kcld] += 0.5*(i==k)*(j==l)*(a==d)*eriMOspin[m,n,e,c]*td[b,e,n,m] - \
                                                                      0.5*(i==k)*(j==l)*(b==d)*eriMOspin[m,n,e,c]*td[a,e,n,m]                   #[53]
                                                    hdd[iajb,kcld] += (i==k)*(j==l)*(a==d)*eriMOspin[m,n,e,c]*ts[e,m]*ts[b,n] - \
                                                                      (i==k)*(j==l)*(b==d)*eriMOspin[m,n,e,c]*ts[e,m]*ts[a,n]                   #[54]
                                                for f in range(nOccupied, spinOrbitals):
                                                    hdd[iajb,kcld] += 0.5*(a==c)*(i==l)*(b==d)*eriMOspin[m,k,e,f]*td[f,e,j,m] - \
                                                                      0.5*(a==c)*(j==l)*(b==d)*eriMOspin[m,k,e,f]*td[f,e,i,m]                   #[58]
                                                    hdd[iajb,kcld] += (a==c)*(i==l)*(b==d)*eriMOspin[m,k,e,f]*ts[f,j]*ts[e,m] - \
                                                                      (a==c)*(j==l)*(b==d)*eriMOspin[m,k,e,f]*ts[f,i]*ts[e,m]                   #[59]
                                        kcld += 1
                        iajb += 1

    else:

        iajb = 0
        for i in range(nOccupied):
            for a in range(nOccupied, spinOrbitals):
                for j in range(nOccupied):
                    for b in range(nOccupied, spinOrbitals):

                        kcld = 0
                        for k in range(nOccupied):
                            for c in range(nOccupied, spinOrbitals):
                                for l in range(nOccupied):
                                    for d in range(nOccupied, spinOrbitals):

                                        hdd[iajb,kcld] += (j==k)*(i==l)*(a==d)*fockMOspin[b,c] - \
                                                          (j==k)*(i==l)*(b==d)*fockMOspin[a,c]                                                  #[50]
                                        hdd[iajb,kcld] += (j==l)*(a==d)*(b==c)*fockMOspin[k,i] - \
                                                          (i==l)*(a==d)*(b==c)*fockMOspin[k,j]                                                  #[55]
                                        kcld += 1
                        iajb += 1
    

    if dialog: print('finished Hdd block in : {:>8.3f} s'.format(time.time()-start))

    eomMatrix = np.bmat([[hss,hsd],[hds,hdd]])

    if dialog:
        print("begin full diagonalization")
        print("matrix dimension:  ", str(len(eomMatrix)) + "x" + str(len(eomMatrix)) )

    eomEVal,eomEVec = sp.linalg.eig(eomMatrix)

    return eomEVal.real, eomEVec


def eommbpt2(fockMOspin, eriMOspin, nOccupied, nVirtual, spinOrbitals, partitioned = False, dialog = True):
    #EOM-MBPT(2) calculation - right eingenvectors

    #get second-order td amplitudes
    td = np.zeros((spinOrbitals, spinOrbitals, spinOrbitals, spinOrbitals))

    for a in range(nOccupied, spinOrbitals):
        for b in range(nOccupied, spinOrbitals):
            for i in range(nOccupied):
                for j in range(nOccupied):
                    td[a,b,i,j] += eriMOspin[a,b,i,j]/(fockMOspin[i,i]+fockMOspin[j,j]-fockMOspin[a,a]-fockMOspin[b,b])

    #get singles-singles block
    nRotate = nOccupied * nVirtual

    start=time.time()

    hss = np.zeros((nRotate, nRotate))
    ia = 0
    for i in range(0,nOccupied):
        for a in range(nOccupied, spinOrbitals):

            kc = 0
            for k in range(nOccupied):
                for c in range(nOccupied, spinOrbitals):
                    
                    hss[ia,kc] += fockMOspin[a,c]*(i==k)                                                                                    #[1]
                    hss[ia,kc] -= fockMOspin[k,i]*(a==c)                                                                                    #[6]    
                    hss[ia,kc] += eriMOspin[a,k,i,c]                                                                                        #[11]
                    for m in range(nOccupied):
                        for e in range(nOccupied, spinOrbitals):
                            hss[ia,kc] -= eriMOspin[m,k,e,c]*td[e,a,i,m]                                                                    #[14]
                            for n in range(nOccupied):
                                hss[ia,kc] -= 0.5*(i==k)*eriMOspin[m,n,c,e]*td[e,a,n,m]                                                     #[4]
                            for f in range(nOccupied, spinOrbitals):
                                hss[ia,kc] -= 0.5*(a==c)*eriMOspin[k,m,e,f]*td[e,f,i,m]                                                     #[9]

                    kc += 1    
            ia += 1

    if dialog: print('finished Hss block in : {:>8.3f} s'.format(time.time()-start))
    start = time.time()

    #get singles-doubles 
    hsd = np.zeros((nRotate, nRotate*nRotate)) 

    ia = 0
    for i in range(nOccupied):
        for a in range(nOccupied, spinOrbitals):

            kcld = 0
            for k in range(nOccupied):
                for c in range(nOccupied, spinOrbitals):
                    for l in range(nOccupied):
                        for d in range(nOccupied, spinOrbitals):

                            hsd[ia,kcld] += 0.5*eriMOspin[a,l,c,d]*(i==k)                                                                   #[18]
                            hsd[ia,kcld] -= 0.5*eriMOspin[k,l,i,d]*(a==c)                                                                   #[20]

                            kcld += 1

            ia += 1

    if dialog: print('finished Hsd block in : {:>8.3f} s'.format(time.time()-start))
    start = time.time()

    #get doubles-singles 
    hds = np.zeros((nRotate*nRotate, nRotate))  

    iajb = 0
    for i in range(nOccupied):
        for a in range(nOccupied, spinOrbitals):
            for j in range(nOccupied):
                for b in range(nOccupied, spinOrbitals):

                    kc = 0
                    for k in range(nOccupied):
                        for c in range(nOccupied, spinOrbitals):

                            hds[iajb,kc] += (i==k)*eriMOspin[a,b,c,j] - (j==k)*eriMOspin[a,b,c,i]                                           #[33]                                                                     
                            hds[iajb,kc] += (b==c)*eriMOspin[k,a,i,j] - (a==c)*eriMOspin[k,b,i,j]                                           #[22]

                            for e in range(nOccupied, spinOrbitals):
                                hds[iajb,kc] += eriMOspin[k,a,c,e]*td[e,b,i,j] -  eriMOspin[k,b,c,e]*td[e,a,i,j]                            #[43]

                                for f in range(nOccupied, spinOrbitals):
                                    hds[iajb,kc] += 0.5*td[e,f,i,j]*( (b==c)*eriMOspin[k,a,e,f] - (a==c)*eriMOspin[e,f,k,b])                #[24]

                            for m in range(nOccupied):
                                hds[iajb,kc] += eriMOspin[k,m,c,j]*td[a,b,m,i] - eriMOspin[k,m,c,i]*td[a,b,m,j]                             #[45]

                                for n in range(nOccupied):
                                    hds[iajb,kc] += 0.5*td[a,b,m,n]*((i==k)*eriMOspin[m,n,c,j] - (j==k)*eriMOspin[m,n,c,i])                 #[35]       

                                for e in range(nOccupied, spinOrbitals):
                                    hds[iajb,kc] += (i==k)*eriMOspin[a,m,c,e]*td[e,b,m,j] - (i==k)*eriMOspin[b,m,c,e]*td[e,a,m,j] - \
                                                    (j==k)*eriMOspin[a,m,c,e]*td[e,b,m,i] + (j==k)*eriMOspin[b,m,c,e]*td[e,a,m,i]           #[34]

                                    hds[iajb,kc] += (b==c)*eriMOspin[k,m,i,e]*td[e,a,m,j] - (b==c)*eriMOspin[k,m,j,e]*td[e,a,m,i] - \
                                                    (a==c)*eriMOspin[k,m,i,e]*td[e,b,m,j] + (a==c)*eriMOspin[k,m,j,e]*td[e,b,m,i]           #[23]

                            kc += 1

                    iajb += 1

    if dialog: print('finished Hds block in : {:>8.3f} s'.format(time.time()-start))
    start = time.time()

    #get doubles-doubles 
    hdd = np.zeros((nRotate*nRotate, nRotate*nRotate))  

    if not partitioned:
        iajb = 0
        for i in range(nOccupied):
            for a in range(nOccupied, spinOrbitals):
                for j in range(nOccupied):
                    for b in range(nOccupied, spinOrbitals):

                        kcld = 0
                        for k in range(nOccupied):
                            for c in range(nOccupied, spinOrbitals):
                                for l in range(nOccupied):
                                    for d in range(nOccupied, spinOrbitals):

                                        hdd[iajb,kcld] += (j==k)*(i==l)*(a==d)*fockMOspin[b,c] - \
                                                          (j==k)*(i==l)*(b==d)*fockMOspin[a,c]                                                  #[50]
                                        hdd[iajb,kcld] += (j==l)*(a==d)*(b==c)*fockMOspin[k,i] - \
                                                          (i==l)*(a==d)*(b==c)*fockMOspin[k,j]                                                  #[55]
                                        hdd[iajb,kcld] += 0.5*(i==k)*(j==l)*eriMOspin[a,b,c,d]                                                  #[60]
                                        hdd[iajb,kcld] += 0.5*(a==c)*(b==d)*eriMOspin[k,l,i,j]                                                  #[64]
                                        hdd[iajb,kcld] += (i==l)*(a==d)*eriMOspin[k,b,c,j] - \
                                                          (j==l)*(a==d)*eriMOspin[k,b,c,i] - \
                                                          (i==l)*(b==d)*eriMOspin[k,a,c,j] + \
                                                          (j==l)*(b==d)*eriMOspin[k,a,c,i]                                                      #[68]
                                        for e in range(nOccupied, spinOrbitals):
                                            hdd[iajb,kcld] += 0.5*(b==d)*eriMOspin[k,l,c,e]*td[e,a,i,j] - \
                                                              0.5*(a==d)*eriMOspin[k,l,c,e]*td[e,b,i,j]                                         #[73] 
                                            for f in range(nOccupied, spinOrbitals):
                                                hdd[iajb,kcld] += 0.25*(a==c)*(b==d)*eriMOspin[k,l,e,f]*td[e,f,i,j]                             #[66]
                                        for m in range(nOccupied):
                                            hdd[iajb,kcld] += 0.5*(j==l)*eriMOspin[k,m,c,d]*td[a,b,m,i] - \
                                                              0.5*(i==l)*eriMOspin[k,m,c,d]*td[a,b,m,j]                                         #[74]
                                            for n in range(nOccupied):
                                                hdd[iajb,kcld] += 0.25*(i==k)*(j==l)*eriMOspin[m,n,c,d]*td[a,b,m,n]                             #[62]
                                            for e in range(nOccupied, spinOrbitals):
                                                hdd[iajb,kcld] += (j==l)*(b==d)*eriMOspin[k,m,c,e]*td[e,a,m,i] - \
                                                                  (i==l)*(b==d)*eriMOspin[k,m,c,e]*td[e,a,m,j] - \
                                                                  (j==l)*(a==d)*eriMOspin[k,m,c,e]*td[e,b,m,i] + \
                                                                  (i==l)*(a==d)*eriMOspin[k,m,c,e]*td[e,b,m,j]                                  #[71]
                                                for n in range(nOccupied):
                                                    hdd[iajb,kcld] += 0.5*(i==k)*(j==l)*(a==d)*eriMOspin[m,n,e,c]*td[b,e,n,m] - \
                                                                      0.5*(i==k)*(j==l)*(b==d)*eriMOspin[m,n,e,c]*td[a,e,n,m]                   #[53]
                                                for f in range(nOccupied, spinOrbitals):
                                                    hdd[iajb,kcld] += 0.5*(a==c)*(i==l)*(b==d)*eriMOspin[m,k,e,f]*td[f,e,j,m] - \
                                                                      0.5*(a==c)*(j==l)*(b==d)*eriMOspin[m,k,e,f]*td[f,e,i,m]                   #[58]

                                        kcld += 1
                        iajb += 1

    else:

        iajb = 0
        for i in range(nOccupied):
            for a in range(nOccupied, spinOrbitals):
                for j in range(nOccupied):
                    for b in range(nOccupied, spinOrbitals):

                        kcld = 0
                        for k in range(nOccupied):
                            for c in range(nOccupied, spinOrbitals):
                                for l in range(nOccupied):
                                    for d in range(nOccupied, spinOrbitals):

                                        hdd[iajb,kcld] += (j==k)*(i==l)*(a==d)*fockMOspin[b,c] - \
                                                          (j==k)*(i==l)*(b==d)*fockMOspin[a,c]                                                  #[50]
                                        hdd[iajb,kcld] += (j==l)*(a==d)*(b==c)*fockMOspin[k,i] - \
                                                          (i==l)*(a==d)*(b==c)*fockMOspin[k,j]                                                  #[55]
                                        kcld += 1
                        iajb += 1

    if dialog: print('finished Hdd block in : {:>8.3f} s'.format(time.time()-start))

    eomMatrix = np.bmat([[hss,hsd],[hds,hdd]])

    if dialog:
        print("begin full diagonalization")
        print("matrix dimension:  ", str(len(eomMatrix)) + "x" + str(len(eomMatrix)) )

    eomEVal,eomEVec = sp.linalg.eig(eomMatrix)

    return eomEVal.real, eomEVec


if __name__ == '__main__':

    #generate a scf calculation
    fileName = '../test/h2.hpf'
    f = open(fileName, 'w')
    hpf = 'name=h2\nmatrix=c\ndiis=off\nunits=angstrom\npost={}\nbasis=3-21g\n\nH1 1 0.00000000  0.00000000 0.00000000000\nH2 1 0.00000000  0.00000000 0.74\nend' 
    f.write(hpf)
    f.close()

    molAtom, molBasis, molData = rhf.mol([], fileName)
    eSCF = rhf.scf(molAtom, molBasis, molData, [])

    #clean up file
    if os.path.exists(fileName):
        os.remove(fileName)

    #occupations
    charge = molData['charge']
    nOccupied = electronCount(molAtom, charge)
    spinOrbitals = len(molBasis) * 2
    nVirtual = spinOrbitals - nOccupied

    #build eriMO spin tensor in physisist notation
    eriMO = buildEriMO(rhf.C, rhf.ERI)
    eriMOspin = buildEriDoubleBar(spinOrbitals, eriMO)  

    #build FockMO in spin formalism
    fockMOspin = buildFockMOspin(spinOrbitals, rhf.C, rhf.fock)

    #get ccsd singles and doubles amplitudes
    iterations = 30
    convergence = 1e-8
    diisStatus = 'on'
    ccsdEnergy, ts, td = cc.scc.ccsd(molAtom, rhf.C, charge, rhf.fock, rhf.ERI, iterations, convergence, rhf.SCFenergy, diisStatus)

    #most significant amplitudes
    maxts = maximumAmplitudes(ts, 5)
    maxtd = maximumAmplitudes(td, 5)

    #do eom-ccsd computation
    eomEVal, eomEVec = eomccsd(fockMOspin, eriMOspin, ts, td, nOccupied, nVirtual, spinOrbitals, partitioned=False)

    from ci import ciDegeneracy
    eomEVal = np.sort(eomEVal * getConstant('hartree->eV'))
    ccsdExcitations = ciDegeneracy(eomEVal) 

    #Gaussian reference (from JJ Goings)
    eomTest = []
    for excitation in eomEVal:
        if excitation > 1.0 and excitation < 50.0: eomTest.append(excitation) 
    gaussianReference = [10.8527,10.8527,10.8527,15.8984,26.4712,26.4712,26.4712,30.5216,31.8814,40.4020,40.4020,40.4020]
    assert np.allclose(eomTest[:12], gaussianReference), 'EOM-CCSD Gaussian check failed - H2'

    #do eom-mbpt(2) computation
    eomEVal, eomEVec = eommbpt2(fockMOspin, eriMOspin, nOccupied, nVirtual, spinOrbitals, partitioned = False)

    eomEVal = np.sort(eomEVal * getConstant('hartree->eV'))
    mbptExcitations = ciDegeneracy(eomEVal)

    #Gaussian reference (from JJ Goings)
    eomTest = []
    for excitation in eomEVal:
        if excitation > 1.0 and excitation < 50.0: eomTest.append(excitation) 
    gaussianReference = [10.6572, 10.6572, 10.6572, 15.7087, 26.2655, 26.2655, 26.2655, 30.2223, 31.6785, 40.2073, 40.2073, 40.2073]
    assert np.allclose(eomTest[:12], gaussianReference) , 'EOM-MBPT2 Gaussian check failed - H2'

    #write results to HTML file
    from view import pre, postSCF, post
    pre('h2','rhf')
    postSCF([[eSCF, ccsdEnergy], [maxts, maxtd], ccsdExcitations, mbptExcitations, [1.0, 50.0]], 'eom')
    post()

