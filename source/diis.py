from __future__ import division
import numpy as np
from numpy.linalg import solve, LinAlgError

class diis_f(object):
    #direct inversion of iterative subspace class

    def __init__(self, size):
        self.size = size
        self.fock_vector  = []
        self.error_vector = []
        self.norm = 0.0

    def append(self, f, d, s, x):
        #update the subspaces respecting capacity of buffer

        self.fock_vector.append(f) 
        if f.ndim == 2:
            fds = np.einsum('im,mn,nj->ij',f, d, s, optimize=True)
            self.error_vector.append(np.einsum('mi,mn,nj->ij', x, (fds - fds.T), x, optimize=True))
        elif f.ndim == 3:
            fds = np.einsum('xim,xmn,nj->xij',f, d, s, optimize=True)
            self.error_vector.append(np.vstack(np.einsum('mi,xmn,nj->xij', x, (fds - fds.transpose(0,2,1)), x, optimize=True)))

        self.norm = np.linalg.norm(self.error_vector[-1])

        #check capacity
        if len(self.fock_vector) > self.size:
            del self.fock_vector[0]
            del self.error_vector[0]


    def build(self, f, d, s, x):
        #compute extrapolated Fock

        #update buffers
        self.append(f, d, s, x)

        #construct B matrix
        nSubSpace = len(self.fock_vector)
        
        #start diis after cache full
        if nSubSpace < self.size: return f

        b = -np.ones((nSubSpace+1,nSubSpace+1))
        b[:-1,:-1] = 0.0 ; b[-1,-1] = 0.0
        for i in range(nSubSpace):
            for j in range(nSubSpace):
                b[i,j] = b[j,i] = np.einsum('ij,ij->',self.error_vector[i], self.error_vector[j], optimize=True)


        #solve for weights
        residuals = np.zeros(nSubSpace+1)
        residuals[-1] = -1

        try:
            weights = np.linalg.solve(b, residuals)
        except np.linalg.LinAlgError as e:
            if 'Singular matrix' in str(e): exit('diis failed with singular matrix')

        #weights should sum to +1
        sum = np.sum(weights[:-1])
        assert np.isclose(sum, 1.0)

        #construct extrapolated Fock
        f = np.zeros_like(f , dtype='float')
        for i in range(nSubSpace):
            f += self.fock_vector[i] * weights[i]

        if f.ndim == 3:
            f = f.reshape(d.shape)

        return f
      
'''
Modified diis class to handle three amplitudes for triples
'''
class diis_c(object):

    def __init__(self, capacity, amplitudes):

        self.buffer = [amplitudes]
        self.errors = []
        self.store  = []
        self.size   = 0
        self.capacity = capacity

    def refresh_store(self, amplitudes):
        #store current amplitude proir to update

        self.store = amplitudes

    def build(self, amplitudes):
        #build the new amplitudes

        self.buffer.append([i.copy() for i in amplitudes])
        n_amplitudes = len(amplitudes)
        error = [(self.buffer[-1][i] - self.store[i]).ravel() for i in range(n_amplitudes)]
        self.errors.append(np.concatenate(error))

        if (len(self.buffer) > self.capacity):
            del self.buffer[0]
            del self.errors[0]
        self.size = len(self.buffer) - 1

        #construct b-matrix
        b = -np.ones((self.size + 1, self.size + 1))
        b[-1, -1] = 0

        n = self.size
        b = np.zeros((n+1,n+1))
        for i in range(0, n):
            for j in range(0, i+1):
                b[j,i] = np.dot(self.errors[i], self.errors[j])
                b[i,j] = b[j,i]
        for i in range(0, n):
            b[n, i] = -1
            b[i, n] = -1

        # Build residual vector
        residual = np.zeros(self.size + 1)
        residual[-1] = -1

        # Solve Pulay equations for weights
        try:
            w = np.linalg.solve(b, residual)
        except np.linalg.LinAlgError as e:
            if 'Singular matrix' in str(e): exit('diis failed with singular matrix')

        # Calculate new amplitudes
        amplitudes = [amplitudes[i]*0.0 for i in range(n_amplitudes)]
        for num in range(self.size):
            for i in range(n_amplitudes): amplitudes[i] += w[num] * self.buffer[num + 1][i]

        return amplitudes
