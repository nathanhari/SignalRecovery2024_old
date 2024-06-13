# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 15:13:59 2024

@author: natha
"""


import cmath
import numpy as np
from FourierTransform import chi, fourier_transform, inverse_fourier_transform

'''
This function takes as input a function with missing values and where the
fourier transform is (treated as) being zero and uses the algorithm to return
the full function.

f: an array of length N. The assumption is that f is a function from Z_N to
C. The values of the array should be either complex numbers from using the
cmath libray or None, representing the missing values.

fourier_zero: this is the values of Z_N for which we are assuming that the
fourier transform of f is zero. There needs to be at least as many values in
here as there are missing values in f.
'''
def recover_with_zero_fourier(f, fourier_zero):
    domain = list(range(len(f)))
    N = len(domain)
    M = [i for i in domain if f[i] is None]
    M_c = [i for i in domain if f[i] is not None]
    
    A = np.array([[chi(-1 * x * w, N) for x in M] for w in fourier_zero])
    b = np.array(
        [sum([chi(-1 * x * w, N) * f[x] for x in M_c]) for w in fourier_zero]
    )
        
    f_found = np.linalg.solve(A, -1 * b)
    
    for i in range(len(M)):
        f[M[i]] = f_found[i]
    
    return(f)


if __name__ == "__main__":
    print("Test Cases")
    wrongs = list()
    for test in range(100):
        N = np.random.randint(100, 1001)
        f_hat = [complex(np.random.rand(), np.random.rand()) for m in range(N)]
        n_zeros = np.random.randint(1, N/5)
        zeros = np.random.choice(range(N), n_zeros, replace=False)
        for m in zeros:
            f_hat[m] = 0
        f = inverse_fourier_transform(f_hat)
        n_missing = n_zeros
        missing = np.random.choice(range(N), n_missing, replace=False)
        f_missing = [f[x] if x not in missing else None for x in range(N)]
        f2 = recover_with_zero_fourier(f_missing, zeros)
        n_wrong = sum(np.abs(np.array(f) - np.array(f2)) > 0.001)
        print('-')
        if n_wrong > 0:
            print('{}/{}/{}'.format(n_wrong, n_missing, N))
            wrongs.append(
                {
                    'f_hat': f_hat,
                    'zeros': zeros,
                    'f': f,
                    'f_missing': f_missing,
                    'f2': f2,
                    'missing': missing
                }
            )