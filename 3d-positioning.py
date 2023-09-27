# This program detects the position of the target from the information of the distances between the antennas and the target.

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# initialize the parameters
# The number of antennas
N_R = 4
# the position of the antennas
p = np.array( [ [ 1, -1, 0], [-1, -1, 0], [0, 1, 0], [0, 0, 2] ] )
# the position of the target
q = np.random.randint(-2000, 2000, (3,) )
# the distance between the antennas and the target
d = np.sqrt( np.sum( (p-q)**2, axis=1) )

#print(d)

# start estimation

def mse( x, d, p ):
    dim = p.shape[0]
    return np.sum( ( np.sqrt( np.sum( (p - np.ones((dim,1),dtype='int')*x)**2, axis=1) ) - d )**2 )

# The following function is the derivative of the mse function (numerical results are incorrect. It must contain bugs)
def dmsedx( x, d, p ):
    [n,m] = p.shape
    r = np.sqrt( np.sum( (p - np.ones((n,1),dtype='int')*x)**2, axis=1) )
    result = np.zeros( (m,) )
    for j in range(m):
        result[j] +=  2 * np.sum( (r - d) / r * ( p[:,j] - x[j]) )  
    
    return result

## obtain the initial guess
## If the distances d are correct, the initial guess gives the perfect estimation. 
def initial_guess(d, p):
    A = 2 * ( p - np.roll(p, -1, axis=0) )
    b = np.sum( p**2 - np.roll(p, -1, axis=0)**2, axis=1 ) - d**2 + np.roll(d, -1)**2
    A = A[0:3,:]
    b = b[0:3]
    x0 = np.linalg.solve( A, b )
    return x0

x0 = initial_guess(d, p)
print(x0)
print(mse(x0, d, p))

#x0 = q + np.random.randint(-100,100, (3,)) #
# x0 = np.array( [ 0, 0, 0] )
# result = sp.optimize.minimize( fun = mse, x0=x0, args=(d, p), jac= dmsedx, method='BFGS' )
result = sp.optimize.minimize( fun = mse, x0=x0, args=(d, p), method='BFGS' )
#result = sp.optimize.minimize( fun = mse, x0=x0, args=(d, p), method='CG' )


print(result.x)
print(q)

print(mse(result.x, d, p))