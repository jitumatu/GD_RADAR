# This program performs a simulation for a radar system using Gabor Division signals with Terminal Collaboration.
# 

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.fftpack as fft

# constants
c = 3e8 # the speed of light
W = 400e6 # bandwidth 400 MHz

pulse = "Gaussian"
# pulse = "Rectangular"

P = 1 # The number of blocks
N_t = 8 # the spreading factor of the time domain
N_f = 8 # the spreading factor of the frequency domain

N = 512 # The number of total time slots in a block
M = 64 # The number of total subcarriers

L = P * N * M # The length of the transmitted signal

F_c = W/N_f # chip bandwidth 50 MHz
T_c = 1/F_c # chip interval 20 ns 
Delta_t = T_c/M # time resolution 625 ps
Delta_f = 1/L/Delta_t # frequency resolution 97.65625 kHz

# the number of terminals
N_terminals = 4

# 3 dimensional positions of the terminals
p = np.array( [ [ 3, -3, 0], [-3, -3, 0], [0, 3, 0], [0, 0, 6] ] ) # Units are in meters
# the position of the target
q = np.random.randint(-1000, 1000, (3,) )
print("The position of the target is: ", q, "meters")

# the distance between the antennas and the target
d = np.sqrt( np.sum( (p-q)**2, axis=1) )

RoundTrip = d + d[0] # the round trip time of the signal; the first terminal is the transmitter
k_d = np.rint( RoundTrip / c / (Delta_t) ).astype(np.int32) # the number of time slots for the signal to travel from the first terminal to the target and back to each terminals
l_D = np.random.randint( - P * N * (M - N_f) // 2, P * N * (M - N_f) // 2) # the Doppler shift

print("The round trip times are: ", RoundTrip, "seconds")
print("k_d=", k_d)
print("l_D=", l_D)


# Define the rectangular or Gaussian waveform
if pulse == "Rectangular":
    g = np.ones( ( M, ) )
    g = np.pad( g,  [0,L-M], 'constant' )
elif pulse == "Gaussian":
    g = 2**(1/4)*np.exp( - np.pi*( np.arange(0, P * N * M, 1 )/M - P * N/2 )**2) # Gaussian waveform
    g = np.roll( g, -( ( P * N - 3 ) * M) // 2 )
#print(g)
#print( g.shape )


G = np.fft.fft(g.reshape( (P * N, M) ), axis=0) # the discrete Zak transform of g.
#print(np.abs(G))

# fig=plt.figure()
# plt.plot(g)
# plt.show()

# Generate the random binary data
X_tf = np.random.randint(0, 2, (P, N_t, N_f)) * 2 - 1 
#print(X_tf)

X_tf = np.pad( X_tf, [(0,0),(0, N - N_t), (0, M - N_f)], 'constant')
X_tf = np.reshape( X_tf, (P*N, M))
X_tf = np.roll( X_tf, -( N_f//2 ), axis=1) # shift the frequency domain
#print(X_tf)

# X is the symplectic Fourier transform 
X = np.fft.ifft( np.fft.fft( X_tf, axis = 0, n = P * N ), axis = 1, n = M ) 

S = X * G # Discrete Zak transform of s 
#print(S.shape)

s = np.fft.ifft( S, axis = 0).flatten() # inverse discrete Zak transform
#print(s)

# fig = plt.figure()
# plt.plot( np.real(s) )
# plt.show()


# additive white Gaussian noise
SNR_in_dB = 40
SNR = 10**(SNR_in_dB/10)
sigma =  np.linalg.norm(s) * np.sqrt( 1 / L / SNR / 2 )
noise = sigma * ( np.random.randn( N_terminals, L ) + 1j * np.random.randn( N_terminals, L ) )
# r = s

# received signal
# add the delay and Doppler shift
# t_d = np.random.randint( (N_t + 2) * M, ( N - N_t - 2) * M)
# f_D = np.random.randint( - P * N * (M - N_f) // 2, P * N * (M - N_f) // 2)

#print(t_d, f_D)
k_d_est = np.zeros( (N_terminals,) )
l_D_est = np.zeros( (N_terminals,) )

# Determine the received signal for each terminal and estimate the delay and Doppler shift
for i in range(N_terminals): 

    r = np.roll( s + noise[i], k_d[i], axis = 0 )
    R = np.fft.fft( r, axis = 0 )
    R = np.roll( R, l_D, axis = 0 )
    r = np.fft.ifft( R, axis = 0 )

    # fig = plt.figure()
    # plt.plot( np.arange(L), np.real( r ) )
    # plt.plot( np.arange(L), np.real( s ) ) 
    # plt.show()

    # S = np.fft.fft( s ) # overwrite S as the frequency domain representation of s 
    # S = np.fft.fftshift( S ) # shift the frequency domain 
    # R = np.fft.fft( r ) # overwrite R as the frequency domain representation of r
    # R = np.fft.fftshift( R ) # shift the frequency domain

    # fig = plt.figure()
    # plt.plot( np.arange(-L/2,L/2), np.real( R ) )
    # plt.plot( np.arange(-L/2,L/2), np.real( S ) ) 
    # plt.show()


    # ambiguity function between r and s
    circ_s = sp.linalg.circulant( s )
    circ_s = circ_s[ :, 0 : N * M ]

    r_s = r * np.conj( circ_s.T ) 
    A = np.fft.fft( r_s, axis = 1 )

    [k_d_est[i], l_D_est[i]] = np.unravel_index( np.argmax( np.abs( A ) ), A.shape )

#    A_zoom = A[t_d_est - 2 * M//N_f : t_d_est + 2 * M//N_f, f_D_est -  2 * P*N//N_t : f_D_est + 2 * P * N//N_t ]

    # fig=plt.figure()
    # plt.imshow( np.abs(A.T), aspect = 'auto', cmap = 'jet' )
    # plt.show()


    # plot the ambiguity function
    # fig=plt.figure()
    # plt.imshow( np.abs(A_zoom.T), aspect = 'auto', cmap = 'jet' )
    # plt.show()

    if l_D_est[i] > L // 2:
        l_D_est[i] = l_D_est[i] - L

    print( k_d_est[i], l_D_est[i] )


# recover the distance from the estimated delay. 
Estimated_d = (k_d_est - k_d_est[0]/2) * c * Delta_t

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

x0 = initial_guess( Estimated_d , p)

print("The initial guess of the target's position is: ", x0)
print("the distance between the initial guess and the true position is: ", np.sqrt( np.sum( (x0-q)**2 ) ), "meters")
print("The MSE is ", mse(x0, Estimated_d, p))

#x0 = q + np.random.randint(-100,100, (3,)) #
# x0 = np.array( [ 0, 0, 0] )
# result = sp.optimize.minimize( fun = mse, x0=x0, args=(d, p), jac= dmsedx, method='BFGS' )
result = sp.optimize.minimize( fun = mse, x0 = x0, args = (Estimated_d, p), method = 'BFGS' )
#result = sp.optimize.minimize( fun = mse, x0=x0, args=(d, p), method='CG' )

#print(result.x)
print("The estimated position of the target is: ", result.x, "meters")
print("the distance between the estimated position and the true position is: ", np.sqrt( np.sum( (result.x-q)**2 ) ), "meters")
print("The MSE is", mse(result.x, d, p))

