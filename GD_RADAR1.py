# This program performs a simulation for a radar system using Gabor Division signals.
# 
# 
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.fftpack as fft

pulse = "Gaussian"
# pulse = "Rectangular"

P = 1 # The number of blocks
N_t = 8 # the spreading factor of the time domain

N_f = 8 # the spreading factor of the frequency domain

N = 512 # The number of total time slots in a block
M = 32 # The number of total subcarriers

L = P * N * M # The length of the transmitted signal

Paths = 4 # The number of paths

# Generate the Gaussian waveform

if pulse == "Rectangular":
    g = np.ones( ( M, ) )
    g = np.pad( g,  [0,L-M], 'constant' )
elif pulse == "Gaussian":
    g = 2**(1/4)*np.exp( - np.pi*( np.arange(0, P * N * M, 1 )/M - P * N/2 )**2) # Gaussian waveform
    g = np.roll( g, -( ( P * N - 3 ) * M) // 2 )
print(g)
print( g.shape )

G = np.fft.fft(g.reshape( (P * N, M) ), axis=0)

print(np.abs(G))

# fig=plt.figure()
# plt.plot(g)
# plt.show()

# Generate the random binary data
X_tf = np.random.randint(0, 2, (P, N_t, N_f))*2-1 
print(X_tf[0])

X_tf = np.pad( X_tf, [(0,0),(0, N - N_t), (0, M - N_f)], 'constant')
X_tf = np.reshape( X_tf, (P*N, M))
X_tf = np.roll( X_tf, -( N_f//2 ), axis=1) # shift the frequency domain
# X is the symplectic Fourier transform 
print(X_tf)

X = np.fft.ifft( np.fft.fft( X_tf, axis = 0, n = P * N ), axis = 1, n = M ) 

S = X * G # Discrete Zak transform of s 
print(S.shape)

s = np.fft.ifft( S, axis = 0).flatten() # inverse discrete Zak transform
print(s)

# fig = plt.figure()
# plt.plot( np.real(s) )
# plt.show()


# additive white Gaussian noise
SNR_in_dB = 20
SNR = 10**(SNR_in_dB/10)
sigma =  np.linalg.norm(s) * np.sqrt( 1 / L / SNR / 2 )
#r = s + sigma * ( np.random.randn( L, ) + 1j * np.random.randn( L, ) )
r = s

# received signal
# add the delay and Doppler shift
t_d = np.random.randint( (N_t + 2) * M, ( N - N_t - 2) * M)
f_D = np.random.randint( - P * N * (M - N_f) // 2, P * N * (M - N_f) // 2)

print(t_d, f_D)

r = np.roll( r, t_d, axis = 0 )
R = np.fft.fft( r, axis = 0 )
R = np.roll( R, f_D, axis = 0 )
r = np.fft.ifft( R, axis = 0 )

fig = plt.figure()
plt.plot( np.arange(L), np.real( r ) )
plt.plot( np.arange(L), np.real( s ) ) 
plt.show()

S = np.fft.fft( s ) # overwrite S as the frequency domain representation of s 
S = np.fft.fftshift( S ) # shift the frequency domain 
R = np.fft.fft( r ) # overwrite R as the frequency domain representation of r
R = np.fft.fftshift( R ) # shift the frequency domain


fig = plt.figure()
plt.plot( np.arange(-L/2,L/2), np.real( R ) )
plt.plot( np.arange(-L/2,L/2), np.real( S ) ) 
plt.show()


# ambiguity function between r and s

circ_s = sp.linalg.circulant( s )
circ_s = circ_s[ :, 0 : N * M ]

r_s = r * np.conj( circ_s.T ) 
A = np.fft.fft( r_s, axis = 1 )

[t_d_est, f_D_est] = np.unravel_index( np.argmax( np.abs( A ) ), A.shape )

A_zoom = A[t_d_est - 2 * M//N_f : t_d_est + 2 * M//N_f, f_D_est -  2 * P*N//N_t : f_D_est + 2 * P * N//N_t ]

# fig=plt.figure()
# plt.imshow( np.abs(A.T), aspect = 'auto', cmap = 'jet' )
# plt.show()


# plot the ambiguity function
fig=plt.figure()
plt.imshow( np.abs(A_zoom.T), aspect = 'auto', cmap = 'jet' )
plt.show()

if f_D_est > L // 2:
    f_D_est = f_D_est - L

print( t_d_est, f_D_est )

