# This program performs a simulation for a radar system using Gabor Division Spread Spectrum signals.
# Accuare estimation is performed after the initial estimation. 
# A terget is randomly located in [-1000, 1000]^3 meters. (Hight of the target may be too high.)
# The number of terminals (drones with antennas) is 4.
# A pulse waveform is transmitted by the 0-th terminal and is reveved by all four terminals.
# We assume that the relative positionning of the four terminals are perfect. 

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time
import sys

FIG_OPTION = True # True: show figures; False: do not show figures

# constants
c = 3e8 # the speed of light
W = 240e6 # bandwidth 10 MHz
f_c = 24e9 # carrier frequency 24 GHz

pulse = "Gaussian"
#pulse = "Rectangular"

P = 1 # The number of blocks
N_t = 8 # the spreading factor of the time domain
N_f = 8 # the spreading factor of the frequency domain

N = 64 # The number of total time slots in a block, which must be greater than N_t.
M = 64 # The number of total subcarriers, which must be greater than N_f.

# N_t = 4 # the spreading factor of the time domain
# N_f = 256 # the spreading factor of the frequency domain

# N = 32 # The number of total time slots in a block, which must be greater than N_t.
# M = 512 # The number of total subcarriers, which must be greater than or equal to N_f.

L = P * N * M # The length of the transmitted signal

U_t = 10 # the oversampling factor of the time domain
U_f = 10 # the oversampling factor of the frequency domain

F_c = W/N_f # chip bandwidth 
T_c = 1/F_c # chip interval 

f_s = M * F_c # sampling frequency 
T_s = 1/f_s # sampling interval 
Delta_f = f_s/L # frequency resolution in FFT 

SNR_in_dB = 40
SNR = 10**(SNR_in_dB/10)

print("N_t = ", N_t)
print("N_f = ", N_f)
print("N = ", N)
print("M = ", M)
print("L = ", L)
print("F_c = ", F_c)
print("T_c = ", T_c)
print("f_s = ", f_s)
print("T_s = ", T_s)
print("Delta_f = ", Delta_f)

print("PRI = ", N * T_c)
print("CPI = ", P * N * T_c)

print("maximum Range = ", c * N * T_c / 2)
print("maximum delay = ", (N-N_t) * T_c)
print("maximum Doppler shift = ", (f_s - W)/ 2)
print("frequency bin width = ", f_s / L)

print("distance resolution = ", c * T_s)


# the number of terminals
# N_terminals = 4
N_terminals = 1 

# 3 dimensional positions of the terminals
if N_terminals == 4:
    p = np.array( [ [ 3, -3, 0], [-3, -3, 0], [0, 3, 0], [0, 0, 6] ] ) # Units are in meters
elif N_terminals == 1:
    p = np.array( [ [0, 0, 0] ] )
# Other numbers of terminals have not been considered yet.

# the position of the target
q = np.random.randint(-1000, 1000, (3,) )
print("The position of the target is: ", q, "meters")

# The following condition is necessary for the target to be detected.
if N < 2 * 1000 * np.sqrt(3) * W / c / N_f:
    print("The number of time slots is too small. The target may not be detected.")
    # sys.exit()

# the distance between the antennas and the target
d = np.sqrt( np.sum( (p-q)**2, axis=1) )

RoundTrip = d + d[0] # the round trip distance of the signal; the first terminal is the transmitter
# t_d = ( RoundTrip / c ).astype(np.float32) # time delay = the round trip time 
t_d = np.random.rand(1,) * (N - 2 * N_t) * T_c + N_t * T_c 
f_D = ( np.random.rand() - 0.5 ) * ( f_s - W )   # Doppler shift 

# to draw an ambiguity function without offsets in time and frequency
t_d = np.array([ N * N_t * T_c //2 ]) 
f_D = 0

k_d = np.rint( t_d / T_s  ).astype(np.int32) # the closest integer delay in T_s unit  
l_D = np.rint( f_D / f_s * L ).astype(np.int32) # the closest frequency bin of the Doppler shift 

print("The round trip times are: ", RoundTrip, "seconds")
print("t_d = ", t_d, ", t_d / T_s=", t_d/T_s)
print("k_d = ", k_d)
print("f_D = ", f_D, ", f_D / f_s * L = ", f_D /f_s * L)
print("l_D = ", l_D)

# Define the rectangular or Gaussian waveform
if pulse == "Rectangular":
    g = np.ones( ( M, ) )
    g = np.pad( g,  [0, L-M], 'constant' )
    L_s = N_t * M # the effectve length of s 
elif pulse == "Gaussian":
    g = 2**(1/4)*np.exp( - np.pi*( np.arange(0, P * N * M, 1 )/M - P * N/2 )**2) # Gaussian waveform
    g = np.roll( g, -( ( P * N - 3 ) * M) // 2 )
    L_s = (N_t + 2) * M # the effectve length of s 
# print(g)
# print( g.shape )
# fig=plt.figure()
# plt.plot(g)
# plt.show()

G = np.fft.fft(g.reshape( (P * N, M) ), axis=0) # the discrete Zak transform of g.
# print(np.abs(G))

# Generate the random binary data
X_tf = np.random.randint(0, 2, (P, N_t, N_f)) * 2 - 1 

# Good example 
# X_tf = [ [ [-1,  1, -1,  1, -1,  1, -1, -1],
#    [-1, -1, -1,  1,  1, -1,  1,  1],
#    [-1,  1,  1, -1, -1, -1,  1,  1],
#    [-1, -1,  1,  1,  1, -1, -1,  1],
#    [-1,  1,  1,  1, -1, -1, -1, -1],
#    [-1,  1,  1, -1, -1, -1,  1,  1],
#    [-1, -1,  1,  1,  1, -1, -1,  1],
#    [ 1,  1,  1,  1, -1,  1,  1, -1] ] ] 

# Bad example
# X_tf =[[[-1, -1,  1, -1, -1, -1,  1,  1],
#   [ 1,  1,  1,  1,  1, -1,  1,  1],
#   [-1, -1, -1, -1, -1,  1, -1, -1],
#   [ 1, -1, -1, -1,  1,  1,  1, -1],
#   [-1, -1,  1,  1,  1,  1, -1, -1],
#   [ 1,  1, -1,  1,  1, -1,  1,  1],
#   [ 1, -1,  1, -1,  1, -1,  1,  1],
#   [-1,  1,  1, -1, -1, -1,  1,  1]]]

print(X_tf)

X_tf = np.pad( X_tf, [(0,0),(0, N - N_t), (0, M - N_f)], 'constant')
X_tf = np.reshape( X_tf, (P*N, M))
X_tf = np.roll( X_tf, -( N_f//2 ), axis=1) # shift the frequency domain

# X is the symplectic Fourier transform 
X = np.fft.ifft( np.fft.fft( X_tf, axis = 0, n = P * N ), axis = 1, n = M ) 

S = X * G # Discrete Zak transform of s 
s = np.fft.ifft( S, axis = 0).flatten().astype(np.complex64) # inverse discrete Zak transform

# print(s)
# fig = plt.figure()
# plt.plot( np.real(s) )
# plt.show()

# additive white Gaussian noise
sigma =  np.linalg.norm(s) * np.sqrt( 1 / L / SNR / 2 )
noise = sigma * ( np.random.randn( N_terminals, L ) + 1j * np.random.randn( N_terminals, L ) )

k_d_est = np.zeros( (N_terminals,), dtype=np.int32() )
l_D_est = np.zeros( (N_terminals,), dtype=np.int32() )

epsilon_t_d = np.zeros( (N_terminals,), dtype=np.int32())
epsilon_f_D = np.zeros( (N_terminals,), dtype=np.int32() )

def make_H_matrix( n, m, tau, nu):
    # generate the ambiguity function matrix

    a = 1/T_s - np.abs(nu) 

    if a > 0:
        r = np.exp ( 1j * np.pi * nu * ( tau + np.arange(n) * T_s ) )
        c = np.exp ( 1j * np.pi * nu * ( tau + n - 1 + np.arange(m) * T_s ) )
        E = sp.linalg.hankel( r, c )

        r = np.sinc( a * ( tau - np.arange(n) * T_s ) )
        c = np.sinc( a * ( tau + np.arange(m) * T_s ) )
        S = sp.linalg.toeplitz( r, c )
        return  ( T_s * a * E * S ).astype(np.complex64)
    else:
        return np.zeros( (n,m) )

def make_A_matrix(r, s):
    # r and s are the received signal and the transmitted signal, respectively.
    # A = A[k,l] is the ambiguity function matrix between r and s.
    # A[k,l] = r @ H[k,l] @ s
    # E = exp( pi * j * k * l / L / U_t / U_f)
    # S = sinc( ( 1 -  |l| / L / U_f ) *  k / U_t ). Only positive l is considered.

    k = np.arange( - U_t * M, 2 * U_t * L_s, 1 )
    k = np.roll(k, k[0])
    l = np.arange( - U_f * N // N_t, U_f * N // N_t, 1 )
    l = np.roll(l, l[0])
    E = np.exp( 1j * np.pi * np.outer( k, l ) / U_t / U_f / L )

    k = np.arange( - U_t * M * (N_t + 3) ,  U_t * M * (N_t + 3), 1 )
    k = np.roll(k, k[0])
    l = np.arange( 0, 2 * U_f * N // N_t, 1 )
    S = np.sinc( np.outer( k / U_t,  1 - l / L / U_f  ) )

    start = time.time()

    A = [ [ r @ ( np.maximum( 1 - np.abs(l) / L / U_f, 0) \
                  * sp.linalg.hankel( E[ k + U_t * np.arange(L_s), l ] , \
                                      E[ k + U_t * np.arange(L_s - 1, 2 * L_s - 1 ), l ] ) \
                  * sp.linalg.toeplitz( S[ k - U_t * np.arange(L_s), np.abs(l) ], \
                                        S[ k + U_t * np.arange(L_s), np.abs(l) ] ) ).conjugate() \
                     @ s.conjugate() for l in range( - N * U_f // N_t, N * U_f // N_t ) ] for k in range( - M * U_t // N_f, M * U_t // N_f ) ]
                     # @ s for l in range( - U_f * N // N_t + 1, U_f * N // N_t ) ] for k in range( - U_t * M // N_f + 1, U_t * M // N_f) ]

    end = time.time()
    print("Computation time of A is ", end - start)
    print("maximum valuse of A is ", np.max(np.abs(A)))
    return np.array(A)

# Determine the received signal for each terminal and estimate the delay and Doppler shift
for i in range(N_terminals): 
    start = time.time()
    H = make_H_matrix(L, L_s, t_d[i], f_D)
    end = time.time()
    print("Computation time of H is ", end - start)

    r = H @ s[:L_s] + noise[i] 

    # ambiguity function between r and s with integer delay and Doppler
    circ_s = sp.linalg.circulant( s )
    circ_s = circ_s[ :, 0 : N * M ]

    r_s = r * np.conj( circ_s.T ) 
    A = np.fft.fft( r_s, axis = 1 )

    [ k_d_est[i], l_D_est[i] ] = np.unravel_index( np.argmax( np.abs( A ) ), A.shape )


    # plot the ambiguity function
    # fig=plt.figure()
    # plt.imshow( np.abs(A.T), aspect = 'auto', cmap = 'jet' )
    # plt.show()

    # A_zoom = A[t_d_est - 2 * M//N_f : t_d_est + 2 * M//N_f, f_D_est -  2 * P*N//N_t : f_D_est + 2 * P * N//N_t ]
    # fig=plt.figure()
    # plt.imshow( np.abs(A_zoom.T), aspect = 'auto', cmap = 'jet' )
    # plt.show()

    if l_D_est[i] > L // 2:
        l_D_est[i] = l_D_est[i] - L

    print("coarse estimation of delay and Doppler", k_d_est[i], l_D_est[i] )

    H = make_H_matrix(L, L, -k_d_est[i] * T_s, - l_D_est[i] * Delta_f) 
    r1 = H @ r 

    if FIG_OPTION:
        fig = plt.figure()
        plt.plot( np.arange(L_s), np.abs( r1[0:L_s] ) )
        plt.savefig("time-frequency_shifted_received_signal.png")
#        plt.show()

        fig = plt.figure()
        plt.plot( np.arange(L_s), np.abs( s[0:L_s] ) )
        plt.savefig("transmitted_signal.png")
 #       plt.show()

    A_oversampled = make_A_matrix(r1[0:L_s], s[0:L_s])

    if FIG_OPTION: 
        fig = plt.figure()
        im = plt.imshow( np.abs(A_oversampled.T), aspect = 'equal', cmap = 'jet', extent=(-8.5, 8.5, -8.5, 8.5) )
        plt.colorbar(im)
        plt.xticks([-8, -6, -4, -2, 0, 2, 4, 6, 8])
        plt.yticks([-8, -6, -4, -2, 0, 2, 4, 6, 8])
        plt.grid()
        plt.xlabel("delay")
        plt.ylabel("Doppler")

        plt.savefig("Conterplot_A_oversampled2.pdf")
 #       plt.show()

    [epsilon_t_d[i], epsilon_f_D[i]] = np.unravel_index( np.argmax( np.abs( A_oversampled ) ), A_oversampled.shape )
    print( epsilon_t_d[i], epsilon_f_D[i] )

    [epsilon_t_d_offset, epsilon_f_D_offset] = A_oversampled.shape

    if FIG_OPTION:
        fig = plt.figure()
        idx = np.arange(A_oversampled.shape[0])
        plt.plot( ( idx - epsilon_t_d_offset//2) / U_t, np.abs(A_oversampled[:, int( epsilon_f_D[i])] ), label = "A_{ss}(tau, 0)", marker="o")
        plt.plot( ( idx - epsilon_t_d_offset//2) / U_t, np.abs( A_oversampled[ epsilon_t_d[i], epsilon_f_D[i]] ) \
                 * np.sinc( 2 * ( idx - epsilon_t_d[i] ) /A_oversampled.shape[0]) , label = "sinc function", marker="x")
        
        # plt.show()
        plt.xlabel("delay (tau) in T_s") 
        plt.ylabel("Absolute value of Ambiguity function")
        plt.grid()
        plt.legend()
        plt.savefig("A_oversampled_in_time_axis.pdf")

        fig = plt.figure()
        idx = np.arange(A_oversampled.shape[1])
        plt.plot( ( idx - epsilon_f_D_offset//2 ) /U_f, np.abs(A_oversampled[ int(epsilon_t_d[i]), :] ) , label="A_{ss}(0, nu)", marker="o" )
        plt.plot( ( idx - epsilon_f_D_offset//2 )/U_f, np.abs( A_oversampled[ epsilon_t_d[i], epsilon_f_D[i]] ) \
                 * np.sinc( 2 * (idx - epsilon_f_D[i])/A_oversampled.shape[1] ) , label="sinc function", marker="x" )
 #       plt.show()
        plt.xlabel("Doppler (nu) in frequency bin") 
        plt.ylabel("Absolute value of Ambiguity function")
        plt.grid()
        plt.legend()
        plt.savefig("A_oversampled_in_frequency_axis.pdf")

    epsilon_t_d[i] -= epsilon_t_d_offset//2
    epsilon_f_D[i] -= epsilon_f_D_offset//2

    print("The estimated delay is ", (k_d_est[i] + epsilon_t_d[i] / U_t) * T_s )
    print("The estimated delay in T_s is ", k_d_est[i] + epsilon_t_d[i] / U_t )
    print("The estimated Doppler shift is ", ( l_D_est[i] + epsilon_f_D[i] / U_f) * Delta_f)
    print("The estimated Doppler shift in frequency bin is ", l_D_est[i]/L + epsilon_f_D[i] / L / U_f, "multipled by L", l_D_est[i] + epsilon_f_D[i] / U_f)


# recover the distance from the estimated delay. 

Estimated_d = (k_d_est - k_d_est[0]/2 + ( epsilon_t_d - epsilon_t_d[0]/2) / U_t ) * c * T_s

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

def new_func(p, q, Estimated_d, mse, initial_guess):
    x0 = initial_guess( Estimated_d , p)
    print("The initial guess of the target's position is: ", x0)
    print("the distance between the initial guess and the true position is: ", np.sqrt( np.sum( (x0-q)**2 ) ), "meters")
    print("The MSE is ", mse(x0, Estimated_d, p))
    return x0

if N_terminals==4:
    x0 = new_func(p, q, Estimated_d, mse, initial_guess)
    result = sp.optimize.minimize( fun = mse, x0 = x0, args = (Estimated_d, p), method = 'BFGS' )
    print(result.x)
    print("The estimated position of the target is: ", result.x, "meters")
    print("the distance between the estimated position and the true position is: ", np.sqrt( np.sum( (result.x-q)**2 ) ), "meters")
    print("The MSE is", mse(result.x, d, p))

