# This program is used to generate Figures 6, 7 and 8 for the paper 
# "2D Sinc Interpolation-Based Fractional Delay and Doppler Estimation Using Time and Frequency Shifted Gaussian Pulses"
# submitted to JC&S'24.

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time
import sys

FIG_OPTION = True # True: show figures; False: do not show figures

# constants
c = 3e8 # the speed of light
W = 240e6 # the bandwidth 240 MHz
f_c = 24e9 # carrier frequency 24 GHz

pulse = "Gaussian"
#pulse = "Rectangular"

# the number of simulations
SIM = 1000

P = 1 # The number of blocks
N_t = 8 # the spreading factor of the time domain
N_f = 8 # the spreading factor of the frequency domain

N = 64 # The number of total time slots in a block, which must be greater than N_t.
M = 16 # The number of total subcarriers, which must be greater than N_f.

L = P * N * M # The length of the transmitted signal

F_c = W/N_f # chip bandwidth 
T_c = 1/F_c # chip interval 

f_s = M * F_c # sampling frequency 
T_s = 1/f_s # sampling interval 
Delta_f = f_s/L # frequency resolution in FFT  = 1/(N*T_C)

SNR_in_dB = [-10, -5, 0, 5, 10, 15, 20, 25, 30] # SNR in dB
# SNR_in_dB = [ 40 ] # SNR in dB
SNR_in_dB = np.array(SNR_in_dB)
SNR_list = 10**(SNR_in_dB/10)

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

# print("t_d = ", t_d, ", t_d / T_s=", t_d/T_s)
# print("l_d = ", l_d)
# print("f_D = ", f_D, ", f_D / f_s * L = ", f_D /f_s * L)
# print("k_D = ", k_D)

N_terminals = 1 # the number of terminals

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
X_tf = [ [ [-1,  1, -1,  1, -1,  1, -1, -1],
   [-1, -1, -1,  1,  1, -1,  1,  1],
   [-1,  1,  1, -1, -1, -1,  1,  1],
   [-1, -1,  1,  1,  1, -1, -1,  1],
   [-1,  1,  1,  1, -1, -1, -1, -1],
   [-1,  1,  1, -1, -1, -1,  1,  1],
   [-1, -1,  1,  1,  1, -1, -1,  1],
   [ 1,  1,  1,  1, -1,  1,  1, -1] ] ] 
# X_tf = np.random.randint(0, 2, (P, N_t, N_f)) * 2 - 1 
X_tf = np.array(X_tf)
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

# show the ambiguity function of s as graph
circ_s = sp.linalg.circulant( s )
circ_s = circ_s[ :, 0 : N * M ]
s_circ_s = s * np.conj( circ_s.T )
A_ss = np.fft.fft( s_circ_s, axis = 1 ) # the ambiguity function of s
A_ss = np.roll(A_ss, ( N*M//2, N*M//2) , axis=(0,1) )
[n, m] = np.unravel_index( np.argmax(np.abs(A_ss) ), A_ss.shape ) 
print("argmax A_ss =", n, m)
A_ss_clip = np.abs(A_ss[n - M//N_f: n + M//N_f+1, m - N//N_f: m + N//N_f+1])
fig = plt.figure() 
plt.xlabel("delay")
plt.ylabel("Doppler")
plt.imshow( A_ss_clip.T, extent=(-2.5, 2.5, -8.5, 8.5) )
plt.xticks([-2, 0, 2])
plt.colorbar()
plt.savefig("A_ss.png")

l_d_est = np.zeros( (N_terminals,), dtype=np.int32() )
k_D_est = np.zeros( (N_terminals,), dtype=np.int32() )

epsilon_t = np.zeros( (N_terminals,), dtype=np.float32() )
epsilon_f = np.zeros( (N_terminals,), dtype=np.float32() )

def make_H_matrix( n, m, tau, nu):
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

def E_A(tau, nu):
    if np.abs(N_f * F_c * tau) > 1 or np.abs(N_t * T_c * nu) > 1:
        return 0
    else:
        return np.sinc(N_f * F_c * tau) * np.sinc(N_t * T_c * nu)
    
def total_error(x, A):
    alpha = x[0]
    epsilon_t = x[1]
    epsilon_f = x[2]
    err = 0
    [n,m] = A.shape
    for k in range(n):
        for l in range(m): 
            #if np.abs( k - (n) // 2 - epsilon_t ) < M// N_f + 1 and np.abs( l - (m) // 2 - epsilon_f ) < N // N_t + 1:
            err += (alpha * E_A( (k - (n) // 2 - epsilon_t) * T_s , ( l - (m) // 2 - epsilon_f) * Delta_f  ) - A[k, l] ) **2 
    return err 

# Determine the received signal for each terminal and estimate the delay and Doppler shift

mean_squared_error_t = np.zeros( (2, len(SNR_list),), dtype=np.float32() )
mean_squared_error_f = np.zeros( (2, len(SNR_list),), dtype=np.float32() )
effective_sim = np.zeros( (len(SNR_list),), dtype=np.int32() )

exec_time = np.zeros( (3, len(SNR_list),), dtype=np.float32() )

for loop in range(SIM):
    print("loop = ", loop)
    t_d = np.random.rand(1,) * (N - 2 * (N_t + 4 )) * T_c + N_t * T_c 
    f_D = ( np.random.rand() - 0.5 ) * ( f_s - ( N_f + 4 ) * F_c ) # Doppler shift 
    # t_d = np.array( [ 1.7098328185172866e-06 ])  
    # f_D = 256867510.3238896* 0.2
    # t_d = np.array([9.347193621687811e-07])
    # f_D = -2683675.961960477
    # t_d = np.array( [165.45494159088807 * T_s] )  
    # f_D = 124.28807455816815 * Delta_f
    


    l_d = np.rint( t_d / T_s ).astype(np.int32) # the closest integer delay in T_s unit  
    k_D = np.rint( f_D / f_s * L ).astype(np.int32) # the closest frequency bin of the Doppler shift 
    noise = np.random.randn( N_terminals, L ) + 1j * np.random.randn( N_terminals, L ) # additive white Gaussian noise

    # print("t_d = ", t_d[0], ", t_d / T_s=", t_d[0]/T_s, "l_d=", l_d[0])
    # print("f_D = ", f_D, ", f_D / f_s * L = ", f_D /f_s * L, "k_D=", k_D)

    for i in range(N_terminals): 
        H = make_H_matrix(L, L_s, t_d[i], f_D)
        for idx_SNR in range(len(SNR_list)):
            # print("SNR = ", 10 * np.log10(SNR), "dB")
            sigma =  np.linalg.norm(s) * np.sqrt( 1 / L / SNR_list[idx_SNR] / 2 )
            r = H @ s[:L_s] + sigma * noise[i] 

            # ambiguity function between r and s with integer delay and Doppler
            exec_time[0, idx_SNR] -= time.process_time_ns()
            circ_s = sp.linalg.circulant( s )
            circ_s = circ_s[ :, 0 : N * M ]
            r_s = r * np.conj( circ_s.T ) 
            A = np.fft.fft( r_s, axis = 1 ) # the ambiguity function between r and s with integer delay and Doppler
            A = np.fft.fftshift( A, axes = 1  )
            exec_time[0, idx_SNR] += time.process_time_ns()

            # coarse estimation of the delay and Doppler shift
            [ l_d_est[i], k_D_est[i] ] = np.unravel_index( np.argmax( np.abs( A ) ), A.shape )
            
            # print("l_d_est[i] =", l_d_est[i], "k_D_est[i] =", k_D_est[i])

            A_clip = np.abs( A[l_d_est[i] - M//N_f : l_d_est[i] + M//N_f + 1, k_D_est[i] - N//N_t : k_D_est[i] + N//N_t + 1] )/np.max(np.abs(A))

            # fig = plt.figure()
            # plt.imshow( A_clip.T )
            # # plt.imshow( np.abs(A[ l_d_est[i] -M//N_f: l_d_est[i] +M//N_f, k_D_est[i] - N//N_t : k_D_est[i] + N//N_t ].T))
            # plt.colorbar()
            # plt.savefig("A{}dB.png".format(SNR_in_dB[idx_SNR]))

            x_0 = np.array([1, 0, 0]) # initial guess of the parameters: alpha, epsilon_t, epsilon_f

            exec_time[1, idx_SNR] -= time.process_time_ns()
            result = sp.optimize.minimize(fun = total_error, x0 = x_0, args = (A_clip), method='BFGS')
            exec_time[1, idx_SNR] += time.process_time_ns()

            epsilon_t[i] = result.x[1]
            epsilon_f[i] = result.x[2]

            k_D_est[i] -= L // 2

            # if l_d_est[i] != l_d:
            #     print("t_d[i]/T_s =", t_d[i]/T_s, "l_d_est =", l_d_est[0], "epsilon_t =", epsilon_t[i])
            # if k_D_est[i] != k_D:
            #     print("f_D/Delta_f =", f_D/Delta_f, "k_D_est =", k_D_est[i], "epsilon_f =", epsilon_f[i])

            # print("delay and Doppler", t_d[i], f_D)
            # print("coarse estimated delay and Doppler", l_d_est[i] * T_s,  k_D_est[i]  * Delta_f)
            # print("fine estimated delay and Doppler", ( l_d_est[i] + epsilon_t[i] ) * T_s , (k_D_est[i] + epsilon_f[i] ) * Delta_f )
            # print("estimation error =", np.abs( t_d[i] - (l_d_est[i] + epsilon_t[i]) * T_s), \
            #     np.abs( f_D - ( k_D_est[i] + epsilon_f[i] ) * Delta_f ) )

            # print("\ndelay and Doppler (in units)", t_d[i]/T_s, f_D / Delta_f)
            # print("coarse estimated delay and Doppler (integers) ", l_d_est[i], k_D_est[i] )
            # print("fine estimated delay and Doppler (in units)", (l_d_est[i] + epsilon_t[i]), (k_D_est[i] + epsilon_f[i]) )
            # print("estimation error (in units) =", np.abs( t_d[i]/T_s - (l_d_est[i] + epsilon_t[i]) ), \
            #      np.abs( f_D / Delta_f - ( k_D_est[i] + epsilon_f[i] ) ) )

            if np.abs( t_d[i]/T_s - (l_d_est[i] + epsilon_t[i]) ) > 0.5:
                print(" t_d[i]/T_s, l_d_est[i], epsilon_t[i] = ", t_d[i]/T_s, l_d_est[i], epsilon_t[i])
                print(" f_D/Delta_f, k_D_est[i], epsilon_f[i] = ", f_D/Delta_f, k_D_est[i], epsilon_f[i])

            mean_squared_error_t[0, idx_SNR] += np.abs( t_d[i] / T_s - (l_d_est[i] + epsilon_t[i]) )**2 
            mean_squared_error_f[0, idx_SNR] += np.abs( f_D/Delta_f - (k_D_est[i] + epsilon_f[i]) )**2 

            # if np.abs( f_D / Delta_f - ( k_D_est[i] + epsilon_f[i] ) ) < 0.5:
            #     effective_sim[idx_SNR] += 1
            #     mean_squared_error_t[0, idx_SNR] += np.abs( t_d[i] / T_s - (l_d_est[i] + epsilon_t[i]) )**2 
            #     mean_squared_error_f[0, idx_SNR] += np.abs( f_D/Delta_f - (k_D_est[i] + epsilon_f[i]) )**2 
            # else: 
            #     print("f_D / Delta_f =", f_D / Delta_f , "k_D_est[i]=", k_D_est[i], "epsilon_f[i]=", epsilon_f[i])

            # the conventional method to estimate the delay and Doppler shift
            # shown in K. Zhang et al. "Radar sensing via OTFS signaling: A delay Doppler signal Pressing perspective"

            exec_time[2, idx_SNR] -= time.process_time_ns()
            k_D_est[i] += L // 2
            a = np.abs( A[ l_d_est[i] + 1, k_D_est[i]          ] )
            b = np.abs( A[ l_d_est[i] - 1, k_D_est[i]          ] )
            c = np.abs( A[ l_d_est[i]         , k_D_est[i] + 1 ] )
            d = np.abs( A[ l_d_est[i]         , k_D_est[i] - 1 ] )
            e = np.abs( A[ l_d_est[i]         , k_D_est[i]     ] )
            k_D_est[i] -= L // 2
            exec_time[2, idx_SNR] += time.process_time_ns()

            # print("a, b, c, d, e =", a, b, c, d, e)

            epsilon_t[i] = (a-b)/ (4*e - 2*a - 2*b)
            epsilon_f[i] = (c-d)/ (4*e - 2*c - 2*d)

            # if a > b:
            #     epsilon_t[i] = a / ( a + e ) 
            # else:
            #     epsilon_t[i] = - b / ( b + e )
            # if c > d:
            #     epsilon_f[i] = c / ( c + e )
            # else:  
            #     epsilon_f[i] = - d / ( d + e )

            # print("quadratic interpolation: fine delay and Doppler (in units)", ( l_d_est[i] + epsilon_t[i] ) , (k_D_est[i] + epsilon_f[i] ) )
            # print("quadratic interpolation: estimation error (in units)=" , np.abs( t_d[i]/T_s - (l_d_est[i] + epsilon_t[i]) ), \
            #      np.abs( f_D / Delta_f - ( k_D_est[i] + epsilon_f[i] )  ) )
        
            # if np.abs( f_D / Delta_f - ( k_D_est[i] + epsilon_f[i] ) ) < 0.5:
            mean_squared_error_t[1, idx_SNR] += np.abs( t_d[i]/T_s - (l_d_est[i] + epsilon_t[i]) )**2 
            mean_squared_error_f[1, idx_SNR] += np.abs( f_D/Delta_f - (k_D_est[i] + epsilon_f[i]) )**2 


#print("effective_sim = ", effective_sim)
mean_squared_error_t /= SIM 
mean_squared_error_f /= SIM 

RMSE_t = np.sqrt( mean_squared_error_t )
RMSE_f = np.sqrt( mean_squared_error_f )

print("RMSE of the delay (in units)")
print(RMSE_t)
print("RMSE of the Doppler shift (in units)")
print(RMSE_f)

print("execution time (s)")
print("coarse estimation =", exec_time[0,:])
print("2D sinc interpolation =", exec_time[1,:])
print("quadratic interpolation =", exec_time[2,:])

if FIG_OPTION:
    fig = plt.figure()
    plt.plot( SNR_in_dB, RMSE_t[0,:], marker = "x", label = "Double Sinc interpolation" )
    plt.plot( SNR_in_dB, RMSE_t[1,:], marker = "+", label = "Quadratic interpolation" )
    plt.xlabel("SNR (dB)")
    plt.ylabel("RMSE of delay estimation")
    plt.legend()
    plt.grid()
    # plt.show()
    plt.savefig("RMSE_delay_estimation7.pdf")

    fig = plt.figure()
    plt.plot( SNR_in_dB, RMSE_f[0,:], marker = "x", label = "Double Sinc interpolation" )
    plt.plot( SNR_in_dB, RMSE_f[1,:], marker = "+", label = "Quadratic interpolation" )
    plt.xlabel("SNR (dB)")
    plt.ylabel("RMSE of Doppler shift estimation")
    plt.legend()
    plt.grid()
    # plt.show()
    plt.savefig("RMSE_Doppler_estimation7.pdf")







# recover the distance from the estimated delay. 

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

