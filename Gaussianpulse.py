# This program is a sample code for plotting the Gaussian waveform.
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time

M = 128 # the number of samples in a chip interval
N = M*128 # the number of samples for a Gaussian waveform (3 chips)

g = 2**(1/4)*np.exp( - np.pi*( np.arange(0, N, 1 )/M - N/2/M )**2) # Gaussian waveform
G = np.fft.fft(g)

G_tilde = G * np.exp( - 4 * 2j * np.pi * np.arange( 0, N, 1 ) / N)
g_tilde = np.fft.ifft(G_tilde)

# fig = plt.figure()
# plt.plot(g)
# #plt.plot(np.abs((g_tilde)))
# plt.plot((g_tilde).real)
# plt.show()

g_tilde = g * np.exp( 40 * 2j * np.pi * np.arange( 0, N, 1 ) / N)

# G = np.fft.fftshift(G)
# fig = plt.figure()
# plt.plot(np.abs(G))
# plt.show()

# Dirichlet = np.sum( np.exp( - 2j * np.pi * np.array( [ [ -k * (m-10.2) for k in range(N) ] for m in range(N) ] ) / N ), axis=1 ) / N
# fig = plt.figure()
# plt.plot(Dirichlet.real)
# plt.show()

def Ambiguity_sinc( n, m, tau, nu):
    if np.abs( nu ) < 1:
        return np.exp( 1j * np.pi * nu * ( tau + m + n ) ) *  (1 - np.abs(nu) ) * np.sinc( (1 - np.abs(nu) ) *  ( tau + m - n ) ) 
    else: 
        return 0

def make_H_matrix( n, m, tau, nu):
    # generate the ambiguity function matrix

    k = 1 - np.abs(nu) 

    if k > 0:
        r = np.exp ( 1j * np.pi * nu * ( tau + np.arange(n) ) )
        c = np.exp ( 1j * np.pi * nu * ( tau + n - 1 + np.arange(m) ) )
        E = sp.linalg.hankel( r, c )

        r = np.sinc( k * ( tau - np.arange(n) ) )
        c = np.sinc( k * ( tau + np.arange(m) ) )
        S = sp.linalg.toeplitz( r, c )
        return  k * E * S
    else:
        return 0


tau = 8.5
nu = 33.5/N # 40.3/N  #4 / N / M

# start = time.time()
# H = np.array( [ [ Ambiguity_sinc( n, m, tau, nu ) for m in range(N) ] for n in range(N) ] )
# end = time.time()
# print("Direct computation time of H is ", end - start)
# y = H @ g

start = time.time()
H = make_H_matrix( N, N, tau, nu )
end = time.time()
print("Impoved computation time of H is ", end - start)

y = H @ g 

# print(np.linalg.norm(y.imag))

fig = plt.figure()
plt.plot(g)
#plt.plot(np.abs((y)))
plt.plot((y).real)
plt.plot(np.abs(y))
plt.show()


H = make_H_matrix( N, N, -tau, -nu )
y = H @ y  

fig = plt.figure()
plt.plot(g)
#plt.plot(np.abs((y)))
plt.plot((y).real)
plt.plot(np.abs(y))
plt.show()

# print(np.linalg.norm(y.imag))


