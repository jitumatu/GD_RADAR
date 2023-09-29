# This program is a sample code for plotting the Gaussian waveform.
import numpy as np
import matplotlib.pyplot as plt

M = 16 # the number of samples in a chip interval
N = M*16 # the number of samples for a Gaussian waveform (3 chips)

g = 2**(1/4)*np.exp( - np.pi*( np.arange(0, N, 1 )/M - N/2/M )**2) # Gaussian waveform
G = np.fft.fft(g)

G_tilde = G * np.exp( - 4 * 2j * np.pi * np.arange( 0, N, 1 ) / N)
g_tilde = np.fft.ifft(G_tilde)

fig = plt.figure()
plt.plot(g)
#plt.plot(np.abs((g_tilde)))
plt.plot((g_tilde).real)
plt.show()


# G = np.fft.fftshift(G)
# fig = plt.figure()
# plt.plot(np.abs(G))
# plt.show()

Dirichlet = np.sum( np.exp( - 2j * np.pi * np.array( [ [ -k * (m-10.2) for k in range(N) ] for m in range(N) ] ) / N ), axis=1 ) / N
fig = plt.figure()
plt.plot(Dirichlet.real)
plt.show()


