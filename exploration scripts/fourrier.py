from scipy.fftpack import fft, fftfreq, fftshift
import matplotlib.pyplot as plt
import numpy as np

# number of signal points
N = 1600
# sample spacing
T = 1.0 / 800.0
x = np.linspace(0.0, N*T, N)
print x.shape
y = np.exp(50.0 * 1.j * 2.0*np.pi*x) + 0.5*np.exp(-80.0 * 1.j * 2.0*np.pi*x)
print y.shape
#y = np.cos(x)

yf = fft(y)
xf = fftfreq(N, T)
xf = fftshift(xf)
yplot = fftshift(yf)
plt.plot(xf, 1.0/N * np.abs(yplot))
plt.grid()
plt.show()