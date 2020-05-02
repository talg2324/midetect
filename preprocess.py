import pywt
from scipy import signal as sp
from matplotlib import pyplot as plt
import numpy as np

def wavelet(signal):

    scales = np.arange(500,100,-25)
    scales = np.hstack([scales, np.arange(100,20,-10)])
    scales = np.hstack([scales, np.arange(20,0,-5)])
    scales = np.hstack([scales, np.arange(5,1,-1)])
    scales = np.hstack([scales, np.arange(1,0.06, -0.01)])

    coeffs, freqs = pywt.cwt(signal, scales, 'morl')

    return np.sum(coeffs, axis=0)
    

    

def visualize_spectrum(signal, label):

    """

    This function uses wavelets of various frequencies to visualize the interesting part of the spectrum relevant to the desired label
    It shouldn't be called in loops as it is computation heavy
    
    """
    

    scales = [500, 250, 150, 100, 50, 20, 1]

    coeffs, freqs = pywt.cwt(signal, scales, 'morl')

    noplts = len(scales) + 2

    plt.subplot(noplts,1,1)

    plt.title('Label - ' + str(label))

    plt.plot(signal)

    plt.subplot(noplts,1,2)

    reconstruction = np.sum(coeffs, axis=0)

    plt.plot(reconstruction)

    for i in range(len(scales)):
        plt.subplot(noplts,1,i+3)

        plt.plot(coeffs[i])

    scales = range(600,1,-25)

    coeffs, freqs = pywt.cwt(signal, scales, 'morl')

    plt.figure(2)

    plt.title('Label' + str(label))

    plt.imshow(coeffs, extent=[0, len(signal), 1, 600], cmap='PRGn', aspect='auto',
            vmax=abs(coeffs).max(), vmin=-abs(coeffs).max())  

    plt.show()