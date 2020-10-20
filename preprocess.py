import pywt
from scipy import signal as sp
from matplotlib import pyplot as plt
import numpy as np

def wavelet(signal, wvlt='sym2', level=6, desired_coeffs='approx'):

    coeffs = pywt.wavedec(signal, 'sym2', 'periodic', level=level-1)

    if desired_coeffs=='approx':

        # Use the approximation coefficients as a feature
        output = sp.resample(coeffs[0], len(signal))
        return minmax(output)

    else:

        # Remove Low Freq Approximation Coefficients
        coeffs[0] = np.zeros_like(coeffs[0])

        # Save the High Freq Detail Signal
        idwt = pywt.waverec(coeffs, 'sym2', 'periodic')

        return minmax(idwt)
        
def minmax(signal):

    return (signal - signal.min()) / (signal.max() - signal.min())

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

def visualize_dwt_filter(signal, wvlt='sym2', level=4):

    coeffs = pywt.wavedec(signal, wvlt, 'periodic', level=level-1)

    not_coeffs = []

    for l in range(level):
        not_coeffs.append(np.zeros_like(coeffs[l]))

    for i in range(1,level):
        not_coeffs[i] = coeffs[i]

    lpf = pywt.waverec(not_coeffs, wvlt, 'periodic')
    hpf = pywt.waverec(not_coeffs, wvlt, 'periodic')

    plt.subplot(311)
    plt.plot(signal)
    plt.title('Signal')
    plt.subplot(312)
    plt.plot(coeffs[0])
    plt.title('Approximation Coefficients')
    plt.subplot(313)
    plt.plot(hpf)
    plt.title('Detail Coefficients')
    
    plt.show()

def visualize_dwt(signal, level=6, wvlt='sym2'):

    coeffs = pywt.wavedec(signal, 'sym2', 'periodic', level=level-1)

    plt.subplot(level,1,1)
    plt.plot(signal)

    for l in range(level-1):

        plt.subplot(level,1,2+l)
        plt.plot(coeffs[l])

    plt.show()