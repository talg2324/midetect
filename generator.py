import numpy as np
import torch

def dummy_data():
    timestep = np.linspace(0,6000,6000)

    signals = []
    labels = []

    for i in range(5000):
        w = np.random.normal()
        if w > 0:
            signal = np.sin(2*np.pi*w*timestep/100)
            signal = torch.tensor(signal)
            signals.append(signal.view(6000,1))
            labels.append([1])
        else:
            signal = np.random.random(size=6000)
            signal = torch.tensor(signal)
            signals.append(signal.view(6000,1))
            labels.append([0])

    return {'Training': (signals, np.array(labels))}
