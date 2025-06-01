import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import entropy

bands = ({ 'delta' : [1,4],
                'theta' : [4, 8],
                'alpha' : [8, 14],
                'beta' : [14, 30]})  

# Klasse um auszuwaehlen welche Features extraiert werden soll,
# signal -> n channels, n samples

def feature_extraction(signals,fs):
   
    feature_matr = list()
    for i,channel in enumerate(signals[:]):
     # Fourier und plv hier einmal berechnen, damit keine Doppelung
    # aber Features einzeln ausgewählt
        f, spectrum = signal.welch(channel,fs)
        plvs = plv_series(2,0.5,channel,fs)
        features = list()
        features.extend(spectral_power(f,spectrum))
        features.extend(mean_spectral_amplitude(f,spectrum))
        features.append(spectral_entropy(f,spectrum))
        features.append(plv_peak(plvs))
        features.append(plv_avg(plvs))
        features.append(plv_power(plvs))
        
        feature_matr.append(np.asarray(features))
        
    feature_matr = np.asarray(feature_matr)
    feature_matr = standardize_matrix(feature_matr)
    
    return feature_matr

# features standardisieren Spaltenweise
def standardize_matrix(matr):
    mean = matr.mean(axis=0, keepdims=True)
    std = matr.std(axis=0, keepdims=True)
    return (matr - mean) / std


# PLV berechnen zur Beurteilung PAC von hig gamma und low freq
def compute_plv (sig,fs):
    b_low, a_low = signal.butter(4, [4/(fs/2), 30/(fs/2)], btype='band')
    low_filtered = signal.filtfilt(b_low,a_low,sig)
    phase_low = np.angle(signal.hilbert(low_filtered))
    
    b_high,a_high = signal.butter(4, [80/(fs/2), 120/(fs/2)], btype='band')
    high_filtered = signal.filtfilt(b_high,a_high,sig)
    amplitude_high = np.abs(signal.hilbert(high_filtered))
    amp_high_phase = np.angle(signal.hilbert(amplitude_high))

    phase_diff = phase_low - amp_high_phase
    complex_phase_diff = np.exp(1j * phase_diff)
    plv = np.abs(np.mean(complex_phase_diff))
    
    return plv
    
def plv_series(window_length, step, sig, fs):

    win_samples = int(window_length * fs)
    step_samples = int(step * fs)
    plvs_list = list()
    
    for start_index in range(0, len(sig) - win_samples, step_samples):
        window = sig[start_index:start_index + win_samples]
        plv = compute_plv(window, fs)
        plvs_list.append(plv)
        plvs = np.asarray(plvs_list)
    return plvs
    
#  Vektor mit spectral power für [0]delta, [1]theta, [2]alpha, [3]beta 
def spectral_power(f,spectrum):
    extracted = list()
    for band, (fmin, fmax) in bands.items():
        idx_band = np.logical_and(f >= fmin, f <= fmax)
        band_power = np.mean(spectrum[idx_band])
        extracted.append(band_power)   
    extracted = np.asarray(extracted)
    return extracted

def mean_spectral_amplitude(f,spectrum):
    extracted = list()
    amp = np.sqrt(spectrum)
    for band, (fmin, fmax) in bands.items():
        idx_band = np.logical_and(f >= fmin, f <= fmax)
        mean_amp = np.mean(amp[idx_band])
        extracted.append(mean_amp)   
    extracted = np.asarray(extracted)
    return extracted

def spectral_entropy(f,spectrum): 
    psd_norm = spectrum / np.sum(spectrum) 
    de = entropy(psd_norm) 
    de /= np.log(len(spectrum))
    return de

def plv_peak(plvs):
    feature = max(plvs)
    return feature

def plv_avg(plvs):
    feature = np.mean(plvs)
    return feature

def plv_power(plvs):
    feature = np.mean(plvs**2)
    return feature