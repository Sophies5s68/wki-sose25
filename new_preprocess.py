import numpy as np 
import scipy.signal as sps
from wettbewerb import get_6montages

# Resampling, um Data auf die gleiche Samplingrate zu bringen

def resample_signal(sig, original_fs, target_fs=256):
    if original_fs == target_fs:
        return sig
    n_samples = int(sig.shape[-1] * target_fs / original_fs)
    if sig.ndim == 1:
        return sps.resample(sig, n_samples)
    elif sig.ndim == 2:
        return np.vstack([sps.resample(ch, n_samples) for ch in sig])
    else:
        raise ValueError("Signal muss 1D oder 2D sein.")

# Notch IIR Filter, um die Netzfrequenz rauszufiltern 

def notch_filter_iir_filtfilt(sig, fs, freq=50.0, Q=30.0):
    w0 = freq / (fs / 2)
    b, a = sps.iirnotch(w0, Q)
    if sig.ndim == 1:
        return sps.filtfilt(b, a, sig)
    elif sig.ndim == 2:
        return np.vstack([sps.filtfilt(b, a, sig[ch]) for ch in range(sig.shape[0])])
    else:
        raise ValueError("Signal must be 1D or 2D.")

        
# Bandpassfilter mit einem IIR Butterworth - Filter 

def bandpass_filter_iir_filtfilt(sig, fs, lowcut=1.0, highcut=120.0, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = sps.butter(order, [low, high], btype='band')
    if sig.ndim == 1:
        return sps.filtfilt(b, a, sig)
    elif sig.ndim == 2:
        return np.vstack([sps.filtfilt(b, a, sig[ch]) for ch in range(sig.shape[0])])
    else:
        raise ValueError("Signal must be 1D or 2D.")

def preprocess_signal_with_montages(channels, data, target_fs, original_fs, ids=0):
    # Holt die 6 Montagen aus dem Datensatz
    montage_names, montage_data, montage_missing = get_6montages(channels, data)
    # Signale werden durch einen Notch Filter gefiltert
    montage_data = notch_filter_iir_filtfilt(montage_data, fs= original_fs)
    # Signale werdenn durch einen Bandpassfilter gefiltert, um nur relevante Frequenzen zu haben
    montage_data = bandpass_filter_iir_filtfilt(montage_data, fs=original_fs)
    # Signale werden auf die gleiche Samplingrate fs = 256 resampled 
    montage_data = resample_signal(montage_data, original_fs, target_fs)
    return montage_names, montage_data, montage_missing, target_fs