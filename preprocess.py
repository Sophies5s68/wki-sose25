"""
File mit allen preprocess-Schritten und einer Methode preprocess, um sie in anderen 
Files aufzurufen
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from pyprep.prep_pipeline import NoisyChannels
import contextlib
import os
import logging

logging.getLogger('pyprep').setLevel(logging.ERROR)  
logging.getLogger('mne').setLevel(logging.ERROR)

def process_eeg (signals, fs, channels, reference):  
    n_channels = signals.shape[0]
    ch_names = channels
    ch_types = ['eeg'] * n_channels
    
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=ch_types,verbose = False)
    raw_sig = mne.io.RawArray(signals, info,verbose = False)
    raw_sig.set_montage('standard_1020',verbose = False)
    #raw_sig.plot(start= 10, duration = 5)
    #spectrum1 = raw_sig.compute_psd()
    #spectrum1.plot(average=True, picks="data", exclude="bads", amplitude=False)

    # Pyprep package um bad channels zu finden

    noisy_detector = NoisyChannels(raw_sig)
    bads = noisy_detector.find_all_bads()
    if bads is None:
        bads = []
    raw_sig.info['bads'] = bads
    bads_removed= raw_sig.copy().interpolate_bads()
    #bads_removed.plot(start= 10, duration = 5)
    #spectrum4 = bads_removed.compute_psd()
    #spectrum4.plot(average=True, picks="data", exclude="bads", amplitude=False)
    # Rereference to average
    referenced_sig, _ = mne.set_eeg_reference(bads_removed, ref_channels='average',verbose = False)
    #referenced_sig.plot(start= 10, duration = 5)
    #spectrum3 = referenced_sig.compute_psd()
    #spectrum3.plot(average=True, picks="data", exclude="bads", amplitude=False)
    
    #sig_data = referenced_sig.get_data(verbose= False)
    # Notch Filter
    referenced_sig = referenced_sig.notch_filter([60, 120],verbose = False)  # Methode falls 50Hz Netz? 
    #referenced_sig.plot(start= 10, duration = 5)
    #spectrum2 = referenced_sig.compute_psd()
    #spectrum2.plot(average=True, picks="data", exclude="bads", amplitude=False)
    
    # Band Pass
    filtered_sig = referenced_sig.filter(1,120,method ='fir',verbose = False)
    #filtered_sig.plot(start= 10, duration = 5)
    #spectrum5 = filtered_sig.compute_psd()
    #spectrum5.plot(average=True, picks="data", exclude="bads", amplitude=False)  
    
    #notch_filtered_sig = notch_filter_iir_filtfilt(sig_data,fs)
    #filtered_sig = bandpass_filter_iir_filtfilt(notch_filtered_sig,fs)
    return filtered_sig.get_data(verbose=False)



def notch_filter_iir_filtfilt(sig, fs, freq=60.0, Q=30.0):
    """
    IIR Notch-Filter mit filtfilt.

    Parameters:
    - signal: 1D oder 2D Array (channels x samples)
    - fs: Samplingfrequenz
    - freq: Frequenz der Kerbe (z.B. 60 Hz)
    - Q: Gütefaktor (höhere Werte = schmalere Kerbe)

    Returns:
    - Gefiltertes Signal
    """
    w0 = freq / (fs / 2)  # Normierte Frequenz
    b, a = signal.iirnotch(w0, Q)

    if sig.ndim == 1:
        filtered = signal.filtfilt(b, a, sig)
    elif sig.ndim == 2:
        filtered = np.zeros_like(sig)
        for ch in range(sig.shape[0]):
            filtered[ch] = signal.filtfilt(b, a, sig[ch])
    else:
        raise ValueError("Signal muss 1D oder 2D sein")

    return filtered


def bandpass_filter_iir_filtfilt(sig, fs, lowcut=1.0, highcut=120.0, order=4):
    """
    Bandpassfilter mit IIR (Butterworth) und filtfilt für Phasenkorrektur.

    Parameters:
    - signal:       1D- oder 2D-Array [n_channels, n_samples] oder [n_samples]
    - fs:           Samplingfrequenz (Hz)
    - lowcut:       untere Grenzfrequenz (Hz)
    - highcut:      obere Grenzfrequenz (Hz)
    - order:        Ordnung des Filters (je höher, desto schärfer)

    Returns:
    - Gefiltertes Signal (gleiche Form wie Input)
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = signal.butter(order, [low, high], btype='band')

    if sig.ndim == 1:
        filtered = signal.filtfilt(b, a, sig)
    elif sig.ndim == 2:
        filtered = np.zeros_like(sig)
        for ch in range(sig.shape[0]):
            filtered[ch] = signal.filtfilt(b, a, sig[ch])
    else:
        raise ValueError("Signal muss 1D oder 2D sein (Kanäle x Samples).")

    return filtered


def process_without_mne(signals, fs, channels, reference,lowest_sampling):  
    n_channels = signals.shape[0]
    ch_names = channels
    ch_types = ['eeg'] * n_channels
    
    
    if reference != 'AR':
        avg = np.mean(signals, axis=0, keepdims=True)
        signals = signals -avg
        
    resampled = []
    if fs != lowest_sampling:
        length_s = signals.shape[1]/fs
        num = int(length_s*lowest_sampling)
        for i,ch in enumerate(signals):
            resampled.append(signal.resample(ch,num))
        resampled = np.asarray(resampled)
    else: resampled = signals
        
    notch_signal = notch_filter_iir_filtfilt(resampled,fs)
    bandpass_signal = bandpass_filter_iir_filtfilt(notch_signal,fs)
    return bandpass_signal