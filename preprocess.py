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

logging.getLogger('pyprep').setLevel(logging.ERROR)  # oder CRITICAL
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
    return filtered_sig.get_data(verbose = False)
