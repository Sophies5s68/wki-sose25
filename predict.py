# -*- coding: utf-8 -*-
"""

Skript testet das vortrainierte Modell


@author:  Maurice Rohr, Dirk Schweickard
"""


import numpy as np
import json
import os
from typing import List, Tuple, Dict, Any
from wettbewerb import get_6montages

# Pakete aus dem Vorlesungsbeispiel
import mne
from scipy import signal as sig
import ruptures as rpt
import torch 
import torch.nn as nn
from CNN_model_copy import CNN_EEG
from new_preprocess import preprocess_signal_with_montages
from new_features import window_prediction, feature_extraction_window
#from CNN_dataset import window_data_evaluate, create_fixed_grid_maps
from glob import glob
from scipy.signal import iirnotch, butter, sosfiltfilt, resample_poly,tf2sos


###Signatur der Methode (Parameter und Anzahl return-Werte) darf nicht verändert werden
def predict_labels(channels : List[str], data : np.ndarray, fs : float, reference_system: str, model_name : str='model.json') -> Dict[str,Any]:
    '''
    Parameters
    ----------
    channels : List[str]
        Namen der übergebenen Kanäle
    data : ndarray
        EEG-Signale der angegebenen Kanäle
    fs : float
        Sampling-Frequenz der Signale.
    reference_system :  str
        Welches Referenzsystem wurde benutzt, "Bezugselektrode", nicht garantiert korrekt!
    model_name : str
        Name eures Models,das ihr beispielsweise bei Abgabe genannt habt. 
        Kann verwendet werden um korrektes Model aus Ordner zu laden
    Returns
    -------
    prediction : Dict[str,Any]
        enthält Vorhersage, ob Anfall vorhanden und wenn ja wo (Onset+Offset)
    '''

#------------------------------------------------------------------------------
# Euer Code ab hier  

    # Initialisiere Return (Ergebnisse)
    seizure_present = True # gibt an ob ein Anfall vorliegt
    seizure_confidence = 0.5 # gibt die Unsicherheit des Modells an (optional)
    onset = 4.2   # gibt den Beginn des Anfalls an (in Sekunden)
    onset_confidence = 0.99 # gibt die Unsicherheit bezüglich des Beginns an (optional)
    offset = 999999  # gibt das Ende des Anfalls an (optional)
    offset_confidence = 0   # gibt die Unsicherheit bezüglich des Endes an (optional)

    # Modell Aufsetzen
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Daten vorbereiten
    window_size = 4
    step_size = 1
    target_fs = 256
    original_fs = fs
 
    
    montage_names, montage_data, montage_missing = get_6montages(channels, data)
    notch_data = notch_filter(montage_data,original_fs)
    band_data = bandpass_filter(notch_data, original_fs)
  
    
    windows, timestamps = window_prediction(band_data, target_fs, window_size, step_size)
    data_for_class = []
    # Feature extraction and brain map calculation
    for win in windows:
        features = feature_extraction_window(win, fs) # shape: (n_channels, n_features)
        assert not np.isnan(features).any(), "NaN in features!"
        x = torch.tensor(features, dtype = torch.float)
        data_for_class.append(x)
        

    # Klassifikation
    predictions_per_window =[]
    with torch.no_grad():
        for feature_matr in data_for_class:
            prob = predictions_ensemble(feature_matr ,model_name, device)
            y_pred = int(prob > 0.5)
            predictions_per_window.append(y_pred)

    seizure_present = False
    seizure_present, onset_candidate = detect_onset(predictions_per_window, timestamps, min_consecutive=2)
    if seizure_present:
        onset = onset_candidate

        
#------------------------------------------------------------------------------  
    prediction = {"seizure_present":seizure_present,"seizure_confidence":seizure_confidence,
                   "onset":onset,"onset_confidence":onset_confidence,"offset":offset,
                   "offset_confidence":offset_confidence}
  
    return prediction # Dictionary mit prediction - Muss unverändert bleiben!
                               
                               
        
def predictions_ensemble(features: torch.Tensor, model_name: str, device: torch.device) -> float:
    file_paths = sorted([os.path.join(model_name, f) for f in os.listdir(model_name) if f.endswith(".pth")])

    probs = []
    with torch.no_grad():
        for path in file_paths:
            model = CNN_EEG(6, 1).to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
            features_new = features.unsqueeze(0).to(device)  # [1, feat_dim]
            output = model(features_new)  # [1]
            prob = torch.sigmoid(output.squeeze())  # scalar in [0, 1]
            probs.append(prob.item())

    return np.mean(probs)  # Ensemble-Mittelwert

def detect_onset(predictions, timestamps, min_consecutive=2):
    predictions = torch.tensor(predictions)
    for i in range(len(predictions) - min_consecutive + 1):
        if torch.all(predictions[i:i+min_consecutive] == 1):
            return True, timestamps[i]
    return False, None



def notch_filter(signal, fs, freq=50.0, Q=30.0):
    w0 = freq / (fs / 2)
    b, a = iirnotch(w0, Q)
    sos = tf2sos(b, a)  # Transferfunktion → SOS
    return sosfiltfilt(sos, signal, axis=-1)


def bandpass_filter(signal, fs, lowcut=1.0, highcut=120.0, order=4):
    sos = sig.butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, signal, axis=-1)

def resample_signal(signal, original_fs, target_fs=256):
    if original_fs == target_fs:
        return signal
    gcd = np.gcd(int(original_fs), int(target_fs))
    up = int(target_fs // gcd)
    down = int(original_fs // gcd)
    return resample_poly(sig, up, down, axis=-1)