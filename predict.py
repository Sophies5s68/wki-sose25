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
from scipy import signal as sps
import ruptures as rpt
import torch 
import torch.nn as nn
from CNN_model import CNN_EEG
from new_preprocess import preprocess_signal_with_montages
from features_predict import window_prediction, feature_extraction_window
#from CNN_dataset import window_data_evaluate, create_fixed_grid_maps
from glob import glob
from scipy.signal import iirnotch, butter, sosfiltfilt, resample_poly, tf2sos


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
    stft_window_size = 1
    stft_overlap = 0.5
    window_size = 4
    step_size = 1
    target_fs = 256
    original_fs = fs
 
    
    montage_names, montage_data, montage_missing,target_fs = preprocess_signal_with_montages(channels, data,target_fs,original_fs) 
    
    windows, timestamps = window_prediction(montage_data, target_fs, window_size, step_size)
    data_for_class = []
    # Feature extraction and brain map calculation
    for win in windows:
        features = feature_extraction_window(win, fs, stft_window_size, stft_overlap) # shape: (n_channels, n_features)
        assert not np.isnan(features).any(), "NaN in features!"
        x = torch.tensor(features, dtype = torch.float)
        data_for_class.append(x)
        
    # Notfallprüfung
    if len(data_for_class) == 0:
        return {
            "seizure_present": False,
            "seizure_confidence": 0.0,
            "onset": 0.0,
            "onset_confidence": 0.0,
            "offset": 0.0,
            "offset_confidence": 0.0
        }
    # Klassifikation
    predictions_per_window =[]
    with torch.no_grad():
        probs = predictions_ensemble(data_for_class ,model_name, device)
        predictions_per_window = [int(p > 0.5) for p in probs]

    seizure_present = False
    seizure_present, onset_candidate = detect_onset(predictions_per_window, timestamps, min_consecutive=2)
    if seizure_present:
        onset = onset_candidate

        
#------------------------------------------------------------------------------  
    prediction = {"seizure_present":seizure_present,"seizure_confidence":seizure_confidence,
                   "onset":onset,"onset_confidence":onset_confidence,"offset":offset,
                   "offset_confidence":offset_confidence}
  
    return prediction # Dictionary mit prediction - Muss unverändert bleiben!
                               
                               
# Methode die mit den 5 abgespeicherten Modellen einen Mehreheitsentscheid macht
# 5 Modelle aus Stratified Fold für robustere Vorhersage
def predictions_ensemble(data_for_class: List[torch.Tensor], model_name: str, device: torch.device) -> List[float]:
    file_paths = sorted([os.path.join(model_name, f) for f in os.listdir(model_name) if f.endswith(".pth")])
    batch_tensor = torch.stack(data_for_class).to(device)
    probs = []

    with torch.no_grad():
        for path in file_paths:
            model = CNN_EEG(6, 1).to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
            outputs = torch.sigmoid(model(batch_tensor)).squeeze(1)
            probs.append(outputs.cpu().numpy())  # shape: (num_windows,)

    ensemble_probs = np.mean(probs, axis=0)  # Mittelwert pro Fenster


    # Sicherstellen, dass es immer eine Liste ist
    if np.isscalar(ensemble_probs):
        return [ensemble_probs]
    else:
        return ensemble_probs.tolist()  # Gib Liste von Wahrscheinlichkeiten zurück


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
    sos = sps.butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, signal, axis=-1)

def resample_signal(signal, original_fs, target_fs=256):
    if original_fs == target_fs:
        return signal
    gcd = np.gcd(int(original_fs), int(target_fs))
    up = int(target_fs // gcd)
    down = int(original_fs // gcd)
    return resample_poly(signal, up, down, axis=-1)