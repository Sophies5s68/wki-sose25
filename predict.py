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
#from CNN_model import CNN_EEG
from new_preprocess import preprocess_signal_with_montages
from new_features import window_prediction, features_prediction_grouped
#from CNN_dataset import window_data_evaluate, create_fixed_grid_maps
from glob import glob
from scipy.signal import iirnotch, butter, sosfiltfilt, resample_poly, tf2sos
from grouped_features import CNN_EEG_Conv2d_muster


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
 
    # Vorverarbeiten der Daten
    montage_names, montage_data, montage_missing, target_fs = preprocess_signal_with_montages(channels, data,target_fs,original_fs) 
    
    # Fenstern der Daten
    windows, timestamps, used = window_prediction(montage_data, target_fs, window_size, step_size)
    data_for_class = []
    # Extrahieren der Features für jedes Fenster
    for win in windows:
        features = features_prediction_grouped(win, target_fs, stft_window_size, stft_overlap) # shape: (n_channels, n_features)
        assert not np.isnan(features).any(), "NaN in features!"
        data_for_class.append(features)
        
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
    
    # Klassifikation durch Mehrheitsentscheid der trainierten Modelle
    predictions_per_window =[]
    with torch.no_grad():
        probs = predictions_ensemble(data_for_class, model_name, device)
    
    # Bestimmen des Onsets falls Anfall erkennat wurde
    seizure_present = False
    seizure_present, onset_candidate = detect_onset(probs, timestamps, min_consecutive=2)
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
    
    '''
    Führt Ensemble-Vorhersagen auf einer gegebenen Liste von EEG-Input-Tensoren durch,
    indem mehrere gespeicherte Modelle geladen und ihre binären Vorhersagen über einen Mehrheitsentscheid
    ausgewertet werden.

    Parameter:
    - data_for_class: Liste von Torch-Tensoren (einzelne EEG-Fenster), die zu einem Batch gestapelt werden.
    - model_name: Pfad zu einem Verzeichnis mit gespeicherten Modellgewichten (*.pth).
    - device: Das Torch-Gerät, auf dem gerechnet wird (z. B. "cuda" oder "cpu").

    Rückgabe:
    - Eine Liste binärer Ensemble-Vorhersagen (0 oder 1) für jedes EEG-Fenster.
    '''
    
    file_paths = sorted([os.path.join(model_name, f) for f in os.listdir(model_name) if f.endswith(".pth")])
    batch_tensor = torch.stack(data_for_class).to(device)
    probs = []
    
    # Threshold pro Modell aus externen Validierungsdatensatz:
    best_thresholds = [0.792446, 0.8888646, 0.5540436, 0.5641926, 0.8125975]
    
    # Vorhersage der einzelnen Modelle mit ihrem individuellen Threshold aus dem Tuning
    with torch.no_grad():
        for idx, path in enumerate(file_paths):
            model = CNN_EEG_Conv2d_muster(4, 1).to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
            outputs = torch.sigmoid(model(batch_tensor)).squeeze(1)
            prob = outputs.cpu().numpy()  # shape: (num_windows,)
            binary_preds = (prob > best_thresholds[idx]).astype(int)
            probs.append(binary_preds)
            
    all_model_probs = np.stack(probs)  # shape: (n_models, n_windows)
    votes_per_window = np.sum(all_model_probs, axis=0)  # shape: (n_windows,)
    ensemble_probs = (votes_per_window >= 3).astype(int)                    


    # Sicherstellen, dass es immer eine Liste ist
    if np.isscalar(ensemble_probs):
        return [ensemble_probs]
    else:
        return ensemble_probs.tolist()  # Gib Liste von Wahrscheinlichkeiten zurück


def detect_onset(predictions, timestamps, min_consecutive=2):
    
    """
    Erkennt den Beginn (Onset) eines Anfalls anhand binärer Vorhersagen über Fenster.

    Parameter:
    - predictions: Liste oder Tensor mit binären Vorhersagen (0 oder 1) pro Fenster.
    - timestamps: Liste mit Zeitpunkten (in Sekunden) zu jedem Fenster.
    - min_consecutive: Minimale Anzahl aufeinanderfolgender positiver Vorhersagen,
                       um einen Anfall zu bestätigen.

    Rückgabe:
    - Tuple (seizure_present: bool, onset_time: float oder None)
      Gibt True und den Zeitpunkt des ersten bestätigten Anfallsfensters zurück,
      falls mindestens min_consecutive Fenster hintereinander positiv sind.
      Sonst False und None.
    """
    
    predictions = torch.tensor(predictions)
    for i in range(len(predictions) - min_consecutive + 1):
        if torch.all(predictions[i:i+min_consecutive] == 1):
            return True, timestamps[i]
    return False, None



