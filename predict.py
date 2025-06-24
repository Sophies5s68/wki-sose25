# -*- coding: utf-8 -*-
"""

Skript testet das vortrainierte Modell


@author:  Maurice Rohr, Dirk Schweickard
"""


import numpy as np
import json
import os
from typing import List, Tuple, Dict, Any
from wettbewerb import get_3montages

# Pakete aus dem Vorlesungsbeispiel
import mne
from scipy import signal as sig
import ruptures as rpt
import torch 
import torch.nn as nn
from CNN_model import CNN_EEG
from new_preprocess import preprocess_signal_with_montages
from new_features import window_prediction, feature_extraction_window
#from CNN_dataset import window_data_evaluate, create_fixed_grid_maps
from glob import glob

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
    
    model = torch.load(model_name, map_location=device)
    model.to(device)
    model.eval()
    
    #Daten vorbereiten
    window_size = 4
    step_size = 1
    target_fs = 256
    original_fs = fs
    processed_input = preprocess_signal_with_montages(channels, data, target_fs, original_fs)
    

    
    windows, timestamps = window_prediction(preprocessed_input, target_fs, window_size, step_size)
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
            predicted_class = predictions_ensemble(feature_matr ,model_name, device)
            predictions_per_window.append(predicted_class)
    
    seizure_present = False
    if 1 in predictions_per_window:
        seizure_present = True
        first_index = predictions_per_window.index(1)
        onset = timestamps[first_index]

        
#------------------------------------------------------------------------------  
    prediction = {"seizure_present":seizure_present,"seizure_confidence":seizure_confidence,
                   "onset":onset,"onset_confidence":onset_confidence,"offset":offset,
                   "offset_confidence":offset_confidence}
  
    return prediction # Dictionary mit prediction - Muss unverändert bleiben!
                               
                               
        
def predictions_ensemble(feature,model_name,device):
    
    file_paths = sorted([os.path.join(model_name, f) for f in os.listdir(model_name) if f.endswith(".pth")])

    probas = torch.zeros(2).to(device)  # 2 Klassen
    with torch.no_grad():
        for path in file_paths:
            model = torch.load(path, map_location=device)
            model.eval()
            output = model(feature)  # shape: [1, 2]
            probas += torch.softmax(output.squeeze(), dim=0)  # → shape: [2]

    prediction = probas / len(file_paths)  # shape: [2]
    y_pred = (prediction[1] > 0.5).long()  # ← sicher!
    return y_pred
