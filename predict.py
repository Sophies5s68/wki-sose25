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
from preprocess import process_without_mne
from features import feature_extraction
from CNN_dataset import window_data_evaluate, create_fixed_grid_maps
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
    '''
    model = torch.load(model_name, map_location=device)
    model.to(device)
    model.eval()
    '''
    #Daten vorbereiten
    window_size = 4.0
    step_size = 1
    standard_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
    n_nodes = len(standard_channels)

    processed_input = process_without_mne(data, fs, channels, reference_system, fs)
    
    channel_map = {ch: idx for idx, ch in enumerate(channels)}
    pad_data = np.zeros((n_nodes, processed_input.shape[1]))
    for j, ch in enumerate(standard_channels):
        if ch in channel_map:
            pad_data[j] = processed_input[channel_map[ch]]
    
    windows = window_data_evaluate(pad_data, fs, window_size, step_size)
    data_for_class = []
    # Feature extraction and brain map calculation
    for win in windows:
        features = feature_extraction(win, fs) # shape: (n_channels, n_features)
        assert not np.isnan(features).any(), "NaN in features!"
        brain_map = create_fixed_grid_maps(features, channels)
        assert not np.isnan(brain_map).any(), "NaN in brain_map!"
        x = torch.tensor(brain_map, dtype = torch.float)
        data_for_class.append(x)
        

    # Klassifikation
    predictions_per_window =[]
    with torch.no_grad():
        for feature_map in data_for_class:
            feature_map = feature_map.unsqueeze(0).to(device)
            #output = model(feature_map)
            #predicted_class = torch.argmax(output, dim=1).item()
            predicted_class = predictions_ensemble(feature_map,model_name,device)
            predictions_per_window.append(predicted_class)
    
    seizure_present = False
    '''
    if 1 in predictions_per_window:
        seizure_present = True
        first_idx = predictions_per_window.index(1)
        time_first = first_idx * step_size
        onset = time_first
    '''
    for i in range(len(predictions_per_window) - 1):
        if predictions_per_window[i] == 1 and predictions_per_window[i + 1] == 1:
            seizure_present = True
            time_first = i * step_size
            onset = time_first
            break
    '''
    # Hier könnt ihr euer vortrainiertes Modell laden (Kann auch aus verschiedenen Dateien bestehen)
    model = MyCNN()
    model.load_state_dict(torch.load(model_name, map_location='cpu'))
    model.eval()

    
    # Wende Beispielcode aus Vorlesung an 
    
    _montage, _montage_data, _is_missing = get_3montages(channels, data)
    signal_std = np.zeros(len(_montage))
    for j, signal_name in enumerate(_montage):
        # Ziehe erste Montage des EEG
        signal = _montage_data[j]
        # Wende Notch-Filter an um Netzfrequenz zu dämpfen
        signal_notch = mne.filter.notch_filter(x=signal, Fs=fs, freqs=np.array([50.,100.]), n_jobs=2, verbose=False)
        # Wende Bandpassfilter zwischen 0.5Hz und 70Hz um Rauschen aus dem Signal zu filtern
        signal_filter = mne.filter.filter_data(data=signal_notch, sfreq=fs, l_freq=0.5, h_freq=70.0, n_jobs=2, verbose=False)
        
        # Berechne short time fourier transformation des Signal: signal_filtered = filtered signal of channel, fs = sampling frequency, nperseg = length of each segment
        # Output f= array of sample frequencies, t = array of segment times, Zxx = STFT of signal
        f, t, Zxx = sig.stft(signal_filter, fs, nperseg=fs * 3)
        # Berechne Schrittweite der Frequenz
        df = f[1] - f[0]
        # Berechne Engergie (Betrag) basierend auf Real- und Imaginärteil der STFT
        E_Zxx = np.sum(Zxx.real ** 2 + Zxx.imag ** 2, axis=0) * df
        
        signal_std[j] = np.std(signal_filter)



        # Erstelle neues Array in der ersten Iteration pro Patient
        if j == 0:
            # Initilisiere Array mit Energiesignal des ersten Kanals
            E_array = np.array(E_Zxx)
        else:
            # Füge neues Energiesignal zu vorhandenen Kanälen hinzu (stack it)
            E_array = np.vstack((E_array, np.array(E_Zxx)))
            
    # Berechne Feature zur Seizure Detektion
    signal_std_max = signal_std.max()
    # Klassifiziere Signal
    seizure_present = signal_std_max>th_opt
    
    # Berechne Gesamtenergie aller Kanäle für jeden Zeitppunkt
    E_total = np.sum(E_array, axis=0)
    # Berechne Stelle der maximalen Energie
    max_index = E_total.argmax()

    # Berechne "changepoints" der Gesamtenergie
    # Falls Maximum am Anfang des Signals ist muss der Onset ebenfalls am Anfang sein und wir können keinen "changepoint" berechnen
    if max_index == 0:
        onset = 0.0
        onset_confidence = 0.2
        
    else:
        # Berechne "changepoint" mit dem ruptures package
        # Setup für  "linearly penalized segmentation method" zur Detektion von changepoints im Signal mi rbf cost function
        algo = rpt.Pelt(model="rbf").fit(E_total)
        # Berechne sortierte Liste der changepoints, pen = penalty value
        result = algo.predict(pen=10)
        #Indices sind ums 1 geshiftet
        result1 = np.asarray(result) - 1
        # Selektiere changepoints vor Maximum
        result_red = result1[result1 < max_index]
        # Falls es mindestens einen changepoint gibt nehmen wir den nächsten zum Maximum
        if len(result_red)<1:
            # Falls keine changepoint gefunden wurde raten wir, dass er "nahe" am Maximum ist
            print('No changepoint, taking maximum')
            onset_index = max_index
        else:
            # Der changepoint entspricht gerade dem Onset 
            onset_index = result_red[-1]
        # Gebe Onset zurück
        onset = t[onset_index]      
     
     
    '''
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
