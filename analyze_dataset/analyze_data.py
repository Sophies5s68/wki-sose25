import os
import json
import csv
from collections import Counter
from typing import Tuple, List
import numpy as np
from wettbewerb import load_references


# Code zum analysieren des vorliegenden Datasets


def extract_patient_id(recording_id: str) -> str:
    return recording_id.split('_')[0]

def analyze_dataset(folder: str = '../shared_data/training', output_prefix: str = 'dataset_stats'):
    total_files = 6265 
    batch_size = 100
    idx = 0

    
    # wichtige Statistiken:
    pos_count = 0
    neg_count = 0
    patient_counter = Counter()
    reference_counter = Counter()
    sampling_freq_counter = Counter()
    channel_stats = []
    data_length_stats = []
    
    # durch alle Daten iterieren und wichtige Werte abspeichern
    while idx < total_files:
        result = load_references(folder=folder, idx=idx)
        if result is None:
            break
        ids, channels, data, s_freqs, references, labels = result

        for i in range(len(ids)):
            label = labels[i][0]
            if label:
                pos_count += 1
            else:
                neg_count += 1

            patient_id = extract_patient_id(ids[i])
            patient_counter[patient_id] += 1

            reference_counter[references[i]] += 1
            sampling_freq_counter[s_freqs[i]] += 1

            channel_stats.append(len(channels[i]))
            data_length_stats.append(data[i].shape[1] if data[i].ndim == 2 else len(data[i]))

        idx += batch_size

    # Zusammenfassung und abspeichern in dict
    stats = {
        "label_distribution": {
            "positive": pos_count,
            "negative": neg_count,
        },
        "reference_systems": dict(reference_counter),
        "sampling_frequencies": dict(sampling_freq_counter),
        "patients": {
            "unique_patient_count": len(patient_counter),
            "total_recordings": sum(patient_counter.values()),
            "top_5_patients": patient_counter.most_common(5)
        },
        "channels": {
            "mean": np.mean(channel_stats),
            "min": int(np.min(channel_stats)),
            "max": int(np.max(channel_stats))
        },
        "data_lengths": {
            "mean": int(np.mean(data_length_stats)),
            "min": int(np.min(data_length_stats)),
            "max": int(np.max(data_length_stats))
        }
    }

    # speichern als JSON
    json_file = output_prefix + '.json'
    with open(json_file, 'w') as f:
        json.dump(stats, f, indent=4)

    # Speichern der verschiedenen CSV Dateien mit den extrahierten Werten
    csv_patient_file = output_prefix + '_recordings_per_patient.csv'
    with open(csv_patient_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Patient_ID', 'Anzahl_Aufnahmen'])
        for patient, count in patient_counter.most_common():
            writer.writerow([patient, count])
    print(" Patienten gespeichert")

    csv_ref_file = output_prefix + '_reference_systems.csv'
    with open(csv_ref_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Referenzsystem', 'Anzahl'])
        for ref, count in reference_counter.items():
            writer.writerow([ref, count])
    print(" Referenzsysteme gespeichert")

    csv_sf_file = output_prefix + '_sampling_frequencies.csv'
    with open(csv_sf_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Sampling_Frequenz_Hz', 'Anzahl_Aufnahmen'])
        for sf, count in sampling_freq_counter.items():
            writer.writerow([sf, count])
    print(" Sampling Frequenzen gespeichert")

if __name__ == "__main__":
    analyze_dataset(folder="../shared_data/training", output_prefix="dataset_stats")
