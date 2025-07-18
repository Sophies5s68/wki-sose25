import numpy as np
import scipy.signal as sps
from scipy.stats import entropy

def window_eeg_data(signal, resampled_fs, seizure_onset, seizure_offset, window_size, step_size):
    """
    Splits EEG signal into overlapping windows and assigns seizure labels.

    Returns:
    - windows: list of np arrays with shape (n_channels, window_samples)
    - labels: list with 0 (no seizure) or 1 (seizure) per window
    - timestamps: list with window start times in seconds
    - used_whole_recording: True if recording was shorter than one window
    """
    if signal.ndim != 2:
        raise ValueError("Signal must be 2D with shape (n_channels, n_samples)")

    window_samples = int(window_size * resampled_fs)
    step_samples = int(step_size * resampled_fs)
    n_channels, n_samples = signal.shape

    windows = []
    labels = []
    timestamps = []
    used_whole_recording = False

    # Check if seizure interval is meaningful
    has_valid_seizure = not (seizure_onset == 0.0 and seizure_offset == 0.0)

    # If recording is shorter than one window, use whole signal as one window
    if n_samples < window_samples:
        windows.append(signal)
        timestamps.append(0.0)
        used_whole_recording = True

        start_sec = 0.0
        end_sec = n_samples / resampled_fs

        if has_valid_seizure:
            label = 1 if (end_sec >= seizure_onset and start_sec <= seizure_offset) else 0
        else:
            label = 0
        labels.append(label)
        return windows, labels, timestamps, used_whole_recording

    # Normal sliding window
    for start in range(0, n_samples - window_samples + 1, step_samples):
        end = start + window_samples
        window = signal[:, start:end]
        windows.append(window)

        start_sec = start / resampled_fs
        end_sec = end / resampled_fs
        timestamps.append(start_sec)

        if has_valid_seizure:
            label = 1 if (end_sec >= seizure_onset and start_sec <= seizure_offset) else 0
        else:
            label = 0

        labels.append(label)

    return windows, labels, timestamps, used_whole_recording