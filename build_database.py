"""
Index the provided song library into song_database.pkl.

Run once with the folder of provided songs:
    python build_database.py "/path/to/EE200 Project Song Database"

The label stored for each song is its filename WITHOUT extension, exactly as
required by the assignment (that label is what the identifier outputs).
"""
import os
import sys
import pickle
from collections import defaultdict

import numpy as np
import librosa
from scipy import signal
from scipy.ndimage import maximum_filter


def generate_spectrogram(audio, fs):
    f, t, Sxx = signal.spectrogram(audio, fs, nperseg=1024, noverlap=512)
    return f, t, 10 * np.log10(Sxx + 1e-10)


def extract_constellation(S):
    local_max = (maximum_filter(S, size=15) == S)
    peaks = local_max & (S > np.percentile(S, 95))
    fi, ti = np.where(peaks)
    return fi, ti


def generate_hashes(freq_idx, time_idx, fan_out=5, target_zone=10):
    hashes = []
    order = np.argsort(time_idx)
    freq_idx, time_idx = freq_idx[order], time_idx[order]
    for i in range(len(time_idx)):
        for j in range(i + 1, min(i + fan_out + 1, len(time_idx))):
            dt = time_idx[j] - time_idx[i]
            if dt <= target_zone:
                hashes.append(((freq_idx[i], freq_idx[j], dt), time_idx[i]))
    return hashes


def main(folder, out="song_database.pkl"):
    database = defaultdict(list)
    files = sorted(f for f in os.listdir(folder)
                   if f.lower().endswith((".mp3", ".wav", ".flac", ".ogg", ".m4a")))
    for k, fname in enumerate(files, 1):
        label = os.path.splitext(fname)[0]
        audio, fs = librosa.load(os.path.join(folder, fname), sr=None, mono=True)
        f, t, S = generate_spectrogram(audio, fs)
        fi, ti = extract_constellation(S)
        for h, anchor_t in generate_hashes(fi, ti):
            database[h].append((label, anchor_t))
        print(f"[{k}/{len(files)}] {label}")
    with open(out, "wb") as fh:
        pickle.dump(dict(database), fh)
    print(f"Wrote {out}: {len(database):,} unique hashes from {len(files)} songs.")


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "songs"
    main(folder)
