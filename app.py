import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy import signal
from scipy.ndimage import maximum_filter
from collections import defaultdict

# ==========================================
# 1. Core Signal Processing Functions
# ==========================================

def generate_spectrogram(audio_signal, fs, nperseg=1024, noverlap=512):
    """Computes the STFT and returns the magnitude spectrogram in dB."""
    frequencies, times, Sxx = signal.spectrogram(
        audio_signal,
        fs,
        window='hann',
        nperseg=nperseg,
        noverlap=noverlap
    )
    # Convert to decibels for easier peak detection
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    return frequencies, times, Sxx_db

def extract_constellation(Sxx_db, filter_size=15, threshold_percentile=90):
    """Finds the strongest time-frequency peaks (local maxima)."""
    # 2D maximum filter to find local peaks
    local_max = maximum_filter(Sxx_db, size=filter_size) == Sxx_db

    # Filter out background noise using a dynamic threshold
    threshold = np.percentile(Sxx_db, threshold_percentile)
    above_background = Sxx_db > threshold

    # Keep only points that are both local maxima AND loud enough
    peaks = local_max & above_background
    freq_indices, time_indices = np.where(peaks)

    return freq_indices, time_indices

def generate_hashes(freq_indices, time_indices, target_zone_width=5, fan_out=3):
    """Pairs nearby peaks into unique hashes: (Freq1, Freq2, Delta Time)."""
    # Sort chronologically
    sort_order = np.argsort(time_indices)
    sorted_time = time_indices[sort_order]
    sorted_freq = freq_indices[sort_order]

    hashes = []
    num_peaks = len(sorted_time)

    for i in range(num_peaks):
        anchor_f = sorted_freq[i]
        anchor_t = sorted_time[i]
        matches_found = 0

        # Look ahead to create pairs
        for j in range(i + 1, num_peaks):
            target_f = sorted_freq[j]
            target_t = sorted_time[j]
            delta_t = target_t - anchor_t

            # If within target zone, create a hash pair
            if 0 < delta_t <= target_zone_width:
                hash_signature = (anchor_f, target_f, delta_t)
                hashes.append((hash_signature, anchor_t))
                matches_found += 1

            # Stop if we hit the fan-out limit or passed the time window
            if matches_found >= fan_out or delta_t > target_zone_width:
                break

    return hashes

# ==========================================
# 2. Database Management
# ==========================================

def build_database(dataset_folder, output_filename="song_database.pkl"):
    """Indexes all .mp3 files in a folder into an inverted hash dictionary."""
    database = {}
    song_files = [f for f in os.listdir(dataset_folder) if f.endswith('.mp3')]

    if not song_files:
        print(f"Error: No .mp3 files found in {dataset_folder}")
        return None

    print(f"Indexing {len(song_files)} songs...")

    for i, filename in enumerate(song_files):
        print(f"[{i+1}/{len(song_files)}] Indexing: {filename}...")
        filepath = os.path.join(dataset_folder, filename)
        song_name = os.path.splitext(filename)[0] # Extract label

        try:
            # Load audio using librosa (handles mp3 and converts stereo to mono)
            audio, fs = librosa.load(filepath, sr=None, mono=True)

           _, _, Sxx_db = generate_spectrogram(audio, fs)
            
            # INCREASE filter_size (e.g., to 40) to find fewer peaks
            # INCREASE threshold_percentile (e.g., to 95) to keep only the loudest peaks
            f_idx, t_idx = extract_constellation(Sxx_db, filter_size=40, threshold_percentile=95)
            
            # DECREASE fan_out (e.g., to 2) to limit the number of pairs per peak
            song_hashes = generate_hashes(f_idx, t_idx, fan_out=2)
            # Populate the inverted index
            for hash_signature, anchor_time in song_hashes:
                if hash_signature not in database:
                    database[hash_signature] = []
                database[hash_signature].append((song_name, anchor_time))

        except Exception as e:
            print(f"Failed to process {filename}: {e}")

    # Save to disk
    with open(output_filename, 'wb') as f:
        pickle.dump(database, f)

    print(f"Database saved! Total unique hashes: {len(database)}")
    return database

def load_database(filename="song_database.pkl"):
    """Loads the pre-computed database from disk."""
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    print("Database not found. Please build it first.")
    return None

# ==========================================
# 3. The Matching Algorithm
# ==========================================

def match_query(query_audio, fs, database):
    """Identifies a song by aligning query hashes against the database."""
    _, _, Sxx_db = generate_spectrogram(query_audio, fs)
    f_idx, t_idx = extract_constellation(Sxx_db)
    query_hashes = generate_hashes(f_idx, t_idx)

    # Dictionary to track time offsets: {song_name: {offset: count}}
    tally = defaultdict(lambda: defaultdict(int))

    for query_hash, query_time in query_hashes:
        if query_hash in database:
            for song_name, db_time in database[query_hash]:
                offset = query_time - db_time
                tally[song_name][offset] += 1

    best_match = "Unknown"
    max_matches = 0

    # Find the song with the most aligned hashes (highest peak in offset histogram)
    for song_name, offsets in tally.items():
        if not offsets:
            continue
        best_offset = max(offsets, key=offsets.get)
        match_count = offsets[best_offset]

        if match_count > max_matches:
            max_matches = match_count
            best_match = song_name

    return best_match, max_matches, tally

# ==========================================
# 4. Execution / Testing Block
# ==========================================

if __name__ == "__main__":
    # --- 1. SET PATHS ---
    # Update these paths based on your local machine / Colab environment
    dataset_folder = '/content/EE200'
    db_file = '/content/song_database.pkl'
    query_path = '/content/EE200/Bohemian Rhapsody.mp3'

    # --- 2. LOAD OR BUILD DATABASE ---
    # If the .pkl file doesn't exist, build it. Otherwise, load it.
    if not os.path.exists(db_file):
        song_db = build_database(dataset_folder, db_file)
    else:
        print("Loading existing database...")
        song_db = load_database(db_file)

    # --- 3. TEST MATCHING ---
    if song_db and os.path.exists(query_path):
        print(f"\nAnalyzing query clip: {query_path}...")
        query_audio, fs = librosa.load(query_path, sr=None, mono=True)

        predicted_song, match_score, all_tallies = match_query(query_audio, fs, song_db)

        print("==================================")
        print(f"IDENTIFIED SONG: {predicted_song}")
        print(f"Confidence (Aligned Hashes): {match_score}")
        print("==================================")
    elif not os.path.exists(query_path):
        print(f"Test clip not found at: {query_path}")
