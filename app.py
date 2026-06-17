import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy import signal
from scipy.ndimage import maximum_filter
from collections import defaultdict
import pandas as pd
import pickle
import os
import tempfile

# ==========================================
# 1. Core Processing Functions
# ==========================================
@st.cache_data 
def load_database(filename="song_database.pkl"):
    """Loads the pre-computed database from disk. Caches it in memory for speed."""
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None

def generate_spectrogram(audio_signal, fs, nperseg=1024, noverlap=512):
    frequencies, times, Sxx = signal.spectrogram(
        audio_signal, fs, window='hann', nperseg=nperseg, noverlap=noverlap
    )
    return frequencies, times, 10 * np.log10(Sxx + 1e-10)

def extract_constellation(Sxx_db, filter_size=15, threshold_percentile=90):
    local_max = maximum_filter(Sxx_db, size=filter_size) == Sxx_db
    above_background = Sxx_db > np.percentile(Sxx_db, threshold_percentile)
    freq_indices, time_indices = np.where(local_max & above_background)
    return freq_indices, time_indices

def generate_hashes(freq_indices, time_indices, target_zone_width=5, fan_out=3):
    sort_order = np.argsort(time_indices)
    sorted_time, sorted_freq = time_indices[sort_order], freq_indices[sort_order]
    hashes = []
    num_peaks = len(sorted_time)
    
    for i in range(num_peaks):
        matches_found = 0
        for j in range(i + 1, num_peaks):
            delta_t = sorted_time[j] - sorted_time[i]
            if 0 < delta_t <= target_zone_width:
                hashes.append(((sorted_freq[i], sorted_freq[j], delta_t), sorted_time[i]))
                matches_found += 1
            if matches_found >= fan_out or delta_t > target_zone_width:
                break
    return hashes

def match_query(query_audio, fs, database):
    _, _, Sxx_db = generate_spectrogram(query_audio, fs)
    f_idx, t_idx = extract_constellation(Sxx_db)
    query_hashes = generate_hashes(f_idx, t_idx)
    
    tally = defaultdict(lambda: defaultdict(int))
    for query_hash, query_time in query_hashes:
        if query_hash in database:
            for song_name, db_time in database[query_hash]:
                tally[song_name][query_time - db_time] += 1
                
    best_match, max_matches = "Unknown", 0
    for song_name, offsets in tally.items():
        if offsets:
            match_count = max(offsets.values())
            if match_count > max_matches:
                max_matches = match_count
                best_match = song_name
                
    return best_match, max_matches, tally, Sxx_db, f_idx, t_idx

# ==========================================
# 2. Streamlit UI Construction
# ==========================================
st.set_page_config(page_title="Sonic Signatures", layout="wide")
st.title("🎵 Sonic Signatures: Audio Identifier")
st.markdown("Upload a short audio clip to identify the song based on its spectrogram footprint.")

# Load the database immediately 
song_db = load_database("song_database.pkl")

# Failsafe if the database isn't uploaded properly
if not song_db:
    st.error("⚠️ Database not found! Please ensure 'song_database.pkl' is uploaded to the root of your GitHub repository.")
else:
    # Build the required two modes
    tab1, tab2 = st.tabs(["Single-Clip Mode", "Batch Mode"])

    # --- MODE 1: SINGLE CLIP ---
    with tab1:
        st.header("Analyze a Single Clip")
        uploaded_file = st.file_uploader("Upload an audio clip (.wav or .mp3)", type=['wav', 'mp3'], key="single")

        if uploaded_file is not None:
            with st.spinner("Analyzing acoustic fingerprint..."):
                # Save uploaded file to a temporary location safely so librosa can read it
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                try:
                    # Load and process
                    audio, fs = librosa.load(tmp_file_path, sr=None, mono=True)
                    predicted_song, match_score, all_tallies, Sxx_db, f_idx, t_idx = match_query(audio, fs, song_db)

                    st.success(f"### 🎶 Identified Song: **{predicted_song}**")
                    st.write(f"**Confidence Score:** {match_score} perfectly aligned hashes")

                    # Visualizations requested by assignment
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig1, ax1 = plt.subplots(figsize=(8, 4))
                        ax1.pcolormesh(Sxx_db, shading='gouraud', cmap='inferno')
                        ax1.scatter(t_idx, f_idx, c='cyan', s=10, edgecolors='black', label='Constellation Peaks')
                        ax1.set_title('Spectrogram & Constellation')
                        ax1.set_ylabel('Frequency Bins')
                        ax1.set_xlabel('Time Frames')
                        st.pyplot(fig1)

                    with col2:
                        fig2, ax2 = plt.subplots(figsize=(8, 4))
                        if predicted_song != "Unknown" and predicted_song in all_tallies:
                            winning_offsets = all_tallies[predicted_song]
                            ax2.bar(list(winning_offsets.keys()), list(winning_offsets.values()), color='springgreen')
                            ax2.set_title(f'Offset Histogram for "{predicted_song}"')
                            ax2.set_xlabel(r'Time Offset ($\Delta t$)')
                            ax2.set_ylabel('Number of Matches')
                        st.pyplot(fig2)
                finally:
                    # Clean up temporary file
                    os.remove(tmp_file_path)

    # --- MODE 2: BATCH MODE ---
    with tab2:
        st.header("Batch Processing")
        st.markdown("Upload multiple query clips to generate an automated `results.csv` file.")
        
        uploaded_files = st.file_uploader("Upload multiple audio clips", type=['wav', 'mp3'], accept_multiple_files=True, key="batch")
        
        if uploaded_files and st.button("Process Batch"):
            results = []
            progress_bar = st.progress(0)
            
            for i, file in enumerate(uploaded_files):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                    tmp_file.write(file.getvalue())
                    tmp_file_path = tmp_file.name
                
                try:
                    audio, fs = librosa.load(tmp_file_path, sr=None, mono=True)
                    prediction, _, _, _, _, _ = match_query(audio, fs, song_db)
                    
                    # Strict formatting: exactly "filename, prediction"
                    results.append({"filename": file.name, "prediction": prediction})
                finally:
                    os.remove(tmp_file_path)
                    
                progress_bar.progress((i + 1) / len(uploaded_files))
                
            # Create DataFrame and display
            df = pd.DataFrame(results)
            st.dataframe(df)
            
            # Convert to CSV and trigger download
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download results.csv",
                data=csv,
                file_name='results.csv',
                mime='text/csv',
            )
