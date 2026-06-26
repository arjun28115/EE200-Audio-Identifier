# Audio Fingerprinting: Sonic Signatures

## Overview
This repository builds a Shazam-style music identifier from scratch. The system is designed to recognize a short, unknown audio clip against a library of 50 indexed songs by leveraging a spectrogram, a sparse constellation of spectral peaks, paired-hash fingerprints, and time-offset voting.

## The Theory: How the Identifier Works

Identifying a song requires tracking how frequency content evolves in time. A single Fourier transform (DFT) of an entire song is not enough because it collapses all temporal structure—it reveals *which* frequencies are present, but not *when*. To build a robust identifier, the pipeline follows four core conceptual steps:

### 1. The Spectrogram and Resolution Trade-off
By sliding a short window along the signal and taking the DFT of each slice, we generate a spectrogram. There is a fundamental uncertainty trade-off: a short window gives sharp time resolution but coarse frequency bins, while a long window blurs time to give fine frequency resolution. This pipeline uses a balanced 1024-sample window to ensure crisp, well-localised peaks.

### 2. The Constellation Map
From the spectrogram, we keep only the local maxima that exceed a high-energy threshold (e.g., the 95th or 98th percentile). This discards approximately 99% of the spectrogram, leaving a sparse, noise-robust "constellation" of standout time-frequency peaks.

### 3. Paired Hashing
To make the fingerprint highly specific, each anchor peak is paired with a few nearby peaks (a fan-out structure) to form compact hashes: `(f1, f2, dt)`. These paired hashes are stored with the anchor's time. A pair is far more specific than a single frequency bin, removing accidental matches and acting as the key to a decisive match.

### 4. Time-Offset Voting (The Alignment Spike)
When a query clip is matched against the candidate songs in the database, we histogram the time offsets (`t_song - t_query`) of its matching hashes. For a true match, nearly every matching hash will agree on a single offset, creating a tall alignment spike. A wrong song will yield only a flat floor of scattered, random matches.

## Robustness Analysis
The reliance on high-energy peaks and paired hashing gives the system specific strengths and vulnerabilities:
* **Additive Noise:** Highly robust. Because peak picking is energy-based, the strongest peaks survive heavy white Gaussian noise, maintaining correct recognition down to ~0 dB SNR.
* **Time Stretch:** Survives modest tempo changes (up to a 5% time stretch) before the `dt` between paired peaks changes too much to match.
* **Pitch Shift:** Highly fragile. Because the fingerprint relies on absolute frequency bins, even a small +0.5 semitone pitch shift moves every peak to a different bin, destroying the matching hashes. 

## The Interactive Application
The identifier logic is wrapped in an interactive Streamlit app designed to visualize the signal-processing steps. It operates in two core modes:
* **Single-clip identification:** Runs the full pipeline on a query clip and visually exposes the intermediate steps, including the spectrogram, the constellation of peaks, and the final offset-alignment histogram that decides the match.
* **Batch identification:** Processes multiple query clips simultaneously and outputs the final predictions into a formatted `results.csv` file, applying a confidence threshold requiring a minimum number of aligned-offset votes and a clear lead over the runner-up.
