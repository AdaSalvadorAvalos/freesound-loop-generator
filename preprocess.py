import torch
import torchaudio
import re
import numpy as np
import os
import json
import random
import librosa
from tqdm import tqdm
import argparse
import shutil
import essentia
from essentia.standard import MonoLoader, RhythmExtractor2013
import subprocess
from scipy.signal import find_peaks 


TARGET_SAMPLE_RATE = 44100
TARGET_DURATION = 16
TARGET_BPM = 120 




# --------

def detect_beats_and_downbeats(audio_path, target_sample_rate=TARGET_SAMPLE_RATE, required_time_sig=4):
    """
    Detect beats and estimate downbeats, but only for files with the required time signature.
    Uses librosa instead of madmom for better compatibility.
    """
    try:
        # First check if the file has the required time signature
        # NOTE: REDUNDANT, but keeping for clarity and skipping logic
        detected_time_sig = estimate_time_signature(audio_path)
        
        if detected_time_sig != required_time_sig:
            print(f"Skipping file - detected {detected_time_sig}/4 time, need {required_time_sig}/4")
            return None, None, None, 0.0
        
        # Load audio with librosa
        y, sr = librosa.load(audio_path, sr=target_sample_rate)
        
        # Track beats
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        
        # Estimate downbeats
        downbeat_indices = estimate_downbeats_librosa(y, sr, beat_frames, detected_time_sig)
        # downbeat_indices = estimate_downbeats_madmom(audio_path)
        
        # Calculate confidence based on beat regularity
        confidence = calculate_beat_confidence(beat_times)
        
        # Fix: Extract scalar value from tempo array to avoid deprecation warning
        tempo_scalar = float(tempo.item()) if hasattr(tempo, 'item') else float(tempo)
        
        return beat_times, downbeat_indices, tempo_scalar, confidence
        
    except Exception as e:
        print(f"Error in beat detection: {e}")
        return None, None, None, 0.0



def estimate_time_signature(audio_path):
    """
    Estimate the time signature (beats per bar) from an audio file
    using beat interval analysis and onset detection.
    Returns the upper number of the time signature (e.g., 3 for 3/4).
    """
    try:
        y, sr = librosa.load(audio_path)

        # Compute onset envelope
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)

        # Beat tracking
        tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        
        if len(beat_frames) < 8:
            return 4  # fallback default

        # Analyze beat strengths
        beat_strengths = onset_env[np.clip(beat_frames, 0, len(onset_env)-1)]
        
        # Detect peaks (potential downbeats)
        peaks, _ = find_peaks(beat_strengths, height=np.mean(beat_strengths))
        
        if len(peaks) > 2:
            intervals = np.diff(peaks)
            if len(intervals) > 0:
                hist, bins = np.histogram(intervals, bins=np.arange(1, 9))
                est = bins[:-1][np.argmax(hist)]
                if est in [3, 4, 6]:
                    return int(est)

        return 4  # fallback
    except Exception as e:
        print(f"Error estimating time signature: {e}")
        return 0


def estimate_downbeats_librosa(y, sr, beat_frames, time_signature=4):
    """
    Estimate downbeats using librosa's onset detection and beat tracking.
    
    Args:
        y: Audio time series
        sr: Sample rate
        beat_frames: Beat frame positions from librosa.beat.beat_track
        time_signature: Number of beats per measure
        
    Returns:
        numpy.array: Indices of estimated downbeats in beat_frames array
    """
    try:
        if len(beat_frames) < time_signature:
            return np.array([0]) if len(beat_frames) > 0 else np.array([])
        
        # Method 1: Use onset strength to find strong beats
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        
        # Ensure beat_frames don't exceed onset_env length
        valid_beat_frames = beat_frames[beat_frames < len(onset_env)]
        
        if len(valid_beat_frames) == 0:
            print("No valid beat frames available for downbeat estimation. Skipping.")
            return np.array([])
            return estimate_downbeats_simple(beat_frames, time_signature)
        
        # Get onset strength at valid beat positions
        beat_strengths = onset_env[valid_beat_frames]
        
        # Method 2: Use spectral features to identify downbeats
        # Downbeats often have more low-frequency energy
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        # Sync chroma to beats - this ensures matching dimensions
        chroma_beats = librosa.util.sync(chroma, valid_beat_frames)
        
        # Downbeats often have stronger tonal clarity
        chroma_strength = np.sum(chroma_beats, axis=0)
        
        # Ensure both arrays have the same length
        min_length = min(len(beat_strengths), len(chroma_strength))
        beat_strengths = beat_strengths[:min_length]
        chroma_strength = chroma_strength[:min_length]
        
        # Combine onset strength and chroma strength
        combined_strength = beat_strengths * 0.7 + chroma_strength * 0.3

        # Look for peaks with minimum distance of time_signature beats
        peaks, _ = find_peaks(combined_strength, distance=time_signature)
        
        if len(peaks) > 0:
            # Refine: ensure we start from a strong beat near the beginning
            first_strong_beat = peaks[0] if peaks[0] < time_signature else 0
            
            # Caluclate downbeat positions
            downbeat_indices = []
            pos = first_strong_beat
            while pos < len(valid_beat_frames):
                downbeat_indices.append(pos)
                pos += time_signature
            
            return np.array(downbeat_indices)
        
    except Exception as e:
        print(f"Error in librosa downbeat estimation: {e}")
        return np.array([])
        return estimate_downbeats_simple(beat_frames, time_signature)

def estimate_downbeats_simple(beat_times_or_frames, time_signature=4):
    """
    Simple downbeat estimation assuming regular time signature.
    
    Args:
        beat_times_or_frames: Array of beat times or frames
        time_signature: Number of beats per measure (default 4 for 4/4 time)
        
    Returns:
        numpy.array: Indices of estimated downbeats
    """
    if len(beat_times_or_frames) < time_signature:
        return np.array([0]) if len(beat_times_or_frames) > 0 else np.array([])
    
    # Simple approach: assume every Nth beat is a downbeat
    downbeat_indices = np.arange(0, len(beat_times_or_frames), time_signature)
    
    return downbeat_indices

def calculate_beat_confidence(beat_times):
    """
    Calculate confidence in beat detection based on regularity.
    
    Args:
        beat_times: Array of beat times in seconds
        
    Returns:
        float: Confidence score between 0 and 1
    """
    if len(beat_times) < 3:
        return 0.0
    
    # Calculate inter-beat intervals
    intervals = np.diff(beat_times)
    
    # Confidence is based on how regular the intervals are
    if len(intervals) == 0:
        return 0.0
    
    mean_interval = np.mean(intervals)
    std_interval = np.std(intervals)
    
    # Normalize confidence: lower standard deviation = higher confidence
    if mean_interval == 0:
        return 0.0
    
    coefficient_of_variation = std_interval / mean_interval
    confidence = max(0.0, 1.0 - coefficient_of_variation * 2)  # Scale appropriately
    
    return min(1.0, confidence)


def align_to_downbeat_and_normalize_beats(audio_path, output_path, target_beats=32, 
                                        target_sample_rate=TARGET_SAMPLE_RATE, confidence_threshold=0.3):
    """
    Align audio to start with a downbeat and normalize to have a specific number of beats.
    FIXED: Always use TARGET_BPM for duration calculation to ensure consistency.
    """
    try:
        # Load original audio
        wav, sr = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        
        # Resample if necessary
        if sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)
            wav = resampler(wav)
            sr = target_sample_rate
        
        # Detect beats and downbeats
        beat_times, downbeat_indices, detected_bpm, confidence = detect_beats_and_downbeats(audio_path, target_sample_rate)
        
        beat_info = {
            'detected_bpm': detected_bpm,
            'confidence': confidence,
            'num_beats': len(beat_times) if beat_times is not None else 0,
            'num_downbeats': len(downbeat_indices) if downbeat_indices is not None else 0,
            'aligned': False
        }
        
        # Always use TARGET_BPM for duration calculation, not detected BPM
        # This ensures consistent output duration regardless of detection accuracy
        target_duration = target_beats * (60.0 / TARGET_BPM)  # Use TARGET_BPM, not detected_bpm
        target_samples = int(target_duration * target_sample_rate)
        
        print(f"Target duration: {target_duration:.2f}s ({target_samples} samples) based on {TARGET_BPM} BPM")
        
        # Check if we have sufficient confidence and beats for alignment
        if (beat_times is None or confidence < confidence_threshold or 
            len(beat_times) < 4 or len(downbeat_indices) == 0):
            print(f"Low confidence ({confidence:.2f}) or insufficient beats detected. Using duration-based processing.")
            
            # Fallback: simple duration-based processing with TARGET_BPM duration
            processed_wav = ensure_exact_length(wav, target_samples)
            torchaudio.save(output_path, processed_wav, target_sample_rate)
            beat_info['target_samples'] = target_samples
            beat_info['actual_samples'] = processed_wav.shape[1]
            return True, processed_wav, beat_info
        
        # Convert beat times to sample indices
        beat_samples = (beat_times * target_sample_rate).astype(int)
        downbeat_samples = beat_samples[downbeat_indices]
        
        # Find the best downbeat that gives us enough audio
        start_sample = None
        
        for downbeat_idx in downbeat_indices:
            potential_start = beat_samples[downbeat_idx]
            
            # Check if we have enough beats after this downbeat
            remaining_beats = len(beat_times) - downbeat_idx
            
            if remaining_beats >= target_beats:
                start_sample = potential_start
                break
        
        # If no suitable downbeat found, use the first one
        if start_sample is None and len(downbeat_samples) > 0:
            start_sample = downbeat_samples[0]
        elif start_sample is None:
            start_sample = 0
        
        # Calculate end sample using TARGET_BPM-based duration
        end_sample = start_sample + target_samples
        
        # Extract the segment, handling edge cases
        if start_sample >= wav.shape[1]:
            # Start is beyond audio length - return original audio
            processed_wav = wav
        elif end_sample <= wav.shape[1]:
            # Perfect case: we have enough audio
            processed_wav = wav[:, start_sample:end_sample]
            beat_info['aligned'] = True
        else:
            # We need to extend the audio
            available_samples = wav.shape[1] - start_sample
            aligned_wav = wav[:, start_sample:]
            
            # Extend by looping if necessary
            if available_samples < target_samples:
                # Calculate how many loops we need
                loops_needed = int(np.ceil(target_samples / available_samples))
                processed_wav = aligned_wav.repeat(1, loops_needed)
            else:
                processed_wav = aligned_wav
            
            beat_info['aligned'] = True
        
        # GUARANTEE exact length
        processed_wav = ensure_exact_length(processed_wav, target_samples)
        
        # Save the processed audio
        torchaudio.save(output_path, processed_wav, target_sample_rate)
        
        beat_info['start_sample'] = start_sample
        beat_info['target_samples'] = target_samples
        beat_info['actual_samples'] = processed_wav.shape[1]
        
        return True, processed_wav, beat_info
        
    except Exception as e:
        print(f"Error in beat alignment: {e}")
        # Fallback: duration-based processing with exact length
        try:
            wav, sr = torchaudio.load(audio_path)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            
            if sr != target_sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)
                wav = resampler(wav)
            
            # Use TARGET_BPM for duration calculation
            target_duration = target_beats * (60.0 / TARGET_BPM)
            target_samples = int(target_duration * target_sample_rate)
            
            processed_wav = ensure_exact_length(wav, target_samples)
            torchaudio.save(output_path, processed_wav, target_sample_rate)
            
            return True, processed_wav, {
                'detected_bpm': TARGET_BPM, 
                'confidence': 0.0, 
                'aligned': False, 
                'target_samples': target_samples, 
                'actual_samples': processed_wav.shape[1]
            }
            
        except Exception as e2:
            print(f"Fallback processing also failed: {e2}")
            return False, None, None

def ensure_exact_length(wav, target_samples):
    """
    Ensure the audio has exactly the target number of samples.
    
    Args:
        wav: Audio tensor of shape (channels, samples)
        target_samples: Desired number of samples
        
    Returns:
        torch.Tensor: Audio with exactly target_samples samples
    """
    current_samples = wav.shape[1]
    
    if current_samples == target_samples:
        return wav
    elif current_samples < target_samples:
        # Need to extend - loop the audio
        loops_needed = int(np.ceil(target_samples / current_samples))
        extended_wav = wav.repeat(1, loops_needed)
        return extended_wav[:, :target_samples]  # Exact truncation
    else:
        # Need to trim
        return wav[:, :target_samples]


def process_wav_with_beat_alignment(file_path, processed_save_path, metadata=None, 
                                  target_duration=10, target_sample_rate=TARGET_SAMPLE_RATE, 
                                  target_bpm=TARGET_BPM, preserve_bpm=False, 
                                  align_beats=True, target_beats=32, confidence_threshold=0.3, required_time_sig=4):
    """
    Enhanced version of process_wav that includes beat alignment and GUARANTEES consistent output length.
    """
    if not os.path.exists(file_path):
        print(f"Error: Input file {file_path} does not exist")
        return None, None, None, 'failed'
    
    # If beat alignment is requested, use the new function
    if align_beats:
        try:
            detected_time_sig = estimate_time_signature(file_path)
            if detected_time_sig != required_time_sig:
                return None, None, {
                    'detected_time_sig': detected_time_sig,
                    'required_time_sig': required_time_sig,
                    'reason': 'time_signature_mismatch'
                }, 'skipped_time_sig'
        except Exception as e:
            print(f"Error detecting time signature for {file_path}: {e}")
            return None, None, None, 'failed'
        
        print(f"Processing with beat alignment: {file_path}")
        
        # First, do the basic tempo conversion if needed
        original_bpm = get_file_bpm(file_path, metadata)
        temp_file = f"temp_before_alignment_{os.path.basename(file_path)}"
        
        # Step 1: Adjust tempo if needed (same as original process_wav)
        if not preserve_bpm and original_bpm and abs(original_bpm - target_bpm) > 1.0:
            tempo_adjusted_file = tempo_convert_ffmpeg(
                file_path, target_bpm, original_bpm, target_sample_rate
            )
            working_file = tempo_adjusted_file
            final_bpm = target_bpm
        else:
            working_file = file_path
            final_bpm = original_bpm if original_bpm else target_bpm
            
            # Just convert sample rate if needed
            info = torchaudio.info(file_path)
            if info.sample_rate != target_sample_rate:
                wav, sr = torchaudio.load(file_path)
                wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)(wav)
                torchaudio.save(temp_file, wav, target_sample_rate)
                working_file = temp_file
        
        # Step 2: Apply beat alignment with guaranteed consistent length
        success, aligned_wav, beat_info = align_to_downbeat_and_normalize_beats(
            working_file, processed_save_path, target_beats=target_beats, 
            target_sample_rate=target_sample_rate, confidence_threshold=confidence_threshold
        )
        
        # Clean up temp files
        temp_files = [
            f"temp_atempo_{os.path.basename(file_path)}",
            f"temp_asetrate_{os.path.basename(file_path)}",
            temp_file
        ]
        for tf in temp_files:
            if os.path.exists(tf) and tf != processed_save_path:
                try:
                    os.remove(tf)
                except:
                    pass
        
        if success:
            target_samples_16sec = int(TARGET_DURATION * target_sample_rate)  # Always 16 seconds
            aligned_wav = ensure_exact_length(aligned_wav, target_samples_16sec)
            torchaudio.save(processed_save_path, aligned_wav, target_sample_rate)
            # Verify the output length
            if beat_info and 'actual_samples' in beat_info:
                expected_samples = int(target_beats * (60.0 / final_bpm) * target_sample_rate)
                actual_samples = beat_info['actual_samples']
                print(f"Expected samples: {expected_samples}, Actual samples: {actual_samples}")
                
                if abs(actual_samples - expected_samples) > 1:  # Allow 1 sample tolerance
                    print(f"Warning: Output length mismatch for {file_path}")
            
            return processed_save_path, final_bpm, beat_info, 'success'
        else:
            return None, None, None, 'failed'
    
    else:
        # Use original process_wav function
        result = process_wav(file_path, processed_save_path, metadata, target_duration, 
                           target_sample_rate, target_bpm, preserve_bpm)
        if result and len(result) >= 2:
            return result[0], result[1], None, 'success'
        else:
            return None, None, None, 'failed'


# Modified version of your process_dataset function to use beat alignment
def process_dataset_with_beat_alignment(input_dir, output_dir, file_list=None, num_files=None, 
                                       metadata=None, preserve_bpm=False, shuffle=False, num_mixes=None,
                                       align_beats=True, target_beats=32, confidence_threshold=0.3):
    """
    Enhanced version of process_dataset that includes beat alignment functionality.
    """
    
    # Create necessary directories
    processed_dir = os.path.join(output_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    # If file_list is provided, use it; otherwise get all WAV files from the input directory
    if file_list:
        wav_files = file_list
    else:
        wav_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.wav', '.WAV'))]
    
    # Shuffle files if requested
    if shuffle:
        print(f"Shuffling {len(wav_files)} files randomly...")
        random.shuffle(wav_files)
        
        # Save the shuffled file list for reproducibility
        shuffled_list_path = os.path.join(output_dir, "shuffled_files_list.json")
        with open(shuffled_list_path, 'w') as f:
            json.dump(wav_files, f, indent=2)
        print(f"Saved shuffled file list to {shuffled_list_path}")
    
    # Limit to the specified number of files if needed
    if num_files is not None:
        wav_files = wav_files[:num_files]
    
    # Process each original file
    original_results = []
    file_mappings = {}
    beat_alignment_stats = []
    skipped_files = []

    
    for wav_file in tqdm(wav_files, desc="Processing original files with beat alignment"):
        file_path = os.path.join(input_dir, wav_file)
        
        try:
            # Clean the filename for saving
            base_name = clean_filename(os.path.basename(wav_file))
            processed_path = os.path.join(processed_dir, f"{base_name}_processed.wav")
            
            # Process the WAV file with beat alignment
            processed_path, final_bpm, beat_info, status = process_wav_with_beat_alignment(
                file_path, 
                processed_path, 
                metadata=metadata,
                preserve_bpm=preserve_bpm,
                align_beats=align_beats,
                target_beats=target_beats,
                confidence_threshold=confidence_threshold
            )
            
            if status == 'success':
                file_mappings[file_path] = {"processed": processed_path}
                if beat_info:
                    beat_alignment_stats.append({
                        "file": wav_file,
                        "beat_info": beat_info,
                        "final_bpm": final_bpm
                    })
                    
            elif status == 'skipped_time_sig':
                skipped_files.append({
                    "file": wav_file, 
                    "time_sig": beat_info.get('detected_time_sig', 'unknown'),
                    "reason": "time_signature_mismatch"
                })
                print(f"Skipped {wav_file} - {beat_info.get('detected_time_sig', '?')}/4 time signature")
                
        except Exception as e:
            print(f"Error processing {wav_file}: {e}")
    
    # Save skipped files info
    if skipped_files:
        skipped_path = os.path.join(output_dir, "skipped_files.json")
        with open(skipped_path, 'w') as f:
            json.dump(skipped_files, f, indent=2)
        print(f"Skipped {len(skipped_files)} files due to time signature")
    
    # Save beat alignment statistics
    if beat_alignment_stats:
        beat_stats_path = os.path.join(output_dir, "beat_alignment_stats.json")
        with open(beat_stats_path, 'w') as f:
            json.dump(beat_alignment_stats, f, indent=2)
        
        # Calculate summary statistics
        aligned_count = sum(1 for stat in beat_alignment_stats if stat["beat_info"]["aligned"])
        total_count = len(beat_alignment_stats)
        avg_confidence = np.mean([stat["beat_info"]["confidence"] for stat in beat_alignment_stats])
        
        summary_stats = {
            "total_files": int(total_count),
            "successfully_aligned": int(aligned_count),
            "alignment_success_rate": float(aligned_count / total_count) if total_count > 0 else 0.0,
            "average_confidence": float(avg_confidence),
            "target_beats": target_beats,
            "confidence_threshold": confidence_threshold
        }
        
        summary_path = os.path.join(output_dir, "beat_alignment_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        print(f"Beat alignment summary: {aligned_count}/{total_count} files successfully aligned")
        print(f"Average confidence: {avg_confidence:.3f}")
    
    return original_results

# ---------------

def run_ffmpeg_command(cmd):
    """Run FFmpeg command with error handling."""
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr
    
def load_metadata(metadata_path):
    """Load BPM information from metadata.json"""
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata

def get_file_id_from_filename(filename):
    """Extract file ID from filename like FSL10K/audio/wav/425998_9497060.wav.wav"""
    # Extract just the base filename without path
    base_name = os.path.basename(filename)
    # Extract the ID part (before the first underscore)
    file_id = base_name.split('_')[0]
    return file_id

def get_bpm_from_metadata(file_path, metadata):
    """Get the BPM for a file from metadata"""
    file_id = get_file_id_from_filename(file_path)
    
    # Look up the file in metadata
    if file_id in metadata:
        if "annotations" in metadata[file_id] and "bpm" in metadata[file_id]["annotations"]:
            return float(metadata[file_id]["annotations"]["bpm"])
    
    # Return None if BPM not found, we'll detect it algorithmically later
    return None

def detect_bpm(file_path):
    """Detect BPM using Essentia's RhythmExtractor2013."""
    try:
        audio = MonoLoader(filename=file_path)()
        rhythm_extractor = RhythmExtractor2013(method="multifeature")
        bpm, _, _, _, _ = rhythm_extractor(audio)
        
        # Apply sanity check
        if 30 <= bpm <= 300:
            return float(bpm)
        else:
            print(f"Detected unreasonable BPM ({bpm}), defaulting to 120.0")
            return 120.0
    except Exception as e:
        print(f"Error detecting BPM: {e}")
        return 120.0
    

def clean_filename(filename):
    base_name = re.sub(r'(\.\w+)+$', '', filename)
    base_name = base_name.replace("_processed", "")
    base_name = base_name.replace("_", "")
    return base_name


def tempo_convert_ffmpeg(input_file, target_bpm, original_bpm, target_sample_rate):
    """
    Convert tempo using FFmpeg with both atempo and asetrate methods,
    then determine which result is closer to the target BPM.
    """
    # Calculate speed ratio
    speed_ratio = target_bpm / original_bpm
    print(f"Speed ratio for conversion: {speed_ratio} (Original: {original_bpm} BPM → Target: {target_bpm} BPM)")
    
    # File paths for the two methods
    output_atempo = f"temp_atempo_{os.path.basename(input_file)}"
    output_asetrate = f"temp_asetrate_{os.path.basename(input_file)}"
    
    # Method 1: FFmpeg atempo
    atempo_chain = []
    remaining = speed_ratio
    
    # Handle speed ratio limits in FFmpeg
    if speed_ratio > 2.0:
        # Chain multiple atempo filters (each with value <= 2.0)
        while remaining > 1.0:
            factor = min(2.0, remaining)
            atempo_chain.append(f"atempo={factor}")
            remaining /= factor
    elif speed_ratio < 0.5:
        # Chain multiple atempo filters (each with value >= 0.5)
        while remaining < 1.0:
            factor = max(0.5, remaining)
            atempo_chain.append(f"atempo={factor}")
            remaining /= factor
    else:
        atempo_chain.append(f"atempo={speed_ratio}")
    
    # Run atempo method
    cmd_atempo = [
        "ffmpeg", "-y", "-i", input_file,
        "-filter:a", ",".join(atempo_chain),
        "-ar", str(target_sample_rate),
        output_atempo
    ]
    
    # print("Running FFmpeg atempo method...")
    success_atempo, _ = run_ffmpeg_command(cmd_atempo)
    
    # Method 2: FFmpeg asetrate
    # Calculate the asetrate value
    original_sr = torchaudio.info(input_file).sample_rate
    asetrate = int(original_sr * speed_ratio)
    
    # Run asetrate method
    cmd_asetrate = [
        "ffmpeg", "-y", "-i", input_file,
        "-filter:a", f"asetrate={asetrate},aresample={target_sample_rate}",
        output_asetrate
    ]
    
    # print("Running FFmpeg asetrate method...")
    success_asetrate, _ = run_ffmpeg_command(cmd_asetrate)
    
    # Determine which method produced the better result
    best_method = None
    best_file = None
    best_error = float('inf')
    
    if success_atempo and os.path.exists(output_atempo):
        bpm_atempo = detect_bpm(output_atempo)
        error_atempo = abs(bpm_atempo - target_bpm)
       #  print(f"atempo method result: {bpm_atempo} BPM (error: {error_atempo:.2f})")
        
        if error_atempo < best_error:
            best_method = "atempo"
            best_file = output_atempo
            best_error = error_atempo
    
    if success_asetrate and os.path.exists(output_asetrate):
        bpm_asetrate = detect_bpm(output_asetrate)
        error_asetrate = abs(bpm_asetrate - target_bpm)
        # print(f"asetrate method result: {bpm_asetrate} BPM (error: {error_asetrate:.2f})")
        
        if error_asetrate < best_error:
            best_method = "asetrate"
            best_file = output_asetrate
            best_error = error_asetrate
    
    if best_method:
        # print(f"Selected {best_method} method (closer to target BPM)")
        return best_file
    else:
        # print("Both tempo conversion methods failed!")
        return input_file  # Return original if both methods fail



def process_wav(file_path, processed_save_path, metadata=None, target_duration=TARGET_DURATION, 
                target_sample_rate=TARGET_SAMPLE_RATE, target_bpm=TARGET_BPM, preserve_bpm=False):
    """
    Process a WAV file to have consistent sample rate and duration.
    If preserve_bpm is False, will convert to target_bpm using the best FFmpeg method.
    """
    if not os.path.exists(file_path):
        print(f"Error: Input file {file_path} does not exist")
        return None, None
    
    # Step 1: Detect original BPM
    original_bpm = get_file_bpm(file_path, metadata)
    
    # print(f"Original file: {file_path}")
    # print(f"Original BPM: {original_bpm}")
    
    # Create a temp copy to work with
    temp_file = f"temp_{os.path.basename(file_path)}"
    
    # Step 2: Adjust tempo if needed
    if not preserve_bpm and abs(original_bpm - target_bpm) > 1.0:
        # print(f"Converting tempo from {original_bpm} to {target_bpm} BPM...")
        tempo_adjusted_file = tempo_convert_ffmpeg(
            file_path, target_bpm, original_bpm, target_sample_rate
        )
        working_file = tempo_adjusted_file
        final_bpm = target_bpm
    else:
        # If preserving BPM or BPMs are already close, just use original
        working_file = file_path
        final_bpm = original_bpm
        
        # Just convert sample rate if needed
        if torchaudio.info(file_path).sample_rate != target_sample_rate:
            wav, sr = torchaudio.load(file_path)
            wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)(wav)
            torchaudio.save(temp_file, wav, target_sample_rate)
            working_file = temp_file
    
    # Step 3: Adjust duration
    wav, sr = torchaudio.load(working_file)
    target_samples = target_duration * target_sample_rate
    
    # Handle mono/stereo
    if wav.shape[0] > 1:
        # Average stereo to mono
        wav = wav.mean(dim=0, keepdim=True)
    
    # Adjust length through padding or trimming
    if wav.shape[1] < target_samples:
        # Loop the audio if needed
        repeats_needed = int(np.ceil(target_samples / wav.shape[1]))
        repeated_audio = torch.tile(wav, (1, repeats_needed))
        wav = repeated_audio[:, :target_samples]
    else:
        # Trim to desired length
        wav = wav[:, :target_samples]
    
    # Step 4: Save the final processed file
    torchaudio.save(processed_save_path, wav, target_sample_rate)
    
    # Clean up temp files
    temp_files = [
        f"temp_atempo_{os.path.basename(file_path)}",
        f"temp_asetrate_{os.path.basename(file_path)}",
        temp_file
    ]
    for tf in temp_files:
        if os.path.exists(tf) and tf != processed_save_path:
            try:
                os.remove(tf)
            except:
                pass
    
    # print(f"Processed file saved to: {processed_save_path}")
    # print(f"Final BPM: {final_bpm}")
    
    return processed_save_path, final_bpm


def get_file_bpm(file_path, metadata=None):
    """
    Get the BPM for a file without processing it
    Returns:
        float: BPM value or None if couldn't be determined
    """
    # Try to get BPM from metadata first
    if metadata:
        bpm = get_bpm_from_metadata(file_path, metadata)
        if bpm is not None:
            return bpm
    
    # If not in metadata, try to detect it
    try:
        # wav, sr = torchaudio.load(file_path)
        # audio_np = wav[0].numpy() if wav.shape[0] > 1 else wav[0].numpy()
        tempo = detect_bpm(file_path)
        # tempo, _ = librosa.beat.beat_track(y=audio_np, sr=sr)
        
        if isinstance(tempo, np.ndarray) and tempo.size > 0:
            return float(tempo.item())
        else:
            return float(tempo)
    except:
        return None

def filter_files_by_bpm(input_dir, metadata=None, min_bpm=120, max_bpm=130, max_files=None):
    """
    Filter files by BPM range
    Args:
        input_dir: Directory containing WAV files
        metadata: Metadata dictionary
        min_bpm: Minimum BPM value to include
        max_bpm: Maximum BPM value to include
        max_files: Maximum number of files to return (None for all)
    Returns:
        List of filenames that fall within the BPM range
    """
    all_wav_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.wav', '.WAV'))]
    
    filtered_files = []
    print(f"Filtering files with BPM between {min_bpm} and {max_bpm}...")
    
    for wav_file in tqdm(all_wav_files, desc="Checking BPM"):
        file_path = os.path.join(input_dir, wav_file)
        bpm = get_file_bpm(file_path, metadata)
        
        if bpm is not None and min_bpm <= bpm <= max_bpm:
            filtered_files.append(wav_file)
            print(f"  Found file with BPM {bpm}: {wav_file}")
            
            if max_files is not None and len(filtered_files) >= max_files:
                break
    
    return filtered_files



if __name__ == "__main__":
    # At the beginning
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    parser = argparse.ArgumentParser(description="Process audio files with Mel spectrograms and one-hot encoded labels")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing WAV files")
    parser.add_argument("--output_dir", type=str, default="MEL_SPEC_output", help="Output directory for processed files")
    parser.add_argument("--metadata", type=str, required=True, help="Path to metadata.json file")
    parser.add_argument("--num_files", type=int, default=None, help="Number of files to process (None for all)")
    parser.add_argument("--preserve_bpm", action="store_true", help="Preserve original BPM instead of normalizing to reference BPM")
    parser.add_argument("--min_bpm", type=float, default=120, help="Minimum BPM value to include")
    parser.add_argument("--max_bpm", type=float, default=130, help="Maximum BPM value to include")
    parser.add_argument("--filter_by_bpm", action="store_true", help="Filter files by BPM range")
    parser.add_argument("--shuffle", action="store_true", help="Randomly shuffle files before processing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    parser.add_argument("--num_mixes", type=int, default=None, help="Number of mixed combinations to create (None for all)")
    parser.add_argument("--align_beats", action="store_true", help="Align audio to start with downbeats")
    parser.add_argument("--target_beats", type=int, default=32, help="Number of beats in output audio")
    parser.add_argument("--confidence_threshold", type=float, default=0.3, help="Minimum confidence for beat alignment")
    
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    if args.shuffle:
        random.seed(args.seed)
        print(f"Random seed set to {args.seed}")
    
    print(f"BPM preservation is {'enabled' if args.preserve_bpm else 'disabled'}")
    print(f"BPM filtering is {'enabled' if args.filter_by_bpm else 'disabled'}")
    print(f"Random shuffling is {'enabled' if args.shuffle else 'disabled'}")
    print(f"Creating {'all possible' if args.num_mixes is None else args.num_mixes} mixed combinations")
    
    # Load metadata
    print(f"Loading metadata from {args.metadata}...")
    metadata = load_metadata(args.metadata)
    print(f"Loaded metadata for {len(metadata)} files")
    
    # Create main output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Filter files by BPM range if requested
    files_to_process = None
    
    if args.filter_by_bpm:
        print(f"Filtering files by BPM range {args.min_bpm}-{args.max_bpm}...")
        files_to_process = filter_files_by_bpm(
            args.input_dir, 
            metadata=metadata, 
            min_bpm=args.min_bpm, 
            max_bpm=args.max_bpm,
            max_files=args.num_files
        )
        print(f"Found {len(files_to_process)} files in BPM range")
        
        # Save the filtered file list
        files_list_path = os.path.join(args.output_dir, f"files_bpm_{args.min_bpm}_{args.max_bpm}.json")
        with open(files_list_path, 'w') as f:
            json.dump(files_to_process, f, indent=2)
    
    
    # Process all files
    print("Processing files...")
    # In your main section, replace the process_dataset call with:
    original_results = process_dataset_with_beat_alignment(
        args.input_dir, 
        args.output_dir, 
        file_list=files_to_process,
        num_files=args.num_files,
        metadata=metadata,
        preserve_bpm=args.preserve_bpm,
        shuffle=args.shuffle,
        align_beats=True,  # Enable beat alignment
        target_beats=32,   # 32 beats = 8 measures of 4/4 time
        confidence_threshold=0.3  # Minimum confidence for alignment
    )
    print(f"Completed processing {len(original_results)} original files")
    
    print(f"All processing complete. Results saved to {args.output_dir}")
