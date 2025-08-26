import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
import numpy as np
from maest import get_maest
import re
from tqdm import tqdm
import json
import os

TARGET_SAMPLE_RATE = 44100


def clean_filename(filename):
    base_name = re.sub(r'(\.\w+)+$', '', filename)
    base_name = base_name.replace("_processed", "")
    base_name = base_name.replace("_", "")
    return base_name

def predict_labels_MAEST(file_path, top_n=5, return_full_probs=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #NOTE: arch="discogs-maest-5s-pw-129e"  USED: arch="discogs-maest-30s-pw-129e-519l"
    model = get_maest(arch="discogs-maest-5s-pw-129e").to(device)
    model.eval()

    # Load and preprocess audio
    wav, sr = torchaudio.load(file_path)
    if sr != TARGET_SAMPLE_RATE:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SAMPLE_RATE)(wav)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    wav = wav.to(device)  # Move to GPU
    # Log-mel spectrogram
    mel_transform = T.MelSpectrogram(
        sample_rate=TARGET_SAMPLE_RATE,
        n_fft=1024,
        hop_length=320,
        n_mels=80
    ).to(device)
    logmel = mel_transform(wav)
    logmel = torch.log1p(logmel)  # shape: [1, 128, time]

    # Add batch and channel dims: [1, 1, 128, T]
    logmel = logmel.unsqueeze(0)

    # Resize to model input shape: [1, 1, 96, 1875]
    # size=(96, 1875) FOR arch="discogs-maest-30s-pw-129e-519l" (96*312) FOR arch="discogs-maest-5s-pw-129e"
    logmel_resized = F.interpolate(logmel, size=(96, 312), mode='bilinear', align_corners=False).to(device)

    # Predict
    with torch.no_grad():
        activations, labels = model.predict_labels(logmel_resized)

    # Convert activations to a PyTorch tensor if they're in numpy format
    if isinstance(activations, np.ndarray):
        activations = torch.tensor(activations).to(device)

    # Uses a sigmoid inside predict_labels and mean each label between [0,1]
    probabilities = activations.squeeze(0).cpu()  # No need for additional activation function
    
    # Get the top N labels based on highest probability for backward compatibility
    top_n_indices = torch.topk(probabilities, top_n).indices
    top_n_probabilities = probabilities[top_n_indices].tolist()
    top_n_labels = [labels[i] for i in top_n_indices]
    
    if return_full_probs:
        # Return both the top N results and the full probability distribution
        return top_n_labels, top_n_probabilities, labels, probabilities
    else:
        # Return just the top N results for backward compatibility
        return top_n_labels, top_n_probabilities, labels
    
def collect_all_labels(input_path, num_files=1):
    """
    Collects all unique genre labels from a sample of files
    to ensure consistent one-hot encoding dimensions.
    Can accept either a directory path or a single file path.
    """
    all_labels_set = set()
    
    # Check if the input_path is a file or directory
    if os.path.isfile(input_path):
        # It's a single file, just use that
        _, _, all_labels = predict_labels_MAEST(input_path)
        return all_labels
    elif os.path.isdir(input_path):
        # It's a directory, process files inside it
        wav_files = [f for f in os.listdir(input_path) if f.lower().endswith(('.wav', '.WAV'))][:num_files]
        
        # Process one file to get the complete label set
        if wav_files:
            _, _, all_labels = predict_labels_MAEST(os.path.join(input_path, wav_files[0]))
            return all_labels
        else:
            raise ValueError("No WAV files found in the specified directory")
    else:
        raise ValueError(f"Invalid path: {input_path} is neither a file nor a directory")
    


processed_dir = "wav_files_5h"
output_dir = "wav_files_5h"
probs_dir = os.path.join(output_dir, "style_probs")
labels_dir = os.path.join(output_dir, "labels")
os.makedirs(probs_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)


# First, collect all possible labels for consistent one-hot encoding
print("Collecting all possible genre labels...")
all_labels = collect_all_labels(output_dir, 1)
print(f"Found {len(all_labels)} unique genre labels for one-hot encoding")

# Save the complete list of labels
labels_master_path = os.path.join(output_dir, "all_labels_master.json")
with open(labels_master_path, 'w') as f:
    json.dump(all_labels, f, indent=2)

wav_files = [f for f in os.listdir(processed_dir) if f.lower().endswith(('.wav', '.WAV'))]

for wav_file in tqdm(wav_files, desc="Processing original files"):
        file_path = os.path.join(processed_dir, wav_file)
        base_name = clean_filename(os.path.basename(wav_file))
        processed_path = os.path.join(processed_dir, f"{base_name}_processed.wav")

        top_n = 5
        labels, probabilities, all_maest_labels, full_probs = predict_labels_MAEST(
            processed_path, top_n=top_n, return_full_probs=True
        )

        probs_path = os.path.join(probs_dir, f"{base_name}_style_probs.pt")
        torch.save(full_probs.cpu(), probs_path)

        # For human readability, also save top probabilities and their corresponding labels
        top_probs_dict = {label: float(prob) for label, prob in zip(labels, probabilities)}

        # Save labels and probabilities as JSON
        label_info = {
            "file": wav_file,
            "labels": labels,
            "top_probabilities": probabilities,
            "top_probs_dict": top_probs_dict
        }
        
        label_path = os.path.join(labels_dir, f"{base_name}_labels.json")
        with open(label_path, 'w') as f:
            json.dump(label_info, f, indent=2)