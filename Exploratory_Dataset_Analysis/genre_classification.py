import torch
import json
import glob
import os
import numpy as np
import shutil
from tqdm import tqdm


def create_genre_folders_and_classify():
    # Load style names
    with open("../wav_files_5h/all_labels_master.json", "r") as f:
        style_names = json.load(f)  # should be a list of 400 names
    
    # Define parent genres we want to organize by
    # parent_genres = ["Electronic", "Rock"]
    parent_genres = ["Electronic", "Rock","Latin","Folk, World, & Country", "Hip Hop","Jazz", "Pop","Funk / Soul", "Classical","Non-Music","Blues","Reggae","Stage & Screen","Brass & Military","Children's Music"]

    # Create main classification directory
    classification_dir = "../wav_files_5h/classified_by_genre"
    os.makedirs(classification_dir, exist_ok=True)
    
    # Create subfolders for each parent genre
    for genre in parent_genres:
        genre_folder = os.path.join(classification_dir, genre.replace("/", "_"))
        os.makedirs(genre_folder, exist_ok=True)
        print(f"Created folder: {genre_folder}")
    
    # Also create an "Other" folder for files that don't fit the main categories
    other_folder = os.path.join(classification_dir, "Other")
    os.makedirs(other_folder, exist_ok=True)
    print(f"Created folder: {other_folder}")
    
    # Get all style probability files
    style_prob_files = glob.glob("../wav_files_5h/style_probs/*_style_probs.pt")
    print(f"\nFound {len(style_prob_files)} style probability files to classify")
    
    # Also get corresponding wav files (assuming they have matching names)
    wav_files = glob.glob("../wav_files_5h/*.wav")
    print(f"Found {len(wav_files)} wav files")
    
    classification_results = {genre: 0 for genre in parent_genres}
    classification_results["Other"] = 0
    
    # Process each style probability file
    for file_path in tqdm(style_prob_files, desc="Classifying files", unit="file"):
        # Load the style probabilities
        style_probs = torch.load(file_path).numpy()  # shape: [400]
        
        # Get the base filename (without path and extension)
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        base_filename = base_filename.replace("_style_probs", "")
        
        # Find the genre with highest cumulative probability
        genre_scores = {}
        
        for genre in parent_genres:
            # Find all style indices that belong to this parent genre
            genre_indices = [i for i, style in enumerate(style_names) 
                           if style.startswith(f"{genre}---")]
            
            if genre_indices:
                # mean probabilities for all styles in this genre
                genre_scores[genre] = np.mean(style_probs[genre_indices])
            else:
                genre_scores[genre] = 0.0
        
        # Find the genre with the highest score
        if max(genre_scores.values()) > 0:
            best_genre = max(genre_scores, key=genre_scores.get)
        else:
            best_genre = "Other"
        
        # Determine destination folder
        dest_genre_folder = best_genre.replace("/", "_")
        dest_folder = os.path.join(classification_dir, dest_genre_folder)
        
        # Copy the style probability file
        style_prob_dest = os.path.join(dest_folder, os.path.basename(file_path))
        shutil.copy2(file_path, style_prob_dest)
        
        # Try to find and copy the corresponding wav file
        possible_wav_names = [
            f"{base_filename}.wav",
            f"{base_filename}_processed.wav"
        ]
        
        wav_copied = False
        for wav_name in possible_wav_names:
            wav_path = os.path.join("../wav_files_5h", wav_name)
            if os.path.exists(wav_path):
                wav_dest = os.path.join(dest_folder, wav_name)
                shutil.copy2(wav_path, wav_dest)
                wav_copied = True
                break
        
        classification_results[best_genre] += 1
        
        # print(f"Classified {base_filename} -> {best_genre} "
        #       f"(score: {genre_scores[best_genre]:.3f}, wav: {'✓' if wav_copied else '✗'})")
    
    # Print summary
    print(f"\n{'='*50}")
    print("CLASSIFICATION SUMMARY")
    print(f"{'='*50}")
    
    for genre, count in classification_results.items():
        percentage = (count / len(style_prob_files)) * 100 if len(style_prob_files) > 0 else 0
        print(f"{genre:15}: {count:4d} files ({percentage:5.1f}%)")
    
    print(f"\nTotal files classified: {sum(classification_results.values())}")
    print(f"Files organized in: {classification_dir}")
    

    # Remove empty genre folders
    print(f"\n{'='*50}")
    print("CLEANING UP EMPTY FOLDERS")
    print(f"{'='*50}")

    removed_folders = 0
    for folder in os.listdir(classification_dir):
        folder_path = os.path.join(classification_dir, folder)
        if os.path.isdir(folder_path) and not os.listdir(folder_path):  # folder is empty
            os.rmdir(folder_path)
            print(f"Removed empty folder: {folder_path}")
            removed_folders += 1

    if removed_folders == 0:
        print("No empty folders found.")
    else:
        print(f"Removed {removed_folders} empty folders.")

    return classification_results, classification_dir

def analyze_genre_distribution():
    """Additional analysis of the genre distribution"""
    # Load style names
    with open("../wav_files_5h/all_labels_master.json", "r") as f:
        style_names = json.load(f)
    
    # Count styles per parent genre
    parent_genre_counts = {}
    for style in style_names:
        if "---" in style:
            parent = style.split("---")[0]
            parent_genre_counts[parent] = parent_genre_counts.get(parent, 0) + 1
    
    print(f"\n{'='*50}")
    print("AVAILABLE GENRES AND STYLE COUNTS")
    print(f"{'='*50}")
    
    for genre, count in sorted(parent_genre_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{genre:20}: {count:3d} sub-styles")
    
    return parent_genre_counts

def remove_pt_files(classification_dir):
    """Remove all .pt files from the classified folders, keeping only .wav files"""
    print(f"\n{'='*50}")
    print("REMOVING .PT FILES")
    print(f"{'='*50}")
    
    # Find all .pt files in the classification directory
    pt_files = glob.glob(os.path.join(classification_dir, "**", "*.pt"), recursive=True)
    
    if not pt_files:
        print("No .pt files found to remove.")
        return
    
    print(f"Found {len(pt_files)} .pt files to remove...")
    
    removed_count = 0
    for pt_file in pt_files:
        try:
            os.remove(pt_file)
            folder_name = os.path.basename(os.path.dirname(pt_file))
            file_name = os.path.basename(pt_file)
            print(f"Removed: {folder_name}/{file_name}")
            removed_count += 1
        except Exception as e:
            print(f"Error removing {pt_file}: {e}")
    
    print(f"\n Successfully removed {removed_count} .pt files")
    print("Only .wav files remain in the classified folders.")

if __name__ == "__main__":
    # First analyze what genres are available
    genre_counts = analyze_genre_distribution()
    
    # Then classify and organize files
    results, output_dir = create_genre_folders_and_classify()
    
    print(f"\n File organization complete!")
    print(f" Check your files in: {output_dir}")
    
    # Ask user if they want to remove .pt files
    response = input("\nDo you want to remove all .pt files from the classified folders? (y/n): ").lower().strip()
    if response in ['y', 'yes']:
        remove_pt_files(output_dir)
    else:
        print("Keeping .pt files. You can remove them later if needed.")