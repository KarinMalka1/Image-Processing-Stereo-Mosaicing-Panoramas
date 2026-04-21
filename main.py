import os
import time
import numpy as np
from PIL import Image

# Import from our local modules
from utils import load_frames_from_folder, extract_frames_from_video, save_panoramas_to_video
from motion import calculate_all_shifts
from panorama import build_all_panoramas, crop_for_alignment

def generate_panorama(input_frames_path, n_out_frames=10):
    """
    Main pipeline entry point: Loads frames, calculates motion shifts across the sequence,
    builds the panoramas, aligns them, and returns them as a list of PIL Images.
    """
    print(f"Loading frames from {input_frames_path}...")
    frames = load_frames_from_folder(input_frames_path)
    if not frames:
        print("Error: No images found.")
        return []

    print(f"Loaded {len(frames)} frames. Calculating shifts...")
    shifts = calculate_all_shifts(frames)

    print(f"Generating {n_out_frames} panoramas simultaneously...")
    raw_panoramas = build_all_panoramas(frames, shifts, n_out_frames)
    
    pil_results = []
    x_ratios = np.linspace(0.0, 1.0, n_out_frames)
    frame_width = frames[0].shape[1]

    print("Aligning and converting to PIL...")
    for i, pano_numpy in enumerate(raw_panoramas):
        ratio = x_ratios[i]
        pano_aligned = crop_for_alignment(pano_numpy, ratio, frame_width)
        pil_img = Image.fromarray(pano_aligned)
        pil_results.append(pil_img)
        
    return pil_results

def main():
    # Make sure to update the path to point to your actual video file before running
    input_video_path = r"C:\Users\ASUS\Documents\University\image_processing\project\viewpoint_input.mp4"      
    frames_dir = "frames"              
    output_video_name = "evening___.mp4"   
    
    if not os.path.exists(frames_dir) or len(os.listdir(frames_dir)) == 0:
        extract_frames_from_video(input_video_path, frames_dir, step=2) 
        
    t0 = time.perf_counter()
    panorama_images = generate_panorama(frames_dir, n_out_frames=10) 
    t1 = time.perf_counter()
    
    print(f"total runtime: {t1-t0:.3f} seconds")
    
    if panorama_images:
        save_panoramas_to_video(panorama_images, output_video_name, fps=4) 

if __name__ == "__main__":
    main()
