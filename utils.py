import os
import cv2
import numpy as np

def load_frames_from_folder(input_dir_path):
    """
    Loads all image frames from a specified directory and converts them from BGR to RGB.
    """
    frames = []
    if not os.path.exists(input_dir_path): return []
    files = sorted([f for f in os.listdir(input_dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    for filename in files:
        path = os.path.join(input_dir_path, filename)
        img = cv2.imread(path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
    return frames

def extract_frames_from_video(video_path, output_folder, step=1):
    """
    Extracts frames from an input video file and saves them to a designated directory.
    Supports skipping frames (step > 1) to improve processing performance.
    """
    if not os.path.exists(output_folder): os.makedirs(output_folder)
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    saved_count = 0
    print(f"Extracting frames from {os.path.basename(video_path)}...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if frame_idx % step == 0:
            filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1
        frame_idx += 1
    cap.release()
    print(f"Finished. Saved {saved_count} frames.")

def save_panoramas_to_video(pil_images, output_path, fps=5):
    """
    Compiles a sequence of PIL images into an MP4 video file.
    """
    if not pil_images:
        print("No images to save to video.")
        return

    width, height = pil_images[0].size
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Saving video to {output_path}...")
    
    for img in pil_images:
        open_cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        if (open_cv_image.shape[1], open_cv_image.shape[0]) != (width, height):
            open_cv_image = cv2.resize(open_cv_image, (width, height))
            
        video_writer.write(open_cv_image)

    video_writer.release()
    print(f"Video saved successfully: {output_path}")
