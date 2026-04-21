import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

def create_dog_space(img, num_octaves=3, scales_per_octave=2, sigma=1.6):
    """
    Constructs a Difference of Gaussians (DoG) pyramid for scale-invariant feature detection.
    """
    dog_pyramid = []
    k = 2 ** (1 / scales_per_octave)
    current_img = img.astype(np.float32)
    
    for o in range(num_octaves):
        octave_gaussians = []
        s_sigma = sigma
        for s in range(scales_per_octave + 3):
            octave_gaussians.append(gaussian_filter(current_img, s_sigma))
            s_sigma *= k
        dog_octave = [octave_gaussians[i+1] - octave_gaussians[i] 
                      for i in range(len(octave_gaussians)-1)]
        dog_pyramid.append(dog_octave)
        current_img = octave_gaussians[-3][::2, ::2]
        
    return dog_pyramid

def find_keypoints(dog_pyramid, threshold=0.04):
    """
    Identifies local maxima in the DoG scale space to find stable keypoints,
    filtering out low-contrast points.
    """
    keypoints = [] 
    for oct_idx, dog_octave in enumerate(dog_pyramid):
        for i in range(1, len(dog_octave) - 1):
            layer = dog_octave[i]
            prev = dog_octave[i-1]
            nxt = dog_octave[i+1]
            
            is_max = np.ones(layer.shape, dtype=bool)
            # 26 neighbors check
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1), (0,0)]:
                if dx == 0 and dy == 0:
                    is_max &= (layer >= prev) & (layer >= nxt)
                    continue
                shifted = np.roll(np.roll(layer, dx, axis=1), dy, axis=0)
                is_max &= (layer >= shifted)
                is_max &= (layer >= np.roll(np.roll(prev, dx, axis=1), dy, axis=0))
                is_max &= (layer >= np.roll(np.roll(nxt, dx, axis=1), dy, axis=0))

            is_max &= np.abs(layer) > threshold
            coords = np.argwhere(is_max)
            for y, x in coords:
                keypoints.append((y, x, oct_idx, i))
    return keypoints

def extract_sift_descriptors(dog_pyramid, keypoints, window_size=16):
    """
    Extracts local gradient histograms around each keypoint to create robust SIFT-like descriptors.
    Normalizes the vectors to improve robustness to illumination changes.
    """
    descriptors = []
    valid_keypoints = []

    for y, x, oct_idx, s_idx in keypoints:
        img = dog_pyramid[oct_idx][s_idx]
        h, w = img.shape

        # Filter points too close to edge
        y_start, y_end = y - window_size//2, y + window_size//2
        x_start, x_end = x - window_size//2, x + window_size//2
        if y_start < 0 or y_end >= h or x_start < 0 or x_end >= w:
            continue
            
        sub_img = img[y_start:y_end, x_start:x_end]
        dy = np.gradient(sub_img, axis=0)
        dx = np.gradient(sub_img, axis=1)
        magnitude = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx) * 180 / np.pi % 360
        
        desc = []
        for i in range(0, window_size, 4):
            for j in range(0, window_size, 4):
                sub_mag = magnitude[i:i+4, j:j+4]
                sub_ang = angle[i:i+4, j:j+4]
                hist, _ = np.histogram(sub_ang, bins=8, range=(0, 360), weights=sub_mag)
                desc.extend(hist)
        
        desc = np.array(desc)
        norm = np.linalg.norm(desc)
        if norm > 0: desc /= norm
        desc[desc > 0.2] = 0.2
        norm = np.linalg.norm(desc)
        if norm > 0: desc /= norm
            
        descriptors.append(desc)
        valid_keypoints.append([x * (2**oct_idx), y * (2**oct_idx)])
        
    return np.array(valid_keypoints), np.array(descriptors)

def my_sift_detect_and_compute(frame):
    """
    Wrapper function to process a single frame: converts to grayscale, builds DoG space,
    finds keypoints, and extracts descriptors. Downscales the image for performance optimization.
    """
    if len(frame.shape) == 3:
        gray = 0.299 * frame[:,:,0] + 0.587 * frame[:,:,1] + 0.114 * frame[:,:,2]
    else:
        gray = frame
    
    h, w = gray.shape
    small_gray = cv2.resize(gray.astype(np.float32), (w // 2, h // 2))
    
    dog = create_dog_space(small_gray)
    kps_raw = find_keypoints(dog)
    pts, descriptors = extract_sift_descriptors(dog, kps_raw)
    
    pts = pts * 2
    
    return pts, descriptors
