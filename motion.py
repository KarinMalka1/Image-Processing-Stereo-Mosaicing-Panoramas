import numpy as np
from features import my_sift_detect_and_compute

def smart_descriptor_matcher(des1, kps1, des2, kps2, img_w, img_h, threshold=0.75):
    """
    Matches descriptors using Euclidean distance and Lowe's ratio test. 
    Matches are constrained by spatial distance (less than 10% of image size) 
    based on the assumption of horizontal camera motion.
    """
    matches = []
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return matches

    max_dx = img_w * 0.1
    max_dy = img_h * 0.1

    for i, (d1, pt1) in enumerate(zip(des1, kps1)):
        x1, y1 = pt1
      
        dx_arr = np.abs(kps2[:, 0] - x1)
        dy_arr = np.abs(kps2[:, 1] - y1)
        
        spatial_mask = (dx_arr < max_dx) & (dy_arr < max_dy)
        
        if not np.any(spatial_mask):
            continue
            
        subset_des2 = des2[spatial_mask]
        subset_indices = np.where(spatial_mask)[0]
        
        distances = np.linalg.norm(subset_des2 - d1, axis=1)
        
        if len(distances) < 2:
            continue
            
        sorted_local_indices = np.argsort(distances)
        best_local_idx = sorted_local_indices[0]
        second_local_idx = sorted_local_indices[1]
        
        best_dist = distances[best_local_idx]
        second_dist = distances[second_local_idx]
        
        # Lowe's Ratio Test
        if best_dist < threshold * second_dist:
            original_idx = subset_indices[best_local_idx]
            matches.append((i, original_idx))
            
    return matches

def apply_ransac(pts_prev, pts_curr, iterations=100, threshold=5.0): 
    """
    Applies the RANSAC algorithm to filter out outliers and estimate a pure 2D 
    translation (dx, dy) vector between two consecutive frames.
    """
    max_inliers = -1
    best_shift = (0, 0)
    num_matches = len(pts_prev)
    if num_matches < 1: return (0, 0)

    for _ in range(iterations):
        idx = np.random.randint(0, num_matches)
        sample_dx = pts_curr[idx, 0] - pts_prev[idx, 0]
        sample_dy = pts_curr[idx, 1] - pts_prev[idx, 1]
 
        expected_curr = pts_prev + np.array([sample_dx, sample_dy])
        distances = np.linalg.norm(pts_curr - expected_curr, axis=1)
        
        inliers_mask = distances < threshold
        num_inliers = np.sum(inliers_mask)
        
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_shift = (np.mean(pts_curr[inliers_mask, 0] - pts_prev[inliers_mask, 0]),
                          np.mean(pts_curr[inliers_mask, 1] - pts_prev[inliers_mask, 1]))
    return best_shift

def calculate_all_shifts(frames):
    """
    Iterates through the video frames to calculate the translation vector between 
    each consecutive pair using SIFT feature matching and RANSAC.
    """
    shifts = [] 
    h, w, c = frames[0].shape
    
    print("   Computing SIFT for first frame...")
    pts_prev, des_prev = my_sift_detect_and_compute(frames[0])
    
    for i in range(1, len(frames)):
        print(f"   Processing match {i}/{len(frames)-1}...")
        pts_curr, des_curr = my_sift_detect_and_compute(frames[i])
        
        dx, dy = 0, 0 
        matches = smart_descriptor_matcher(des_prev, pts_prev, des_curr, pts_curr, w, h)
        
        if len(matches) > 4:
            matched_pts_prev = np.array([pts_prev[m[0]] for m in matches])
            matched_pts_curr = np.array([pts_curr[m[1]] for m in matches])
            dx, dy = apply_ransac(matched_pts_prev, matched_pts_curr)
        
        shifts.append((dx, dy))
        pts_prev, des_prev = pts_curr, des_curr
        
    return shifts
