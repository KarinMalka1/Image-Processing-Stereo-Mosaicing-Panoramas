import numpy as np

def build_panorama_at_angle(frames, shifts, x_ratio):
    """
    Constructs a single panoramic image at a specific viewing angle by extracting 
    vertical strips from the frame sequence and blending them on a global canvas.
    """
    h, w, c = frames[0].shape
    
    y_positions = [0]
    cum_dy = 0
    for _, dy in shifts:
        cum_dy += dy
        y_positions.append(cum_dy)
    
    min_dy, max_dy = min(y_positions), max(y_positions)
    total_dx = sum([abs(s[0]) for s in shifts])
    
    canvas_w = int(total_dx) + w + 100
    canvas_h = int(h + max_dy - min_dy) + 100
    
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    weight_mask = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    
    current_x = 0.0
    global_y_offset = abs(min_dy) + 50
    margin = 5 

    for i, frame in enumerate(frames[:-1]):
        dx = abs(shifts[i][0])
        if abs(dx) < 0.1: 
            dx = 2.0 

        strip_width = int(dx + 2 * margin)
        safe_zone_start = strip_width // 2
        safe_zone_end = w - (strip_width // 2)

        cut_center = int(safe_zone_start + (safe_zone_end - safe_zone_start) * x_ratio)
        
        start_col = int(cut_center - strip_width // 2)
        s_start = max(0, start_col)
        s_end = min(w, start_col + strip_width)
        
        strip = frame[:, s_start:s_end].astype(np.float32)
        s_h, s_w, _ = strip.shape
        mask = np.ones((s_h, s_w, 3), dtype=np.float32)
        
        if s_w > 2 * margin:
            fade_curve = np.linspace(0, 1, margin)
            mask[:, :margin] = fade_curve[None, :, None]
            mask[:, -margin:] = fade_curve[::-1][None, :, None]
        
        target_x = int(current_x)
        paste_y = int(global_y_offset - y_positions[i])
        
        y_start, y_end = max(0, paste_y), min(paste_y + s_h, canvas_h)
        x_start, x_end = max(0, target_x), min(target_x + s_w, canvas_w)
        
        act_h, act_w = y_end - y_start, x_end - x_start
        if act_h > 0 and act_w > 0:
            s_y, s_x = y_start - paste_y, x_start - target_x
            canvas[y_start:y_end, x_start:x_end] += strip[s_y:s_y+act_h, s_x:s_x+act_w] * mask[s_y:s_y+act_h, s_x:s_x+act_w]
            weight_mask[y_start:y_end, x_start:x_end] += mask[s_y:s_y+act_h, s_x:s_x+act_w]
            
        current_x += dx
        
    weight_mask[weight_mask == 0] = 1.0
    final_pano = np.clip(canvas / weight_mask, 0, 255).astype(np.uint8)
    final_pano = final_pano[:, :int(current_x)]
    return final_pano

def build_all_panoramas(frames, shifts, n_out_frames):
    """
    Optimized function to simultaneously build multiple panoramas representing 
    a sweeping camera movement (changing viewpoints).
    """
    h, w, c = frames[0].shape
    
    y_positions = [0]
    cum_dy = 0
    for _, dy in shifts:
        cum_dy += dy
        y_positions.append(cum_dy)
    
    min_dy, max_dy = min(y_positions), max(y_positions)
    total_dx = sum([abs(s[0]) for s in shifts])
    
    canvas_w = int(total_dx) + w + 100
    canvas_h = int(h + max_dy - min_dy) + 100
    
    canvases = [np.zeros((canvas_h, canvas_w, 3), dtype=np.float32) for _ in range(n_out_frames)]
    weight_masks = [np.zeros((canvas_h, canvas_w, 1), dtype=np.float32) for _ in range(n_out_frames)]
    
    x_ratios = np.linspace(0.0, 1.0, n_out_frames)
    
    current_x = 0.0
    global_y_offset = abs(min_dy) + 50
    margin = 5 

    for i, frame in enumerate(frames[:-1]):
        dx = abs(shifts[i][0])
        if abs(dx) < 0.1: dx = 2.0 

        strip_width = int(dx + 2 * margin)
        safe_zone_start = strip_width // 2
        safe_zone_end = w - (strip_width // 2)
        
        start_col_base = safe_zone_start 
        span = safe_zone_end - safe_zone_start
        
        target_x = int(current_x)
        paste_y = int(global_y_offset - y_positions[i])
        
        for j, ratio in enumerate(x_ratios):
            cut_center = int(start_col_base + span * ratio)
            
            s_start_col = int(cut_center - strip_width // 2)
            s_start = max(0, s_start_col)
            s_end = min(w, s_start_col + strip_width)
            
            strip = frame[:, s_start:s_end].astype(np.float32)
            s_h, s_w, _ = strip.shape
            
            if s_w <= 0: continue

            mask = np.ones((s_h, s_w, 1), dtype=np.float32)
            if s_w > 2 * margin:
                fade = np.linspace(0, 1, margin).astype(np.float32)
                mask[:, :margin, 0] = fade[None, :]
                mask[:, -margin:, 0] = fade[::-1][None, :]
            
            y_start, y_end = max(0, paste_y), min(paste_y + s_h, canvas_h)
            x_start, x_end = max(0, target_x), min(target_x + s_w, canvas_w)
            
            act_h, act_w = y_end - y_start, x_end - x_start
            
            if act_h > 0 and act_w > 0:
                s_y, s_x = y_start - paste_y, x_start - target_x
                
                canvases[j][y_start:y_end, x_start:x_end] += strip[s_y:s_y+act_h, s_x:s_x+act_w] * mask[s_y:s_y+act_h, s_x:s_x+act_w]
                weight_masks[j][y_start:y_end, x_start:x_end] += mask[s_y:s_y+act_h, s_x:s_x+act_w]

        current_x += dx

    final_panos = []
    final_width = int(current_x)
    
    for k in range(n_out_frames):
        w_mask = weight_masks[k]
        w_mask[w_mask == 0] = 1.0
        pano = np.clip(canvases[k] / w_mask, 0, 255).astype(np.uint8)
        pano = pano[:, :final_width]
        final_panos.append(pano)
        
    return final_panos

def crop_for_alignment(pano, ratio, frame_width):
    """
    Crops the final panorama to ensure consistent alignment and framing across the output sequence.
    """
    h, w, c = pano.shape

    if w < 2 * frame_width:
        return pano
        
    max_crop = int(frame_width * 0.9)
    crop_left = int(max_crop * (1.0 - ratio))
    crop_right = int(max_crop * ratio)
   
    new_start = crop_left
    new_end = w - crop_right
    
    if new_end > new_start:
        return pano[:, new_start:new_end]
    else:
        return pano
