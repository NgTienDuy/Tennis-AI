# visualization/drawer.py
import cv2
import numpy as np

def draw_velocity(frame, speed_p1, speed_p2):
    h, w = frame.shape[:2]
    panel_w = 250
    panel_h = 100
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - panel_w, h - panel_h - 50), (w, h - 50), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
    
    cv2.putText(frame, "VELOCITY (km/h)", (w - panel_w + 10, h - panel_h - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.putText(frame, f"P1: {speed_p1:.1f}", (w - panel_w + 10, h - panel_h + 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2) 
    cv2.putText(frame, f"P2: {speed_p2:.1f}", (w - panel_w + 10, h - panel_h + 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2) 
    return frame

def draw_court_lines(img, color=(255, 255, 255), thickness=2):
    """
    Vẽ sân tennis chuẩn. 
    Lưu ý: Toạ độ dưới đây dựa trên CourtReference gốc (khoảng 1117x2408).
    Để vẽ đẹp trên canvas tuỳ ý, ta cần scale và offset.
    """
    # Toạ độ gốc từ court_reference
    base_lines = [
        ((286, 561), (1379, 561)),   # Top Baseline
        ((286, 2935), (1379, 2935)), # Bottom Baseline
        ((286, 1748), (1379, 1748)), # Net
        ((286, 561), (286, 2935)),   # Left Side
        ((1379, 561), (1379, 2935)), # Right Side
        ((423, 561), (423, 2935)),   # Left Singles
        ((1242, 561), (1242, 2935)), # Right Singles
        ((832, 1110), (832, 2386)),  # Center Serve
        ((423, 1110), (1242, 1110)), # Top Service Line
        ((423, 2386), (1242, 2386))  # Bottom Service Line
    ]

    h_img, w_img = img.shape[:2]
    
    # Tính toán scale để fit vào canvas hiện tại
    # Kích thước bao quanh sân gốc: Rộng ~1100, Cao ~2400
    # Ta thêm padding để không bị cắt
    target_h = h_img * 0.9
    scale = target_h / 3000 # 3000 là chiều cao ước lượng bao gồm lề của hệ toạ độ gốc
    
    offset_x = (w_img - 1665 * scale) / 2 # Canh giữa theo chiều ngang (1665 là chiều rộng gốc)
    offset_y = (h_img - 3500 * scale) / 2 # Canh giữa theo chiều dọc

    # Hàm transform cục bộ
    def tr(pt):
        return (int(pt[0] * scale + offset_x), int(pt[1] * scale + offset_y))

    for line in base_lines:
        p1 = tr(line[0])
        p2 = tr(line[1])
        cv2.line(img, p1, p2, color, thickness)
        
    return img, scale, offset_x, offset_y

def draw_minimap(frame, p1_pos, p2_pos, ball_pos=None, 
                 p1_trail=[], p2_trail=[], ball_trail=[], 
                 ref_dims=(3600, 1800)): # Tăng kích thước canvas mặc định
    
    # 1. Tạo Canvas (Cao x Rộng)
    minimap_canvas = np.zeros((ref_dims[0], ref_dims[1], 3), dtype=np.uint8)
    
    # 2. Vẽ sân và lấy thông số scale/offset để map các đối tượng khác
    minimap_canvas, scale_draw, off_x, off_y = draw_court_lines(minimap_canvas, thickness=15)

    # 3. Resize để ghép vào video
    frame_h = frame.shape[0]
    final_scale = frame_h / ref_dims[0]
    minimap_w = int(ref_dims[1] * final_scale)
    minimap_resized = cv2.resize(minimap_canvas, (minimap_w, frame_h))

    # Helper map toạ độ: Từ Hệ quy chiếu gốc -> Canvas vẽ -> Resize hiển thị
    def map_coords(pos):
        if pos is None or np.isnan(pos).any(): return None
        # Bước 1: Map vào canvas vẽ (dùng scale/offset của hàm draw_court_lines)
        x_canvas = pos[0] * scale_draw + off_x
        y_canvas = pos[1] * scale_draw + off_y
        
        # Bước 2: Map vào ảnh hiển thị cuối cùng
        x_final = int(x_canvas * final_scale)
        y_final = int(y_canvas * final_scale)
        return (x_final, y_final)

    # 4. Vẽ Trails
    for i, pos in enumerate(p1_trail):
        pt = map_coords(pos)
        if pt: cv2.circle(minimap_resized, pt, 2, (0, 0, 255), -1)
            
    for i, pos in enumerate(p2_trail):
        pt = map_coords(pos)
        if pt: cv2.circle(minimap_resized, pt, 2, (255, 0, 255), -1)

    for i, pos in enumerate(ball_trail):
        pt = map_coords(pos)
        if pt: cv2.circle(minimap_resized, pt, 2, (0, 255, 255), -1)

    # 5. Vẽ Vị trí hiện tại
    mp1 = map_coords(p1_pos)
    mp2 = map_coords(p2_pos)
    mb = map_coords(ball_pos)
    
    if mp1: cv2.circle(minimap_resized, mp1, 8, (0, 0, 255), -1)
    if mp2: cv2.circle(minimap_resized, mp2, 8, (255, 0, 255), -1)
    if mb: cv2.circle(minimap_resized, mb, 5, (0, 255, 255), -1)

    return minimap_resized