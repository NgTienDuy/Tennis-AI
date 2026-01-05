import os
import json
import matplotlib.pyplot as plt
import numpy as np
import cv2
from visualization.drawer import draw_court_lines

def save_text_reports(stats_dict, output_folder):
    json_path = os.path.join(output_folder, 'match_report.json')
    def convert(o):
        if isinstance(o, np.generic): return o.item()
        raise TypeError
    
    with open(json_path, 'w') as f:
        json.dump(stats_dict, f, indent=4, default=convert)
    
    txt_path = os.path.join(output_folder, 'match_report.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=== TENNIS MATCH ANALYSIS REPORT ===\n")
        for k, v in stats_dict.items():
            if isinstance(v, dict):
                f.write(f"\n[{k}]\n")
                for sub_k, sub_v in v.items():
                    f.write(f"  {sub_k}: {sub_v}\n")
            else:
                f.write(f"{k}: {v}\n")
    print(f"Report files saved to: {output_folder}")

def save_dashboard_image(heatmap, stats_dict, output_path):
    fig = plt.figure(figsize=(14, 8))
    
    # 1. Heatmap
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.axis('off')
    ax1.set_title("Player Movement Heatmap", fontsize=14, pad=20)
    im = ax1.imshow(heatmap, cmap='hot', interpolation='bilinear')
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    # 2. Stats
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axis('off')
    ax2.set_title("Match Statistics", fontsize=16, fontweight='bold', pad=20)
    
    text_content = []
    text_content.append(f"Duration: {stats_dict.get('Total Time (s)', 0):.1f} s")
    text_content.append(f"Total Frames: {stats_dict.get('Total Frames', 0)}")
    text_content.append("")
    
    text_content.append("PLAYER PERFORMANCE")
    text_content.append("-" * 30)
    text_content.append(f"Top Player Dist:    {stats_dict.get('Top Player Dist (m)', 0):.2f} m")
    text_content.append(f"Bottom Player Dist: {stats_dict.get('Bottom Player Dist (m)', 0):.2f} m")
    
    if 'Velocity' in stats_dict:
        text_content.append(f"Max Speed P1:       {stats_dict['Velocity'].get('Avg P1', 0):.1f} km/h")
        text_content.append(f"Max Speed P2:       {stats_dict['Velocity'].get('Avg P2', 0):.1f} km/h")
    text_content.append("")

    text_content.append("SYSTEM METRICS")
    text_content.append("-" * 30)
    text_content.append(f"Court Confidence:   {stats_dict.get('Court Score', 0):.1f}")
    text_content.append(f"Detection Quality:  {stats_dict.get('Court Accuracy (%)', 0):.1f}%")

    y_pos = 0.95
    for line in text_content:
        font_weight = 'bold' if line.isupper() else 'normal'
        fontsize = 12 if not line.isupper() else 13
        ax2.text(0.05, y_pos, line, transform=ax2.transAxes, 
                 fontsize=fontsize, fontweight=font_weight, fontfamily='monospace')
        y_pos -= 0.05

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def save_ball_trajectory(ball_positions, output_path, court_matrices):
    h, w = 2000, 1000 
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas, _, _, _ = draw_court_lines(canvas, thickness=5)
    
    if not court_matrices: return

    valid_points = []
    inv_matrix = court_matrices[0] if len(court_matrices) > 0 else None

    for i, pos in enumerate(ball_positions):
        if pos[0] is None: continue
        current_inv = court_matrices[i] if i < len(court_matrices) else inv_matrix
        
        if current_inv is not None:
            pt_np = np.array([[[pos[0], pos[1]]]], dtype=np.float32)
            try:
                pt_out = cv2.perspectiveTransform(pt_np, current_inv)[0][0]
                
                sx = w / 1117.0 
                sy = h / 2408.0 
                scale = min(sx, sy) * 0.9 
                
                dx = (w - 1117*scale) / 2
                dy = (h - 2408*scale) / 2
                
                x_2d = int(pt_out[0] * scale + dx)
                y_2d = int(pt_out[1] * scale + dy)
                
                # [FIX CRASH] Kiểm tra toạ độ hợp lệ trong canvas
                if 0 <= x_2d < w and 0 <= y_2d < h:
                    valid_points.append((x_2d, y_2d))
            except:
                continue

    if len(valid_points) > 1:
        for i in range(len(valid_points) - 1):
            pt1 = valid_points[i]
            pt2 = valid_points[i+1]
            dist = np.linalg.norm(np.array(pt1)-np.array(pt2))
            
            # [FIX CRASH] Đảm bảo kiểu int và không quá xa
            if dist < 200: 
                try:
                    cv2.line(canvas, pt1, pt2, (0, 255, 255), 2, cv2.LINE_AA)
                except:
                    pass
                
    for pt in valid_points:
        cv2.circle(canvas, pt, 3, (0, 165, 255), -1)

    cv2.imwrite(output_path, canvas)