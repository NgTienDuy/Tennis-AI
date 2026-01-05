import os
import cv2
import pandas as pd
import numpy as np
import time
import pickle
from utils.general import load_config, check_dir, get_device
from utils.video_utils import get_video_properties

# Import Modules
from modules.court import CourtDetector
from modules.player import PlayerDetector
from modules.ball import BallDetector
from modules.pose import PoseExtractor

# Import Analysis & Viz
from analysis.smoothing import DataSmoother
from analysis.velocity import VelocityCalculator
from analysis.stats import MatchStatistics
from visualization.drawer import draw_velocity, draw_minimap
from visualization.stickman import StickmanAnimator
from visualization.dashboard import save_dashboard_image, save_text_reports, save_ball_trajectory

def main():
    print(">>> [INIT] Loading Configuration...")
    config = load_config("configs/config.yaml")
    
    real_device = get_device(config['system']['device'])
    config['system']['device'] = real_device.type
    
    base_output = config['paths']['output_folder']
    dir_videos = os.path.join(base_output, 'videos')
    dir_reports = os.path.join(base_output, 'reports')
    dir_packages = os.path.join(base_output, 'packages')
    
    check_dir(dir_videos)
    check_dir(dir_reports)
    check_dir(dir_packages)
    
    checkpoint_path = os.path.join(dir_packages, "detection_checkpoint6.pkl")
    input_path = config['paths']['input_video']
    
    cap = cv2.VideoCapture(input_path)
    fps_orig, length_orig, width, height = get_video_properties(cap)
    cap.release()
    
    TARGET_FPS = 24.0 # Resample xuống 24fps
    
    print(f"    - Video: {input_path}")
    print(f"    - Original: {width}x{height} | {fps_orig} FPS | {length_orig} Frames")

    # =========================================================================
    # PASS 1: DETECTION (LOGIC MỚI)
    # =========================================================================
    data_store = {}

    if os.path.exists(checkpoint_path):
        print(f"\n>>> [PASS 1] Found Checkpoint. Loading data...")
        with open(checkpoint_path, 'rb') as f:
            data_store = pickle.load(f)
        
        court_detector = CourtDetector(config)
        court_detector.court_warp_matrix = data_store.get("court_warp_matrix", [])
        court_detector.game_warp_matrix = data_store.get("court_matrices", [])
        
    else:
        print(f"\n>>> [PASS 1] Running Detection (Smart Logic)...")
        
        court_detector = CourtDetector(config)
        player_detector = PlayerDetector(config)
        ball_detector = BallDetector(config)
        pose_extractor_p1 = PoseExtractor(config)
        pose_extractor_p2 = PoseExtractor(config)
        
        cap = cv2.VideoCapture(input_path)
        start_time = time.time()
        
        current_frame_idx = 0
        target_frame_accum = 0.0
        ratio = fps_orig / TARGET_FPS
        processed_count = 0
        processed_indices = []
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # Logic Resampling 24 FPS
            if current_frame_idx >= int(target_frame_accum):
                
                # 1. Court
                if len(court_detector.court_warp_matrix) == 0:
                    court_detector.detect(frame)
                else:
                    court_detector.track_court(frame)
                    
                # 2. Player (GỌI HÀM MỚI)
                player_detector.detect_all(frame, court_detector)
                
                # 3. Ball
                ball_detector.detect_ball(frame)
                
                # 4. Pose
                p1_box = player_detector.player_1_boxes[-1] if player_detector.player_1_boxes else None
                pose_extractor_p1.extract_pose(frame, p1_box)
                
                p2_box = player_detector.player_2_boxes[-1] if player_detector.player_2_boxes else None
                pose_extractor_p2.extract_pose(frame, p2_box)
                
                processed_indices.append(current_frame_idx)
                processed_count += 1
                target_frame_accum += ratio
                
                if processed_count % 20 == 0:
                    print(f"    Processed {processed_count} frames...", end='\r')

            current_frame_idx += 1

        cap.release()
        total_time = time.time() - start_time
        print(f"\n    Done in {total_time:.2f}s")
        
        data_store = {
            "p1_boxes": player_detector.player_1_boxes,
            "p2_boxes": player_detector.player_2_boxes,
            "ball_positions": ball_detector.xy_coordinates,
            "court_matrices": court_detector.game_warp_matrix,
            "court_warp_matrix": court_detector.court_warp_matrix,
            "pose_data_p1": pose_extractor_p1.data,
            "pose_data_p2": pose_extractor_p2.data,
            "processed_frames_indices": processed_indices,
            "total_time": total_time
        }
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(data_store, f)

    # =========================================================================
    # PROCESSING & ANALYSIS
    # =========================================================================
    print("\n>>> [ANALYSIS] Generating Reports & Stickman Video...")
    
    stickman_vid_path = os.path.join(dir_videos, 'stickman_2players6.avi')
    
    if True: 
        if 'pose_extractor_p1' not in locals():
            pose_extractor_p1 = PoseExtractor(config)
            pose_extractor_p1.data = data_store.get("pose_data_p1", [])
        if 'pose_extractor_p2' not in locals():
            pose_extractor_p2 = PoseExtractor(config)
            pose_extractor_p2.data = data_store.get("pose_data_p2", [])

        df_p1 = pose_extractor_p1.save_csv(dir_reports, 'pose_p1_6.csv')
        df_p2 = pose_extractor_p2.save_csv(dir_reports, 'pose_p2_6.csv')

        smoother = DataSmoother()
        smooth_p1 = smoother.smooth_pose_data(df_p1)
        smooth_p2 = smoother.smooth_pose_data(df_p2)
        
        stickman_anim = StickmanAnimator(
            output_path=stickman_vid_path,
            fps=TARGET_FPS, width=width, height=height
        )
        stickman_anim.create_video(smooth_p1, smooth_p2)

    stats_engine = MatchStatistics(
        p1_boxes=data_store["p1_boxes"],
        p2_boxes=data_store["p2_boxes"],
        ball_positions=data_store["ball_positions"],
        inverse_matrices=data_store["court_matrices"], 
        fps=TARGET_FPS
    )
    
    p1_dist, p2_dist, heatmap, coverage = stats_engine.run_analysis()
    match_duration = len(data_store["p1_boxes"]) / TARGET_FPS
    p1_avg_speed = (p1_dist/1000) / (match_duration/3600) if match_duration > 0 else 0
    p2_avg_speed = (p2_dist/1000) / (match_duration/3600) if match_duration > 0 else 0

    report_data = {
        "Total Time (s)": data_store.get("total_time", 0),
        "Total Frames": len(data_store["p1_boxes"]),
        "Court Score": getattr(court_detector, 'court_score', 0),
        "Court Accuracy (%)": getattr(court_detector, 'court_accuracy', 0),
        "Top Player Dist (m)": round(p2_dist, 2),
        "Bottom Player Dist (m)": round(p1_dist, 2),
        "Velocity": {
            "Avg P1": round(p1_avg_speed, 2),
            "Avg P2": round(p2_avg_speed, 2)
        }
    }
    
    save_dashboard_image(heatmap, report_data, os.path.join(dir_reports, 'dashboard6.png'))
    save_text_reports(report_data, dir_reports)
    try:
        save_ball_trajectory(data_store["ball_positions"], os.path.join(dir_reports, 'trajectory6.png'), data_store["court_matrices"])
    except: pass

    # =========================================================================
    # PASS 2: RENDERING FINAL VIDEO
    # =========================================================================
    print(f"\n>>> [RENDER] Creating Final Video (.avi)...")
    
    final_vid_path = os.path.join(dir_videos, 'finalmatch6.avi')
    
    court_img_ref = court_detector.court_reference.court
    # Tính toán kích thước Minimap CHẴN
    minimap_scale = height / court_img_ref.shape[0]
    minimap_w = int(court_img_ref.shape[1] * minimap_scale)
    # [FIX] Đảm bảo chiều rộng minimap là số chẵn
    if minimap_w % 2 != 0: minimap_w += 1
    
    output_w = width + minimap_w if config['visualization']['show_minimap'] else width
    # [FIX] Đảm bảo chiều rộng video output là số chẵn
    if output_w % 2 != 0: output_w += 1
    
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(final_vid_path, fourcc, TARGET_FPS, (output_w, height))
    
    velocity_calc = VelocityCalculator(TARGET_FPS)
    
    p1_boxes = data_store["p1_boxes"]
    p2_boxes = data_store["p2_boxes"]
    ball_positions = data_store["ball_positions"]
    court_warp_matrices = data_store["court_warp_matrix"]
    court_game_matrices = data_store["court_matrices"]
    processed_indices = set(data_store.get("processed_frames_indices", []))
    
    if not processed_indices:
        ratio = fps_orig / TARGET_FPS
        accum = 0.0
        for _ in range(len(p1_boxes)):
            processed_indices.add(int(accum))
            accum += ratio

    trail_p1, trail_p2, trail_ball = [], [], []
    TRAIL_LEN = 20
    frame_idx_orig = 0
    data_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if frame_idx_orig in processed_indices:
            if len(court_warp_matrices) > 0:
                valid_idx = min(data_idx, len(court_warp_matrices)-1)
                court_detector.court_warp_matrix = court_warp_matrices
                frame = court_detector.add_court_overlay(frame, frame_num=valid_idx)
            
            p1_b = p1_boxes[data_idx] if data_idx < len(p1_boxes) else None
            p2_b = p2_boxes[data_idx] if data_idx < len(p2_boxes) else None
            
            if p1_b is not None and p1_b[0] is not None:
                x1, y1, x2, y2 = map(int, p1_b)
                cv2.rectangle(frame, (x1, y1), (x2, y2), config['visualization']['colors']['player_1'], 2)
            
            if p2_b is not None and p2_b[0] is not None:
                x1, y1, x2, y2 = map(int, p2_b)
                cv2.rectangle(frame, (x1, y1), (x2, y2), config['visualization']['colors']['player_2'], 2)

            bx, by = None, None
            if data_idx < len(ball_positions):
                pos = ball_positions[data_idx]
                if pos[0] is not None:
                    bx, by = int(pos[0]), int(pos[1])
                    cv2.circle(frame, (bx, by), 5, config['visualization']['colors']['ball'], -1)

            if config['visualization']['show_minimap']:
                game_idx = min(data_idx, len(court_game_matrices)-1)
                warp_idx = min(data_idx, len(court_warp_matrices)-1)
                inv_matrix = court_game_matrices[game_idx] if len(court_game_matrices) > 0 else None
                warp_matrix = court_warp_matrices[warp_idx] if len(court_warp_matrices) > 0 else None
                
                v1, v2 = velocity_calc.calculate(p1_b, p2_b, warp_matrix)
                if config['visualization']['show_velocity']:
                    frame = draw_velocity(frame, v1, v2)

                p1_c, p2_c, b_c = None, None, None
                if inv_matrix is not None:
                    if p1_b is not None and p1_b[0] is not None:
                        f_pos = np.array([[[ (p1_b[0]+p1_b[2])/2, p1_b[3] ]]], dtype=np.float32)
                        p1_c = cv2.perspectiveTransform(f_pos, inv_matrix)[0][0]
                        trail_p1.append(p1_c)
                    
                    if p2_b is not None and p2_b[0] is not None:
                        f_pos = np.array([[[ (p2_b[0]+p2_b[2])/2, p2_b[3] ]]], dtype=np.float32)
                        p2_c = cv2.perspectiveTransform(f_pos, inv_matrix)[0][0]
                        trail_p2.append(p2_c)
                        
                    if bx is not None:
                        b_pos = np.array([[[bx, by]]], dtype=np.float32)
                        b_c = cv2.perspectiveTransform(b_pos, inv_matrix)[0][0]
                        trail_ball.append(b_c)

                trail_p1 = trail_p1[-TRAIL_LEN:]
                trail_p2 = trail_p2[-TRAIL_LEN:]
                trail_ball = trail_ball[-TRAIL_LEN:]

                minimap_img = draw_minimap(frame, p1_c, p2_c, b_c, 
                                         p1_trail=trail_p1, p2_trail=trail_p2, ball_trail=trail_ball)
                
                # [FIX RESIZE & MERGE]
                # Ép kích thước minimap về đúng minimap_w đã tính ở trên
                minimap_img = cv2.resize(minimap_img, (minimap_w, height))
                if len(minimap_img.shape) == 2: minimap_img = cv2.cvtColor(minimap_img, cv2.COLOR_GRAY2BGR)
                
                # Ghép ảnh: Đảm bảo tổng chiều rộng bằng output_w
                frame = np.hstack((frame, minimap_img))
                
                # Double check kích thước trước khi ghi
                if frame.shape[1] != output_w:
                    frame = cv2.resize(frame, (output_w, height))

            out.write(frame)
            data_idx += 1
            if data_idx % 20 == 0:
                print(f"    Rendered {data_idx} frames...", end='\r')

        frame_idx_orig += 1
        
    cap.release()
    out.release()
    print(f"\n\n>>> ALL TASKS COMPLETED!")
    print(f"Files organized in: {base_output}")

if __name__ == "__main__":
    main()