import cv2
import numpy as np
import pandas as pd
import os

class StickmanAnimator:
    def __init__(self, output_path, fps, width, height):
        self.output_path = output_path
        self.fps = fps
        self.width = width
        self.height = height
        self.line_connections = [
            (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
        ]

    def create_video(self, df_p1, df_p2):
        print(f"Creating Stickman Animation: {self.output_path}")
        
        # [FIX] Sử dụng MJPG và .avi để đảm bảo tương thích
        fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
        out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))
        
        max_len = max(len(df_p1), len(df_p2))
        
        for i in range(max_len):
            canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            if i < len(df_p1):
                self._draw_skeleton(canvas, df_p1.iloc[i], (0, 255, 0), (0, 200, 0))
            if i < len(df_p2):
                self._draw_skeleton(canvas, df_p2.iloc[i], (255, 0, 255), (200, 0, 200))
            
            out.write(canvas)
            
        out.release()
        print("Stickman video saved.")

    def _draw_skeleton(self, canvas, row, color_point, color_line):
        points = {}
        valid = True
        
        MAX_BONE_LEN = self.width * 0.25 

        for i in range(17):
            col_x = self._df_col_name(i, 'x')
            col_y = self._df_col_name(i, 'y')
            if col_x not in row.index: 
                valid = False; break
            px = row[col_x]
            py = row[col_y]
            
            if pd.notna(px) and pd.notna(py):
                if 0 <= px < self.width and 0 <= py < self.height:
                    points[i] = (int(px), int(py))
                    cv2.circle(canvas, (int(px), int(py)), 3, color_point, -1)
        
        if valid:
            for pair in self.line_connections:
                if pair[0] in points and pair[1] in points:
                    pt1 = points[pair[0]]
                    pt2 = points[pair[1]]
                    dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
                    if dist < MAX_BONE_LEN:
                        cv2.line(canvas, pt1, pt2, color_line, 2)

    def _df_col_name(self, idx, axis):
        names = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 
                 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 
                 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 
                 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
        return f"{names[idx]}_{axis}"