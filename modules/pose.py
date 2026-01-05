# modules/pose.py
import cv2
import torch
import torchvision
from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights
import numpy as np
import pandas as pd
import os

class PoseExtractor:
    def __init__(self, config):
        self.device = torch.device(config['system']['device'])
        self.dtype = torch.cuda.FloatTensor if self.device.type == 'cuda' else torch.FloatTensor
        
        # Load Model
        self.model = torchvision.models.detection.keypointrcnn_resnet50_fpn(
            weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
        )
        self.model.type(self.dtype)
        self.model.eval()
        
        self.min_score = config['detection']['pose']['min_score']
        self.kp_threshold = config['detection']['pose']['keypoint_threshold']
        
        self.data = []
        self.COCO_KEYPOINTS = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]

    def extract_pose(self, frame, player_box):
        # Kiểm tra box hợp lệ
        is_invalid = False
        if player_box is None:
            is_invalid = True
        elif isinstance(player_box, (list, np.ndarray)) and len(player_box) == 0:
            is_invalid = True
        elif player_box[0] is None:
            is_invalid = True

        if is_invalid:
            self.data.append([None]*34)
            return
            
        box = player_box
        x1, y1, x2, y2 = map(int, box)
        margin = 50
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1-margin), max(0, y1-margin)
        x2, y2 = min(w, x2+margin), min(h, y2+margin)
        
        patch = frame[y1:y2, x1:x2]
        if patch.size == 0: 
            self.data.append([None]*34)
            return

        img_t = patch.transpose((2, 0, 1)) / 255
        img_tensor = torch.from_numpy(img_t).unsqueeze(0).type(self.dtype)
        
        with torch.no_grad():
            output = self.model(img_tensor)
            
        kp_data = [None] * 34
        # Lấy người có điểm cao nhất
        if len(output[0]['scores']) > 0 and output[0]['scores'][0] > self.min_score:
            keypoints = output[0]['keypoints'][0].detach().cpu().numpy()
            scores = output[0]['keypoints_scores'][0].detach().cpu().numpy()
            
            flat_kp = []
            for i in range(17):
                k_x, k_y, k_v = keypoints[i]
                s = scores[i]
                if s > self.kp_threshold: 
                    # Map toạ độ từ patch về ảnh gốc
                    flat_kp.append(k_x + x1)
                    flat_kp.append(k_y + y1)
                else:
                    flat_kp.append(None)
                    flat_kp.append(None)
            kp_data = flat_kp
            
        self.data.append(kp_data)

    def save_csv(self, output_folder, filename='stickman_data.csv'):
        cols = []
        for k in self.COCO_KEYPOINTS:
            cols.append(k + '_x')
            cols.append(k + '_y')
        df = pd.DataFrame(self.data, columns=cols)
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        out_path = os.path.join(output_folder, filename)
        df.to_csv(out_path, index=False)
        return df