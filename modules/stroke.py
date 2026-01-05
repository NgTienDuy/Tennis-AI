# modules/stroke.py
import torch
import torch.nn as nn
import torchvision
import numpy as np
import cv2
from torchvision import transforms

class Identity(nn.Module):
    def forward(self, x): return x

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = torchvision.models.inception_v3(pretrained=True)
        self.feature_extractor.fc = Identity()
    def forward(self, x): return self.feature_extractor(x)

class LSTM_model(nn.Module):
    def __init__(self, num_classes, input_size=2048, num_layers=3, hidden_size=90, dtype=torch.FloatTensor):
        super().__init__()
        self.dtype = dtype
        self.LSTM = nn.LSTM(input_size, hidden_size, num_layers, bias=True, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = x.type(self.dtype)
        h0 = torch.zeros(3, x.size(0), 90).type(self.dtype)
        c0 = torch.zeros(3, x.size(0), 90).type(self.dtype)
        output, _ = self.LSTM(x, (h0, c0))
        output = output[:, -1, :] # Last hidden state
        scores = self.fc(output)
        return scores

class StrokeRecognition:
    def __init__(self, config):
        self.device = torch.device(config['system']['device'])
        # Tự động chọn kiểu FloatTensor phù hợp (Cuda hoặc CPU)
        if self.device.type == 'cuda':
            self.dtype = torch.cuda.FloatTensor 
        else:
            self.dtype = torch.FloatTensor
        
        # 1. Feature Extractor (InceptionV3)
        self.feature_extractor = FeatureExtractor().to(self.device).type(self.dtype).eval()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # 2. LSTM Classifier
        self.model = LSTM_model(num_classes=3, dtype=self.dtype).to(self.device).eval()
        
        # Load weights
        # [FIXED] Thêm weights_only=False
        saved_state = torch.load(
            config['paths']['stroke_weights'], 
            map_location=self.device, 
            weights_only=False
        )
        self.model.load_state_dict(saved_state['model_state'])
        
        self.frames_features_seq = None
        self.max_seq_len = 55
        self.labels = ['Forehand', 'Backhand', 'Service/Smash']
        self.softmax = nn.Softmax(dim=1)

    def add_frame(self, frame, player_box):
        """Extract features from player crop"""
        if player_box is None or player_box[0] is None: return
        x1, y1, x2, y2 = map(int, player_box)
        margin = 50
        h, w = frame.shape[:2]
        crop = frame[max(0, y1-margin):min(h, y2+margin), max(0, x1-margin):min(w, x2+margin)]
        
        if crop.size == 0: return
        
        crop = cv2.resize(crop, (299, 299))
        img_t = crop.transpose((2, 0, 1)) / 255
        img_tensor = torch.from_numpy(img_t).unsqueeze(0).type(self.dtype)
        img_tensor = self.normalize(img_tensor)
        
        with torch.no_grad():
            feature = self.feature_extractor(img_tensor).unsqueeze(1)
            
        if self.frames_features_seq is None:
            self.frames_features_seq = feature
        else:
            self.frames_features_seq = torch.cat([self.frames_features_seq, feature], dim=1)
            
        # Keep window size
        if self.frames_features_seq.size(1) > self.max_seq_len:
            self.frames_features_seq = self.frames_features_seq[:, 1:, :]

    def predict(self):
        if self.frames_features_seq is None or self.frames_features_seq.size(1) < 10:
            return None, None
            
        with torch.no_grad():
            scores = self.model(self.frames_features_seq)
            probs = self.softmax(scores).cpu().numpy()[0]
            
        label = self.labels[np.argmax(probs)]
        return label, probs
    
    def reset(self):
        self.frames_features_seq = None