import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image, ImageDraw

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pad, bias=True, bn=True):
        super().__init__()
        if bn:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=pad, bias=bias),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=pad, bias=bias),
                nn.ReLU()
            )
    def forward(self, x): return self.block(x)

class BallTrackerNet(nn.Module):
    def __init__(self, out_channels=256, bn=True):
        super().__init__()
        self.out_channels = out_channels
        # Encoder
        self.encoder = nn.Sequential(
            ConvBlock(9, 64, 3, 1, bn=bn), ConvBlock(64, 64, 3, 1, bn=bn), nn.MaxPool2d(2, 2),
            ConvBlock(64, 128, 3, 1, bn=bn), ConvBlock(128, 128, 3, 1, bn=bn), nn.MaxPool2d(2, 2),
            ConvBlock(128, 256, 3, 1, bn=bn), ConvBlock(256, 256, 3, 1, bn=bn), ConvBlock(256, 256, 3, 1, bn=bn), nn.MaxPool2d(2, 2),
            ConvBlock(256, 512, 3, 1, bn=bn), ConvBlock(512, 512, 3, 1, bn=bn), ConvBlock(512, 512, 3, 1, bn=bn)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2), ConvBlock(512, 256, 3, 1, bn=bn), ConvBlock(256, 256, 3, 1, bn=bn), ConvBlock(256, 256, 3, 1, bn=bn),
            nn.Upsample(scale_factor=2), ConvBlock(256, 128, 3, 1, bn=bn), ConvBlock(128, 128, 3, 1, bn=bn),
            nn.Upsample(scale_factor=2), ConvBlock(128, 64, 3, 1, bn=bn), ConvBlock(64, 64, 3, 1, bn=bn),
            ConvBlock(64, self.out_channels, 3, 1, bn=bn)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, testing=False):
        features = self.encoder(x)
        output = self.decoder(features)
        if testing: output = self.softmax(output)
        return output

    def inference(self, frames: torch.Tensor):
        self.eval()
        with torch.no_grad():
            if len(frames.shape) == 3: frames = frames.unsqueeze(0)
            if next(self.parameters()).is_cuda: frames = frames.cuda()
            output = self(frames, True)
            output = output.argmax(dim=1).detach().cpu().numpy()
            if self.out_channels == 2: output *= 255
            x, y = self.get_center_ball(output)
        return x, y

    def get_center_ball(self, output):
        output = output.reshape((360, 640)).astype(np.uint8)
        heatmap = cv2.resize(output, (640, 360))
        ret, heatmap = cv2.threshold(heatmap, 127, 255, cv2.THRESH_BINARY)
        circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2, maxRadius=7)
        if circles is not None and len(circles) == 1:
            return int(circles[0][0][0]), int(circles[0][0][1])
        return None, None

def combine_three_frames(frame1, frame2, frame3, width, height):
    img1 = cv2.resize(frame1, (width, height)).astype(np.float32)
    img2 = cv2.resize(frame2, (width, height)).astype(np.float32)
    img3 = cv2.resize(frame3, (width, height)).astype(np.float32)
    imgs = np.concatenate((img1, img2, img3), axis=2)
    return np.rollaxis(imgs, 2, 0)

class BallDetector:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['system']['device'])
        self.model = BallTrackerNet(out_channels=config['detection']['ball']['out_channels'])
        
        # Load weights
        # [FIXED] Thêm weights_only=False để fix lỗi PyTorch 2.6+
        saved_state = torch.load(
            config['paths']['tracknet_weights'], 
            map_location=self.device, 
            weights_only=False
        )
        self.model.load_state_dict(saved_state['model_state'])
        self.model.eval().to(self.device)

        self.current_frame = None
        self.last_frame = None
        self.before_last_frame = None
        self.video_width = None
        self.video_height = None
        
        self.model_w = config['detection']['ball']['model_input_width']
        self.model_h = config['detection']['ball']['model_input_height']
        self.threshold_dist = config['detection']['ball']['threshold_dist']
        
        self.xy_coordinates = np.array([[None, None], [None, None]])

    def detect_ball(self, frame):
        if self.video_width is None:
            self.video_height, self.video_width = frame.shape[:2]
        
        self.last_frame = self.before_last_frame
        self.before_last_frame = self.current_frame
        self.current_frame = frame.copy()

        x, y = None, None
        if self.last_frame is not None:
            frames = combine_three_frames(self.current_frame, self.before_last_frame, self.last_frame, self.model_w, self.model_h)
            frames = (torch.from_numpy(frames) / 255).to(self.device)
            x, y = self.model.inference(frames)
            
            if x is not None:
                x = x * (self.video_width / self.model_w)
                y = y * (self.video_height / self.model_h)
                
                # Check outlier
                if self.xy_coordinates[-1][0] is not None:
                    if np.linalg.norm(np.array([x, y]) - self.xy_coordinates[-1]) > self.threshold_dist:
                        x, y = None, None
        
        self.xy_coordinates = np.append(self.xy_coordinates, np.array([[x, y]]), axis=0)
        return x, y