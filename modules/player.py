import cv2
import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import numpy as np
from utils.sort import Sort

class PlayerDetector:
    def __init__(self, config):
        self.device = torch.device(config['system']['device'])
        self.dtype = torch.cuda.FloatTensor if self.device.type == 'cuda' else torch.FloatTensor
        
        # Load model
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        )
        self.model.type(self.dtype)
        self.model.eval()
        
        # Tăng nhẹ threshold lên lại để bớt nhiễu (0.2 -> 0.3)
        self.score_thresh = 0.3 
        
        self.player_1_boxes = [] # Bottom
        self.player_2_boxes = [] # Top
        
        # Tracker
        self.tracker_p1 = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
        self.tracker_p2 = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

    def detect_all(self, frame, court_detector):
        height, width = frame.shape[:2]
        
        # 1. Detect Full HD
        image_t = frame.transpose((2, 0, 1)) / 255
        image_tensor = torch.from_numpy(image_t).unsqueeze(0).type(self.dtype)
        
        with torch.no_grad():
            preds = self.model(image_tensor)
            
        raw_boxes = []
        raw_scores = []
        
        # 2. Lấy thông tin sân để lọc
        net_y = height // 2
        inv_matrix = None # Ma trận Frame -> Sân Chuẩn
        
        if len(court_detector.game_warp_matrix) > 0:
            inv_matrix = court_detector.game_warp_matrix[-1]
            try:
                # Lấy đường lưới
                if court_detector.net is not None:
                    net_y = (court_detector.net[1] + court_detector.net[3]) / 2
            except: pass

        for box, label, score in zip(preds[0]['boxes'], preds[0]['labels'], preds[0]['scores']):
            if label == 1 and score > self.score_thresh: 
                b = box.detach().cpu().numpy()
                s = score.detach().cpu().numpy()
                
                # [BỘ LỌC QUAN TRỌNG] Kiểm tra vị trí địa lý
                if self._is_in_court(b, inv_matrix, width, height):
                    raw_boxes.append(b)
                    raw_scores.append(s)
                
        # 3. Phân loại Top/Bottom
        candidates_p1 = [] 
        candidates_p2 = [] 
        
        for b, s in zip(raw_boxes, raw_scores):
            foot_y = b[3]
            # Bottom (P1)
            if foot_y > net_y:
                candidates_p1.append(np.append(b, s))
            # Top (P2)
            else:
                # Lọc thêm: Người Top không được quá to (nếu to là người Bottom bị nhận nhầm)
                box_h = b[3] - b[1]
                if box_h < (height * 0.4): 
                    candidates_p2.append(np.append(b, s))

        # 4. Tracking
        # P1
        dets_p1 = np.array(candidates_p1) if len(candidates_p1) > 0 else np.empty((0, 5))
        tracked_p1 = self.tracker_p1.update(dets_p1)
        
        box_p1 = [None, None, None, None]
        if len(tracked_p1) > 0:
            box_p1 = max(tracked_p1, key=lambda x: (x[2]-x[0])*(x[3]-x[1]))[:4]
        self.player_1_boxes.append(box_p1)

        # P2
        dets_p2 = np.array(candidates_p2) if len(candidates_p2) > 0 else np.empty((0, 5))
        tracked_p2 = self.tracker_p2.update(dets_p2)
        
        box_p2 = [None, None, None, None]
        if len(tracked_p2) > 0:
            # Chọn người gần trục giữa sân nhất (Center X)
            center_x = width / 2
            best_p2 = min(tracked_p2, key=lambda x: abs((x[0]+x[2])/2 - center_x))
            box_p2 = best_p2[:4]
            
        self.player_2_boxes.append(box_p2)

    def _is_in_court(self, box, inv_matrix, img_w, img_h):
        """
        Kiểm tra xem box có nằm trong khu vực sân tennis hợp lệ không.
        Sử dụng ma trận Inverse Homography để map toạ độ chân về Sân Chuẩn.
        """
        if inv_matrix is None:
            # Fallback: Nếu chưa tìm được sân, dùng quy tắc biên đơn giản
            # Loại bỏ box quá sát mép trái/phải (nơi thường có biển quảng cáo/trọng tài)
            x_center = (box[0] + box[2]) / 2
            if x_center < (img_w * 0.1) or x_center > (img_w * 0.9): return False
            return True

        # Toạ độ chân người chơi trên ảnh
        foot_x = (box[0] + box[2]) / 2
        foot_y = box[3]
        
        # Transform về hệ toạ độ Sân Chuẩn (Reference Court)
        # Sân chuẩn (trong court.py) có kích thước tổng khoảng: Rộng 1665, Cao 3500
        # (Bao gồm cả lề sân)
        
        pt = np.array([[[foot_x, foot_y]]], dtype=np.float32)
        try:
            pt_out = cv2.perspectiveTransform(pt, inv_matrix)[0][0]
            ref_x, ref_y = pt_out[0], pt_out[1]
            
            # Kích thước sân tham chiếu (hardcode từ court.py)
            REF_W = 1117 + 274 * 2 # ~1665
            REF_H = 2408 + 549 * 2 # ~3500
            
            # Kiểm tra biên: Cho phép lấn ra ngoài một chút (Margin)
            MARGIN = 200 # pixel trong hệ tham chiếu
            
            if (-MARGIN < ref_x < REF_W + MARGIN) and (-MARGIN < ref_y < REF_H + MARGIN):
                return True
            else:
                return False # Nằm ngoài sân (khán đài, biển quảng cáo)
        except:
            return True # Nếu lỗi tính toán, tạm chấp nhận