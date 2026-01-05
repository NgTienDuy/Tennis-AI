# analysis/velocity.py
import numpy as np
import cv2

class VelocityCalculator:
    def __init__(self, fps):
        self.fps = fps
        self.prev_position_p1 = None
        self.prev_position_p2 = None
        self.speed_p1 = 0.0 # km/h
        self.speed_p2 = 0.0 # km/h
        
    def calculate(self, p1_box, p2_box, homography_matrix):
        """
        Calculate speed for both players
        Returns: (speed_p1, speed_p2) in km/h
        """
        self.speed_p1 = self._calc_single_speed(p1_box, homography_matrix, is_p1=True)
        self.speed_p2 = self._calc_single_speed(p2_box, homography_matrix, is_p1=False)
        return self.speed_p1, self.speed_p2

    def _calc_single_speed(self, box, matrix, is_p1):
        if box is None or box[0] is None or matrix is None:
            return 0.0
            
        # Calculate feet position (center bottom of box)
        feet_x = (box[0] + box[2]) / 2
        feet_y = box[3]
        
        # Transform to meters (using Homography)
        point_pixel = np.array([[[feet_x, feet_y]]], dtype=np.float32)
        point_meter = cv2.perspectiveTransform(point_pixel, matrix)[0][0]
        
        # Tennis court scale (simplified mapping from pixel space 1117x2408 to real meters)
        # Real court length ~23.77m, width ~10.97m. 
        # The reference image is scaled. We need a conversion factor.
        # Assuming reference court height 2408 pixels corresponds to 23.77m + margins
        # Let's approximate: 1 pixel ~ 0.015 meters (Need calibration for precision)
        SCALE_FACTOR = 0.015 
        
        current_pos = point_meter * SCALE_FACTOR
        
        speed = 0.0
        prev_pos = self.prev_position_p1 if is_p1 else self.prev_position_p2
        
        if prev_pos is not None:
            distance = np.linalg.norm(current_pos - prev_pos) # meters
            # Speed = Distance / Time
            # Time between frames = 1/FPS
            speed_mps = distance * self.fps 
            speed_kmh = speed_mps * 3.6
            
            # Filter noise (human cannot run > 40km/h easily on court)
            if speed_kmh > 45: speed_kmh = 0 
            speed = speed_kmh

        # Update previous position
        if is_p1: self.prev_position_p1 = current_pos
        else: self.prev_position_p2 = current_pos
        
        return speed