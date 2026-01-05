import numpy as np
import pandas as pd
from scipy import signal

class DataSmoother:
    def __init__(self, window_length=7, polyorder=2):
        self.window_length = window_length
        self.polyorder = polyorder

    def smooth_pose_data(self, pose_data_df):
        """Smooth pose keypoints dataframe"""
        df = pose_data_df.copy()
        
        # [FIX CRITICAL ERROR] Chuyển đổi toàn bộ cột sang dạng số (Numeric)
        # Các giá trị None/chuỗi sẽ biến thành NaN để Interpolate hiểu được
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # [FIX WARNING] Sử dụng infer_objects() để tránh cảnh báo tương lai
        df = df.infer_objects(copy=False)
        
        # Interpolate missing values (Nội suy các điểm bị mất)
        df = df.interpolate(method='linear', limit_direction='both')
        
        # [FIX WARNING] Thay thế method='bfill' bằng .bfill() và .ffill()
        df = df.bfill().ffill()
        
        # Apply Savitzky-Golay filter (Làm mượt chuyển động)
        for col in df.columns:
            try:
                # Chỉ smooth nếu đủ dữ liệu
                if df[col].notna().sum() > self.window_length:
                    df[col] = signal.savgol_filter(df[col], self.window_length, self.polyorder)
            except:
                pass 
        return df

    def smooth_trajectory(self, points):
        """Smooth a list of (x, y) points"""
        points = np.array(points)
        x = points[:, 0]
        y = points[:, 1]
        
        nans, x_idx = np.isnan(x), lambda z: z.nonzero()[0]
        if nans.any():
            # Kiểm tra nếu toàn bộ là NaN thì không làm gì được
            if nans.all():
                return points
            
            x[nans] = np.interp(x_idx(nans), x_idx(~nans), x[~nans])
            y[nans] = np.interp(x_idx(nans), x_idx(~nans), y[~nans])
            
        try:
            if len(x) > self.window_length:
                x_smooth = signal.savgol_filter(x, self.window_length, self.polyorder)
                y_smooth = signal.savgol_filter(y, self.window_length, self.polyorder)
                return np.column_stack((x_smooth, y_smooth))
            else:
                return points
        except:
            return points