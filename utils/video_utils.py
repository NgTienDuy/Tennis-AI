# utils/video_utils.py
import cv2
import numpy as np

def get_video_properties(video):
    """
    Lấy thông số FPS, Tổng số frame, Chiều rộng, Chiều cao
    """
    if not video.isOpened():
        raise ValueError("Video không mở được!")

    fps = video.get(cv2.CAP_PROP_FPS)
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    return fps, length, width, height

def crop_center(image):
    """
    Cắt trung tâm ảnh (giữ nguyên tỷ lệ)
    """
    shape = image.shape[:-1]
    max_size_index = np.argmax(shape)
    diff1 = abs((shape[0] - shape[1]) // 2)
    diff2 = shape[max_size_index] - shape[1 - max_size_index] - diff1
    return image[:, diff1: -diff2] if max_size_index == 1 else image[diff1: -diff2, :]