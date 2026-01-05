# utils/general.py
import torch
import yaml
import os

def get_device(config_device_str=None):
    """
    Tự động chọn thiết bị.
    Ưu tiên config, nhưng nếu máy không có CUDA thì tự fallback về CPU.
    """
    # Kiểm tra xem CUDA có thực sự khả dụng không
    cuda_available = torch.cuda.is_available()
    
    # Nếu config yêu cầu cuda nhưng máy không có -> Cảnh báo và về CPU
    if config_device_str == 'cuda' and not cuda_available:
        print("WARNING: Cấu hình yêu cầu 'cuda' nhưng không tìm thấy GPU hoặc PyTorch CPU-only.")
        print(">>> Switching to CPU automatically.")
        return torch.device('cpu')
    
    # Nếu config là cuda và máy có cuda
    if config_device_str == 'cuda' and cuda_available:
        print(">>> Using CUDA (GPU)")
        return torch.device('cuda')
        
    # Mặc định
    print(">>> Using CPU")
    return torch.device('cpu')

def load_config(config_path):
    """
    Đọc file YAML cấu hình
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Không tìm thấy file config tại: {config_path}")
    
    # 🔧 FIX LỖI UnicodeDecodeError
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config

def check_dir(path):
    """
    Kiểm tra và tạo thư mục nếu chưa tồn tại
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
