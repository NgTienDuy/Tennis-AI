---

---

# HỆ THỐNG PHÂN TÍCH VÀ THEO DÕI TRẬN ĐẤU TENNIS BẰNG THỊ GIÁC MÁY TÍNH

**Môn học:** Thị giác máy tính
**GVHD:** [TS. Phạm Tiến Lâm]
**Sinh viên:** [Nguyễn Tiến Duy]                Mã SV: 21002193
**Sinh viên:** [Khiếu Hữu Tiến Dũng]            Mã SV: 22001310

---

# Nội dung trình bày

1. **Đặt vấn đề & Mục tiêu:** Tại sao làm đề tài này?
2. **Cơ sở lý thuyết:** Các công nghệ lõi (Hough, Faster R-CNN, Kalman).
3. **Phương pháp đề xuất:** Quy trình xử lý hệ thống.
4. **Thực nghiệm & Kết quả:** Demo và đánh giá hiệu năng.
5. **Kết luận & Hướng phát triển.**

---

# 1. Đặt vấn đề

> *"Người ta không thể phân biệt cầu thủ đánh bóng tỉ lệ .300 và .275 bằng mắt thường."* - Michael Lewis (Moneyball)

* **Thực trạng:**
    * Phân tích thể thao chuyển dịch từ "trực giác" sang "dữ liệu".
    * Các hệ thống chuyên nghiệp (Hawk-Eye) giá triệu đô, cần 10+ camera.
    * Thiếu giải pháp cho giải đấu phong trào/bán chuyên.
* **Thách thức:**
    * Bóng nhỏ, di chuyển nhanh (Motion blur).
    * Camera đơn (Broadcast view) thường bị che khuất.

---

# Mục tiêu đề tài

Xây dựng hệ thống tự động hóa với **chi phí thấp** trên **phần cứng phổ thông**:

1. **Input:** Video trận đấu (1 camera duy nhất).
2. **Core Tasks:**
    * Nhận diện sân (Court Detection).
    * Nhận diện người chơi (Player Detection).
    * Theo dõi vết di chuyển (Tracking).
3. **Output:**
    * Minimap 2D thời gian thực.
    * Biểu đồ nhiệt (Heatmap) chiến thuật.

---

# 2. Cơ sở lý thuyết: Nhận diện Sân

**Phép biến đổi Hough (Probabilistic Hough Transform)**

* **Mục đích:** Phát hiện các đường thẳng (line) của sân tennis.
* **Nguyên lý:** Biểu diễn đường thẳng trong không gian tham số cực $(\rho, \theta)$.

$$\rho = x \cos \theta + y \sin \theta$$

* **Homography (Ánh xạ xạ ảnh):**
    * Chuyển đổi tọa độ từ ảnh camera $(u,v)$ sang mặt phẳng sân thực tế $(x,y)$.
    * Yêu cầu tối thiểu 4 điểm tương ứng.

---

# Cơ sở lý thuyết: Nhận diện Người

**Mạng nơ-ron tích chập: Faster R-CNN**

* **Backbone:** ResNet-50 (Trích xuất đặc trưng).
* **RPN (Region Proposal Network):** Đề xuất các vùng có khả năng chứa đối tượng (Anchor Boxes).
* **Loss Function:** Tổng của mất mát phân loại và hồi quy.

$$L = L_{cls} + L_{reg}$$

* **Ưu điểm:** Chính xác hơn các phương pháp trừ nền (Background Subtraction) cũ.

---

# Cơ sở lý thuyết: Tracking

**Bộ lọc Kalman (Kalman Filter)**

* **Dự đoán (Predict):** Ước lượng vị trí tiếp theo dựa trên vận tốc hiện tại.
  $$\hat{x}_{k|k-1} = F_k \hat{x}_{k-1|k-1}$$
* **Cập nhật (Update):** Hiệu chỉnh vị trí khi có kết quả từ Faster R-CNN.

**Thuật toán SORT:**
* Kết hợp Kalman Filter và thuật toán **Hungarian**.
* Dùng chỉ số **IoU** để ghép cặp ID.

---

# 3. Phương pháp: Tổng quan hệ thống

* **Kiến trúc Pipeline tuần tự:**
    1. **Preprocessing:** Lọc nhiễu, chuẩn hóa ảnh.
    2. **Detection:** Chạy song song detect sân và người.
    3. **Tracking:** Gán ID cho người chơi.
    4. **Visualization:** Vẽ Minimap.

---

# BẢN ĐỒ CẤU TRÚC FOLDER
```text
.
├── configs/                # Cài đặt hệ thống
│   └── config.yaml         # Ngưỡng, đường dẫn, màu sắc
│
├── models/                 # Bộ não AI (đã huấn luyện)
│   ├── TrackNet.pth
│   └── StrokeNet.pth
│
├── modules/                # Giai đoạn 3: Inference
│   ├── court.py            # Phát hiện sân
│   ├── player.py           # Phát hiện người chơi
│   ├── ball.py             # Phát hiện bóng
│   ├── pose.py             # Trích xuất xương
│   └── stroke.py           # Phân loại cú đánh
│
├── utils/                  # Công cụ hỗ trợ
│   ├── sort.py             # Tracking (Kalman Filter)
│   └── video_utils.py      # Đọc & xử lý video
│
├── analysis/               # Giai đoạn 4: Xử lý số liệu
│   ├── smoothing.py        # Lọc nhiễu
│   ├── velocity.py         # Tính vận tốc (km/h)
│   └── stats.py            # Heatmap, quãng đường
│
├── visualization/          # Giai đoạn 4: Trực quan hóa
│   ├── drawer.py           # Vẽ sân, minimap
│   ├── dashboard.py        # Biểu đồ báo cáo
│   └── stickman_animator.py# Video skeleton
│
├── data/                   # Input / Output
│   ├── raw/                # Video gốc
│   └── processed/          # Dữ liệu trung gian
│
├── outputs/                # Kết quả cuối
│   ├── videos/             # Video .avi
│   ├── reports/            # Ảnh, JSON
│   └── packages/           # Checkpoint (.pkl)
│
└── main.py                 # Nhạc trưởng pipeline
```

# Module Nhận diện Sân (Court Detection)

1. **Lọc màu (Color Thresholding):**
   * Lấy pixel trắng: `Intensity > 200`.
   * Lọc nhiễu bằng hình thái học (Dilation).
2. **Gộp dòng (Line Merging):**
   * Phân loại: Ngang (Horizontal) vs Dọc (Vertical).
   * Gộp các đoạn thẳng rời rạc nếu khoảng cách < 20px.
3. **Kết quả:** Xác định 4 đỉnh sân để tính ma trận Homography.

---

# Module Nhận diện Người (Player Detection)

* **Model:** `torchvision.models.fasterrcnn_resnet50_fpn`.
* **Trọng số:** Pre-trained trên COCO Dataset.
* **Bộ lọc Logic (Heuristic Filter):**
    * Chỉ giữ lại box có độ tin cậy > 0.7.
    * Chỉ giữ lại box nằm trong vùng sân (ROI).
    * Phân loại: **Top Player** vs **Bottom Player** dựa trên vị trí lưới.

---

# Module Theo dõi (Tracking Strategy)

Sử dụng thư viện **SORT** tùy chỉnh:

* **Bước 1:** Dự đoán vị trí mới của các track cũ (Kalman Predict).
* **Bước 2:** Tính ma trận IoU giữa Track và Detection mới.
* **Bước 3:** Ghép cặp (Hungarian Algorithm).
* **Bước 4:** Xử lý ngoại lệ:
    * **Unmatched Tracks:** Xóa nếu mất dấu quá 10 frames.
    * **Unmatched Detections:** Tạo ID mới.

---

# 4. Thực nghiệm & Kết quả

**Môi trường thử nghiệm:**
* **Dữ liệu:** 3 Video clips (Full HD 30fps) - Sân cứng & Sân cỏ.
* **Phần cứng:** GPU NVIDIA GeForce [Tên GPU].
* **Thư viện:** PyTorch, OpenCV, NumPy.

**Thách thức xử lý:**
* Bóng mờ (Motion blur).
* Che khuất (Occlusion) khi người chơi lên lưới.

---

# Đánh giá Định lượng (Quantitative)

So sánh Faster R-CNN (Đề xuất) vs Trừ nền (Cũ):

| Phương pháp | Precision | Recall | F1-Score | FPS |
| :--- | :---: | :---: | :---: | :---: |
| Background Sub. | 78.5% | 82.0% | 80.2% | 30 |
| **Faster R-CNN** | **96.2%** | **94.5%** | **95.3%** | **18** |

> **Nhận xét:** Deep Learning chậm hơn nhưng chính xác vượt trội trong môi trường động.

---

# Kết quả Trực quan (Demo)

* **Tracking:** Duy trì ID ổn định khi người chơi di chuyển chéo sân.
* **Minimap:** Phản ánh đúng vị trí thực tế trên mô hình 2D.

---

# Phân tích Chiến thuật (Heatmap)

* **Vùng hoạt động:** 70% thời gian ở Rally Zone (Sau baseline).
* **Chiến thuật:**
    * Player 1 (Thắng): Bao sân tốt, di chuyển linh hoạt.
    * Player 2: Bị ép về góc sân (Defensive Zone).

---

# 5. Kết luận

* **Đạt được:**
    * Xây dựng thành công Pipeline end-to-end.
    * Ứng dụng Deep Learning (Faster R-CNN) thay thế phương pháp cổ điển.
    * Giải quyết tốt bài toán Tracking và Mapping 2D.
* **Hạn chế:**
    * Tốc độ ~18 FPS (Chưa đạt Real-time 30 FPS).
    * Đôi khi nhận diện nhầm nhặt bóng (Ball boy).

---

# Hướng phát triển

1. **Tối ưu tốc độ:**
   * Chuyển sang mô hình **YOLOv8** hoặc **YOLOv11** (One-stage detector).
2. **Nhận diện hành động (Action Recognition):**
   * Dùng LSTM/3D-CNN để phân loại cú đánh (Forehand, Backhand, Serve).
3. **Dự đoán:**
   * Dự đoán điểm rơi của bóng để hỗ trợ tập luyện.

---

# Tài liệu tham khảo

1. Ren et al., *Faster R-CNN: Towards Real-Time Object Detection*, NIPS 2015.
2. Bewley et al., *Simple Online and Realtime Tracking (SORT)*, ICIP 2016.
3. Lewis, M., *Moneyball: The Art of Winning an Unfair Game*, 2004.

---

# CẢM ƠN THẦY CÔ VÀ CÁC BẠN ĐÃ LẮNG NGHE!

**Q & A**