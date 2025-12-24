# Mô hình AI phân tích hành vi người chơi Tennis / Cầu lông

utils.py
    Hỗ trợ xử lý ảnh (crop)
    Thiết lập device (CPU/GPU)
    Lấy thông số video
    Định nghĩa kết nối ketpoints (stickman)

datasets.py
    Nạp dữ liệu cho 3 nhiệm vụ:
        ThetisDataset: Xử lý THETIS (Nhận diện hành động)
        StrokesDataset: Xử lý dữ liệu phân loại cú đánh
        TrackNetDataset: Xử lý dữ liệu theo dõi bóng bằng cách ghép 3 frame liên tiếp

court_reference.py
    Chứa tọa độ chuẩn của sân Tennis (đường biên, lưới...)
    Chứa các hàm tạo mask hoặc ảnh tham chiếu sân

court_detection.py
    Xử lý ảnh: Thresholding và lọc màu để tách đường line trắng của sân
    Phát hiện đường thẳng bằng biến đổi Hough (Hough Transform)
    Tìm ma trận chuyển đổi (Homography) mà khớp với các giao điểm trên ảnh với mô hình chuẩn (court_reference) để ánh xạ từ video sang 2D
    Tracking: Theo dõi vị trí sân qua các frame tiếp theo và giảm tải tính toán

sort.py
    Thuật toán SORT:
        Kalman Filter: Dự đoán vị trí của đối tượng trong frame tiếp theo dựa trên tốc độ hiện tại
        IoU Matching: So sánh bounding box ở frame hiện tại cới dự đoán từ frame trước đó để duy trì ID đối tượng
    -> Theo dõi người chơi và đảm bảo danh tính

detection.py
    Sử dụng Faster R-CNN (ResNet50) để phát hiện người chới
        Player 1: Dựa vào vùng ROI nửa dưới sân và chọn box lớn nhất gần vị trí cũ nhất
        Player 2: Sử dụng SORT để theo dõi IO, sử dụng thuật toán lọc để loại bỏ người nhặt bóng hoặc khán giả

statistics.py
    Phân tích và trực quan hóa dữ liệu
        Heatmap: Mật độ di chuyển của vận động viên
        Tổng quãng đường di chuyển của từng người chơi
        Vẽ các biểu đồ này chồng lên hình ảnh sân tham chiếu

create_data.py
    Tạo dataset từ video gốc
        Cắt vùng ảnh: Tự động phát hiện người chơi và cắt vùng patch quanh họ
        Làm mượt: Sử dụng bộ lọc Savitzky-Golay để làm mượt quỹ đạo di chuyển của khung hình, giúp đầu ra video ổn định, phục vụ việc huấn luyện mô hình

stroke_recognition.py
    Nhận diện cứ đánh
        CNN (Inception V3) dùng để trích suất đặc trưng hình ảnh từng frame
        RNN (LSTM) học sự phụ thuộc theo thời gian của chuỗi các đặc trưng này (Forehand, Backhand, Service/Smash)

trainer.py
    Huấn luyện mô hình nhận diện cú đánh
        Quản lý training loop: epochs, loss, accuracy cho train và valid
        Sử dụng Adam Optimizer và giảm learning-rate nếu loss không cải thiện
        Đánh  giá bằng accuracy trên tập test và vẽ confusion-matrix xem có nhầm lẫn cú đánh nào không

ball_tracker_net.py
    Kiến trúc mạng BallTrackerNet
        Input: 9 kênh màu (3 frame liên tiếp) để mô hình học thông tin chuyển động của bóng
        Encoder: Trích xuất đặc trưng và giảm kích thước không gian
        Decoder: Khôi phục kích thước gốc để tạo Heatmap, nơi điểm có giá trị cao nhất chính là vị trí quả bóng

ball_detection.py
    Phát hiện bóng trong video thực tế
        Ghép 3 frame liên tiếp thành 1 input
        Đưa qua mạng neural để được tọa độ bóng
        Lọc nhiễu hoặc threshold_dist và lưu tọa độ để vẽ lên video hoặc biểu đồ phân tích

pose.py
    Sử dụng Keypoint R-CNN (được huấn luyện sẵn trên tập COCO) để trích xuất dạng người
        Phát hiện keypoints: Tọa độ các điểm khớp (mũi, mắt, vai...) trên cơ thể người chơi
        Vẽ stickman: Nối các điểm khớp theo quy tắc đã định nghĩa (line_connection) để tạo thành hình nhân, giúp trực quan hóa hành động của vận động viên

smooth.py
    Xử lý dáng người sau khi trích xuất
        Hampel Filter: Phát hiện và loại bỏ các nhiễu ngoại lai (outliers)
        Savitzky-Golay Filter: Làm mượt chuyển động các điểm khớp, giúp stickman chuyển động tự nhiên hơn

process.py



5 nhóm chức năng:
    Định vị sân:
        court_reference + court_detection
    Theo dõi người chơi:
        detection + sort + pose + smooth
    Theo dõi bóng:
        ball_tracker_net + ball_detection + datasets (TrackNet)
    Nhân diện cú đánh:
        stroke_recognition + trainer + datasets (Strokes)
    Tổng hợp và báo cáo:
        process + statistics + utils + create_data
