from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from PIL import Image
import base64
from ultralytics import YOLO
from flask_cors import CORS, cross_origin
from datetime import datetime

app = Flask(__name__)

# Đảm bảo mô hình YOLO được load đúng
detect_model = YOLO('best.pt')

# Các lớp (class names) và màu sắc tương ứng
class_names = ['Cháy lá', 'Đốm nâu', 'Đạo ôn']
class_names2 = ['Chay la', 'Dom nau', 'Dao on']
class_names_number = ['0', '1', '2']
colors_classnames = [(0, 0, 255), (0, 255, 255), (19, 69, 139)]

# Cấu hình thư mục lưu ảnh
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# Hàm vẽ bounding boxes lên ảnh
def draw_bounding_boxes(image, boxes, detected_cls_idx, detected_confidences, class_names):
    global colors_classnames
    for box, cls, conf in zip(boxes, detected_cls_idx, detected_confidences):
        x1, y1, x2, y2 = map(int, box)
        label = f"{class_names[int(cls)]}: {conf:.2f}"
        
        # Vẽ bounding box
        color = colors_classnames[int(cls)]
        thickness = 2
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # Vẽ nhãn (label)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        text_size = cv2.getTextSize(label, font, font_scale, 1)[0]
        text_x, text_y = x1, y1 - 10  # Vị trí vẽ nhãn
        cv2.rectangle(image, (text_x, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5), color, -1)
        cv2.putText(image, label, (text_x, text_y), font, font_scale, (0, 0, 0), 1)
    
    return image


@app.route('/api/predict', methods=['POST'])
@cross_origin(origins='*')
def predict():
    global detect_model
    global class_names_number

    uploaded_file = request.files['file']
    uploaded_file = cv2.imread(uploaded_file)

    result = detect_model.predict(uploaded_file, save=False)
    
    boxes = result[0].boxes.xyxy.tolist()
    detected_cls_idx = result[0].boxes.cls.tolist()
    detected_confidences = result[0].boxes.conf.tolist()

    image_with_boxes = draw_bounding_boxes(uploaded_file, boxes, detected_cls_idx, detected_confidences, class_names2)

    _, encoded_image = cv2.imencode('.jpg', image_with_boxes)
    encoded_image = base64.b64encode(encoded_image).decode('utf-8')

    rounded_numbers = []
    for number in detected_confidences:
        rounded_numbers.append(round(number, 3))

    return jsonify({
        'detected_image': encoded_image,
        'detected_classes': detected_cls_idx,
        'confidences': rounded_numbers
    })

# Hàm cắt ảnh thành các ô nhỏ
def crop_image(image_path, tile_size=200):
    img_tmp = cv2.imread(image_path)  # Đọc ảnh trực tiếp từ file trên ổ đĩa

    if img_tmp is None:
        raise ValueError("Không thể đọc ảnh từ file.")

    height, width, _ = img_tmp.shape

    cropped_images = []
    coords = []

    # Cắt ảnh thành các ô nhỏ (tiles)
    for top in range(0, height, tile_size):
        for left in range(0, width, tile_size):
            tile = img_tmp[top:min(top + tile_size, height), left:min(left + tile_size, width)]
            cropped_images.append(tile)
            coords.append((left, top))  # Lưu lại vị trí của tile

    return cropped_images, coords, width, height

# Hàm ghép lại hình ảnh từ các tiles đã nhận diện
def stitch_images(cropped_images, coords, img_width, img_height, tile_size=200):
    # Tạo một ảnh trống để ghép các phần lại với nhau
    stitched_image = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    for i, (left, top) in enumerate(coords):
        tile_height, tile_width, _ = cropped_images[i].shape
        stitched_image[top:top + tile_height, left:left + tile_width] = cropped_images[i]

    return stitched_image

@app.route('/api/predict-field', methods=['POST'])
@cross_origin(origins='*')
def predict_field():
    global detect_model, class_names

    # Nhận file ảnh từ request
    uploaded_file = request.files['file']
    
    # Tạo tên file duy nhất dựa trên thời gian hiện tại
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{timestamp}_{secure_filename(uploaded_file.filename)}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Lưu file vào ổ đĩa
    uploaded_file.save(file_path)

    # Kiểm tra nếu ảnh đã được lưu thành công
    if not os.path.exists(file_path):
        return jsonify({'error': 'Failed to save the image file.'}), 400

    # Đọc lại ảnh từ file đã lưu
    img = cv2.imread(file_path)

    # Kiểm tra ảnh đọc được không
    if img is None:
        return jsonify({'error': 'Failed to decode image.'}), 400

    # Lấy chiều rộng và chiều cao của ảnh
    h, w = img.shape[:2]
    area_big = h * w

    # Cắt ảnh thành các tile nhỏ
    try:
        tiles, coords, img_width, img_height = crop_image(file_path, tile_size=200)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    detected_tiles = []
    bounding_boxes_info = {i: [] for i in range(len(class_names))}

    # Xử lý từng tile
    for idx, tile in enumerate(tiles):
        # Giả sử detect_model là mô hình YOLO hoặc tương tự
        results = detect_model(tile)
        boxes = results[0].boxes.xyxy.tolist()
        detected_cls_idx = results[0].boxes.cls.tolist()
        detected_confidences = results[0].boxes.conf.tolist()

        # Lưu thông tin bounding boxes theo từng lớp
        for box, cls, conf in zip(boxes, detected_cls_idx, detected_confidences):
            x1, y1, x2, y2 = map(int, box)
            class_name = class_names[int(cls)]
            bounding_boxes_info[int(cls)].append({
                'class': class_name,
                'bbox': (x1, y1, x2, y2),
                'confidence': conf,
                'area': (x2 - x1) * (y2 - y1)
            })

        # Vẽ bounding box lên từng tile
        tile_with_boxes = draw_bounding_boxes(
            tile.copy(),
            boxes,
            detected_cls_idx,
            detected_confidences,
            class_names2
        )
        detected_tiles.append(tile_with_boxes)

    # Ghép lại các tile đã được nhận diện
    stitched_image = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    for i, (left, top) in enumerate(coords):
        tile_height, tile_width, _ = detected_tiles[i].shape
        stitched_image[top:top + tile_height, left:left + tile_width] = detected_tiles[i]
    
    # Mã hóa ảnh đã ghép lại thành base64 để gửi về
    _, encoded_image = cv2.imencode('.jpg', stitched_image)
    encoded_image = base64.b64encode(encoded_image).decode('utf-8')

    classes_info = []
    for cls, boxes in bounding_boxes_info.items():
        class_name = class_names[cls]
        class_area = sum([box_info['area'] for box_info in boxes])
        class_area_percent = (class_area / area_big) * 100
        class_count = len(boxes)  # Tổng số lượng bounding boxes của lớp này

        classes_info.append({
            'class_name': class_name,
            'total_area': class_area,
            'area_percentage': class_area_percent,
            'count': class_count
        })

    # Xóa file ảnh sau khi xử lý xong
    os.remove(file_path)

    # Trả về kết quả dưới dạng JSON
    return jsonify({
        'detected_image': encoded_image,
        'bounding_boxes_info': bounding_boxes_info,
        'classes_info': classes_info,
        'isDetect': 1 if any(bounding_boxes_info.values()) else 0
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2409)
