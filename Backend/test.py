import cv2
import numpy as np
import base64
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image
import random

# Định nghĩa màu sắc cho các lớp
def generate_colors(num_classes):
    """Tạo danh sách màu sắc ngẫu nhiên cho các lớp."""
    return [tuple(random.randint(0, 255) for _ in range(3)) for _ in range(num_classes)]

# Định nghĩa màu sắc toàn cục
colors_classnames = None

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

def crop_image(image_file, tile_size=200):
    img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    height, width, _ = img.shape

    cropped_images = []
    coords = []

    # Cắt ảnh thành các ô nhỏ (tiles)
    for top in range(0, height, tile_size):
        for left in range(0, width, tile_size):
            tile = img[top:min(top + tile_size, height), left:min(left + tile_size, width)]
            cropped_images.append(tile)
            coords.append((left, top))  # Lưu lại vị trí của tile

    return cropped_images, coords, width, height

def stitch_images(cropped_images, coords, img_width, img_height, tile_size=200):
    stitched_image = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    for i, (left, top) in enumerate(coords):
        tile_height, tile_width, _ = cropped_images[i].shape
        stitched_image[top:top + tile_height, left:left + tile_width] = cropped_images[i]

    return stitched_image

def predict_field(detect_model, class_names, uploaded_file):
    """
    Nhận diện bệnh trên cánh đồng từ file ảnh.

    Args:
        detect_model: Mô hình nhận diện YOLO.
        class_names: Danh sách tên các lớp.
        uploaded_file: File ảnh đầu vào (PIL Image hoặc tương tự).

    Returns:
        dict: Kết quả nhận diện gồm:
            - detected_image: Ảnh đã vẽ bounding boxes, mã hóa base64.
            - percent_disease: Phần trăm từng loại bệnh.
            - isDetect: Cờ xác định có phát hiện bệnh hay không (1: Có, 0: Không).
    """
    tiles, coords, img_width, img_height = crop_image(uploaded_file, tile_size=200)
    detected_tiles = []
    detected_cls_all = []

    for idx, tile in enumerate(tiles):
        # Nhận diện đối tượng trên tile
        results = detect_model(tile)

        # Lấy thông tin bounding box, lớp và độ tự tin
        boxes = results[0].boxes.xyxy.tolist()
        detected_cls_idx = results[0].boxes.cls.tolist()
        detected_cls_all.append(detected_cls_idx)
        detected_confidences = results[0].boxes.conf.tolist()

        # Vẽ bounding box lên tile
        tile_with_boxes = draw_bounding_boxes(
            tile.copy(),  # Sao chép để không ảnh hưởng đến dữ liệu gốc
            boxes,
            detected_cls_idx,
            detected_confidences,
            class_names
        )
        detected_tiles.append(tile_with_boxes)  # Lưu tile đã vẽ bounding boxes

    # Ghép lại các tiles đã được nhận diện
    stitched_image = stitch_images(detected_tiles, coords, img_width, img_height, tile_size=200)

    # Tính toán thống kê
    flattened_list = [item for sublist in detected_cls_all for item in sublist]
    total_count = 0
    count_disease = []

    for i in range(len(class_names)):
        mark = float(i)
        count = flattened_list.count(mark)
        count_disease.append(count)
        total_count += count

    percent_disease = []
    for j in range(len(class_names)):
        if total_count > 0:
            percent = count_disease[j] / total_count
            temp = [class_names[j], percent * 100]
            percent_disease.append(temp)

    isDetect = 1 if percent_disease else 0

    # Mã hóa ảnh cuối cùng thành base64
    _, encoded_image = cv2.imencode('.jpg', stitched_image)
    encoded_image = base64.b64encode(encoded_image).decode('utf-8')

    return {
        'detected_image': encoded_image,
        'percent_disease': percent_disease,
        'isDetect': isDetect
    }

def display_results(image, percent_disease):
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Ảnh đã nhận diện")
    plt.show()

    print("Phần trăm bệnh được phát hiện:")
    for disease, percent in percent_disease:
        print(f"- {disease}: {percent:.2f}%")

def main():
    image_path = "domanu (2).jpg"
    model_path = "best.pt"

    print("Đang tải mô hình...")
    detect_model = YOLO(model_path)
    class_names = detect_model.names

    global colors_classnames
    colors_classnames = generate_colors(len(class_names))

    print("Đang tải ảnh...")
    with open(image_path, "rb") as image_file:
        uploaded_file = image_file

        print("Đang nhận diện bệnh...")
        result = predict_field(detect_model, class_names, uploaded_file)

        detected_image = cv2.imdecode(
            np.frombuffer(base64.b64decode(result['detected_image']), dtype=np.uint8),
            cv2.IMREAD_COLOR
        )

        display_results(detected_image, result['percent_disease'])

        if result['isDetect']:
            print("Bệnh đã được phát hiện.")
        else:
            print("Không phát hiện bệnh nào.")

if __name__ == "__main__":
    main()
