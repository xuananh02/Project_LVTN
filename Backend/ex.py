from ultralytics import YOLO

detect_model = YOLO('best.pt')

classnames = ['fruit-bores', 'grasshopper', 'leafroller', 'maybug', 'moth', 'red-spider', 'mealybug-disease', 'snail', 'stag-beetle', 'stinkbug'] 

result = detect_model.predict('grasshopper_201.jpg', save=False)

detected_cls_idx = result[0].boxes.cls.tolist()

for index in detected_cls_idx:
    detected_cls = classnames[int(index)]
    print(detected_cls)