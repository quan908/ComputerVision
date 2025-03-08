import cv2 
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import numpy as np
import matplotlib.pyplot as plt

def compute_iou(box1, box2):
    """ 计算两个边界框的IoU """
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2
    
    xi1, yi1 = max(x1, x1_p), max(y1, y1_p)
    xi2, yi2 = min(x2, x2_p), min(y2, y2_p)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)
    
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

cap = cv2.VideoCapture("24.mp4")
model = YOLO("yolo11n-seg.pt")
resize_scale = 0.5
conf_threshold = 0.5
iou_threshold = 0.5  # IoU 阈值，可调整

motorcycle_count = 0  # 统计摩托车数量
tracked_moto_ids = set()  # 存储已统计的摩托车ID

while True:
    ret, im0 = cap.read()
    if not ret:
        print("at the end of the video.")
        break
    
    im0 = cv2.resize(im0, (int(im0.shape[1] * resize_scale), int(im0.shape[0] * resize_scale)))
    annotator = Annotator(im0, line_width=3)
    result = model.track(im0, persist=True)
    
    motorcycles = []  # 存储摩托车框
    persons = []  # 存储行人框
    
    if result[0].boxes.id is not None:
        for i, box in enumerate(result[0].boxes):
            label = model.names[int(box.cls[0])]
            bbox = box.xyxy[0].tolist()
            obj_id = box.id.tolist()[0]
            
            if label == "motorcycle":
                motorcycles.append((bbox, obj_id))
                if obj_id not in tracked_moto_ids:
                    tracked_moto_ids.add(obj_id)
                    motorcycle_count += 1  # 统计唯一摩托车数量
            elif label == "person":
                persons.append((bbox, obj_id))
    
    for moto_box, moto_id in motorcycles:
        merged_box = list(moto_box)  # 初始化摩托车框
        for person_box, person_id in persons:
            if compute_iou(moto_box, person_box) > iou_threshold:
                merged_box[0] = min(merged_box[0], person_box[0])
                merged_box[1] = min(merged_box[1], person_box[1])
                merged_box[2] = max(merged_box[2], person_box[2])
                merged_box[3] = max(merged_box[3], person_box[3])
        
        annotator.box_label(merged_box, label=f"Moto-{moto_id}", color=colors(moto_id, True))
    
    # 显示摩托车数量
    cv2.putText(im0, f"Motorcycles: {motorcycle_count}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("frame", im0)
    if cv2.waitKey(1) & 0xff == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()