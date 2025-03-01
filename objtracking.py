import cv2 
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

import numpy as np
import matplotlib.pyplot as plt
 

cap = cv2.VideoCapture("2.mp4")
model = YOLO("yolo11n-seg.pt")

while True:
    ret, im0 = cap.read()

    annotator = Annotator(im0,line_width=3)

    if not ret:
        print("at the end of the video.")
        break

    result = model.track(im0,persist=True)
    if result[0].boxes.id is not None and result[0].masks is not None:
        masks = result[0].masks.xy
        ids = result[0].boxes.id.tolist()
        
        for mask,id in zip(masks,ids):           
            annotator.seg_bbox(mask=mask,mask_color = colors(id,True),label = str(id))
   

    cv2.imshow("frame", im0)
    if cv2.waitKey(1) & 0xff == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()

