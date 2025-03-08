from ultralytics import YOLO

model = YOLO("best.pt")
model.predict(source="ragnarok.mp4", show = True,
               save = True,line_width = 2,
              show_labels = True, show_conf= True,
              classes = [0,1], conf = 0.60)
