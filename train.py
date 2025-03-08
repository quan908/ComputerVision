from ultralytics import YOLO

model = YOLO("yolo11m.pt")

model.train(data = "config.yaml", imgsz = 640, epochs = 20, batch = 8)