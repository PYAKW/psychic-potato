from ultralytics import YOLO

#LoadModel
model = YOLO('yolov8n.pt')

#train
model.train(
    data = 'datasetpantograph\data.yaml',
    epochs = 30,
    imgsz = 640,
)   