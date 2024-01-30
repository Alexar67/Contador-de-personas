import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import random

model = YOLO("models/yolov8n.pt", task="detect")

colors = random.choices(range(256), k=1000)

def draw_results(image, image_results, show_id=False):
    annotator = Annotator(image.copy())
    for box in image_results.boxes:
        b = box.xyxy[0]
        cls = int(box.cls)
        conf = float(box.conf)
        label = f"{model.names[cls]} {round(conf*100, 2)}"
        if show_id:
            label += f' id:{int(box.id)}'
        if cls == 0 and conf >= 0.35:
            annotator.box_label(b, label, color=colors[int(box.id):int(box.id)+2])
    image_annotated = annotator.result()
    return image_annotated

cap = cv2.VideoCapture(0)  # Use the default camera (change to 1, 2, etc. for additional cameras)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results_track = model.track(frame, conf=0.40, classes=0, tracker="botsort.yaml", persist=True, verbose=False)
    image = draw_results(frame, results_track[0], show_id=True)
    
    cv2.imshow('Live Camera with Detection', image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
