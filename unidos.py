import cv2
import json
import numpy as np
from shapely.geometry import Polygon, Point
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import random

model = YOLO("models/yolov8n.pt", task="detect")

colors = random.choices(range(256), k=1000)

def draw_results(image, image_results, areas, show_id=False):
    annotator = Annotator(image.copy())
    
    people_in_areas = {area_id: 0 for area_id in areas.keys()}  # Inicializar el contador para cada área
    
    for result in image_results:
        for box in result.boxes:
            b = box.xyxy[0]
            cls = int(box.cls)
            conf = float(box.conf)
            label = f"{model.names[cls]} {round(conf*100, 2)}"
            if show_id and box.id is not None:  
                label += f' id:{int(box.id)}'
            
            # Check if the detected person is inside any of the selected areas
            person_point = Point(box.xyxy[0][0], box.xyxy[0][1])
            inside_area = any(area.contains(person_point) for area in areas.values())
            
            if cls == 0 and conf >= 0.35:
                annotator.box_label(b, label, color=colors[int(box.id):int(box.id)+2] if box.id is not None else None)
                
                # Incrementar el contador solo si la persona está dentro de alguna área
                for area_id, area_polygon in areas.items():
                    if inside_area:
                        people_in_areas[area_id] += 1
    
    for area_id, count in people_in_areas.items():
        print(f"People in Area {area_id}: {count}")
    
    image_annotated = annotator.result()
    return image_annotated

def get_viz(cam, points):
    """Función para obtener visualización con polígonos"""
    cam_vis = cam.copy()
    alpha = 0.5
    overlay = camera_image_original.copy()
    for camid in points_cameras.keys():
        pts = np.array(points_cameras[camid], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(overlay, [pts], True, (0, 0, 255), thickness=3)
    cv2.addWeighted(overlay, alpha, cam_vis, 1 - alpha, 0, cam_vis)
    return cam_vis

def visualize_camera():
    """Función para visualizar la cámara"""
    camera_image_visualization = get_viz(camera_image_original, points_cameras)
    cv2.imshow('camera', camera_image_visualization)

def mouse_camera(event, x, y, flags, param):
    """Función ligada a la visualización de la cámara para realizar callbacks"""
    if event == cv2.EVENT_LBUTTONDOWN: 
        if not camera_index in points_cameras.keys():
            points_cameras[camera_index] = [[x, y]]
        else:
            points_cameras[camera_index].append([x, y])
        visualize_camera()

def count_people_in_areas(frame, areas):
    results_track = model.track(frame, conf=0.40, classes=0, tracker="botsort.yaml", persist=True, verbose=False)
    
    for area_id, area_points in areas.items():
        area_polygon = Polygon(area_points)
        people_in_area = 0
        
        for result in results_track:
            for box in result.boxes:
                person_point = Point(box.xyxy[0][0], box.xyxy[0][1])
                if area_polygon.contains(person_point):
                    people_in_area += 1
        
        print(f"People in Area {area_id}: {people_in_area}")

cap = cv2.VideoCapture(0)  # Use the default camera (change to 1, 2, etc. for additional cameras)

# Obtener el tamaño del frame de la cámara
camw_ = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
camh_ = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cv2.namedWindow('camera', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('camera', mouse_camera)

index = 0
camera_index = 'Area' + f"--{index}"
points_cameras = {}
areas_to_count = {}
count_people = False

while True:
    ret, camera_image_original = cap.read()
    if not ret:
        break

    visualize_camera()

    k = cv2.waitKey(1)
    if k == ord('q'):
        break
    elif k == ord('c'):
        if len(points_cameras[camera_index]) > 0:
            points_cameras[camera_index] = points_cameras[camera_index][:-1]
        else:
            print("La cámara no tiene más puntos")
        visualize_camera()
    elif k == ord('n'):
        index += 1
        camera_index = 'Area' + f"--{index}"
        points_cameras[camera_index] = []
    elif k == 13:  # Enter key
        areas_to_count = {k: Polygon(v) for k, v in points_cameras.items()}  # Convert points to Polygons
        count_people = True

    if count_people:
        count_people_in_areas(camera_image_original, areas_to_count)
        image_annotated = draw_results(camera_image_original, model(camera_image_original), areas_to_count)
        cv2.imshow('Live Camera with Detection', image_annotated)

cap.release()
cv2.destroyAllWindows()
