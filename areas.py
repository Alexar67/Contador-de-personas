import cv2
import os
import json
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
import numpy as np

def get_viz(cam,points):
    """funcion para obtener visualizacion con poligonos"""
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
    """Funcion para visulizar la camara"""
    camera_image_visualization = get_viz(camera_image_original,points_cameras)
    cv2.imshow('camera',camera_image_visualization)

def mouse_camera(event, x, y, flags, param):
    """Funcion ligada a la visulizacion de la cámara para realizar callbaks"""
    if event == cv2.EVENT_LBUTTONDOWN: # boton izquierdo del mouse, EVENT_MBUTTONDOWN es el boton del medio (rueda)
        if not camera_index in points_cameras.keys():
            points_cameras[camera_index] = [[x, y]]
        else:
            points_cameras[camera_index].append([x, y])
        visualize_camera()

img_path            = "frame.jpg"
index               = 0
video_filename      = img_path.split(os.sep)[-1][:-4]
camera              = img_path.split(os.sep)[-1][:-4]
points_cameras          = {}

# Load Images
camera_image_original       = cv2.imread(img_path)
camera_image_visualization  = camera_image_original.copy()
camh_, camw_                = camera_image_original.shape[:2]

cv2.namedWindow('camera', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('camera', mouse_camera)

camera_index = 'Area'+f"--{index}"
points_cameras[camera_index]=[]

visualize_camera()

k = cv2.waitKey(0)
while k != ord('q'):
    if k == ord('c'):
        if len(points_cameras[camera_index])>0:
            points_cameras[camera_index] = points_cameras[camera_index][:-1]
        else:
            print("CAMARA no tiene más puntos")
        visualize_camera()
    if k == ord('n'):        
        index+=1
        camera_index = 'Area'+f"--{index}"
        points_cameras[camera_index]=[]
        
    k=cv2.waitKey(0)

cv2.destroyAllWindows()

with open(img_path.replace(".jpg",".json"), 'w') as fp:
        json.dump(points_cameras, fp, indent=4, sort_keys=True)