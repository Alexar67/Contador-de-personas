# Conteo de Personas en un Video con Detección de Objetos

Este script permite contar personas en un video utilizando detección de objetos mediante el modelo YOLO (You Only Look Once). Selecciona áreas de interés en el video y realiza un seguimiento de las personas dentro de esas áreas.

## Requisitos

- Python 3
- OpenCV
- Numpy
- Shapely
- Ultralytics

## Instalación de Dependencias

```bash
pip install numpy opencv-python-headless shapely 'git+https://github.com/ultralytics/yolov5.git'
