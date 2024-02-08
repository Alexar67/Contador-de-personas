# Contador de Personas en tiempo real y de un video

Esta es una aplicación desarrollada en Python utilizando **CustomTkinter** que es una biblioteca de interfaz de usuario de Python basada en **Tkinter** para la interfaz gráfica y otras bibliotecas como OpenCV, Pillow y Shapely para la detección y conteo de personas en imágenes y videos.

## Requisitos

- Python 3.x
- OpenCV (cv2)
- Pillow (PIL)
- Shapely
- Ultralytics YOLO
- CustomTkinter
- Tkinter
- NumPy

## Instalación de Dependencias

Puedes instalar las dependencias utilizando pip:

**CustomTkinter**
```
pip3 install customtkinter
```
Puede encontrar la documentación oficial de CustomTkinter aquí:

**➡️ https://customtkinter.tomschimansky.com/documentation**

**Ultralytics**
```
pip install ultralytics
```
**Shapely**
```
pip install shapely
```
**Demás dependencias:**
```
pip install Pillow
pip install opencv-python-headless
pip install numpy
pip install shapely
```
## Ejecución de la Aplicación

Para ejecutar la aplicación, simplemente ejecute el script Python `Contador_de_personas.py`.

***Nota:***
Los demás codigos: `camara_detector_sin interfaz` y `video_detector_sin interfaz` son los códigos por separados sin interfaz y no son necesarios para coprrer el programa principal `Contador_de_personas.py`.

## Funcionalidades

La aplicación cuenta con las siguientes funcionalidades:

- **Conteo de Personas en Tiempo Real**: Utiliza la cámara de tu computadora para detectar y contar personas en tiempo real.
- **Detección y Conteo de Personas en un Video**: Permite cargar un archivo de video para detectar y contar personas en el mismo.
- **Selección de Áreas de Interés**: Permite al usuario seleccionar áreas específicas en la imagen o video donde desea realizar el conteo de personas.
- **Interfaz Gráfica Intuitiva**: La aplicación cuenta con una interfaz gráfica fácil de usar que proporciona instrucciones claras sobre cómo utilizar cada funcionalidad.

## Instrucciones para el conteo de personas en tiempo real

- Haz clic en 'Abrir cámara' para visualizar la imagen de la cámara y seleccionar áreas de detección.
- Utiliza el clic izquierdo para seleccionar una zona.
- Para corregir un trazo incorrecto, utiliza la tecla 'c'.
- Una vez seleccionada un área, para elegir otra, utiliza la tecla 'n'.
- Después de haber seleccionado todas las áreas necesarias, presiona 'Enter' para iniciar el conteo de personas.
- El programa mostrará el recuento de personas en la ventana de la consola, correspondiente a las áreas seleccionadas.
- Para cerrar el programa presionar la tecla 'q'.

## Instrucciones para el conteo de personas de un video seleccionado

- Utiliza los botones 'Seleccionar video' y 'Reproducir video' para elegir el archivo de video y verificar que sea el correcto.
- Con el botón 'Iniciar' se mostrará un fotograma del video para que puedas seleccionar las áreas de interés.
- Utiliza clics izquierdos sucesivos para marcar las áreas deseadas.
- En caso de trazos incorrectos, corrige presionando la tecla 'c'.
- Para seleccionar una nueva área, presiona la tecla 'n'.
- Una vez que hayas marcado todas las áreas necesarias, presiona 'Enter' para iniciar la identificación y conteo de personas y 'q'para cerrar.

¡Disfruta de la experiencia de contar personas con esta aplicación!

