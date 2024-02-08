# Importación de módulos necesarios
import customtkinter
import os
from PIL import Image
import cv2
import numpy as np
import random
from shapely.geometry import Polygon, Point
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from tkinter import filedialog

# Configuración de la apariencia de la interfaz personalizada
customtkinter.set_appearance_mode('light')

# Definición de la clase principal de la aplicación
class App(customtkinter.CTk):

    #********************************************* Función para contador de personas en tiempo real **************************************

    # Funcipon principal para iniciar el contador de personas en tiempo real
    def camara(self):
        # Cargar el modelo YOLO para detección de objetos
        model = YOLO("models/yolov8n.pt", task="detect")
        # Colores aleatorios para etiquetar las personas detectadas
        colors = random.choices(range(256), k=1000)

        # Función para dibujar los resultados de detección en la imagen
        def draw_results(image, image_results, areas, show_id=True):
            annotator = Annotator(image.copy())
            
            for result in image_results:
                for box in result.boxes:
                    b = box.xyxy[0]
                    cls = int(box.cls)
                    conf = float(box.conf)
                    label = f"{model.names[cls]} {round(conf*100, 2)}"
                    if show_id and box.id is not None:
                        label += f' id:{int(box.id)}'

                        # Dibujar un cuadro alrededor de la persona detectada
                        if cls == 0 and conf >= 0.35:
                            annotator.box_label(b, label, color=colors[int(box.id):int(box.id)+2] if box.id is not None else None)
            
            # Obtener la imagen con las anotaciones
            image_annotated = annotator.result()
            return image_annotated

        # Función para visualizar la cámara con las áreas seleccionadas
        def get_viz(cam, points_cameras):
            """Función para obtener visualización con polígonos"""
            cam_vis = cam.copy()
            alpha = 0.5
            overlay = cam.copy()
            for camid in points_cameras.keys():
                pts = np.array(points_cameras[camid], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(overlay, [pts], True, (0, 0, 255), thickness=3)
            cv2.addWeighted(overlay, alpha, cam_vis, 1 - alpha, 0, cam_vis)
            return cam_vis

        # Función para visualizar la imagen de la cámara
        def visualize_camera():
            ret, camera_image_original = cap.read()
            if not ret:
                return None

            camera_image_visualization = get_viz(camera_image_original, points_cameras)
            cv2.imshow('video', camera_image_visualization)

        # Función de detección del evento del mouse en la cámara
        def mouse_camera(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN: 
                if not camera_index in points_cameras.keys():
                    points_cameras[camera_index] = [[x, y]]
                else:
                    points_cameras[camera_index].append([x, y])
                visualize_camera()

        # Función para contar personas dentro de las áreas seleccionadas
        def count_people_in_areas(frame, areas):
            results_track = model.track(frame, conf=0.40, classes=0, tracker="botsort.yaml", persist=True, verbose=False)
            text_y_position = 20

            for area_id, area_points in areas.items():
                area_polygon = Polygon(area_points)
                people_in_area = 0
                people_ids = []  # Lista para almacenar las IDs de las personas en el área
                
                for result in results_track:
                    for box in result.boxes:
                        person_point = Point(box.xyxy[0][0], box.xyxy[0][1])
                        if area_polygon.contains(person_point):
                            people_in_area += 1
                            if box.id is not None:
                                people_ids.append(int(box.id))  # Agregar la ID de la persona
                            
                # Mostrar el recuento de personas y sus IDs en la pantalla del video y en concola            
                print(f"Personas en el {area_id}: {people_in_area}   IDs: {people_ids}")
                text = f"Personas en el {area_id}: {people_in_area}   IDs: {people_ids}"
                cv2.putText(frame, text, (10, text_y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
                # Incrementar la posición vertical para el próximo área
                text_y_position += 20

        cap = cv2.VideoCapture(0)  # Usar la cámara del computador (puede necesitar ajustes dependiendo del número de cámara)

        # Capturar la imagen inicial de la cámara
        ret, camera_image_original = cap.read()
        if not ret:
            print("Error al leer la cámara.")
            exit()

        # Configuración de la ventana de video y del evento del mouse
        cv2.namedWindow('video', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('video', 800, 600)  
        cv2.setMouseCallback('video', mouse_camera)

        index = 0
        camera_index = 'Area' + f"-{index}"
        points_cameras = {}
        areas_to_count = {}
        count_people = False

        # Bucle para seleccionar áreas de interés en la cámara
        while True:
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
                camera_index = 'Area' + f"-{index}"
                points_cameras[camera_index] = []
            elif k == 13:  # Enter key
                areas_to_count = {k: Polygon(v) for k, v in points_cameras.items()}
                areas_to_draw = list(areas_to_count.values())
                count_people = True

            if count_people:
                break
        
        # Bucle para detectar y contar personas en tiempo real, ademas indica el nombre del área en cada área
        while True:
            ret, camera_image_original = cap.read()
            if not ret:
                break

            visualize_camera()

            count_people_in_areas(camera_image_original, areas_to_count)
            image_annotated = draw_results(camera_image_original, model(camera_image_original), areas_to_count)
            
            for area_id, area_polygon in areas_to_count.items():
                pts = np.array(area_polygon.exterior.coords, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(image_annotated, [pts], True, (0, 255, 0), thickness=2)
                
                centroid = np.array(area_polygon.centroid.coords[0], np.int32)
                cv2.putText(image_annotated, f' {area_id}', tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            cv2.imshow('video', image_annotated)

            k = cv2.waitKey(1)
            if k == ord('q'):
                break

        # Liberar la cámara y cerrar todas las ventanas        
        cap.release()
        cv2.destroyAllWindows()

    #********************************************* Función para contador de personas de un video **************************************

    # Funcipon principal para iniciar el contador de personas en un video
    def video(self):
        # Cargar el modelo YOLO para detección de objetos
        model = YOLO("models/yolov8n.pt", task="detect")
        # Colores aleatorios para etiquetar las personas detectadas
        colors = random.choices(range(256), k=1000)
        # Diccionario para almacenar las áreas a contar
        areas_to_count = {}  
        # Lista para almacenar las áreas dibujadas
        areas_to_draw = [] 
        # Bandera para controlar el conteo de personas
        count_people = False

        # Crear la ventana con un nombre específico
        cv2.namedWindow('video', cv2.WINDOW_NORMAL)

        # Especificar la posición y el tamaño de la ventana
        cv2.moveWindow('video', 100, 100)  # Cambia los valores según tus necesidades
        cv2.resizeWindow('video', 800, 600)  # Cambia los valores según tus necesidades

        # Función para dibujar los resultados de detección en la imagen
        def draw_results(image, image_results, areas, show_id=True):
            annotator = Annotator(image.copy())
            
            for result in image_results:
                for box in result.boxes:
                    b = box.xyxy[0]
                    cls = int(box.cls)
                    conf = float(box.conf)
                    label = f"{model.names[cls]} {round(conf*100, 2)}"
                    if show_id and box.id is not None:  
                        label += f' id:{int(box.id)}'
                        
                        # Dibujar un cuadro alrededor de la persona detectada
                        if cls == 0 and conf >= 0.35:
                            annotator.box_label(b, label, color=colors[int(box.id):int(box.id)+2] if box.id is not None else None)

            # Obtener la imagen con las anotaciones
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
            cv2.imshow('video', camera_image_visualization)

        def mouse_camera(event, x, y, flags, param):
            """Función ligada a la visualización de la cámara para realizar callbacks"""
            if event == cv2.EVENT_LBUTTONDOWN: 
                if not camera_index in points_cameras.keys():
                    points_cameras[camera_index] = [[x, y]]
                else:
                    points_cameras[camera_index].append([x, y])
                visualize_camera()

        # Función para contar personas dentro de las áreas seleccionadas
        def count_people_in_areas(frame, areas):
            results_track = model.track(frame, conf=0.40, classes=0, tracker="botsort.yaml", persist=True, verbose=False)
            text_y_position = 20

            for area_id, area_points in areas.items():
                area_polygon = Polygon(area_points)
                people_in_area = 0
                people_ids = []  # Lista para almacenar las IDs de las personas en el área
                
                for result in results_track:
                    for box in result.boxes:
                        person_point = Point(box.xyxy[0][0], box.xyxy[0][1])
                        if area_polygon.contains(person_point):
                            people_in_area += 1
                            if box.id is not None:
                                people_ids.append(int(box.id))  # Agregar la ID de la persona

                # Mostrar el recuento de personas y sus IDs en la pantalla del video y en consola            
                print(f"Personas en el {area_id}: {people_in_area}   IDs: {people_ids}")
                text = f"Personas en el {area_id}: {people_in_area}   IDs: {people_ids}"
                cv2.putText(frame, text, (10, text_y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
                # Incrementar la posición vertical para el próximo área
                text_y_position += 20

        global video
        # Ruta del video a procesar
        video_path = video  
        cap = cv2.VideoCapture(video_path)

        # Leer el primer frame para la selección de áreas
        ret, camera_image_original = cap.read()
        if not ret:
            print("Error al leer el primer frame del video.")
            exit()

        cv2.namedWindow('video', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('video', mouse_camera)

        index = 0
        camera_index = 'Area' + f"-{index}"
        points_cameras = {}
        areas_to_count = {}
        count_people = False

        while True:
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
                camera_index = 'Area' + f"-{index}"
                points_cameras[camera_index] = []
            elif k == 13:  # Enter key
                areas_to_count = {k: Polygon(v) for k, v in points_cameras.items()}  # Convert points to Polygons
                areas_to_draw = list(areas_to_count.values())  # Almacena las áreas dibujadas
                count_people = True

            if count_people:
                break

        # Continuar con la detección de personas en las áreas seleccionadas
        while True:
            ret, camera_image_original = cap.read()
            if not ret:
                break

            visualize_camera()

            count_people_in_areas(camera_image_original, areas_to_count)
            image_annotated = draw_results(camera_image_original, model(camera_image_original), areas_to_count)
            
            # Dibujar las áreas seleccionadas continuamente
            for area_id, area_polygon in areas_to_count.items():
                pts = np.array(area_polygon.exterior.coords, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(image_annotated, [pts], True, (0, 255, 0), thickness=2)
                
                # Agregar texto de identificación cerca del área
                centroid = np.array(area_polygon.centroid.coords[0], np.int32)
                cv2.putText(image_annotated, f' {area_id}', tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            cv2.imshow('video', image_annotated)

            k = cv2.waitKey(1)
            if k == ord('q'):
                break
        # Liberar la cámara y cerrar todas las ventanas            
        cap.release()
        cv2.destroyAllWindows()

    # Función para reproducir el video seleccionado
    def reproducir_video(self):
        global video
        if video:
            self.label_info_video.configure(text=f"Reproduciendo: {os.path.basename(video)}")
            os.startfile(video)

    # Función para seleccionar un video de la galería
    def seleccionar_video(self):
        global video
        ruta_video = filedialog.askopenfilename(filetypes=[("Videos", "*.mp4;*.avi;*.webm")])
        if ruta_video:
            video = ruta_video  # Almacena la ruta del video seleccionado
            self.label_info_video.configure(text=f"Video seleccionado: {os.path.basename(video)}")
            self.third_frame_button_2.configure(state="normal")

#********************************************VENTANA PRINCIPAL*************************************************************
    # Dimensiones predeterminadas de la ventana principal
    width = 700
    height = 450
    
    # Método de inicialización de la clase
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
         # Configuración inicial de la ventana principal
        self.title("Contador de Personas")
        self.geometry(f"{self.width}x{self.height}")
        self.resizable(False, False)
    
        # Configuración del diseño de cuadrícula: 1 fila, 2 columnas
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # Cargar las imágenes para el modo oscuro y para el modo claro
        image_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_images")
        self.caratula_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "caratula.png")), size=(430, 430))

        self.logo_image = customtkinter.CTkImage(light_image=Image.open(os.path.join(image_path, "logo.png")),dark_image=Image.open(os.path.join(image_path, "logow.png")), size=(20, 20))
        self.chat_image = customtkinter.CTkImage(light_image=Image.open(os.path.join(image_path, "info.png")),dark_image=Image.open(os.path.join(image_path, "infow.png")), size=(20, 20))
        self.imagen_image = customtkinter.CTkImage(light_image=Image.open(os.path.join(image_path, "camara.png")),dark_image=Image.open(os.path.join(image_path, "camaraw.png")), size=(20, 20))
        self.audio_image = customtkinter.CTkImage(light_image=Image.open(os.path.join(image_path, "videos.png")),dark_image=Image.open(os.path.join(image_path, "videosw.png")), size=(20, 20))

        # Crear el frame de navegación
        self.navigation_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.navigation_frame.grid(row=0, column=0, sticky="nsew")
        self.navigation_frame.grid_rowconfigure(7, weight=1)

        # Creación y configuración del label en el frame de navegación
        self.navigation_frame_label = customtkinter.CTkLabel(self.navigation_frame, text="  Menú de opciones", image=self.logo_image, compound="left", font=customtkinter.CTkFont(size=15, weight="bold"))
        self.navigation_frame_label.grid(row=0, column=0, padx=20, pady=20)

        # Creación de botones en el frame de navegación
        self.home_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Información",fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),image=self.chat_image, anchor="w", command=self.home_button_event)
        self.home_button.grid(row=1, column=0, sticky="ew")

        self.frame_2_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Contador en tiempo real", fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),image=self.imagen_image, anchor="w", command=self.frame_2_button_event) 
        self.frame_2_button.grid(row=2, column=0, sticky="ew")

        self.frame_3_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Contador de un video",fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),image=self.audio_image, anchor="w", command=self.frame_3_button_event)
        self.frame_3_button.grid(row=3, column=0, sticky="ew")
        
        #Apariencia de la ventana
        self.appearance_mode_menu = customtkinter.CTkOptionMenu(self.navigation_frame, values=["Light", "Dark", "System"],command=self.change_appearance_mode_event)
        self.appearance_mode_menu.grid(row=7, column=0, padx=20, pady=20, sticky="s")

        #************************************* Frame de información de la App ********************************************************

        # Creación del frame de información
        self.home_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.home_frame.grid_columnconfigure(0, weight=1)

        # Creación del label en el frame de información para mostrar la imagen de la carátula
        self.home_frame_large_image_label = customtkinter.CTkLabel(self.home_frame, text="", image=self.caratula_image)
        self.home_frame_large_image_label.grid(row=0, column=0, padx=20, pady=10)

        #********************************* Frame para el Contador de personas en tiempo real ************************************************

        #Creación y configuraciones del frame Contador de personas en tiempo real
        self.second_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")

        # Configuración de la etiqueta del título
        self.label_title = customtkinter.CTkLabel(self.second_frame, text="Contador de personas en tiempo real", anchor="center", justify="center", font=customtkinter.CTkFont(size=13, weight="bold"))
        self.label_title.grid(row=1, column=0, padx=20, pady=(10, 0))

        # Configuración de la etiqueta de las instrucciones
        self.label_instructions = customtkinter.CTkLabel(self.second_frame, text="Instrucciones:", anchor="w", justify="left", font=customtkinter.CTkFont(size=13, weight="bold"))
        self.label_instructions.grid(row=2, column=0, padx=10, pady=(10, 0))

        # Añadir las instrucciones
        instructions = [
            "- Haz clic en 'Abrir cámara' para visualizar la imagen de la cámara y seleccionar áreas de detección.",
            "- Utiliza el clic izquierdo para seleccionar una zona.",
            "- Para corregir un trazo incorrecto, utiliza la tecla 'c'.",
            "- Una vez seleccionada un área, para elegir otra, utiliza la tecla 'n'.",
            "- Después de haber seleccionado todas las áreas necesarias, presiona 'Enter' para iniciar el conteo de personas.",
            "- El programa mostrará el recuento de personas en la ventana de la consola, correspondiente a las áreas seleccionadas.",
            "- Para cerrar la ventana presionar la tecla 'q'."
            ]

        # Agregar las instrucciones al texto de la etiqueta
        instructions_text = "\n\n".join(instructions)
        self.label_pasos = customtkinter.CTkLabel(self.second_frame, text=instructions_text, anchor="w", justify="left", wraplength=400)
        self.label_pasos.grid(row=3, column=0, padx=50, pady=(10, 0))

        # Creación y configuración del botón rara abrir la cámara
        self.second_frame_button_1 = customtkinter.CTkButton(self.second_frame, text="Abrir camara", command=self.camara)
        self.second_frame_button_1.grid(row=5, column=0, padx=150, pady=(50,0), sticky="ew", columnspan=2)

        #************************************ Frame para el Contador de personas de un video *****************************************************

        #Creación y configuraciones del frame Contador de personas de un video
        self.third_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.third_frame.grid_columnconfigure(0, weight=1)

        # Configuración de la etiqueta del título
        self.label_title = customtkinter.CTkLabel(self.third_frame, text="Conteo de personas de un video", anchor="center", justify="center", font=customtkinter.CTkFont(size=13, weight="bold"))
        self.label_title.grid(row=1, column=0, padx=20, pady=(10, 0), columnspan=2)

        # Configuración de la etiqueta de las instrucciones
        self.label_instructions = customtkinter.CTkLabel(self.third_frame, text="Instrucciones:", anchor="w", justify="left", font=customtkinter.CTkFont(size=13, weight="bold"))
        self.label_instructions.grid(row=2, column=0, padx=10, pady=(10, 0))

        # Añadir las instrucciones
        instructions2 = [
            "- Utiliza los botones 'Seleccionar video' y 'Reproducir video' para elegir el archivo de video y verificar que sea el correcto.",
            "- Con el botón 'Iniciar' se mostrará un fotograma del video para que puedas seleccionar las áreas de interés.",
            "- Utiliza clics izquierdos sucesivos para marcar las áreas deseadas.",
            "- En caso de trazos incorrectos, corrige presionando la tecla 'c'.",
            "- Para seleccionar una nueva área, presiona la tecla 'n'.",
            "- Una vez que hayas marcado todas las áreas necesarias, presiona 'Enter' para iniciar la identificación y conteo de personas y 'q' para cerrar."
        ]
        
        # Agregar las instrucciones al texto de la etiqueta
        instructions_text = "\n\n".join(instructions2)
        self.label_pasos = customtkinter.CTkLabel(self.third_frame, text=instructions_text, anchor="w", justify="left", wraplength=400)
        self.label_pasos.grid(row=3, column=0, padx=30, pady=(10, 0), columnspan=2)

        # Agragar los botones de seleccionar video, reproducir e iniciar
        self.label_info_video = customtkinter.CTkLabel(self.third_frame, text="Seleccione un video ", anchor="w", font=customtkinter.CTkFont(size=13, weight="bold"))
        self.label_info_video.grid(row=4, column=0, padx=20, pady=(10, 0), columnspan=2)

        self.third_frame_button_1 = customtkinter.CTkButton(self.third_frame, text="Seleccionar video", command=self.seleccionar_video)
        self.third_frame_button_1.grid(row=5, column=0, padx=(50, 10), pady=(10, 0), sticky="ew")

        self.third_frame_button_2 = customtkinter.CTkButton(self.third_frame, command=self.reproducir_video)
        self.third_frame_button_2.grid(row=5, column=1, padx=(0, 50), pady=(10, 0), sticky="ew")
        self.third_frame_button_2.configure(state="disabled", text="Reproducir video")

        self.third_frame_button_3 = customtkinter.CTkButton(self.third_frame, text="Iniciar", command=self.video)
        self.third_frame_button_3.grid(row=6, column=0, padx=50, pady=(10, 0), sticky="ew", columnspan=2)

        # Frame por defecto home que es el de información
        self.select_frame_by_name("home")

    # Función para seleccionar un frame según el nombre dado
    def select_frame_by_name(self, name):
        # Configuración de color de botones según el frame seleccionado
        self.home_button.configure(fg_color=("gray75", "gray25") if name == "home" else "transparent")
        self.frame_2_button.configure(fg_color=("gray75", "gray25") if name == "frame_2" else "transparent")
        self.frame_3_button.configure(fg_color=("gray75", "gray25") if name == "frame_3" else "transparent")

        # Moastrar el frame seleccionado
        if name == "home":
            self.home_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.home_frame.grid_forget()
        if name == "frame_2":
            self.second_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.second_frame.grid_forget()
        if name == "frame_3":
            self.third_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.third_frame.grid_forget()

    # Manejadores de eventos para los botones del menú de opciones
    def home_button_event(self):
        self.select_frame_by_name("home")

    def frame_2_button_event(self):
        self.select_frame_by_name("frame_2")

    def frame_3_button_event(self):
        self.select_frame_by_name("frame_3")

    # Función para cambiar la apariencia de la aplicación (modo claro, oscuro o sistema)
    def change_appearance_mode_event(self, new_appearance_mode):
        customtkinter.set_appearance_mode(new_appearance_mode)

# Punto de entrada principal de la aplicación
if __name__ == "__main__":
    app = App()
    app.mainloop()