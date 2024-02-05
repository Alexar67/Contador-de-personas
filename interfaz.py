import customtkinter
import os
from PIL import Image
import cv2
import numpy as np
from shapely.geometry import Polygon, Point
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import random
from shapely.geometry import Polygon, Point
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

customtkinter.set_appearance_mode('dark')
model = YOLO("models/yolov8n.pt", task="detect")
colors = random.choices(range(256), k=1000)

class App(customtkinter.CTk):

    # Funciones necesarias para el conteo de personas en tiempo real con la cámara del computador
    def draw_results(self, image, image_results, areas, show_id=True):
        annotator = Annotator(image.copy())
        
        people_in_areas = {area_id: [] for area_id in areas.keys()}
        
        for result in image_results:
            for box in result.boxes:
                b = box.xyxy[0]
                cls = int(box.cls)
                conf = float(box.conf)
                label = f"{model.names[cls]} {round(conf*100, 2)}"
                if show_id and box.id is not None:
                    label += f' id:{int(box.id)}'
                
                    person_point = Point(box.xyxy[0][0], box.xyxy[0][1])
                    inside_area = any(area.contains(person_point) for area in areas.values())
                    
                    if cls == 0 and conf >= 0.35:
                        annotator.box_label(b, label, color=colors[int(box.id):int(box.id)+2] if box.id is not None else None)
                        
                        for area_id, area_polygon in areas.items():
                            if inside_area:
                                people_in_areas[area_id].append(int(box.id))
        
        for area_id, ids in people_in_areas.items():
            print(f"People in Area {area_id}: {len(ids)} with IDs {ids}")
        
        image_annotated = annotator.result()
        return image_annotated

    def get_viz(self, cam, points_cameras):
        cam_vis = cam.copy()
        alpha = 0.5
        overlay = cam.copy()
        for camid in self.points_cameras.keys():
            pts = np.array(self.points_cameras[camid], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(overlay, [pts], True, (0, 0, 255), thickness=3)
        cv2.addWeighted(overlay, alpha, cam_vis, 1 - alpha, 0, cam_vis)
        return cam_vis

    def visualize_camera(self):
        ret, camera_image_original = self.cap.read()
        if not ret:
            return None

        camera_image_visualization = self.get_viz(camera_image_original, self.points_cameras)
        cv2.imshow('video', camera_image_visualization)

    def mouse_camera(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN: 
            if not self.camera_index in self.points_cameras.keys():
                self.points_cameras[self.camera_index] = [[x, y]]
            else:
                self.points_cameras[self.camera_index].append([x, y])
            self.visualize_camera()

    def count_people_in_areas(self, frame, areas):
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

    def camara(self):

        cv2.namedWindow('video', cv2.WINDOW_NORMAL)
        # Especificar la posición y el tamaño de la ventana
        cv2.resizeWindow('video', 800, 600)  # Cambia los valores según tus necesidades
        cv2.setMouseCallback('video', self.mouse_camera)
        self.cap = cv2.VideoCapture(0)  # Usar la cámara del computador (puede necesitar ajustes dependiendo del número de cámara)

        camw_ = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        camh_ = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        ret, camera_image_original = self.cap.read()
        if not ret:
            print("Error al leer la cámara.")
            exit()

        index = 0
        self.camera_index = 'Area' + f"--{index}"
        self.points_cameras = {}
        areas_to_count = {}
        count_people = False

        while True:
            self.visualize_camera()

            k = cv2.waitKey(1)
            if k == ord('q'):
                break
            elif k == ord('c'):
                if len(self.points_cameras[self.camera_index]) > 0:
                    self.points_cameras[self.camera_index] = self.points_cameras[self.camera_index][:-1]
                else:
                    print("La cámara no tiene más puntos")
                self.visualize_camera()
            elif k == ord('n'):
                index += 1
                self.camera_index = 'Area' + f"--{index}"
                self.points_cameras[self.camera_index] = []
            elif k == 13:  # Enter key
                areas_to_count = {k: Polygon(v) for k, v in self.points_cameras.items()}
                areas_to_draw = list(areas_to_count.values())
                count_people = True

            if count_people:
                break

        while True:
            ret, camera_image_original = self.cap.read()
            if not ret:
                break

            self.visualize_camera()

            self.count_people_in_areas(camera_image_original, areas_to_count)
            image_annotated = self.draw_results(camera_image_original, model(camera_image_original), areas_to_count)
            
            for area_id, area_polygon in areas_to_count.items():
                pts = np.array(area_polygon.exterior.coords, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(image_annotated, [pts], True, (0, 255, 0), thickness=2)
                
                centroid = np.array(area_polygon.centroid.coords[0], np.int32)
                cv2.putText(image_annotated, f'Area {area_id}', tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            cv2.imshow('video', image_annotated)

            k = cv2.waitKey(1)
            if k == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    width = 700
    height = 450
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.title("Contador de Personas")
        self.geometry(f"{self.width}x{self.height}")
        self.resizable(False, False)
    
        # set grid layout 1x2
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # load images with light and dark mode image
        image_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_images")
        self.logo_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "blue.png")), size=(26, 26))
        self.mp3 = customtkinter.CTkImage(Image.open(os.path.join(image_path, "audio.jpg")), size=(150, 200))

        self.chat_image = customtkinter.CTkImage(light_image=Image.open(os.path.join(image_path, "chat_dark.png")),dark_image=Image.open(os.path.join(image_path, "chat_light.png")), size=(20, 20))
        self.imagen_image = customtkinter.CTkImage(light_image=Image.open(os.path.join(image_path, "imagenes.png")),dark_image=Image.open(os.path.join(image_path, "imagenesw.png")), size=(20, 20))
        self.audio_image = customtkinter.CTkImage(light_image=Image.open(os.path.join(image_path, "audio.png")),dark_image=Image.open(os.path.join(image_path, "audiow.png")), size=(20, 20))
        self.video_image = customtkinter.CTkImage(light_image=Image.open(os.path.join(image_path, "videos.png")),dark_image=Image.open(os.path.join(image_path, "videosw.png")), size=(20, 20))

        # create navigation frame
        self.navigation_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.navigation_frame.grid(row=0, column=0, sticky="nsew")
        self.navigation_frame.grid_rowconfigure(7, weight=1)

        #frame de menu de opciones
        self.navigation_frame_label = customtkinter.CTkLabel(self.navigation_frame, text="  Menú de opciones", image=self.logo_image, compound="left", font=customtkinter.CTkFont(size=15, weight="bold"))
        self.navigation_frame_label.grid(row=0, column=0, padx=20, pady=20)

        #OPCIONES DEL MENÚ
        self.home_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Información",fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),image=self.chat_image, anchor="w", command=self.home_button_event)
        self.home_button.grid(row=1, column=0, sticky="ew")

        self.frame_2_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Video Detección", fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),image=self.imagen_image, anchor="w", command=self.frame_2_button_event) 
        self.frame_2_button.grid(row=2, column=0, sticky="ew")

        self.frame_3_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Subir video",fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),image=self.audio_image, anchor="w", command=self.frame_3_button_event)
        self.frame_3_button.grid(row=3, column=0, sticky="ew")

        self.frame_4_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Datos",fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),image=self.video_image, anchor="w", command=self.frame_4_button_event)
        self.frame_4_button.grid(row=4, column=0, sticky="ew")
        
        #apariencia de la ventana
        self.appearance_mode_menu = customtkinter.CTkOptionMenu(self.navigation_frame, values=["Light", "Dark", "System"],command=self.change_appearance_mode_event)
        self.appearance_mode_menu.grid(row=7, column=0, padx=20, pady=20, sticky="s")

        # Frame de información de la App ****************************************************************************************************+
        self.home_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        

        # Frame para la detección del video *************************************************************************************+
        self.second_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")

        # Configuración de la etiqueta del título
        self.label_title = customtkinter.CTkLabel(self.second_frame, text="Conteo de personas", anchor="center", justify="center")
        self.label_title.grid(row=1, column=0, padx=20, pady=(10, 0))

        # Configuración de la etiqueta de las instrucciones
        self.label_instructions = customtkinter.CTkLabel(self.second_frame, text="Instrucciones:", anchor="w", justify="center")
        self.label_instructions.grid(row=2, column=0, padx=20, pady=(10, 0))

        # Añadir las instrucciones
        instructions = [
            "- Haz clic en 'Abrir cámara' para visualizar la imagen de la cámara y seleccionar áreas de detección.",
            "- Utiliza el clic izquierdo para seleccionar una zona.",
            "- Para corregir un trazo incorrecto, utiliza la tecla 'c'.",
            "- Una vez seleccionada un área, para elegir otra, utiliza la tecla 'n'.",
            "- Después de haber seleccionado todas las áreas necesarias, presiona 'Enter' para iniciar el conteo de personas.",
            "- El programa mostrará el recuento de personas en la ventana de la consola, correspondiente a las áreas seleccionadas.",
            "- Para cerrar el programa presionar la tecla 'q'."
            ]

        # Agregar las instrucciones al texto de la etiqueta
        instructions_text = "\n\n".join(instructions)
        self.label_pasos = customtkinter.CTkLabel(self.second_frame, text=instructions_text, anchor="w", justify="left", wraplength=400)
        self.label_pasos.grid(row=3, column=0, padx=50, pady=(10, 0))

        self.second_frame_button_1 = customtkinter.CTkButton(self.second_frame, text="Abrir camara", command=self.camara)
        self.second_frame_button_1.grid(row=5, column=0, padx=150, pady=(50,0), sticky="ew", columnspan=2)


        # Frame para la detección de personas en un video *************************************************************************************+
        self.third_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.third_frame.grid_rowconfigure(0, weight=1)
        self.third_frame.grid_columnconfigure(0, weight=1)

        self.video_mostrar = None

        self.label_info_video = customtkinter.CTkLabel(self.third_frame, text="Seleccione un video ", anchor="w")
        self.label_info_video.grid(row=1, column=0, padx=20, pady=(10, 0))

        self.third_frame_button_1 = customtkinter.CTkButton(self.third_frame, text="Seleccionar video")
        self.third_frame_button_1.grid(row=2, column=0, padx=20, pady=(20,0), sticky="ew")

        self.third_frame_button_2 = customtkinter.CTkButton(self.third_frame)
        self.third_frame_button_2.grid(row=3, column=0, padx=20, pady=(20,20), sticky="ew")
        self.third_frame_button_2.configure(state="disabled", text="Reproducir video")

        self.third_frame_button_3 = customtkinter.CTkButton(self.third_frame, text="Enviar video")
        self.third_frame_button_3.grid(row=3, column=1, padx=20, pady=20, sticky="nsew")

        # audio frame **************************************************************************************
        self.cuarto_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")

        # select default frame
        self.select_frame_by_name("home")

    def select_frame_by_name(self, name):
        # set button color for selected button
        self.home_button.configure(fg_color=("gray75", "gray25") if name == "home" else "transparent")
        self.frame_2_button.configure(fg_color=("gray75", "gray25") if name == "frame_2" else "transparent")
        self.frame_3_button.configure(fg_color=("gray75", "gray25") if name == "frame_3" else "transparent")
        self.frame_4_button.configure(fg_color=("gray75", "gray25") if name == "frame_4" else "transparent")

        # show selected frame
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
        if name == "frame_4":
            self.cuarto_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.cuarto_frame.grid_forget()

    def home_button_event(self):
        self.select_frame_by_name("home")

    def frame_2_button_event(self):
        self.select_frame_by_name("frame_2")

    def frame_3_button_event(self):
        self.select_frame_by_name("frame_3")

    def frame_4_button_event(self):
        self.select_frame_by_name("frame_4")

    def change_appearance_mode_event(self, new_appearance_mode):
        customtkinter.set_appearance_mode(new_appearance_mode)


if __name__ == "__main__":
    app = App()
    app.mainloop()