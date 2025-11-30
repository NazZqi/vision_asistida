import cv2
import torch
import numpy as np
import time
from vision_asistida import select_device

# --- 1. CONFIGURACIÓN DE ARCHIVOS Y PARÁMETROS ---
INPUT_VIDEO_PATH = "VID_20251022_170843.mp4"   # <-- ¡Poner el nombre de tu video aquí!
OUTPUT_VIDEO_PATH = "resultado_profundidad.mp4" # <-- Nombre del video que se creará

# Parámetros del detector (ajústalos como en el script anterior)
DANGER_DEPTH_THRESHOLD = 0.5
OBSTRUCTION_PERCENT_THRESHOLD = 0.1
# ---------------------------------------------------

# --- 2. Configuración Inicial (Modelos e IA) ---
device = select_device()
print(f"Usando dispositivo: {device}")

print("Cargando modelo MiDaS (profundidad)...")
midas_model_type = "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", midas_model_type)
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform
print("Modelos cargados.")

# --- 3. Configuración del Lector y Escritor de Video ---
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: No se pudo abrir el video '{INPUT_VIDEO_PATH}'")
    exit()

# Obtener propiedades del video de entrada
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Definir el codec y crear el objeto VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

print(f"Procesando video: {INPUT_VIDEO_PATH} ({total_frames} fotogramas)")
print(f"Guardando resultado en: {OUTPUT_VIDEO_PATH}")

frame_count = 0
start_time = time.time()

# --- 4. Bucle Principal de Procesamiento ---
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("...Procesamiento completado.")
            break # Termina el bucle si no hay más fotogramas

        frame_count += 1
        
        # --- Lógica de Detección (idéntica al script anterior) ---
        
        # 1. Definir Zonas
        (frame_height, frame_width) = frame.shape[:2]
        zone_y1 = int(frame_height * 0.25)
        zone_y2 = int(frame_height * 0.75)
        
        zone_width = frame_width // 3
        zone1_x1, zone1_x2 = 0, zone_width
        zone2_x1, zone2_x2 = zone_width, zone_width * 2
        zone3_x1, zone3_x2 = zone_width * 2, frame_width
        
        zones = {
            "izquierda": (zone1_x1, zone_y1, zone1_x2, zone_y2),
            "al frente": (zone2_x1, zone_y1, zone2_x2, zone_y2),
            "derecha": (zone3_x1, zone_y1, zone3_x2, zone_y2)
        }
        
        # 2. Ejecutar MiDaS
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = transform(img_rgb).to(device)
        with torch.no_grad():
            prediction = midas(input_batch)
            depth_map = torch.nn.functional.interpolate(
                prediction.unsqueeze(1), size=frame.shape[:2], mode="bicubic", align_corners=False,
            ).squeeze()
        
        depth_map = depth_map.cpu().numpy()
        
        depth_min, depth_max = depth_map.min(), depth_map.max()
        if depth_max - depth_min > 0:
            depth_normalized = (depth_map - depth_min) / (depth_max - depth_min)
        else:
            depth_normalized = np.zeros(depth_map.shape)

        # 3. Analizar Zonas y Dibujar
        overlay = frame.copy()

        for zone_name, (x1, y1, x2, y2) in zones.items():
            depth_zone = depth_normalized[y1:y2, x1:x2]
            
            dangerous_pixels = np.sum(depth_zone > DANGER_DEPTH_THRESHOLD)
            total_pixels_in_zone = depth_zone.size
            
            # Evitar división por cero si la zona es de tamaño 0
            if total_pixels_in_zone == 0:
                obstruction_ratio = 0
            else:
                obstruction_ratio = dangerous_pixels / total_pixels_in_zone
            
            if obstruction_ratio > OBSTRUCTION_PERCENT_THRESHOLD:
                # Zona de Peligro (ROJO)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
            else:
                # Zona Segura (VERDE)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
        
        # Mezclar la capa de visualización con el frame original
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # --- 5. Escribir el fotograma procesado en el video de salida ---
        video_writer.write(frame)

        # Imprimir progreso en la consola
        if frame_count % 100 == 0:
            print(f"  Procesado fotograma {frame_count} / {total_frames}")

except KeyboardInterrupt:
    print("Interrupción recibida. Finalizando video...")

finally:
    # --- 6. Limpieza y guardado ---
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    
    end_time = time.time()
    print("¡Video de demostración creado exitosamente!")
    print(f"Archivo guardado en: {OUTPUT_VIDEO_PATH}")
    print(f"Tiempo total de procesamiento: {end_time - start_time:.2f} segundos.")
