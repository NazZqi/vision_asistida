import cv2
import pyttsx3
import time
import threading
import torch
import numpy as np
import argparse
import sys

# --- Configuración Clave ---
DANGER_DEPTH_THRESHOLD = 0.5
DEFAULT_ANNOUNCEMENT_COOLDOWN = 3.0
OBSTRUCTION_PERCENT_THRESHOLD = 0.1

# --- Argumentos CLI ---
parser = argparse.ArgumentParser(description="Detector de obstáculos por profundidad (MiDaS).")
parser.add_argument("--camera", "-c", type=str, default="0",
                    help="Índice de la cámara (0, 1, ...) o ruta a un archivo de vídeo.")
parser.add_argument("--cooldown", "-d", type=float, default=DEFAULT_ANNOUNCEMENT_COOLDOWN,
                    help="Cooldown en segundos entre anuncios de voz.")
args = parser.parse_args()

# Detectar si se recibió un índice de cámara (int) o una ruta de vídeo (string)
source_arg = args.camera
use_camera_index = False
try:
    camera_index = int(source_arg)
    use_camera_index = True
except Exception:
    camera_index = source_arg  # será tratado como ruta/URL

ANNOUNCEMENT_COOLDOWN = args.cooldown

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")
if use_camera_index:
    print(f"Fuente de vídeo: cámara index={camera_index} | Cooldown de anuncios: {ANNOUNCEMENT_COOLDOWN}s")
else:
    print(f"Fuente de vídeo: archivo='{camera_index}' | Cooldown de anuncios: {ANNOUNCEMENT_COOLDOWN}s")

print("Cargando modelo MiDaS (profundidad)...")
midas_model_type = "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", midas_model_type)
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform
print("Modelos cargados. Iniciando cámara...")

# Abrir fuente: si es índice, pasar int; si es ruta, pasar string
cap = cv2.VideoCapture(camera_index if use_camera_index else camera_index)
if not cap.isOpened():
    if use_camera_index:
        print(f"Error: No se pudo abrir la cámara (index={camera_index}).")
        print("Prueba otro índice de cámara: 0, 1, 2, ...")
    else:
        print(f"Error: No se pudo abrir el archivo de vídeo/URL: {camera_index}")
        print("Comprueba la ruta/URL y los códecs soportados por OpenCV.")
    sys.exit(1)

last_announcement_time = time.time()
last_alert_message = ""

def join_spanish(items):
    items = list(items)
    if not items: return ""
    if len(items) == 1: return items[0]
    if len(items) == 2: return " y ".join(items)
    return ", ".join(items[:-1]) + " y " + items[-1]

def announce_in_thread(text):
    def _worker(t):
        try:
            engine = pyttsx3.init()
            engine.say(t)
            engine.runAndWait()
        except Exception as e:
            print(f"Error en el motor de voz: {e}")
    th = threading.Thread(target=_worker, args=(text,), daemon=True)
    th.start()

print("Iniciando detector de obstáculos por profundidad... Presiona 'q' o Ctrl+C para salir.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al leer fotograma.")
            break

        (frame_height, frame_width) = frame.shape[:2]
        zone_y1 = int(frame_height * 0.25)
        zone_y2 = int(frame_height * 0.75)
        
        zone_width = frame_width // 3
        zone1_x1 = 0
        zone1_x2 = zone_width
        
        zone2_x1 = zone_width
        zone2_x2 = zone_width * 2
        
        zone3_x1 = zone_width * 2
        zone3_x2 = frame_width
        
        zones = {
            "izquierda": (zone1_x1, zone_y1, zone1_x2, zone_y2),
            "al frente": (zone2_x1, zone_y1, zone2_x2, zone_y2),
            "derecha": (zone3_x1, zone_y1, zone3_x2, zone_y2)
        }

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

        alert_messages = []
        overlay = frame.copy()

        for zone_name, (x1, y1, x2, y2) in zones.items():
            depth_zone = depth_normalized[y1:y2, x1:x2]
            dangerous_pixels = np.sum(depth_zone > DANGER_DEPTH_THRESHOLD)
            total_pixels_in_zone = depth_zone.size
            
            obstruction_ratio = dangerous_pixels / total_pixels_in_zone
            
            if obstruction_ratio > OBSTRUCTION_PERCENT_THRESHOLD:
                alert_messages.append(zone_name)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
            else:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
        
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        current_time = time.time()
        
        if alert_messages:
            announcement = "¡Cuidado! Obstáculo " + join_spanish(alert_messages)
            if (current_time - last_announcement_time > ANNOUNCEMENT_COOLDOWN) or (announcement != last_alert_message):
                print(f"ALERTA: {announcement}")
                announce_in_thread(announcement)
                last_announcement_time = current_time
                last_alert_message = announcement
        else:
            last_alert_message = ""

        cv2.imshow("Detector de Obstáculos por Profundidad (Presiona 'q')", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupción recibida. Saliendo...")
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Programa terminado.")