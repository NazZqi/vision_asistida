import cv2
import pyttsx3
import time
import threading
import torch
import numpy as np
import argparse
import sys

# --- Utilidades de dispositivo ---
def select_device(preferred: str = None):
    """
    Devuelve el mejor dispositivo disponible:
    - prioriza CUDA si está disponible
    - luego Metal/MPS (Apple Silicon)
    - luego DirectML (Windows/AMD/Intel, requiere torch-directml)
    - por último CPU
    Si se pasa `preferred`, se intenta respetar (p.ej. "cuda:0", "mps", "directml", "cpu").
    """
    def _mps_available():
        return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available() and torch.backends.mps.is_built()

    def _directml_device():
        try:
            import torch_directml  # type: ignore
            return torch_directml.device()
        except Exception:
            return None

    if preferred:
        normalized = preferred.strip().lower()
        if normalized.startswith("cuda"):
            if torch.cuda.is_available():
                return torch.device(preferred if preferred != "" else "cuda")
            print("[Device] Se solicitó CUDA pero no está disponible, usando la mejor alternativa.")
        elif normalized in ("mps", "metal"):
            if _mps_available():
                return torch.device("mps")
            print("[Device] Se solicitó MPS pero no está disponible, usando la mejor alternativa.")
        elif normalized in ("directml", "dml"):
            dml = _directml_device()
            if dml is not None:
                return dml
            print("[Device] Se solicitó DirectML pero no está disponible, usando la mejor alternativa.")
        elif normalized == "cpu":
            return torch.device("cpu")
        else:
            print(f"[Device] Dispositivo solicitado '{preferred}' no reconocido, usando autodetección.")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_available():
        return torch.device("mps")
    dml = _directml_device()
    if dml is not None:
        return dml
    return torch.device("cpu")

# --- Configuración Clave ---
DANGER_DEPTH_THRESHOLD = 0.5
DEFAULT_ANNOUNCEMENT_COOLDOWN = 3.0
OBSTRUCTION_PERCENT_THRESHOLD = 0.1
YOLO_CONFIDENCE_THRESHOLD = 0.35
APPROACH_DELTA = 0.05  # diferencia mínima en profundidad normalizada para considerar acercamiento
TRACKING_GRID_SIZE = 80  # píxeles para agrupar detecciones cercanas

MONITORED_CLASSES = {
    "person",
    "car",
    "bus",
    "truck",
    "motorcycle",
    "bicycle",
    "traffic light",
    "stop sign",
    "train",
}

CLASS_TRANSLATIONS_ES = {
    "person": "persona",
    "car": "auto",
    "bus": "autobús",
    "truck": "camión",
    "motorcycle": "moto",
    "bicycle": "bicicleta",
    "traffic light": "semáforo",
    "stop sign": "señal de alto",
    "train": "tren",
}

# --- Argumentos CLI ---
# Definir el parser en el ámbito global está bien
parser = argparse.ArgumentParser(description="Detector de obstáculos por profundidad (MiDaS).")
parser.add_argument("--camera", "-c", type=str, default="0",
                    help="Índice de la cámara (0, 1, ...) o ruta a un archivo de vídeo.")
parser.add_argument("--cooldown", "-d", type=float, default=DEFAULT_ANNOUNCEMENT_COOLDOWN,
                    help="Cooldown en segundos entre anuncios de voz.")
parser.add_argument("--device", type=str, default=None,
                    help="Forzar dispositivo (p.ej. 'cuda:0', 'mps', 'directml', 'cpu'). Por defecto autodetecta.")

# --- Funciones (Definiciones) ---
# Estas funciones deben estar fuera para que puedan ser importadas
def join_spanish(items):
    items = list(items)
    if not items: return ""
    if len(items) == 1: return items[0]
    if len(items) == 2: return " y ".join(items)
    return ", ".join(items[:-1]) + " y " + items[-1]

_tts_lock = threading.Lock()
_tts_engine = None


def _get_tts_engine():
    """Inicializa pyttsx3 una sola vez y selecciona voz en español si está disponible."""
    global _tts_engine
    if _tts_engine is not None:
        return _tts_engine
    engine = pyttsx3.init()
    try:
        voices = engine.getProperty("voices") or []
        es_voice_id = None
        for v in voices:
            lang = " ".join(v.languages or [])
            name = v.name or ""
            if "es" in lang.lower() or "span" in name.lower():
                es_voice_id = v.id
                break
        if es_voice_id:
            engine.setProperty("voice", es_voice_id)
    except Exception as e:
        print(f"[TTS] No se pudo seleccionar voz en español: {e}")

    try:
        engine.setProperty("volume", 1.0)
        default_rate = engine.getProperty("rate")
        if default_rate:
            engine.setProperty("rate", max(140, min(190, default_rate)))
    except Exception as e:
        print(f"[TTS] No se pudieron ajustar volumen/velocidad: {e}")

    _tts_engine = engine
    return _tts_engine


def announce_in_thread(text):
    def _worker(t):
        try:
            engine = _get_tts_engine()
            # El lock evita superponer voces cuando llegan mensajes seguidos.
            with _tts_lock:
                engine.say(t)
                engine.runAndWait()
        except Exception as e:
            print(f"Error en el motor de voz: {e}")
    th = threading.Thread(target=_worker, args=(text,), daemon=True)
    th.start()


def direction_from_center(x_center, y_center, frame_width, frame_height):
    """Devuelve una etiqueta corta en español según la posición en la cuadrícula 3x3."""
    zone_width = frame_width // 3
    zone_height = frame_height // 3

    if x_center < zone_width:
        horiz = "izquierda"
    elif x_center < zone_width * 2:
        horiz = "centro"
    else:
        horiz = "derecha"

    if y_center < zone_height:
        vert = "arriba"
    elif y_center < zone_height * 2:
        vert = "centro"
    else:
        vert = "abajo"

    if vert == "centro" and horiz == "centro":
        return "en el centro"
    return f"{vert} {horiz}"


# --- Lógica Principal de Ejecución ---
def main():
    # El código que se ejecuta (como parsear args) va DENTRO de main()
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

    device = select_device(args.device)
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
    print("Cargando modelo YOLOv5 (detección de objetos)...")
    yolo = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    yolo.to(device)
    yolo.eval()

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
    last_object_depths = {}

    print("Iniciando detector de obstáculos por profundidad... Presiona 'q' o Ctrl+C para salir.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error al leer fotograma.")
                break

            (frame_height, frame_width) = frame.shape[:2]
            
            # Cuadrícula 3x3 para analizar toda la imagen
            zone_width = frame_width // 3
            zone_height = frame_height // 3
            zones = {
                "arriba izquierda": (0, 0, zone_width, zone_height),
                "arriba centro": (zone_width, 0, zone_width * 2, zone_height),
                "arriba derecha": (zone_width * 2, 0, frame_width, zone_height),
                "centro izquierda": (0, zone_height, zone_width, zone_height * 2),
                "centro": (zone_width, zone_height, zone_width * 2, zone_height * 2),
                "centro derecha": (zone_width * 2, zone_height, frame_width, zone_height * 2),
                "abajo izquierda": (0, zone_height * 2, zone_width, frame_height),
                "abajo centro": (zone_width, zone_height * 2, zone_width * 2, frame_height),
                "abajo derecha": (zone_width * 2, zone_height * 2, frame_width, frame_height),
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

            object_alerts = []
            try:
                yolo_results = yolo(img_rgb, size=640)
                detections = yolo_results.xyxy[0].cpu().numpy()
                for det in detections:
                    x1, y1, x2, y2, conf, cls_id = det
                    if conf < YOLO_CONFIDENCE_THRESHOLD:
                        continue
                    class_name = yolo_results.names[int(cls_id)]
                    if class_name not in MONITORED_CLASSES:
                        continue

                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    x1 = max(x1, 0)
                    y1 = max(y1, 0)
                    x2 = min(x2, frame_width - 1)
                    y2 = min(y2, frame_height - 1)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    depth_region = depth_normalized[y1:y2, x1:x2]
                    if depth_region.size == 0:
                        continue
                    box_depth_value = float(np.max(depth_region))

                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    tracking_key = f"{class_name}-{center_x // TRACKING_GRID_SIZE}-{center_y // TRACKING_GRID_SIZE}"
                    prev_depth = last_object_depths.get(tracking_key)
                    approaching = prev_depth is not None and (box_depth_value - prev_depth) > APPROACH_DELTA
                    last_object_depths[tracking_key] = box_depth_value

                    is_close = box_depth_value > DANGER_DEPTH_THRESHOLD
                    spanish_label = CLASS_TRANSLATIONS_ES.get(class_name, class_name)
                    direction_label = direction_from_center(center_x, center_y, frame_width, frame_height)

                    label_text = f"{spanish_label} {conf:.0%}"
                    color = (0, 0, 255) if is_close else ((0, 255, 255) if approaching else (0, 255, 0))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label_text, (x1, max(0, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    if is_close or approaching:
                        desc = f"{spanish_label} {direction_label}"
                        if approaching and is_close:
                            desc += " muy cerca y acercándose"
                        elif approaching:
                            desc += " acercándose"
                        else:
                            desc += " muy cerca"
                        object_alerts.append((box_depth_value, desc))
            except Exception as e:
                print(f"Error en detección de objetos: {e}")

            current_time = time.time()
            
            if object_alerts:
                object_alerts.sort(key=lambda x: x[0], reverse=True)
                descriptions = [desc for _, desc in object_alerts]
                announcement = "¡Cuidado! " + join_spanish(descriptions)
                if (current_time - last_announcement_time > ANNOUNCEMENT_COOLDOWN) or (announcement != last_alert_message):
                    print(f"ALERTA: {announcement}")
                    announce_in_thread(announcement)
                    last_announcement_time = current_time
                    last_alert_message = announcement
            elif alert_messages:
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

# Esta línea le dice a Python que solo ejecute main() 
# si el script es llamado directamente (ej. "python vision_asistida.py")
# y NO cuando es importado.
if __name__ == "__main__":
    main()
