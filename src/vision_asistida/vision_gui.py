import sys
import time

import cv2
import torch
import numpy as np
import requests  # geocodificar + pasos de ruta
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QSizePolicy,
    QDialog,
    QListWidget,
    QInputDialog,
    QLineEdit,
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon
from PyQt5.QtWebEngineWidgets import QWebEngineView
import qtawesome as qta

from vision_asistida import (
    DANGER_DEPTH_THRESHOLD,
    DEFAULT_ANNOUNCEMENT_COOLDOWN,
    OBSTRUCTION_PERCENT_THRESHOLD,
    YOLO_CONFIDENCE_THRESHOLD,
    APPROACH_DELTA,
    TRACKING_GRID_SIZE,
    MONITORED_CLASSES,
    CLASS_TRANSLATIONS_ES,
    join_spanish,
    announce_in_thread,
    direction_from_center,
)

# Mapa inicial: solo muestra Valparaíso sin ruta
MAP_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    html, body, #map { height: 100%; margin: 0; padding: 0; }
  </style>
</head>
<body>
  <div id="map"></div>
  <script>
    const map = L.map('map').setView([-33.0458, -71.6168], 15);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap contributors'
    }).addTo(map);
  </script>
</body>
</html>
"""


class VisionGUI(QWidget):
    def __init__(self, camera_index=0, announcement_cooldown=DEFAULT_ANNOUNCEMENT_COOLDOWN):
        super().__init__()

        self.setWindowTitle("OJOPIOJO - Vision Asistida")
        self.camera_index = camera_index
        self.announcement_cooldown = announcement_cooldown

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[GUI] Usando dispositivo: {self.device}")

        self.midas = None
        self.midas_transform = None
        self.yolo = None

        self.last_announcement_time = time.time()
        self.last_alert_message = ""
        self.last_object_depths = {}
        self.alert_history = []
        self.locations = [
            {"name": "Casa", "ubicacion": "Casa", "calle": "", "numero": "", "ciudad": "", "extra": ""},
            {"name": "Trabajo", "ubicacion": "Trabajo", "calle": "", "numero": "", "ciudad": "", "extra": ""},
            {"name": "Clinica", "ubicacion": "Clinica", "calle": "", "numero": "", "ciudad": "", "extra": ""},
        ]
        self.screen_alert = ""

        self.cap = None

        self.primary = "#f2d24a"  # amarillo ligeramente mas oscuro
        self.dark_bg = "#FFFFFF"
        self.text_dark = "#1F1300"
        self.text_light = "#1F1300"
        self.success = "#0B6E4F"
        self.warn = "#F4A261"
        self.danger = "#D7263D"
        self.secondary = "#bc3fde"
        self.icon_size = 22

        # pasos de ruta para audio
        self.route_steps = []
        self.current_route_step = 0

        self._build_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.load_models()
        self.set_status("Modelos cargados, listo para iniciar camara", level="ok")
        announce_in_thread("Sistema listo. Vocalización activada.")

    # ---------- ICONOS ----------
    def micon(self, name: str):
        try:
            return qta.icon(name, color="white")
        except Exception as e:
            print(f"[GUI] No se pudo cargar icono '{name}': {e}")
            return QIcon()
    # ----------------------------

    # ---------- GEOCODING ----------
    def geocode_address(self, query: str):
        """
        Convierte una dirección de texto en (lat, lon) usando Nominatim (OpenStreetMap).
        Devuelve None si falla.
        """
        url = "https://nominatim.openstreetmap.org/search"
        try:
            resp = requests.get(
                url,
                params={"q": query, "format": "json", "limit": 1},
                headers={"User-Agent": "OJOPIOJO/1.0"},
                timeout=10,
            )
            data = resp.json()
            if not data:
                return None
            lat = float(data[0]["lat"])
            lon = float(data[0]["lon"])
            return lat, lon
        except Exception as e:
            print(f"[GUI] Error geocodificando '{query}': {e}")
            return None
    # -------------------------------

    # ---------- AUDIO RUTA ----------
    def set_route_instructions(self, steps):
        """Guarda la lista de pasos de la ruta y lee el primero."""
        self.route_steps = steps or []
        self.current_route_step = 0
        if self.route_steps:
            announce_in_thread(self.route_steps[0])
            self.set_status("Leyendo primera indicación de la ruta", level="info")
            self.set_route_instruction_box(self.route_steps[0], 0)
        else:
            self.set_status("Ruta sin instrucciones de voz disponibles", level="warn")
            self.set_route_instruction_box(None)

    def speak_next_step(self):
        """Lee el siguiente paso de la ruta."""
        if not self.route_steps:
            self.set_status("No hay ruta activa", level="warn")
            self.set_route_instruction_box(None)
            return

        if self.current_route_step < len(self.route_steps) - 1:
            self.current_route_step += 1

        text = self.route_steps[self.current_route_step]
        announce_in_thread(text)
        self.set_status(
            f"Paso {self.current_route_step + 1}/{len(self.route_steps)}",
            level="info",
        )
        self.set_route_instruction_box(text, self.current_route_step)

    def speak_prev_step(self):
        """Lee el paso anterior de la ruta."""
        if not self.route_steps:
            self.set_status("No hay ruta activa", level="warn")
            self.set_route_instruction_box(None)
            return

        if self.current_route_step > 0:
            self.current_route_step -= 1

        text = self.route_steps[self.current_route_step]
        announce_in_thread(text)
        self.set_status(
            f"Paso {self.current_route_step + 1}/{len(self.route_steps)}",
            level="info",
        )
        self.set_route_instruction_box(text, self.current_route_step)

    def repeat_step(self):
        """Repite el paso actual."""
        if not self.route_steps:
            self.set_status("No hay ruta activa", level="warn")
            self.set_route_instruction_box(None)
            return

        text = self.route_steps[self.current_route_step]
        announce_in_thread(text)
        self.set_route_instruction_box(text, self.current_route_step)
    # -------------------------------

    def set_route_instruction_box(self, text, step_index=None):
        """Actualiza el recuadro de indicaciones de ruta."""
        if not text:
            self.route_instruction.setText("Sin ruta activa. Indica un origen y destino.")
            return
        prefix = ""
        if step_index is not None and self.route_steps:
            prefix = f"Paso {step_index + 1}/{len(self.route_steps)}: "
        self.route_instruction.setText(prefix + text)

    def _build_ui(self):
        base_font = QFont("Segoe UI", 10)
        self.setFont(base_font)
        self.setStyleSheet(
            f"""
            QWidget {{
                background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 #fffaf0, stop:1 #ffffff);
                color: {self.text_light};
                font-family: 'Segoe UI';
                letter-spacing: 0.2px;
            }}
            QLabel#Title {{
                color: {self.primary};
                font-size: 28px;
                font-weight: 800;
                letter-spacing: 1px;
            }}
            QLabel#Subtitle {{
                color: #5c5750;
                font-size: 12px;
            }}
            QLabel#VideoFrame {{
                background-color: #fdf7e4;
                border: 2px solid {self.primary};
                border-radius: 12px;
                padding: 12px;
            }}
            QLabel#StatusChip {{
                padding: 6px 12px;
                border-radius: 14px;
                font-weight: 700;
                letter-spacing: 0.4px;
                font-size: 11px;
            }}
            QLabel#AlertBanner {{
                color: #7a0f1d;
                font-weight: 700;
                padding-left: 4px;
            }}
            QLabel#AlertIcon {{
                padding-right: 6px;
            }}
            QPushButton {{
                padding: 14px 18px;
                border-radius: 14px;
                border: none;
                font-size: 14px;
                font-weight: 700;
            }}
            QPushButton#Primary {{
                background-color: {self.primary};
                color: {self.text_dark};
            }}
            QPushButton#Primary:hover {{
                background-color: #e5c33c;
            }}
            QPushButton#Secondary {{
                background-color: {self.secondary};
                color: #ffffff;
                border: 1px solid #a22dcc;
            }}
            QPushButton#Secondary:hover {{
                background-color: #a22dcc;
            }}
            QPushButton#Danger {{
                background-color: {self.danger};
                color: #fff;
            }}
            QPushButton#Danger:hover {{
                background-color: #bf1f33;
            }}
            QDialog {{
                background-color: #ffffff;
            }}
            QListWidget {{
                background-color: #fdfaf3;
                border: 1px solid #e8decf;
                border-radius: 10px;
                padding: 8px;
            }}
            QListWidget::item {{
                padding: 6px;
                border-radius: 6px;
            }}
            QListWidget::item:selected {{
                background-color: #f2d24a;
                color: {self.text_dark};
            }}
            QLineEdit {{
                padding: 8px;
                border: 1px solid #e0d9cd;
                border-radius: 8px;
                background: #ffffff;
            }}
            QLabel#RouteBox {{
                background-color: #fff7d6;
                border: 1px solid #e6c531;
                border-radius: 12px;
                padding: 12px;
                font-weight: 700;
                line-height: 1.4em;
            }}
            """
        )

        self.brand_label = QLabel("OJOPIOJO")
        self.brand_label.setObjectName("Title")
        self.subtitle_label = QLabel("Vision asistida en tiempo real")
        self.subtitle_label.setObjectName("Subtitle")

        self.status_chip = QLabel("Cargando modelos...")
        self.status_chip.setObjectName("StatusChip")
        self.alert_icon = QLabel()
        self.alert_icon.setObjectName("AlertIcon")
        self.alert_label = QLabel("")
        self.alert_label.setObjectName("AlertBanner")
        self.alert_icon.setVisible(False)
        self.alert_label.setVisible(False)

        header_left = QVBoxLayout()
        header_left.addWidget(self.brand_label)
        header_left.addWidget(self.subtitle_label)

        header = QHBoxLayout()
        header.addLayout(header_left)
        header.addStretch()
        header.addWidget(self.status_chip, alignment=Qt.AlignRight)
        alert_wrap = QHBoxLayout()
        alert_wrap.setSpacing(4)
        alert_wrap.addWidget(self.alert_icon)
        alert_wrap.addWidget(self.alert_label)
        header.addLayout(alert_wrap)

        self.video_label = QLabel("Video no iniciado")
        self.video_label.setObjectName("VideoFrame")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(900, 540)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Mapa principal embebido
        self.map_view = QWebEngineView()
        self.map_view.setMinimumHeight(260)
        self.map_view.setHtml(MAP_HTML)

        self.btn_start = QPushButton("Iniciar")
        self.btn_start.setObjectName("Primary")
        self.btn_start.setIcon(self.micon("mdi.play"))

        self.btn_locations = QPushButton("Ubicaciones preferidas")
        self.btn_locations.setObjectName("Secondary")
        self.btn_locations.setIcon(self.micon("mdi.map-marker"))

        self.btn_home = QPushButton("Historial de alertas")
        self.btn_home.setObjectName("Secondary")
        self.btn_home.setIcon(self.micon("mdi.history"))

        self.btn_map_view = QPushButton("Mapa")
        self.btn_map_view.setObjectName("Secondary")
        self.btn_map_view.setIcon(self.micon("mdi.map"))

        self.btn_exit = QPushButton("Salir")
        self.btn_exit.setObjectName("Danger")
        self.btn_exit.setIcon(self.micon("mdi.logout"))

        for btn in (self.btn_start, self.btn_locations, self.btn_home, self.btn_map_view, self.btn_exit):
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.btn_start.clicked.connect(self.start_camera)
        self.btn_locations.clicked.connect(self.open_map)
        self.btn_home.clicked.connect(self.go_home)
        self.btn_map_view.clicked.connect(self.open_main_map)
        self.btn_exit.clicked.connect(self.close)

        self.route_instruction = QLabel("Sin ruta activa. Indica un origen y destino.")
        self.route_instruction.setObjectName("RouteBox")
        self.route_instruction.setWordWrap(True)
        self.route_instruction.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        # controles de navegación de ruta
        self.btn_route_prev = QPushButton("Paso anterior")
        self.btn_route_prev.setObjectName("Secondary")
        self.btn_route_prev.setIcon(self.micon("mdi.arrow-left-bold"))

        self.btn_route_next = QPushButton("Paso siguiente")
        self.btn_route_next.setObjectName("Primary")
        self.btn_route_next.setIcon(self.micon("mdi.arrow-right-bold"))

        self.btn_route_prev.clicked.connect(self.speak_prev_step)
        self.btn_route_next.clicked.connect(self.speak_next_step)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_locations)
        btn_layout.addWidget(self.btn_home)
        btn_layout.addWidget(self.btn_map_view)
        btn_layout.addWidget(self.btn_exit)
        btn_layout.addStretch()

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(24, 18, 24, 18)
        main_layout.setSpacing(14)
        main_layout.addLayout(header)
        main_layout.addWidget(self.video_label, stretch=3)
        main_layout.addWidget(self.map_view)
        main_layout.addWidget(self.route_instruction)

        route_controls = QHBoxLayout()
        route_controls.setSpacing(8)
        route_controls.addWidget(self.btn_route_prev)
        route_controls.addWidget(self.btn_route_next)
        route_controls.addStretch()

        main_layout.addLayout(route_controls)
        main_layout.addLayout(btn_layout, stretch=1)

        self.setLayout(main_layout)
        self.resize(1220, 900)

    def load_models(self):
        print("[GUI] Cargando modelo MiDaS (profundidad)...")
        midas_model_type = "MiDaS_small"
        self.midas = torch.hub.load("intel-isl/MiDaS", midas_model_type)
        self.midas.to(self.device)
        self.midas.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.midas_transform = midas_transforms.small_transform

        print("[GUI] Cargando modelo YOLOv5 (deteccion de objetos)...")
        self.yolo = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
        self.yolo.to(self.device)
        self.yolo.eval()

        print("Modelos cargados correctamente.")

    def set_status(self, text, level="info"):
        colors = {
            "info": (self.primary, self.text_dark),
            "ok": (self.success, "#E8F6EF"),
            "warn": (self.warn, self.text_dark),
            "danger": (self.danger, "#ffffff"),
        }
        bg, fg = colors.get(level, colors["info"])
        chip_style = (
            f"background-color: {bg}; color: {fg}; padding: 6px 10px; "
            "border-radius: 12px; font-weight: 700; letter-spacing: 0.3px; font-size: 11px;"
        )
        self.status_chip.setStyleSheet(chip_style)
        self.status_chip.setText(text)
        self.set_alert("")

    def start_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            self.set_status(f"No se pudo abrir camara {self.camera_index}", level="danger")
            return

        self.set_status(f"Camara {self.camera_index} encendida", level="ok")
        self.timer.start(30)

    def stop_camera(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.video_label.setText("Video detenido")
        self.set_status("Camara detenida", level="warn")

    def open_map(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Ubicaciones destacadas")
        dialog.setStyleSheet(self.styleSheet())

        list_widget = QListWidget()
        list_widget.setSpacing(6)

        add_btn = QPushButton("Agregar direccion")
        add_btn.setObjectName("Primary")
        add_btn.setIcon(self.micon("mdi.map-marker-plus"))
        add_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        edit_btn = QPushButton("Editar seleccion")
        edit_btn.setObjectName("Secondary")
        edit_btn.setIcon(self.micon("mdi.map-marker-edit"))
        edit_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        delete_btn = QPushButton("Eliminar seleccion")
        delete_btn.setObjectName("Danger")
        delete_btn.setIcon(self.micon("mdi.delete-forever"))
        delete_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        route_btn = QPushButton("Iniciar recorrido")
        route_btn.setObjectName("Primary")
        route_btn.setIcon(self.micon("mdi.walk"))
        route_btn.setEnabled(False)

        close_btn = QPushButton("Cerrar")
        close_btn.setObjectName("Secondary")
        close_btn.setIcon(self.micon("mdi.close"))
        close_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        def format_loc(loc):
            ciudad = loc.get("ciudad", "")
            calle = loc.get("calle", "")
            numero = loc.get("numero", "")
            detalle = f"{calle} {numero}".strip()
            tail = f"{detalle} · {ciudad}".strip(" ·")
            return f"{loc.get('name','')} - {tail}" if tail else loc.get("name", "")

        def refresh_list():
            list_widget.clear()
            for loc in self.locations:
                list_widget.addItem(format_loc(loc))

        def prompt_location(existing=None):
            dlg = QDialog(dialog)
            dlg.setWindowTitle("Datos de ubicacion")
            dlg.setStyleSheet(self.styleSheet())

            name_input = QLineEdit(existing.get("name", "") if existing else "")
            ubic_input = QLineEdit(existing.get("ubicacion", "") if existing else "")
            calle_input = QLineEdit(existing.get("calle", "") if existing else "")
            numero_input = QLineEdit(existing.get("numero", "") if existing else "")
            ciudad_input = QLineEdit(existing.get("ciudad", "") if existing else "")
            extra_input = QLineEdit(existing.get("extra", "") if existing else "")

            form = QVBoxLayout()

            def row(label, widget):
                h = QHBoxLayout()
                h.addWidget(QLabel(label))
                h.addWidget(widget)
                form.addLayout(h)

            row("Nombre", name_input)
            row("Ubicacion", ubic_input)
            row("Calle", calle_input)
            row("Numero", numero_input)
            row("Ciudad", ciudad_input)
            row("Referencia/extra", extra_input)

            btn_accept = QPushButton("Guardar")
            btn_accept.setObjectName("Primary")
            btn_cancel = QPushButton("Cancelar")
            btn_cancel.setObjectName("Secondary")

            btns = QHBoxLayout()
            btns.addStretch()
            btns.addWidget(btn_accept)
            btns.addWidget(btn_cancel)
            form.addLayout(btns)

            dlg.setLayout(form)

            result = {}

            def accept():
                result.update(
                    {
                        "name": name_input.text().strip(),
                        "ubicacion": ubic_input.text().strip(),
                        "calle": calle_input.text().strip(),
                        "numero": numero_input.text().strip(),
                        "ciudad": ciudad_input.text().strip(),
                        "extra": extra_input.text().strip(),
                    }
                )
                dlg.accept()

            btn_accept.clicked.connect(accept)
            btn_cancel.clicked.connect(dlg.reject)
            dlg.exec_()
            return result if result.get("name") else None

        def add_location():
            data = prompt_location()
            if data:
                self.locations.append(data)
                refresh_list()

        def edit_location():
            row = list_widget.currentRow()
            if row < 0:
                return
            current = self.locations[row]
            data = prompt_location(current)
            if data:
                self.locations[row] = data
                refresh_list()

        def delete_location():
            row = list_widget.currentRow()
            if row < 0:
                return
            self.locations.pop(row)
            refresh_list()
            route_btn.setEnabled(False)

        def start_route():
            row = list_widget.currentRow()
            if row < 0:
                return
            location = self.locations[row]
            destino = location.get("name", "ubicacion seleccionada")
            self.set_status(f"Iniciando recorrido hacia {destino}", level="ok")
            dialog.accept()

        def on_selection_changed():
            route_btn.setEnabled(list_widget.currentRow() >= 0)

        list_widget.currentRowChanged.connect(on_selection_changed)
        add_btn.clicked.connect(add_location)
        edit_btn.clicked.connect(edit_location)
        delete_btn.clicked.connect(delete_location)
        route_btn.clicked.connect(start_route)
        close_btn.clicked.connect(dialog.accept)
        refresh_list()

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Ubicaciones favoritas"))
        layout.addWidget(list_widget)

        buttons_row = QHBoxLayout()
        buttons_row.addWidget(add_btn)
        buttons_row.addWidget(edit_btn)
        buttons_row.addWidget(delete_btn)
        layout.addLayout(buttons_row)

        bottom_row = QHBoxLayout()
        bottom_row.addWidget(route_btn)
        bottom_row.addWidget(close_btn)
        layout.addLayout(bottom_row)

        dialog.setLayout(layout)
        dialog.resize(460, 380)
        dialog.exec_()

    def go_home(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Historial de alertas")
        dialog.setStyleSheet(self.styleSheet())

        list_widget = QListWidget()
        list_widget.addItems(self.alert_history if self.alert_history else ["Sin mensajes hasta ahora"])
        list_widget.setSpacing(6)

        close_btn = QPushButton("Cerrar")
        close_btn.setObjectName("Secondary")
        close_btn.setIcon(self.micon("mdi.close"))
        close_btn.clicked.connect(dialog.accept)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Mensajes lanzados"))
        layout.addWidget(list_widget)
        layout.addWidget(close_btn)

        dialog.setLayout(layout)
        dialog.resize(420, 320)
        dialog.exec_()

    def open_main_map(self):
        """
        Diálogo que pide Punto A (origen) y Punto B (destino) como texto,
        geocodifica y actualiza el mapa principal con una ruta peatonal.
        """
        dialog = QDialog(self)
        dialog.setWindowTitle("Ruta segura")
        dialog.setStyleSheet(self.styleSheet())

        origen_input = QLineEdit()
        destino_input = QLineEdit()
        origen_input.setPlaceholderText("Ej: Avenida Brasil 1480, Valparaíso")
        destino_input.setPlaceholderText("Ej: DUOC UC Sede Valparaíso")

        form = QVBoxLayout()

        def row(label, widget):
            h = QHBoxLayout()
            h.addWidget(QLabel(label))
            h.addWidget(widget)
            form.addLayout(h)

        row("Punto A (origen)", origen_input)
        row("Punto B (destino)", destino_input)

        btn_start_route = QPushButton("Comenzar ruta")
        btn_start_route.setObjectName("Primary")
        btn_close = QPushButton("Cerrar")
        btn_close.setObjectName("Secondary")

        buttons = QHBoxLayout()
        buttons.addStretch()
        buttons.addWidget(btn_start_route)
        buttons.addWidget(btn_close)
        form.addLayout(buttons)

        dialog.setLayout(form)
        dialog.resize(500, 200)

        def start_route():
            origen_txt = origen_input.text().strip()
            destino_txt = destino_input.text().strip()

            if not origen_txt or not destino_txt:
                self.set_status("Completa origen y destino", level="danger")
                return

            origen_coords = self.geocode_address(origen_txt)
            if origen_coords is None:
                self.set_status("No se pudo encontrar el origen", level="danger")
                return

            destino_coords = self.geocode_address(destino_txt)
            if destino_coords is None:
                self.set_status("No se pudo encontrar el destino", level="danger")
                return

            self.update_main_map(origen_coords, destino_coords)
            self.set_status("Ruta segura calculada", level="ok")
            dialog.accept()

        btn_start_route.clicked.connect(start_route)
        btn_close.clicked.connect(dialog.reject)
        dialog.exec_()

    # ---------- OBTENER PASOS DE OSRM ----------
    def build_route_instructions(self, origin, destination):
        """
        Pide a OSRM la ruta a pie entre origin y destination y genera
        instrucciones de texto simples (en inglés pero entendibles).
        """
        o_lat, o_lon = origin
        d_lat, d_lon = destination
        url = f"https://router.project-osrm.org/route/v1/foot/{o_lon},{o_lat};{d_lon},{d_lat}"
        params = {
            "overview": "false",
            "steps": "true",
            "alternatives": "false",
        }
        steps_text = []

        def format_step(step):
            """Crea una indicación en español para el paso."""
            man = step.get("maneuver", {}) or {}
            mtype = (man.get("type") or "").lower()
            modifier = (man.get("modifier") or "").lower()
            street = step.get("name") or ""
            distance = step.get("distance")

            def dist_phrase():
                if distance:
                    dist_m = int(round(distance))
                    return f" Avanza aproximadamente {dist_m} metros."
                return ""

            dir_words = {
                "left": "gira a la izquierda",
                "slight left": "gira levemente a la izquierda",
                "sharp left": "gira pronunciado a la izquierda",
                "right": "gira a la derecha",
                "slight right": "gira levemente a la derecha",
                "sharp right": "gira pronunciado a la derecha",
                "uturn": "haz un giro en U",
                "straight": "continúa recto",
            }

            street_part = f" hacia {street}" if street else ""
            corner_phrase = " en la esquina" if modifier in dir_words else ""

            if mtype == "depart":
                return f"Inicia avanzando por {street or 'la ruta'}.{dist_phrase()}"
            if mtype in {"turn", "end of road"}:
                action = dir_words.get(modifier, "gira")
                return f"{action}{corner_phrase}{street_part}.{dist_phrase()}"
            if mtype == "continue":
                action = dir_words.get(modifier, "continúa recto")
                return f"{action}{street_part}.{dist_phrase()}"
            if mtype == "roundabout":
                exit_no = man.get("exit")
                exit_part = f" toma la salida {exit_no}" if exit_no else ""
                return f"En la rotonda, {dir_words.get(modifier, 'toma la salida')}{exit_part}{street_part}.{dist_phrase()}"
            if mtype == "arrive":
                return "Has llegado a tu destino."

            # fallback usando instrucción original
            instr = man.get("instruction") or step.get("instruction") or "Sigue la ruta."
            return instr + dist_phrase()

        try:
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()
            routes = data.get("routes")
            if not routes:
                return steps_text
            legs = routes[0].get("legs", [])
            for leg in legs:
                for step in leg.get("steps", []):
                    instr = format_step(step)
                    if instr:
                        steps_text.append(instr)
        except Exception as e:
            print(f"[GUI] Error obteniendo instrucciones OSRM: {e}")
        return steps_text
    # -------------------------------------------

    def update_main_map(self, origin, destination):
        """
        Actualiza el mapa principal para mostrar una ruta de A a B
        usando OSRM en modo 'foot' (caminando) y genera instrucciones de voz.
        """
        o_lat, o_lon = origin
        d_lat, d_lon = destination
        html = f"""
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width, initial-scale=1.0">
          <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
          <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
          <link rel="stylesheet" href="https://unpkg.com/leaflet-routing-machine@3.2.12/dist/leaflet-routing-machine.css" />
          <script src="https://unpkg.com/leaflet-routing-machine@3.2.12/dist/leaflet-routing-machine.min.js"></script>
          <style>html, body, #map {{ height: 100%; margin: 0; padding: 0; }}</style>
        </head>
        <body>
          <div id="map"></div>
          <script>
            const map = L.map('map').setView([{o_lat}, {o_lon}], 15);
            L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
              maxZoom: 19,
              attribution: '&copy; OpenStreetMap contributors'
            }}).addTo(map);
            const control = L.Routing.control({{
              waypoints: [
                L.latLng({o_lat}, {o_lon}),
                L.latLng({d_lat}, {d_lon})
              ],
              router: L.Routing.osrmv1({{
                serviceUrl: 'https://router.project-osrm.org/route/v1/foot'
              }}),
              lineOptions: {{
                styles: [{{color: '#1d6ae5', weight: 5}}]
              }},
              show: false,
              collapsible: true,
              addWaypoints: false,
              draggableWaypoints: false
            }}).addTo(map);
            map.on('routingerror', function() {{
              alert('No se pudo calcular la ruta de caminata.');
            }});
          </script>
        </body>
        </html>
        """
        self.map_view.setHtml(html)

        # construir lista de pasos y activar audio
        steps = self.build_route_instructions(origin, destination)
        if not steps:
            steps = ["Ruta iniciada. Sigue caminando por la ruta indicada en el mapa."]
        self.set_route_instructions(steps)

    def set_alert(self, text):
        self.screen_alert = text
        if text:
            icon = self.style().standardIcon(self.style().SP_MessageBoxWarning).pixmap(22, 22)
            self.alert_icon.setPixmap(icon)
            self.alert_icon.setVisible(True)
            self.alert_label.setText(text)
            self.alert_label.setVisible(True)
            if not self.alert_history or self.alert_history[-1] != text:
                self.alert_history.append(text)
        else:
            self.alert_label.clear()
            self.alert_icon.clear()
            self.alert_icon.setVisible(False)
            self.alert_label.setVisible(False)

    def update_frame(self):
        if self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.set_status("No se pudo leer frame", level="danger")
            return

        frame = self.process_frame_with_models(frame)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(pixmap)

    def process_frame_with_models(self, frame):
        frame_height, frame_width = frame.shape[:2]

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
        input_batch = self.midas_transform(img_rgb).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_batch)
            depth_map = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
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
            yolo_results = self.yolo(img_rgb, size=640)
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
                y2 = max(min(y2, frame_height - 1), 0)
                if x2 <= x1 or y2 <= y1:
                    continue

                depth_region = depth_normalized[y1:y2, x1:x2]
                if depth_region.size == 0:
                    continue
                box_depth_value = float(np.max(depth_region))

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                tracking_key = f"{class_name}-{center_x // TRACKING_GRID_SIZE}-{center_y // TRACKING_GRID_SIZE}"
                prev_depth = self.last_object_depths.get(tracking_key)
                approaching = prev_depth is not None and (box_depth_value - prev_depth) > APPROACH_DELTA
                self.last_object_depths[tracking_key] = box_depth_value

                is_close = box_depth_value > DANGER_DEPTH_THRESHOLD
                spanish_label = CLASS_TRANSLATIONS_ES.get(class_name, class_name)
                direction_label = direction_from_center(center_x, center_y, frame_width, frame_height)

                label_text = f"{spanish_label} {conf:.0%}"
                color = (0, 0, 255) if is_close else ((0, 255, 255) if approaching else (0, 255, 0))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    label_text,
                    (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

                if is_close or approaching:
                    desc = f"{spanish_label} {direction_label}"
                    if approaching and is_close:
                        desc += " muy cerca y acercandose"
                    elif approaching:
                        desc += " acercandose"
                    else:
                        desc += " muy cerca"
                    object_alerts.append((box_depth_value, desc))

        except Exception as e:
            print(f"[GUI] Error en deteccion de objetos: {e}")

        current_time = time.time()
        if object_alerts:
            object_alerts.sort(key=lambda x: x[0], reverse=True)
            descriptions = [desc for _, desc in object_alerts]
            announcement = "Cuidado! " + join_spanish(descriptions)
            if (current_time - self.last_announcement_time > self.announcement_cooldown) or (
                announcement != self.last_alert_message
            ):
                print(f"ALERTA: {announcement}")
                announce_in_thread(announcement)
                self.last_announcement_time = current_time
                self.last_alert_message = announcement
            self.set_alert(announcement)
        elif alert_messages:
            announcement = "Cuidado! Obstaculo " + join_spanish(alert_messages)
            if (current_time - self.last_announcement_time > self.announcement_cooldown) or (
                announcement != self.last_alert_message
            ):
                print(f"ALERTA: {announcement}")
                announce_in_thread(announcement)
                self.last_announcement_time = current_time
                self.last_alert_message = announcement
            self.set_alert(announcement)
        else:
            self.last_alert_message = ""
            self.set_alert("")

        return frame

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VisionGUI(camera_index=0)
    window.show()
    sys.exit(app.exec())
