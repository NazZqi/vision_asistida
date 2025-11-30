# Vision Asistida

Vision Asistida is a Python application designed to assist users by analyzing video input for depth and obstacle detection using advanced machine learning models. The application leverages the MiDaS model for depth estimation and provides real-time feedback on potential hazards in the user's environment.

## Features

- Real-time depth analysis using the MiDaS model.
- Obstacle detection with visual and audio alerts.
- Language translation for object recognition from English to Spanish.

## Installation

To install the necessary dependencies, run the following command:

1. Instala PyTorch y torchvision según tu GPU/CPU (ver https://pytorch.org/get-started/locally/). Ejemplos:
   - CPU: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
   - CUDA 12.6: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`
   - DirectML (AMD/Intel en Windows): `pip install torch-directml` (además de torch/torchvision)
2. Instala el resto: `pip install -r requirements.txt`

## Usage

1. Ensure your camera is connected and accessible.
2. Run the main application:

```
python src/vision_asistida.py
```

Puedes forzar el dispositivo con `--device`, por ejemplo:

```
python src/vision_asistida/vision_asistida.py --device mps
python src/vision_asistida/vision_asistida.py --device directml
```

3. Follow the audio and visual prompts to navigate safely.

## Testing

To run the unit tests, execute:

```
pytest tests/test_basic.py
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them.
4. Push your branch and create a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
