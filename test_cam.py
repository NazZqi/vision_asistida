import cv2

def probar_indices(max_index=5):
    print("Probando índices de cámara de 0 a", max_index)
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"✅ Cámara encontrada en índice {i}")
            ret, frame = cap.read()
            if ret:
                print(f"   -> Pude leer un frame en el índice {i}")
            else:
                print(f"   -> No pude leer frame en {i}, pero la cámara se abrió.")
            cap.release()
        else:
            print(f"❌ No se pudo abrir la cámara en índice {i}")

if __name__ == "__main__":
    probar_indices(5)
