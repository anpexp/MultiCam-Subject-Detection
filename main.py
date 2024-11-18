import cv2
import tensorflow as tf
import numpy as np
import socket
import pickle

# Configuración de la cámara (simulada)
cap = cv2.VideoCapture('PERSONA CORRIENDO.mp4')

# Carga del modelo MobileNetV2 pre-entrenado
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)  
 
predictions = Dense(1, activation='sigmoid')(x)  
  # Solo una salida para detección de persona (0 o 1)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  


# Configuración de la comunicación por sockets
HOST = 'localhost'  # Reemplaza con la IP correcta
PORT = 5000
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar fotograma. Saliendo")
        break
    # Preprocesamiento de la imagen para MobileNetV2
    img = cv2.resize(frame, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

    # Detección de personas
    prediction = model.predict(img)

    # Lógica para activar la alarma (ajusta el umbral según tus necesidades)
    if prediction[0][0] > 0.5:  # Asume un umbral de confianza de 0.7 para detectar persona
        print("Persona detectada!")

        # Envía la imagen al otro Raspberry Pi
        data = pickle.dumps(frame)


        data_size = len(data).to_bytes(4, byteorder='big')  # 4 bytes para el tamaño
        s.sendall(data_size)
        
        s.sendall(data)

    # Visualización (opcional)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
s.close()