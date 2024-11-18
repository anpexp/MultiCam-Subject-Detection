import socket
import pickle
import cv2

# Configuración del socket servidor
HOST = 'localhost'
PORT = 5000

# Crea el socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(1)  # Espera una conexión

print("Esperando conexión...")
conn, addr = s.accept()  # Acepta la conexión
print(f"Conexión establecida desde {addr}")

while True:

    data_size = conn.recv(4)  # Asume que el tamaño se envía en 4 bytes
    if not data_size:
        break
    data_size = int.from_bytes(data_size, byteorder='big')
    
    # Recibe los datos en trozos de 4096 bytes
    data = b""
    while len(data) < data_size:
        packet = conn.recv(4096)
        if not packet: break
        data += packet
    
    # Deserializa la imagen recibida
    frame = pickle.loads(data)

    # Muestra la imagen recibida (opcional)
    cv2.imshow('Imagen recibida', frame)
    cv2.waitKey(1)

conn.close()
s.close()