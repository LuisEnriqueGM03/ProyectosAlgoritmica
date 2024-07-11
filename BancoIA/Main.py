import math
import cv2
from ultralytics import YOLO
import torch
import psycopg2
from psycopg2 import OperationalError, sql

# Función para crear la conexión a la base de datos
def create_connection():
    try:
        conn = psycopg2.connect(
            dbname="banco",
            user="postgres",
            password="root",
            host="localhost",
            port="5432"
        )
        print("Connection to PostgreSQL DB successful")
        return conn
    except OperationalError as e:
        print(f"The error '{e}' occurred")
        return None

# Autenticación del usuario
def login(conn, correo, contraseña):
    cur = conn.cursor()
    cur.execute("SELECT id, nombre, apellido FROM usuario WHERE correo=%s AND contraseña=%s", (correo, contraseña))
    user = cur.fetchone()
    cur.close()
    if user:
        print("Login exitoso")
        return user[0]
    else:
        print("Correo o contraseña incorrectos")
        return None

# Registro del usuario
def register(conn, nombre, apellido, correo, contraseña):
    try:
        cur = conn.cursor()
        cur.execute("INSERT INTO usuario (nombre, apellido, correo, contraseña) VALUES (%s, %s, %s, %s) RETURNING id",
                    (nombre, apellido, correo, contraseña))
        user_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        print("Registro exitoso")
        return user_id
    except psycopg2.Error as e:
        print(f"Error al registrar usuario: {e}")
        return None

# Clase BankIA
class BankIA:
    def __init__(self, user_id):
        self.user_id = user_id
        self.conn = create_connection()

        # VideoCapture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)
        self.cap.set(4, 720)

        # Utilizar PyTorch con la GPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # Modelos
        self.billModel = YOLO('BilleteV1.pt')

        # Clases de billetes
        self.clsBillBank = ['200bs', '20bs', '100bs', '10bs', '50bs']

        # Balance total y balance temporal
        self.total_balance = 0
        self.temp_balance = 0

    def balance_process(self, bill_type):
        if bill_type == '10bs':
            return 10
        elif bill_type == '20bs':
            return 20
        elif bill_type == '50bs':
            return 50
        elif bill_type == '100bs':
            return 100
        elif bill_type == '200bs':
            return 200
        return 0

    def prediction_model(self, frame):
        """
        Realiza la predicción y devuelve el frame y el balance detectado.
        """
        results = self.billModel(frame, device=self.device)
        for res in results:
            boxes = res.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                if x1 < 0: x1 = 0
                if y1 < 0: y1 = 0
                if x2 < 0: x2 = 0
                if y2 < 0: y2 = 0

                cls = int(box.cls[0])
                conf = math.ceil(box.conf[0])

                bill_type = self.clsBillBank[cls]
                text_obj = f'{self.clsBillBank[cls]} {int(conf * 100)}%'
                balance = self.balance_process(bill_type)

                size_obj, thickness_obj = 0.75, 1
                frame = self.draw_text(frame, (0, 255, 0), text_obj, x1, y1, size_obj, thickness_obj, back=True)
                frame = self.draw_area(frame, (0, 255, 0), x1, y1, x2, y2)

                return frame, balance
        return frame, 0

    def draw_area(self, img, color, xi, yi, xf, yf):
        img = cv2.rectangle(img, (xi, yi), (xf, yf), color, 1, 1)
        return img

    def draw_text(self, img, color, text, xi, yi, size, thickness, back=False):
        sizetext = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, size, thickness)
        dim = sizetext[0]
        baseline = sizetext[1]
        if back:
            img = cv2.rectangle(img, (xi, yi - dim[1] - baseline), (xi + dim[0], yi + baseline - 7), (0, 0, 0), cv2.FILLED)
        img = cv2.putText(img, text, (xi, yi - 5), cv2.FONT_HERSHEY_DUPLEX, size, color, thickness)
        return img

    def update_balance(self):
        cur = self.conn.cursor()
        cur.execute("UPDATE balance SET dinero = dinero + %s WHERE usuario_id = %s", (self.total_balance, self.user_id))
        self.conn.commit()
        cur.close()
        print(f"Balance de {self.total_balance} actualizado en la base de datos para el usuario {self.user_id}")
        self.total_balance = 0

    def bancoIA(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("No se pudo leer el frame de la cámara")
                break

            clean_frame = frame.copy()
            frame, balance = self.prediction_model(clean_frame)

            text_balance = f'Saldo temporal: {self.temp_balance} $'
            frame = self.draw_text(frame, (0, 255, 0), text_balance, 10, 30, 0.60, 1, back=False)

            cv2.imshow("Banco IA", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('a'):
                self.temp_balance += balance
                print(f"Saldo de {balance} bs guardado temporalmente.")
            if key == ord('s'):
                self.total_balance += self.temp_balance
                self.update_balance()
                print("Programa finalizado.")
                break
            if key == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    conn = create_connection()
    if conn is not None:
        # Menú principal para iniciar sesión o registrarse
        while True:
            print("1. Iniciar sesión")
            print("2. Registrarse")
            choice = input("Seleccione una opción: ")

            if choice == '1':
                correo = input("Ingrese su correo: ")
                contraseña = input("Ingrese su contraseña: ")
                user_id = login(conn, correo, contraseña)
                if user_id:
                    bank_ia = BankIA(user_id)
                    bank_ia.bancoIA()
                    break
            elif choice == '2':
                nombre = input("Ingrese su nombre: ")
                apellido = input("Ingrese su apellido: ")
                correo = input("Ingrese su correo: ")
                contraseña = input("Ingrese su contraseña: ")
                user_id = register(conn, nombre, apellido, correo, contraseña)
                if user_id:
                    bank_ia = BankIA(user_id)
                    bank_ia.bancoIA()
                    break
            else:
                print("Opción no válida. Intente de nuevo.")
