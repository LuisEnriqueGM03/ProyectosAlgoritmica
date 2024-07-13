import tkinter as tk
from tkinter import messagebox
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
    cur.execute("SELECT id, nombre, apellido, (SELECT dinero FROM balance WHERE usuario_id = usuario.id) as balance FROM usuario WHERE correo=%s AND contraseña=%s", (correo, contraseña))
    user = cur.fetchone()
    cur.close()
    if user:
        print("Login exitoso")
        return user
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
    def __init__(self, user):
        self.user_id = user[0]
        self.user_name = user[1]
        self.balance = user[3]
        self.conn = create_connection()

        # VideoCapture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)
        self.cap.set(4, 720)

        # Configuración del dispositivo para modelos .pt y .onnx
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # Ruta del modelo
        model_path = 'BilleteV1'  # Reemplaza con el camino correcto a tu modelo .onnx o .pt
        self.billModel = YOLO(model_path)

        # Mover modelo a GPU si es un modelo .pt y CUDA está disponible
        if model_path.endswith('.pt') and self.device == 'cuda':
            self.billModel.to(self.device)

        # Clases de billetes y sus colores
        self.clsBillBank = ['200bs', '20bs', '100bs', '10bs', '50bs']
        self.bill_colors = {
            '10bs': (0, 255, 0),  # Verde
            '20bs': (0, 165, 255),  # Naranja
            '50bs': (255, 0, 255),  # Morado
            '100bs': (0, 0, 255),  # Rojo
            '200bs': (42, 42, 165)  # Café
        }

        # Balance total y balance temporal
        self.total_balance = self.balance
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
        results = self.billModel(frame)
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
                color = self.bill_colors[bill_type]
                text_obj = f'{self.clsBillBank[cls]} {int(conf * 100)}%'
                balance = self.balance_process(bill_type)

                size_obj, thickness_obj = 0.75, 1
                frame = self.draw_text(frame, color, text_obj, x1, y1, size_obj, thickness_obj, back=True)
                frame = self.draw_area(frame, color, x1, y1, x2, y2)

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
        cur.execute("UPDATE balance SET dinero = dinero + %s WHERE usuario_id = %s RETURNING dinero", (self.temp_balance, self.user_id))
        new_balance = cur.fetchone()[0]
        self.conn.commit()
        cur.close()
        self.total_balance = new_balance
        return new_balance

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
                new_balance = self.update_balance()
                self.show_balance_popup(self.temp_balance, new_balance)
                self.temp_balance = 0  # Reiniciar el saldo temporal después de actualizar
                break
            if key == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def show_balance_popup(self, added_balance, new_balance):
        popup = tk.Tk()
        popup.title("Balance Actualizado")

        label_info = tk.Label(popup, text=f"Usuario: {self.user_name}\nSe añadieron: {added_balance} bs\nSu balance actual es de: {new_balance} bs")
        label_info.pack(pady=10)

        button_frame = tk.Frame(popup)
        button_frame.pack(pady=10)

        volver_button = tk.Button(button_frame, text="Volver al inicio", command=lambda: self.return_to_dashboard(popup))
        volver_button.grid(row=0, column=0, padx=10)

        aceptar_button = tk.Button(button_frame, text="Aceptar", command=popup.destroy)
        aceptar_button.grid(row=0, column=1, padx=10)

        popup.mainloop()

    def return_to_dashboard(self, popup):
        popup.destroy()
        dashboard = Dashboard(tk.Tk(), (self.user_id, self.user_name, '', self.total_balance))

class Dashboard:
    def __init__(self, root, user):
        self.root = root
        self.user = user
        self.root.title("Dashboard")

        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(pady=20)

        self.label_name = tk.Label(self.main_frame, text=f"Bienvenido, {self.user[1]}")
        self.label_name.grid(row=0, column=0, columnspan=2)

        self.label_balance = tk.Label(self.main_frame, text=f"Balance actual: {self.user[3]} bs")
        self.label_balance.grid(row=1, column=0, columnspan=2, pady=10)

        self.cargar_saldo_button = tk.Button(self.main_frame, text="Cargar Saldo", command=self.start_bankia)
        self.cargar_saldo_button.grid(row=2, column=0, columnspan=2)

    def start_bankia(self):
        self.root.destroy()
        bank_ia = BankIA(self.user)
        bank_ia.bancoIA()

class LoginApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Login/Register")
        self.conn = create_connection()

        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(pady=20)

        self.label_title = tk.Label(self.main_frame, text="Bienvenido al Banco IA")
        self.label_title.grid(row=0, column=0, columnspan=2)

        self.label_correo = tk.Label(self.main_frame, text="Correo:")
        self.label_correo.grid(row=1, column=0)
        self.entry_correo = tk.Entry(self.main_frame)
        self.entry_correo.grid(row=1, column=1)

        self.label_contraseña = tk.Label(self.main_frame, text="Contraseña:")
        self.label_contraseña.grid(row=2, column=0)
        self.entry_contraseña = tk.Entry(self.main_frame, show="*")
        self.entry_contraseña.grid(row=2, column=1)

        self.login_button = tk.Button(self.main_frame, text="Iniciar sesión", command=self.login)
        self.login_button.grid(row=3, column=0, columnspan=2, pady=10)

        self.register_button = tk.Button(self.main_frame, text="Registrarse", command=self.register)
        self.register_button.grid(row=4, column=0, columnspan=2)

    def login(self):
        correo = self.entry_correo.get()
        contraseña = self.entry_contraseña.get()
        user = login(self.conn, correo, contraseña)
        if user:
            self.root.destroy()
            dashboard = Dashboard(tk.Tk(), user)
        else:
            messagebox.showerror("Error", "Correo o contraseña incorrectos")

    def register(self):
        reg_window = tk.Toplevel(self.root)
        reg_window.title("Registro")

        reg_frame = tk.Frame(reg_window)
        reg_frame.pack(pady=20)

        tk.Label(reg_frame, text="Nombre:").grid(row=0, column=0)
        entry_nombre = tk.Entry(reg_frame)
        entry_nombre.grid(row=0, column=1)

        tk.Label(reg_frame, text="Apellido:").grid(row=1, column=0)
        entry_apellido = tk.Entry(reg_frame)
        entry_apellido.grid(row=1, column=1)

        tk.Label(reg_frame, text="Correo:").grid(row=2, column=0)
        entry_correo = tk.Entry(reg_frame)
        entry_correo.grid(row=2, column=1)

        tk.Label(reg_frame, text="Contraseña:").grid(row=3, column=0)
        entry_contraseña = tk.Entry(reg_frame, show="*")
        entry_contraseña.grid(row=3, column=1)

        def submit_registration():
            nombre = entry_nombre.get()
            apellido = entry_apellido.get()
            correo = entry_correo.get()
            contraseña = entry_contraseña.get()
            user_id = register(self.conn, nombre, apellido, correo, contraseña)
            if user_id:
                messagebox.showinfo("Éxito", "Registro exitoso")
                reg_window.destroy()
            else:
                messagebox.showerror("Error", "Error al registrar usuario")

        submit_button = tk.Button(reg_frame, text="Registrar", command=submit_registration)
        submit_button.grid(row=4, column=0, columnspan=2, pady=10)

if __name__ == "__main__":
    root = tk.Tk()
    app = LoginApp(root)
    root.mainloop()
