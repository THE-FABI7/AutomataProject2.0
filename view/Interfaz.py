import json
from tkinter import Tk, filedialog
import tkinter
import networkx as nx
import matplotlib.pyplot as plt
import tkinter
from tkinter import ttk
from PIL import Image, ImageTk
import tkinter as tk

from controller.AutomataController import load_nfa_from_json, load_nfa_from_txt


def crear_interfaz():
    root = tkinter.Tk()
    root.title("Interfaz del Autómata")

    # Cargar imagen de fondo
    image = Image.open("./image/fondolindo.jpg")
    # Ajustar el tamaño de la imagen de fondo
    new_width = root.winfo_screenwidth()
    new_height = root.winfo_screenheight()
    image = image.resize((new_width, new_height), Image.ANTIALIAS)
    background_image = ImageTk.PhotoImage(image)

    # Configurar un widget Label con la imagen de fondo
    background_label = tk.Label(root, image=background_image)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)

    # Estilos personalizados
    style = ttk.Style()
    style.configure("TFrame",
                    background="black")
    style.configure("Green.TButton",
                    background="#006400",
                    foreground="Black")
    style.configure("Red.TButton",
                    background="#8B0000",
                    foreground="Black")

    # Contenedor principal
    main_frame = ttk.Frame(root, style="TFrame")
    main_frame.pack(expand=True)

    # Espacio en blanco para centrar los botones
    spacer_frame = ttk.Frame(main_frame, style="TFrame")
    spacer_frame.pack(expand=True, padx=50, pady=50)

    # Botón de cargar autómata
    cargar_button = ttk.Button(spacer_frame, text="Cargar Autómata(json)", command=load_nfa_from_json, style="Green.TButton")
    cargar_button.pack(pady=20, ipadx=10, ipady=10)


    cargar_button = ttk.Button(spacer_frame, text="Cargar Autómata(txt)", command=load_nfa_from_txt, style="Green.TButton")
    cargar_button.pack(pady=20, ipadx=10, ipady=10)



    # Separador
    separator = ttk.Separator(main_frame, orient="horizontal")
    separator.pack(fill="x", pady=10)
    # Botón de cerrar
    cerrar = ttk.Button(main_frame, text="Cerrar", command=root.destroy, style="Red.TButton")
    cerrar.pack(pady=10, ipadx=10, ipady=10)

    # Centrar la ventana en la pantalla
    window_width = root.winfo_width()
    window_height = root.winfo_height()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x_position = int((screen_width - window_width) / 2)
    y_position = int((screen_height - window_height) / 2)
    root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

    root.mainloop()