import tkinter as tk
from tkinter import messagebox

def somar():
    try:
        num1 = float(entry1.get())
        num2 = float(entry2.get())
        resultado = num1 + num2
        messagebox.showinfo("Resultado", f"A soma é: {resultado}")
    except ValueError:
        messagebox.showerror("Erro", "Por favor, insira números válidos")

# Criar a janela principal
root = tk.Tk()
root.title("Calculadora Simples")

# Criar os componentes
label1 = tk.Label(root, text="Número 1:")
label1.pack()

entry1 = tk.Entry(root)
entry1.pack()

label2 = tk.Label(root, text="Número 2:")
label2.pack()

entry2 = tk.Entry(root)
entry2.pack()

button = tk.Button(root, text="Somar", command=somar)
button.pack()

# Executar a aplicação
root.mainloop()
