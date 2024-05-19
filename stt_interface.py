import tkinter as tk
from tkinter import ttk
import speech_recognition as sr

def recognize_speech():
    lang_name = language_var.get()  # Isso pega o nome do idioma, como "Português (Brasil)"
    lang_code = language_choices.get(lang_name, 'en-US')  # Isso converte o nome para o código do idioma
    with sr.Microphone() as source:
        print("Fale algo:")
        audio = recognizer.listen(source)
    try:
        # Agora usa o código do idioma correto
        result = recognizer.recognize_google(audio, language=lang_code)
        result_label.config(text="Resultado: " + result)
    except sr.UnknownValueError:
        result_label.config(text="Não foi possível entender o áudio")
    except sr.RequestError as e:
        result_label.config(text="Erro no serviço de reconhecimento: {0}".format(e))

# Configuração da interface gráfica
root = tk.Tk()
root.title("Reconhecimento de Fala")

recognizer = sr.Recognizer()

language_var = tk.StringVar()
language_choices = {'Português (Brasil)': 'pt-BR', 'Português (Portugal)': 'pt-PT', 'English (US)': 'en-US', 'Español': 'es-ES'}
language_var.set('en-US')  # define o padrão como Inglês

language_label = tk.Label(root, text="Escolha o idioma:")
language_label.pack()

language_menu = ttk.Combobox(root, textvariable=language_var, values=list(language_choices.keys()))
language_menu.pack()

recognize_button = tk.Button(root, text="Reconhecer Fala", command=recognize_speech)
recognize_button.pack()

result_label = tk.Label(root, text="Resultado:")
result_label.pack()

root.mainloop()
