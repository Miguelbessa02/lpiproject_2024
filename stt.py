import speech_recognition as sr
import subprocess
import json

recognizer = sr.Recognizer()

languages = {
    '1': ('Português (Brasil)', 'pt-BR'),
    '2': ('Português (Portugal)', 'pt-PT'),
    '3': ('English (US)', 'en-US'),
    '4': ('Español', 'es-ES')
}

print("Escolha o idioma:")
for key, value in languages.items():
    print(f"{key}: {value[0]}")

language_choice = input("Digite o número do idioma: ")
language = languages.get(language_choice, ('English (US)', 'en-US'))[1]

with sr.Microphone() as source:
    print("Fale algo:")
    audio = recognizer.listen(source)

try:
    recognized_text = recognizer.recognize_google(audio, language=language)
    print("Você disse: " + recognized_text)
    # Chamar tradutor.py passando o texto reconhecido
    result = subprocess.run(['python', 'tradutor.py', recognized_text], capture_output=True, text=True)
    print("Traduzido para Português: " + result.stdout)
except sr.UnknownValueError:
    print("Não pude entender o áudio")
except sr.RequestError as e:
    print("Não foi possível solicitar resultados do serviço de reconhecimento de fala: {}".format(e))
