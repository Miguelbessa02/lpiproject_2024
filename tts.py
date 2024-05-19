from gtts import gTTS
import os

# Texto para ser convertido em fala
text = "Olá, como você está hoje?"
tts = gTTS(text=text, lang='pt')

# Salvar o arquivo de áudio
tts.save("hello.mp3")

# Reproduzir o arquivo de áudio
os.system("start hello.mp3")
