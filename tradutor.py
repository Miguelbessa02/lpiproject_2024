import requests
import sys

def translate_text(text, source_lang='auto', target_lang='pt'):
    url = "https://libretranslate.de/translate"
    params = {
        "q": text,
        "source": source_lang,
        "target": target_lang,
        "format": "text"
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=params, headers=headers)
    translated_text = response.json()['translatedText']

    # Pós-processamento para ajustar para pt-PT
    translations_adjustments = {
        "mouse": "rato",
        "computer": "computador",
        "car": "carro"
    }
    for word, translation in translations_adjustments.items():
        translated_text = translated_text.replace(word, translation)

    return translated_text

if __name__ == "__main__":
    if len(sys.argv) > 1:
        text_to_translate = sys.argv[1]
        print(translate_text(text_to_translate))
