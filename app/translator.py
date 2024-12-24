from googletrans import Translator

def translate_text(text, src="de", dest="en"):
    translator = Translator()
    return translator.translate(text, src=src, dest=dest).text
