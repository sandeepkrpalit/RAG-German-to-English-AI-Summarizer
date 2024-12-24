from googletrans import Translator

translator = Translator()

def translate_text(german_text, src="de", dest="en"):
    """
    Translates German text into English.

    Args:
        german_text (str): Text in German to be translated.
        src (str): Source language.
        dest (str): Target language.

    Returns:
        str: Translated English text.
    """
    result = translator.translate(german_text, src=src, dest=dest)
    return result.text
