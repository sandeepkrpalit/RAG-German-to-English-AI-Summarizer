from transformers import MarianMTModel, MarianTokenizer

# Initialize the MarianMT model and tokenizer for German to English translation
model_name = 'Helsinki-NLP/opus-mt-de-en'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate_text(german_text, src='de', dest='en'):
    # Ensure text is not None or empty
    if not german_text or not isinstance(german_text, str):
        return ""

    try:
        # Tokenize the input German text
        inputs = tokenizer(german_text, return_tensors="pt", padding=True)
        
        # Perform translation
        translated = model.generate(**inputs)
        
        # Decode the translated tokens to text
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        
        return translated_text
    except Exception as e:
        print(f"Error translating text: {e}")
        return ""
