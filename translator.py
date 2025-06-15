from deep_translator import GoogleTranslator

def translate_to_english(text, source_lang='auto'):
    return GoogleTranslator(source=source_lang, target='en').translate(text)
