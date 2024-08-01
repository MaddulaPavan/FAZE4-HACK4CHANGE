import speech_recognition as sr
from googletrans import Translator
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gtts import gTTS
import gradio as gr
import tempfile
import os

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

recognizer = sr.Recognizer()
translator = Translator()

def natural_language_understanding(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    processed_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return " ".join(processed_tokens)

def translate_text(text, target_language):
    translated = translator.translate(text, dest=target_language)
    return translated.text

def text_to_speech(text):
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        return fp.name

def process_input(input_text, input_audio, feature, target_language, output_language):
    if input_audio is not None:

        try:
            with sr.AudioFile(input_audio) as source:
                audio = recognizer.record(source)
            input_text = recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return "Could not understand audio", None
        except sr.RequestError:
            return "Could not request results from speech recognition service", None
        except Exception as e:
            return f"An error occurred: {str(e)}", None

    if not input_text:
        return "No input provided", None

    processed_text = natural_language_understanding(input_text)

    if feature == "Translation":
        result = translate_text(processed_text, target_language)
    elif feature == "Voice Command":
        result = "Voice command feature not implemented in this example"

    elif feature == "Transcription":
        result = processed_text
    else:
        result = "Invalid feature selected"

    if output_language:
        result = translate_text(result, output_language)

    return result, None

def tts_function(text):
    if text:
        return text_to_speech(text)
    return None


with gr.Blocks() as iface:
    gr.Markdown("# The Advanced Multi-Faceted Chatbot")
    gr.Markdown("Enter text or speak to interact with the chatbot. Choose a feature and specify languages for translation if needed.")

    with gr.Row():
        input_text = gr.Textbox(label="Input Text")
        input_audio = gr.Audio(label="Input Audio", type="filepath")

    with gr.Row():
        feature = gr.Radio(["Translation", "Voice Command", "Transcription"], label="Feature")
        target_language = gr.Textbox(label="Target Language ")
        output_language = gr.Textbox(label="Output Language ")

    submit_button = gr.Button("Process")

    result_text = gr.Textbox(label="Result")
    tts_button = gr.Button("Convert to Speech")
    audio_output = gr.Audio(label="Audio Output")

    submit_button.click(
        process_input,
        inputs=[input_text, input_audio, feature, target_language, output_language],
        outputs=[result_text, audio_output]
    )

    tts_button.click(
        tts_function,
        inputs=[result_text],
        outputs=[audio_output]
    )


iface.launch()
