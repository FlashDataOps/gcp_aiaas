import streamlit as st
import speech_recognition as sr
from melo.api import TTS
import os

def speech_to_text():
    rec = sr.Recognizer()
    
    if st.button('Hablar'):
        st.write("Di algo...")
        
        with sr.Microphone() as source:
            try:
                rec.adjust_for_ambient_noise(source)
                audio = rec.listen(source)
                
                to_text = rec.recognize_google(audio, language='es-ES')
                st.success(f"Texto reconocido: {to_text}")
                generate_audio_response()
            
            except sr.UnknownValueError:
                st.error("No se pudo entender el audio")

def generate_audio_response():
    text = "Los datos más relevantes a día de hoy son: las ventas han crecido un 2%, las previsiones de alojamiento son de 18900 personas para la próxima semana y la ocupación media de los hoteles ronda el 85%."
    
    speed = 1.0
    device = 'cpu'
    
    model = TTS(language='ES', device=device)
    speaker_ids = model.hps.data.spk2id
    
    output_path = 'response.wav'
    model.tts_to_file(text, speaker_ids['ES'], output_path, speed=speed)
    
    with open(output_path, 'rb') as audio_file:
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/wav')
    
    os.remove(output_path)

def main():
    st.title("Reconocimiento de Voz con Respuesta de Audio (Melo TTS)")
    speech_to_text()

if __name__ == "__main__":
    main()