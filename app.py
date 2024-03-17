from model import *
import asyncio
import requests
import pygame
import speech_recognition


def wav_record():
    r = speech_recognition.Recognizer()

    while True:
        print("Recording... (Press Enter to stop)")
        with speech_recognition.Microphone() as source:
            audio = r.listen(source)

            if (input() == ""): 
                print("Finish Recording! Saving...")
                file_name = f"audios/recorded_audio.wav"

                with open(file_name, "wb") as f:
                    f.write(audio.get_wav_data())
                break  

    return os.getcwd() + f"/{file_name}"


def play_audio(audio_path):
    pygame.mixer.init()
    pygame.mixer.music.load(audio_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)


async def by_voice(): 
    try:
        chain = qa_bot()

        # For the Speech-to-Text Capability:
        user_audio_path = wav_record()

        stt_data = {
            "file_path": user_audio_path,
        }

        stt_response = requests.post("http://localhost:3050/stt-quechua", json=stt_data)
        if stt_response.status_code == 200: 
            print("...")
        else: 
            print(f"Error: {stt_response.text}")

        user_transcription = f"> {stt_response.text}"
        print(user_transcription)

        # === CHATBOT PROCESS ===
        user_language =  'qu' #detect_language(user_transcription)
        final_query = user_transcription

        if user_language != 'en':
            final_query = translate_google(user_transcription, 'en')
    
        res = await chain.acall(final_query)
        answer = res["result"]

        final_answer = translate_google(answer, user_language) if user_language != 'en' else answer
        print(f"Kutichiy: {final_answer}")

        # For the Text-to-Speech Capability:
        tts_data = {
            "text": final_answer
        }

        tts_response = requests.post("http://localhost:3050/tts-quechua", json=tts_data)

        if tts_response.status_code == 200: 
            print("...")
            play_audio(os.getcwd() + "/audios/audio_generated.wav")
            print("Ñam!!")
            
        else: 
            print(f"Error: {tts_response.text}")

    except Exception as e: 
        print(f"Error: {e}")


async def main():
    print("Allin hamusqaykichik Hampi Kunka Yanapakuqman")
    await by_voice()

        
if __name__ == "__main__":
    asyncio.run(main())
