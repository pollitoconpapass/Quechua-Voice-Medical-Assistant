# === HERE WILL GO ALL THE ENDPOINTS OF THE TEXT-TO-SPEECH AND SPEECH-TO-TEXT == 
import io
import os
import torch
import uvicorn
import torchaudio
import numpy as np
import soundfile as sf
from fastapi import FastAPI
from jsonschema import validate
from transformers import VitsModel, AutoTokenizer
from transformers import Wav2Vec2ForCTC, AutoProcessor


app = FastAPI()


# === TEXT-TO-SPEECH CAPABILITY ===
@app.post("/tts-quechua")
async def tts(data: dict):
    global audio_count
    try:
        schema = {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "audio-name": {"type": "string"},
            },
            "required": ["text"]
        }

        validate(instance=data, schema=schema)

        text = data["text"]

        # === LOAD THE HUGGING FACE MODELS ===
        selected_quechua = "facebook/mms-tts-quz"  
        model = VitsModel.from_pretrained(selected_quechua)  # -> loads the model itself
        tokenizer = AutoTokenizer.from_pretrained(selected_quechua)   # ... token

        # === GENERATE THE AUDIO FILE ===
        inputs = tokenizer(text, return_tensors="pt") 

        with torch.no_grad():  # -> ensures no gradients (as only inference)
            output = model(**inputs).waveform.cpu().numpy()  # -> generates the audio w model and inputs
        output = output / np.max(np.abs(output))  # -> normalizes the waveform

        # === SAVE THE AUDIO FILE ===
        filename = "audio_generated.wav"

        filepath = os.path.join("audios", filename)
        rate = int(model.config.sampling_rate)
        sf.write(filepath, output.T, rate, subtype='PCM_16')
        return filepath
    
    except Exception as e:
        return {"error": str(e)}


# === SPEECH-TO-TEXT CAPABILITY ===
@app.post("/stt-quechua")
async def stt(data: dict):
    try:
        model_id = "facebook/mms-1b-all"
        processor = AutoProcessor.from_pretrained(model_id)
        model = Wav2Vec2ForCTC.from_pretrained(model_id)

        wav_file_path = data["file_path"]
        audio_data, original_sampling_rate = torchaudio.load(wav_file_path)
        resampled_audio_data = torchaudio.transforms.Resample(original_sampling_rate, 16000)(audio_data)

        processor.tokenizer.set_target_lang("quz")
        model.load_adapter("quz")

        inputs = processor(resampled_audio_data.numpy(), sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs).logits

        ids = torch.argmax(outputs, dim=-1)[0]
        transcription = processor.decode(ids)

        return transcription
    
    except Exception as e:
        return {"error": str(e)}
    

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=3050, log_level="info")
