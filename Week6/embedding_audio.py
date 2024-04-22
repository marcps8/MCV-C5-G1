import librosa
import numpy as np

import whisper


from transformers import WhisperModel, WhisperFeatureExtractor

from transformers import WhisperProcessor, WhisperForConditionalGeneration


import torch.nn as nn
import torch 

def extract_features(file_name: str):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    # Maximum embedding size allowed by the model (=128)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=128)
    mfccs_processed = np.mean(mfccs.T, axis=0)
     
    # Shape is (128)
    return mfccs_processed




@torch.no_grad()
def extract_features_from_whisper(file_name: str):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 

    model = WhisperModel.from_pretrained("openai/whisper-base")

    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small",  sampling_rate=sample_rate)

    inputs = feature_extractor(audio, return_tensors="pt", sampling_rate = sample_rate)

    input_features = inputs.input_features

    decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
    last_hidden_state = model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state
    
    return torch.mean(last_hidden_state, dim=1).squeeze()

# Example
if __name__ == "__main__":
    test = "/ghome/group01/MCV-C5-G1/Week6/data/train/1/-2qsCrkXdWs.001.wav"
    audio, sample_rate = librosa.load(test, res_type='kaiser_fast', sr=16000) 

    #melt = (extract_features_from_whisper(test))
    #m = extract_features(test)

    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
    input_features = processor(
    audio, sampling_rate=sample_rate, return_tensors="pt").input_features
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    print(transcription[0])