import os
import sys
import subprocess

import librosa
import numpy as np

import gradio as gr

from common.conf import get_conf
conf = get_conf()

# 日志
from common.log import get_logger
logger = get_logger()

def generate_audio(
        slices,
        _rate,
        _volume,
        _lang,
        _gender,
        sample_rate=44100
    ):
    '''
    _lang = 'Auto'
    '''
    audio_cached = conf.cache_path + '/audio'
    os.makedirs(audio_cached, exist_ok=True)
    
    _rate = f"+{int(_rate*100)}%" if _rate >= 0 else f"{int(_rate*100)}%"
    _volume = f"+{int(_volume*100)}%" if _volume >= 0 else f"{int(_volume*100)}%"
    _gender = "Male" if _gender == "男" else "Female"
    
    audio_files = []
    for idx, text in enumerate(slices):
        _output = conf.cache_path + f'/audio/{idx}.wav'
        audio_files.append(_output)
        subprocess.run([sys.executable, "edgetts/tts.py", text, _lang, _rate, _volume, _gender, _output], check=True)
        
    # 读取音频文件,并对齐
    audio_list = []
    for a_f in audio_files:
        audio16bit_arr = load_audio(a_f, rate=sample_rate)
        lenght, com_audio = complete_audio(audio16bit_arr, sample_rate=sample_rate)
        audio_list.append((lenght, com_audio))
        
    return audio_list
    
def load_audio(audio_file, rate):
    '''
    读取音频文件，并进行重采样
    '''
    y, sr = librosa.load(audio_file)
    resampled_y = librosa.resample(y, orig_sr=sr, target_sr=rate)
    
    result = convert_to_16_bit_wav(resampled_y)     # float32 t0 int16
    
    return result

def concat_audios(audios, audio_file, sample_rate=44100, fmt='wav'):
    concat_audio = np.concatenate(audios)
    gr.processing_utils.audio_to_file(sample_rate, concat_audio, audio_file, format=fmt)
    
    return audio_file

def convert_to_16_bit_wav(data):
    # Based on: https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.write.html
    warning = "Trying to convert audio automatically from {} to 16-bit int format."
    if data.dtype in [np.float64, np.float32, np.float16]:
        logger.warn(warning.format(data.dtype))
        data = data / np.abs(data).max()
        data = data * 32767
        data = data.astype(np.int16)
    elif data.dtype == np.int32:
        logger.warn(warning.format(data.dtype))
        data = data / 65538
        data = data.astype(np.int16)
    elif data.dtype == np.int16:
        pass
    elif data.dtype == np.uint16:
        logger.warn(warning.format(data.dtype))
        data = data - 32768
        data = data.astype(np.int16)
    elif data.dtype == np.uint8:
        logger.warn(warning.format(data.dtype))
        data = data * 257 - 32768
        data = data.astype(np.int16)
    else:
        raise ValueError(
            "Audio data cannot be converted automatically from "
            f"{data.dtype} to 16-bit int format."
        )
    return data
        
        
def complete_audio(audio, sample_rate=44100):
    '''
    音频补齐（整数秒），后面增加空白音频。
    bremainder : 是否补齐；true，补齐，false，不补齐
    '''
    
    lenght = audio.shape[0]
    
    quotient, remainder = divmod(lenght, sample_rate)
    complement = int(sample_rate - remainder)
        
    silence = np.zeros(complement, dtype=np.int16)
    
    ret_audio = np.concatenate((audio, silence))
    
    return quotient+1, ret_audio