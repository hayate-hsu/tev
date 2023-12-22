import numpy as np
import torch

import gradio as gr

from .infer import infer, load_voc_model, get_hps
# from .utils import get_hparams_from_file
from common.conf import get_conf
conf = get_conf()

from worker import text

# hps = get_hparams_from_file(conf.voc_conf)

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
    

def generate_audio(
        slices,
        sdp_ratio,
        noise_scale,
        noise_scale_w,
        length_scale,
        speaker,
        language,
    ):
    '''
    '''
    audio_list = []
    
    # load hps
    hps = get_hps(conf.voc_conf)
    silence = np.zeros(hps.data.sampling_rate // 2, dtype=np.int16)
    
    # load voc model
    model = load_voc_model(conf.voc_model, device=conf.device, hps=hps)
    
    with torch.no_grad():
        for idx, piece in enumerate(slices):
            # 文本语言处理
            piece_au_list = []
            sentences_list = text.split_by_language(piece)
            for sentences, lang in sentences_list:
                lang = lang.upper()
                if lang == "JA":
                    lang = "JP"
                
                audio = infer(
                    sentences,
                    sdp_ratio=sdp_ratio,
                    noise_scale=noise_scale,
                    noise_scale_w=noise_scale_w,
                    length_scale=length_scale,
                    sid=speaker,
                    language=lang,
                    hps=hps,
                    model=model,
                    device=conf.device,
                )
            
                # 音频对齐，取整
                audio16bit = gr.processing_utils.convert_to_16_bit_wav(audio)
                piece_au_list.append(audio16bit)
            piece_audio = np.concatenate(piece_au_list)
            lenght, com_audio = complete_audio(piece_audio, sample_rate=hps.data.sampling_rate)
            
            audio_list.append((lenght, com_audio))
            
            # 保存音频片段
            # tmp_audio_path = '{}{}_{}.{}'.format('./tmp/', idx, piece, 'wav')
            
            # audio16bit = gr.processing_utils.convert_to_16_bit_wav(audio)
            # ret = np.concatenate((audio16bit, silence))
            # gr.processing_utils.audio_to_file(sample_rate, ret, tmp_audio_path, format=fmt)
            # audio_list.append(tmp_audio_path)
            # audio16bit = gr.processing_utils.convert_to_16_bit_wav(audio)
            # audio_list.append(audio16bit)
            # audio_list.append(silence)  # 将静音添加到列表中
    return audio_list


def concat_audios(audios, audio_file, sample_rate=44100, fmt='wav'):
    concat_audio = np.concatenate(audios)
    gr.processing_utils.audio_to_file(sample_rate, concat_audio, audio_file, format=fmt)
    
    return audio_file
