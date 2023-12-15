import os
import sys
from typing import Any, Callable, Dict, List, Tuple
import shutil
import datetime

# 日志
from common.log import get_logger
logger = get_logger()

from common.conf import get_conf
conf = get_conf()

from .text import split_text
from . import video


from voc import audio

from worker.search import find_image, find_video

def synthesis(texts, video_folder, img_folder, **kwargs):
    '''
    video_first: 默认视频优先
    补充空白长度：
    '''
    # 初始化工作环境
    init_env()
    
    docs = split_text(texts)
    
    logger.info(f'split_text: {docs}')
    
    audio_results = audio.generate_audio(
        docs, 
        sdp_ratio=0.2, 
        noise_scale=0.6,
        noise_scale_w=0.8,
        length_scale=1.0,
        speaker='fangqi',
        language='ZH'
        )
    
    # 视频搜索
    video_results = find_video(texts, video_folder)
    
    # 图片搜索
    image_results = find_image(texts, img_folder)
    
    docs_videos = []        # 记录每段文本对应的视频文件
    
    # 处理视频剪辑
    for idx, doc in enumerate(docs):
        if doc == conf.negativate_class:        # 
            continue
        
        lenght = audio_results[idx][0]      #获取对应文本的音频长度，根据音频长度确定裁剪的视频片段长度    
        lenght_total =  lenght
        videos = video_results.get(doc, []) # 有可能检索不到视频片段
        
        # 置信度降序排列
        r_videos = sorted(videos, key=lambda item:item['score'], reverse=True)
        
        # 处理视频
        ret_videos = []
        for item in r_videos:
            left, right = item['leftIndex'], item['rightIndex']
            
            # 处理计算
            if (right+1) - left >= lenght:
                item['rightIndex'] = left+lenght-1
                ret_videos.append(item)         #视频片段足够长，剪辑这部分即可
                lenght = 0
                break
            else:
                ret_videos.append(item) 
                lenght = lenght - (right + 1 - left)
                
        # ret_videos : 视频片段，进行视频剪辑
        r_videos = cut_fragment(ret_videos, idx, doc, conf.cache_path)
        
        logger.info('find: {}, from videos, lenght: {}, left: {}\nvideos file: {}\n'.format(doc, lenght_total-lenght, lenght, r_videos))

        # 处理图片,将图片均匀的分布在 时长lenght的视频片段。视频合成
        # 视频长度不足，需要搜索图片来补充
        imgs = image_results.get(doc, [])       # 有可能检索不到图片
        if lenght & len(imgs):
            r_imgs = sorted(imgs, key=lambda item:item['score'], reverse=True)  
            r_imgs = [img.url for img in r_imgs]
            
            output = conf.cache_path + "/{}_image_tov.mp4".format(idx)
            video.imgs_to_video(r_imgs, lenght, output)
            r_videos.append(output)
            lenght = 0
            
            logger.info('find: {}, from images:{}, lenght:{}\n, video file:{}\n'.format(doc, r_imgs, lenght, output))
            
        # 剩余长度不为0，意味着检索到的视频和图片不足以满足视频长度,以黑频替代，后续人工处理
        if lenght:
            logger.warning(f'found {doc} results 不足以满足视频剪辑,将使用空白帧填充\n')
            
            bk_img = conf.bkg
            output = conf.cache_path + "/{}_bkg_image.mp4".format(idx)
            video.imgs_to_video([bk_img], lenght, output)
            r_videos.append(output)
            lenght = 0
            logger.info('find: {}, from blackground, lenght:{}\n, video file:{}\n'.format(doc, lenght, output))
            # raise Exception
            
        # concat video
        doc_video = concat_fragments(r_videos, idx, doc, conf.cache_path)
        docs_videos.append(doc_video)
        
    # 拼接处理音频
    audio_file = conf.cache_path + '/audio.wav'
    audios = [ item[1] for item in audio_results]
    audio_file = audio.concat_audios(audios, audio_file)
    
    # 拼接处理视频
    ret_video = concat_fragments(docs_videos, -1, docs, conf.cache_path)
    
    # 视频+音频合成 输出目标视频, 视频名称由result+当前时间组成
    now =  datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')
    result_v = './output' + f'/result_{now}.mp4'          # 添加第一段文本内容
    video.compose(ret_video, audio_file, result_v)
    
    return ret_video     
    
def cut_fragment(fragments, i, doc, cache_path):
    r_videos = []
    for idx,item in enumerate(fragments):
        left, right = item['leftIndex'], item['rightIndex']
        # duration = right - left 
        start = video.getTime(left) # 将其转换为标准时间
        
        max_index = item['maxImage']['index']
        uri = item['maxImage']['uri']
        
        output = cache_path + "/{}_{}.mp4".format(i, idx)

        logger.info('text:{}, cut video:{} from: {} to: {}. output:{}'.format(doc, uri, left, right, output))
        video.cutVideo(start,right+1-left, uri, output) # 对视频进行切分，视频分段是 包含两边的
        r_videos.append(output)
        
    return r_videos

def concat_fragments(videos, idx, doc, cache_path):
    '''
    拼接视频
    '''
    doc_video = cache_path + '/{}_video.mp4'.format(idx)
    if len(videos) > 1:        
        video.concat_videos(videos, doc_video)
    elif len(videos) == 1:
        shutil.copy(videos[0], doc_video)
    else:
        logger.warning('text:{}, found no videos'.format(doc))
        raise Exception(msg='text:{}, found no videos'.format(doc))
        
    return doc_video


def init_env():
    '''
    清空缓存
    '''
    try:
        shutil.rmtree(conf.cache_path)
    except FileNotFoundError:
        pass
    os.makedirs(conf.cache_path)
    os.makedirs(conf.output_path, exist_ok=True)
    

