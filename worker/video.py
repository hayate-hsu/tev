import os
import sys
import glob
import shutil

from typing import Any, Callable, Dict, List, Tuple


import numpy as np
# import cv2

import ffmpeg

from .image import adjust_image_size
from db import dao

# 日志
from common.log import get_logger
logger = get_logger()

# 根据秒数还原 例如 10829s 转换为 03:04:05
def getTime(t: int):
    m,s = divmod(t, 60)
    h, m = divmod(m, 60)
    t_str = "%02d:%02d:%02d" % (h, m, s)
    return t_str

ffmpeg_video_args = {
    'format':'rawvideo',
    'pix_fmt':'rgb24',
    'frame_pts':True,
    'vsync':0,
    'vf': f'fps=1.0',           # 帧率（设置为1，则每秒取一帧）
}

def extract_video_frames(video_path:str, ffmpeg_args:Dict):
    '''
        提取视频帧，根据vf（默认1.0），导出相应帧数
    '''
    video_frames = []
    try:
        # get width and height
        video = ffmpeg.probe(video_path)['streams'][0]
        w, h = ffmpeg_args.get('s', f'{video["width"]}x{video["height"]}').split('x')
        w = int(w)
        h = int(h)
        out, _ = (
            ffmpeg.input(video_path)
            .output('pipe:', **ffmpeg_args)
            .run(capture_stdout=True, quiet=True)
        )

        video_frames = np.frombuffer(out, np.uint8) #.reshape([-1, h, w, 3])
        # print(video_frames.shape)
        video_frames = video_frames.reshape([-1, h, w, 3])
    except ffmpeg.Error as e:
        logger.error('Frame extraction failed, {}, {}'.format(video_path, e.stderr), exc_info=True)

    return video_frames

def extract(video_path, max_size=224, ffmpeg_args=ffmpeg_video_args):
    '''
        video_path: 视频文件路径
        max_size: 图片视图大小（最大边）
    '''
    frame_tensors = extract_video_frames(video_path, ffmpeg_args)
    chunks = []
    print(frame_tensors.shape)
    for idx, frame_tensor in enumerate(frame_tensors):
        image = adjust_image_size(frame_tensor, max_size)
        chunks.append(image)
       
    return video_path, chunks

def extract_video_from_folder(video_path, ext='*.mp4'):
    vf, imgs = [], []           # 用于存储检索到的video文件 以及其对应的图片帧
    for fn in glob.glob(os.path.join(video_path, ext)):
        f, img_frames = extract(fn)
        
        vf.append(f)
        imgs.append(img_frames)
    
    return vf, imgs

def load_video(db, video_path, ext='*.mp4'):
    vf, imgs = [], []           # 用于存储检索到的video文件 以及其对应的图片帧
    for fn in glob.glob(os.path.join(video_path, ext)):
        fn = fn.replace('\\', '/')
        queries = {'url':{'$eq': fn}}
        if dao.filter_one(db, queries):       # 视频文件已经load并处理，其关键帧信息已经保存在db中
            continue
        f, img_frames = extract(fn)
        vf.append(f)
        imgs.append(img_frames)
    return vf, imgs

def imgs_to_video(images, lenght, output):
    '''
    给定图片列表，根据图片顺序，以及给定目标视频长度，合成视频流。
    '''
    # 复制images 至 cache 目录
    cache_path = './cache/images'
    try:
        os.rmdir(cache_path)      # 删除换成目录
    except:
        pass
    
    os.makedirs(cache_path, exist_ok=True)
    input_images = ''
    for idx, img in enumerate(images):
        dst = '{}/image_{:02d}.jpg'.format(cache_path, idx)
        shutil.copy(img, dst)
        
        input_images += f'-i {dst} '
        
    frame_rate = len(images)/lenght
   
    # -s 1080x1920
    # 拼接缓存目录的的 图片，进行有序拼接
    logger.info(f'图片至视频：ffmpeg -framerate {frame_rate} -f image2 -i ./cache/images/image_%02d.jpg -c:v libx264 -t {lenght} -r 30 -pix_fmt yuv420p {output}')
    os.system(f'ffmpeg -framerate {frame_rate} -f image2 -i ./cache/images/image_%02d.jpg -c:v libx264 -t {lenght} -r 30 -pix_fmt yuv420p {output}')
    
def init_blk_img(height, width, save_path):
    from PIL import Image
    arr = np.zeros((height,width), dtype=np.uint8)
    img = Image.fromarray(arr)
    
    img.save(save_path)
    

def concat_videos(videos, output):
    '''
    拼接多条视频至目标视频
    '''
    input_v = ''
    concat_v = ''
    for idx,video in enumerate(videos):
        input_v += f'-i {video} '
        concat_v += f'[{idx}:v]'
        
    concat_v += f'concat={len(videos)}:v=1:a=0[outv]'
    
    logger.info(f'拼接视频：ffmpeg {input_v} -filter_complex "{concat_v}" -map "[outv]" -strict -2 {output}')
    os.system(f'ffmpeg {input_v} -filter_complex "{concat_v}" -map "[outv]" -strict -2 {output}')
    
    
# 根据传入的时间戳位置对视频进行截取
def cutVideo(start_t: str, length: int, input: str, output: str):
    """
    start_t: 起始位置
    length: 持续时长
    input: 视频输入位置
    output: 视频输出位置
    """
    # os.system(f'ffmpeg -ss {start_t} -i {input} -t {length} -c:v copy -c:a copy -y {output}')
    logger.info(f'cut video: ffmpeg -ss {start_t} -i {input} -t {length} -c:v copy -y {output}')
    os.system(f'ffmpeg -ss {start_t} -i {input} -t {length} -c:v copy -y {output}')
    
def compose(video_f, audio_f, output):
    '''
    音频、视频组合
    '''
    logger.info(f'compose video&audio: ffmpeg -i {video_f} -i {audio_f} -c:v copy -c:a aac -strict experimental {output}')
    os.system(f'ffmpeg -i {video_f} -i {audio_f} -c:v copy -c:a aac -strict experimental {output}')
