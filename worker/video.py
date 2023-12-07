import os
import sys
import glob

from typing import Any, Callable, Dict, List, Tuple


import numpy as np
import PIL
from PIL import Image
# import cv2

import ffmpeg

from .image import adjust_image_size
from db import dao

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
        print('Frame extraction failed, {}, {}'.format(video_path, e.stderr))

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
        image = adjust_image_size(frame_tensor, max_size=224)
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
        queries = {'url':{'$eq': fn}}
        if dao.filter_one(db, queries):       # 视频文件已经load并处理，其关键帧信息已经保存在db中
            continue
        f, img_frames = extract(fn)
        vf.append(f)
        imgs.append(img_frames)
    return vf, imgs