import os
import sys

from typing import Any, Callable, Dict, List, Tuple
import shutil
import glob

import numpy as np
import cv2
from PIL import Image

from docarray import DocList
from docarray.documents import ImageDoc

from common.log import get_logger
logger = get_logger()

from common.conf import get_conf
conf = get_conf()

from db import dao

def adjust_image_size(image, max_size=224):
    height, width = image.shape[:2]
    if height > width:
        if height > max_size:
            height, width = max_size, int(max_size / height * width)
    else:
        if width > max_size:
            height, width = int(max_size / width * height), max_size
    image = cv2.resize(image, (width, height))
    return image

def load_imgs(path, max_size:int=224):
    fa,imgs = [],[]                   # 文件路径，图片内容
    for fn in glob.glob(os.path.join(path, '*.jpg')):
        # img = Image.fromarray(dimg.tensor)
        # image = cv2.imread(fn, cv2.IMREAD_COLOR)  # 文件路径转换为gbk，兼容中文路径
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.imdecode(np.fromfile(fn, dtype=np.uint8), -1)       # RGB格式
        image = adjust_image_size(image, max_size=max_size)   # 模型执行 prepossessor 会处理缩放，为了减少内存占用，这里也做处理
        
        # item = {'uri': fn, 'img':np.asarray(img).astype('uint8')}
        fa.append(fn)
        imgs.append(image)          # image np.ndarray
        
    return fa, imgs

def load_imgs2(path, max_size:int=224):
    imgs = DocList[ImageDoc]()
    for fn in glob.glob(os.path.join(path, '*.jpg')):
        img = ImageDoc(url=fn)
        img.tensor = adjust_image_size(img.url.load())
        imgs.append(img)
    
    return imgs


def load_imgs3(image_path, db):
    '''
        搜索并加载图片，如果图片已经存储在db中，则跳过，未存储， 则读取图片并处理
    '''
    try:
        imgs = []
        for fn in glob.glob(os.path.join(image_path, '*.jpg')):
            # find fn ?
            fn = fn.replace('\\', '/')
            if dao.filter_one(db, {'url':{'$eq':fn}}):  # 图片已经保持在库中
                continue
            img = ImageDoc(url=fn)
            img.tensor = adjust_image_size(img.url.load())
            imgs.append(img)
    except Exception as e: 
        logger.error('读取图片错误', exc_info=True)
    return imgs

def create_blk_img(blk_img, width, height):
    '''
    根据宽x高，创建黑色背景图片
    '''
    arr = np.zeros((height,width), dtype=np.uint8)
    img = Image.fromarray(arr)

    img.save(blk_img)
    
def scale_image(img_path, output, width, height):
    '''
    对图片进行等比例缩放，至少一边达到目标（width，height）要求；
    不足部分用黑色补充。
    '''
    img = Image.open(img_path)
    w,h = img.size
    
    if w==width and h==height:
        # 图片与目标大小一致
        img.close()
        shutil.copy(img_path, output)
        return 
    
    scale = min(width/w, height/h)
    
    nw, nh = int(w*scale, h*scale)
    
    img = img.resize((nw,nh), Image.BICUBIC)
    
    new_img = Image.new('RGB', (width, height), (0,0,0))
    new_img.paste(img, ((w-nw)//2, (h-nh)//2))
    
    new_img.save(output)
    


        