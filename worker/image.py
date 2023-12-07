import os
import sys

from typing import Any, Callable, Dict, List, Tuple

import math
import glob

import numpy as np
import PIL
from PIL import Image
import cv2

from docarray import DocList
from docarray.documents import ImageDoc

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
    imgs = []
    for fn in glob.glob(os.path.join(image_path, '*.jpg')):
        # find fn ?
        if dao.filter_one(db, {'url':{'$eq':fn}}):  # 图片已经保持在库中
            continue
        img = ImageDoc(url=fn)
        img.tensor = adjust_image_size(img.url.load())
        imgs.append(img)
        
    return imgs

def find_from_vs(path, vs):
    # load from vs , if not found ,load from disk
    for fn in glob.glob(os.path.join(path, '*.jpg')):
        query = {'uri': {'$eq': fn}}
        docs =  vs.filter(query)           # 路径名为uri图片的features
        if len(docs) == 1:
            pass
        else:
            pass
        