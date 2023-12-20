import os
import sys
from typing import Any, Callable, Dict, List, Tuple
import functools

import PIL
from PIL import Image
import numpy as np

from docarray import DocList

import torch

import cn_clip.clip as clip
from cn_clip.clip import load_from_name

# 日志
from common.log import get_logger
logger = get_logger()

from common.conf import get_conf
conf = get_conf()

from .text import split_text
from .image import load_imgs3
from .video import load_video

from db import dao
from document import ImageFeature

@functools.lru_cache
def load_chinese_clip(
    name: str = "ViT-H-14", download_root='./') -> Tuple[torch.nn.Module, Callable[[PIL.Image.Image], torch.Tensor]]:
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = conf.device
    model, preprocess = load_from_name(name, device=device, download_root=download_root)
    model.eval()
    
    logger.info('load chinese clip, model: {}, device:{}'.format(name, device))
    return model.to(device), preprocess

def encode_text(model, docs, device):
    '''
    '''
    docs_token = clip.tokenize(docs).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(docs_token)
    
    return text_features

def encode_image(model, preprocess, imgs, device):
    img_features = None
    
    with torch.no_grad():
        for index, img in enumerate(imgs):
            image = preprocess(Image.fromarray(img)).unsqueeze(0).to(device)
            img_feature = model.encode_image(image)
            
            if img_features != None: 
                img_features = torch.cat([img_features,img_feature], dim=0)
            else:
                img_features = img_feature  
    
    return img_features

def get_similarity(model, preprocess, imgs, texts, device='cuda'):
    '''
        probs : 如果texts，带有负向类型（other），则删除other 对应的列
    '''
    img_features = encode_image(model, preprocess, imgs, device=device)    
                
    text_features = encode_text(model, texts, device=device)                   # texts : 分割后的文本数组
        
    # 对特征进行归一化，请使用归一化后的图文特征用于下游任务
    img_features /= img_features.norm(dim=-1, keepdim=True) 
    text_features /= text_features.norm(dim=-1, keepdim=True)    
    
    # cosine similarity as logits
    logit_scale = model.logit_scale.exp()
    logits_per_image = logit_scale * img_features @ text_features.t()
    logits_per_text = logits_per_image.t()
    
    probs = logits_per_image.softmax(dim=-1).cpu().detach().numpy()
    
    # 如果最后一个类别是 other，则最后返回可信度时删除other列对应的可信度数据 
    if texts[-1] == conf.negativate_class:
        probs = probs[:,0:len(texts)-1]
    return probs     # 去掉unknown 列结果

def get_similarity2(model, imgs_f, texts_f, device='cuda'):
    # cosine similarity as logits
    imgs_f = imgs_f.to(device).to(torch.float32)
    texts_f = texts_f.to(device).to(torch.float32)
    
    logit_scale = model.logit_scale.exp()
    logits_per_image = logit_scale * imgs_f @ texts_f.t()
    logits_per_text = logits_per_image.t()
    
    probs = logits_per_image.softmax(dim=-1).cpu().detach().numpy()
    
    return probs

def getMultiRange(results: list, threshold = 0.5, dValue=0.1, maxCount: int = 3, length: int = 10):
    '''
        results : 
        thod: 视频帧score > thod, 才会被检索出
        dValue： 视频帧score - 最大帧score差值 要在 dvalue范围内
        maxCount： 最多寻着几个片段?
    '''
    ignore_range = {}
    index_list = []
    for i in range(maxCount):
        maxItem = getNextMaxItem(results, ignore_range, threshold=threshold)
        if maxItem is None:
            break
        # print(maxItem["score"])
        leftIndex, rightIndex, maxImage = getRange(maxItem, results, dValue, ignore_range, length=length)
        index_list.append({
            "leftIndex": leftIndex,
            "rightIndex": rightIndex,
            "maxImage": maxImage,
            "score": maxImage['score']
        })
        # 将已经提取的片段，添加到忽略列表
        if maxImage["uri"] in ignore_range:
            ignore_range[maxImage["uri"]] += list(range(leftIndex, rightIndex + 1))     
        else:
            ignore_range[maxImage["uri"]] = list(range(leftIndex, rightIndex + 1))
    # print(ignore_range)
    return index_list


def getNextMaxItem(results: list, ignore_range, threshold=0.5):
    '''
        获取图片比对结果中，最接近的图片（忽略掉已经选取的）。
    '''
    maxItem = None
    for item in results:
        if item["uri"] in ignore_range and item["index"] in ignore_range[item["uri"]]:
            continue
        if item['score'] < threshold:
            continue
        if maxItem is None:
            maxItem = item
        if item["score"] > maxItem["score"]:
            maxItem = item
    return maxItem

def getRange(maxItem, result: list, dValue = 0.1, ignore_range = None, length = 10):
    
    # 记录置信度最高的图片信息
    maxImageScore = maxItem["score"]
    maxImageUri = maxItem["uri"]
    maxIndex = maxItem["index"]
    leftIndex = maxIndex
    rightIndex = maxIndex
    
    has_ignore_range = ignore_range is not None

    d_result = list(filter(lambda x: x["uri"] == maxImageUri, result))      # 获取与maxitem图片置信度最高的 同视频下的其他图片
    d_result = sorted(d_result, key=lambda item: item['index'])             # 将视频关键帧按照时序排列
    for i in range(maxIndex):
        prev_index = maxIndex - 1 - i
        if has_ignore_range and prev_index in ignore_range:
            break
        # print(maxImageScore, thod, maxImageUri, maxIndex)
        if d_result[prev_index]["score"] >= maxImageScore - dValue:       # 图片执行都与maxItem置信度相差小于0.1，则视频左边界回退1
            leftIndex = prev_index
        else:
            break

    for i in range(maxIndex+1, len(d_result)):
        if has_ignore_range and i in ignore_range:
            break
        if d_result[i]["score"] >= maxImageScore - dValue:
            rightIndex = i
        else:
            break
    if (rightIndex - leftIndex) > 2*length:
        # 视频片段过长，降低（socre差值）进行进一步细化
        return getRange(maxItem, result, dValue/2, ignore_range)
    return leftIndex, rightIndex, d_result[maxIndex]

def encode_video(model, preprocess, video_file, frames, device):
    '''
    '''
    video_f = DocList[ImageFeature]([])
    
    for idx, frame in enumerate(frames):
        with torch.no_grad():
            image_token = preprocess(Image.fromarray(frame)).unsqueeze(0).to(device)
            img_feature = model.encode_image(image_token)
            img_feature /= img_feature.norm(dim=-1, keepdim=True) 
            
            #
            folder = os.path.dirname(video_file)
            video_f.append(ImageFeature(uid=idx, url=video_file, folder=folder, embedding=torch.squeeze(img_feature)))      # shape [1024]
        
    return video_f

def construct_features(img_fs):
    '''
    将img_fs, 列表 转换为 特征的矩阵。
    即： list -> tensor([n,1024])
    '''
    img_features = None
    for idx, img_f in enumerate(img_fs):
        embedding = torch.unsqueeze(torch.from_numpy(img_f.embedding), 0)
        if img_features is not None:
            img_features = torch.cat([img_features,embedding], dim=0)
        else:
            img_features = embedding
            
    return img_features

def update_image_db(model, preprocess, image_paths, db):
    for image_path in image_paths:
        image_path = image_path.replace('\\', '/')          # 转换为同意格式
        images = load_imgs3(image_path, db)     # 加载数据库中未保存的图片
        
        img_fs = DocList[ImageFeature]([])
        
        # 将图片进行处理
        for idx, img in enumerate(images):
            with torch.no_grad():
                image_token = preprocess(Image.fromarray(img.tensor)).unsqueeze(0).to(conf.device)
                img_feature = model.encode_image(image_token)
                img_feature /= img_feature.norm(dim=-1, keepdim=True) 
                
                #
                url = img.url.replace('\\', '/')
                folder = os.path.dirname(url)
                img_fs.append(ImageFeature(uid=idx, url=url, folder=folder, embedding=torch.squeeze(img_feature)))      # shape [1024
        
        if len(img_fs):
            db.index(img_fs)            # 添加新图片至数据库中
    
def find_image(text, image_paths, **kwargs):
    '''
    image_paths : 图片目录列表
    '''
    threshold = kwargs.get('threshold', conf.threshold)
    topn = kwargs.get('itopN', conf.img_top_n)   
    
    # 加载模型
    model, preprocess = load_chinese_clip(conf.clip_model_name, conf.download_root)
    # 读取数据库
    db, _ = dao.load_db()
    
    texts = split_text(text, sentence_size=conf['sentence_size'])
    
    # 加载图片至数据库
    update_image_db(model, preprocess, image_paths, db)
    
    text_features = encode_text(model, texts, device=conf.device)                   # texts : 分割后的文本数组
    # 对特征进行归一化，请使用归一化后的图文特征用于下游任务
    text_features /= text_features.norm(dim=-1, keepdim=True) 
    
    # # queries  = []
    # matches, scores = db.find_batched(text_features.cpu(), search_field='embedding', limit=10)
    # queries = {'folder':{'$eq':image_path}}
    img_fs = []
    if image_paths:
        # 目标目录不为空，则搜索指定目录
        for image_path in image_paths:
            image_path.replace('\\', '/')
            queries = {'folder':{'$eq':image_path}}
            f_docs = dao.filter_all(db, queries)
            if f_docs:
                img_fs.extend(f_docs)
    else:
        # 目标目录为空，则全部搜索
        queries = {'url':{'$gte':0}}
        img_fs = dao.filter_all(db, queries)     # 查找所有
        
    if not img_fs:
        logger.warning(f'in {image_paths} found {texts}, no results\n')
        return {}               
    
    img_features = construct_features(img_fs)       # 将doclist[ImageFeature] , 变换为tensor([n,1024]);

    probs = get_similarity2(model, img_features, text_features, device=conf.device)
    
    results = get_image_by_probs(probs, texts, img_fs, threshold=threshold, topn=topn)
    
    return results


def get_image_by_probs(probs, texts, img_fs, threshold=0.5, topn=3):
    '''
        基于probs 获取置信度高的图片
    '''
    results = {}            # {'天鹅':[{uri, score:}, ]}
    
    for idx, doc in enumerate(texts):
        if doc == conf.negativate_class:    # other, 则跳过
            continue
        text_prob = probs[:,idx]
        indexs = np.argpartition(text_prob, topn)[-topn:]
        ret = []
        for i in indexs:
            if text_prob[i] < threshold:
                continue
            ret.append({'uri':img_fs[i].url, 'score':float(text_prob[i])})
        results[doc] = ret
        
    logger.info('search {}， found {}\n'.format(texts, results))
    
    return results

def update_video_db(model, preprocess, db, video_paths):
    for video_path in video_paths:
        video_path = video_path.replace('\\', '/')
        # 读取视频文件
        vf, v_frames = load_video(db, video_path)       # 
        
        for idx, f in enumerate(vf):
            # 
            video_f = encode_video(model, preprocess, f, v_frames[idx], device=conf.device)    
            if video_f:
                db.index(video_f)   

def find_video(text,video_paths, **kwargs):
    '''
    '''
    threshold = kwargs.get('threshold', conf.threshold)
    dvalue = kwargs.get('dvalue', conf.d_value)   
    topn = kwargs.get('vtopN', conf.video_top_n)   
    length = kwargs.get('lenght', conf.max_lenght)   
      
    # 加载模型
    model, preprocess = load_chinese_clip(conf.clip_model_name, conf.download_root)
    # 读取数据库
    _, db = dao.load_db()
    
    statements = split_text(text, sentence_size=conf['sentence_size'])
    
    # 读取并更新视频文件至数据库
    update_video_db(model, preprocess, db, video_paths)
       
    frames_f = [] 
    if video_paths:
        for video_path in video_paths:
            video_path.replace('\\', '/')
            queries = {'folder':{'$eq':video_path}}         # 注意文件目录windows 与linux的区别
            v_docs = dao.filter_all(db, queries) 
            if v_docs:
                frames_f.extend(v_docs)
    else:
        # 给定目录为空，过滤处理全部视频帧
        # filter all video frame features
        queries = {'uid':{'$gte':0}}
        # queries = {'folder':{'$eq':vedio_path}}
        frames_f = dao.filter_all(db, queries)          # 全部帧
        
    if not frames_f:
        logger.warning(f'in {video_paths} found {statements}, no results\n')
        return {}
    
    text_features = encode_text(model, statements, device=conf.device)                   # texts : 分割后的文本数组
    # 对特征进行归一化，请使用归一化后的图文特征用于下游任务
    text_features /= text_features.norm(dim=-1, keepdim=True) 
    
    frames_features = construct_features(frames_f)
    
    probs = get_similarity2(model, frames_features, text_features, device=conf.device)
    
    s_results = {}
    for i,kw in enumerate(statements):
        results = []
        if kw==conf.negativate_class:
            continue
        kw_probs = probs[:,i].tolist()
        for j,prob in enumerate(kw_probs):
            result = {
                'score':float(prob),
                # 'index':idx,
                'uri':frames_f[j].url,
                'index':frames_f[j].uid,
                # 'url':f,
            }
            results.append(result)
            
        index_list = getMultiRange(results, threshold=threshold, dValue=dvalue, maxCount=topn, length=length) #
        # s_results.append([kw, index_list])
        s_results[kw] = index_list
        
    logger.info('find video results: {}'.format(s_results))

    return s_results


