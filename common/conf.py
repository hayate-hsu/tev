import json
import os
import uuid
import functools

available_setting = {
    "audio":"voc",                       # 语音转换模型，voc：使用自训练模型；ms-tts: 使用微软在线tts服务
    "device": "cuda",

    "clip_model_name": "ViT-H-14",       # chinese clip model： 选择的模型
    "download_root": "./",             # 模型默认下载目录，如果目录没有模型文件，则下载，有的话直接加载
    
    # 文本
    "sentence_size": 256,                # 文本分段最大长度
    "negativate_class": "other",         # 增加'other' 类型检索 
    
    # 图片
    "max_size": 224,                     # 图片压缩大小，一般与cn_clip加载模型相关
    
    
    # 视频

    "top_n": 3,                         # 搜索最匹配的 top_n 条记录
    "threshold": 0.6,                   # 置信度：0.6，近提取socre > threshold 的帧
    "d_value": 0.1,                     # 左、右帧与最大socre（>threshold）的差值，如果小于d_value，则是相近的帧

    "max_lenght": 30,                    # 最大视频长度或秒数（每秒提取一帧处理），
    
    # 向量存储地址
    "image_vs":"./data/image_vs/",       # 图片存储地址   
    "video_vs":"./data/video_vs/",       # 视频存储地址
    
    # voc 模块配置文件
    "voc_conf":"./conf/voc.conf",
    
    "cache_path":"./cache",             # 中间视频、音频缓存目录,每次工作前应清空缓存
    "output_path": "./output",    
    "bkg": "D:\\ai\\video\\tev\\src\\image\\bkg.jpg",       # 黑色背景图片，用于生成空白视频
    
    # 模型镜像源，默认huggingface，使用openi镜像源需指定openi_token
    "mirror": "openi",
    "openi_token": "",  # openi token
    
    "tmp_dir":'./tmp/'                  #
}

class Config(dict):
    def __init__(self, d: dict = {}):
        super().__init__(d)
        # user_datas: 用户数据，key为用户名，value为用户数据，也是dict
        self.user_datas = {}

    def __getitem__(self, key):
        # if key not in available_setting:
        #     raise Exception("key {} not in available_setting".format(key))
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        # if key not in available_setting:
        #     raise Exception("key {} not in available_setting".format(key))
        return super().__setitem__(key, value)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError as e:
            return default
        except Exception as e:
            raise e
        
    def __getattr__(self, name):
        try:
            return self.get(name)
        except KeyError as e:
            raise AttributeError(msg='{} not in {}'.format(name, self))

# config = Config()

def load_config(config_path):
    if not os.path.exists(config_path):
        config_path = "./common/config-template.json"

    config_str = read_file(config_path)

    # 将json字符串反序列化为dict类型
    config = Config(json.loads(config_str))
    
    return config
    
def read_file(path):
    with open(path, mode="r", encoding="utf-8") as f:
        return f.read()

@functools.lru_cache
def get_conf(conf_path='./data/conf/config.json'):
    conf = load_config(config_path=conf_path)
    
    if 'hps' not in conf:
        from voc.utils import get_hparams_from_file
        hps = get_hparams_from_file(conf.voc_conf)
    conf.hps = hps  
    
    return conf

if __name__ == '__main__':
    config = get_conf()
    
    print(config)
    