# TEV 
基于给定文本，搜索给定目录下的图片、视频文件，找到相近的图片或视频片段。进而即将检索到的视频片段或图片，裁剪成视频。
- 视频片段&图片检索： 当指定目录时，加载指定目录视频|图片， 并在此目录下进行素材检索；如果未指定视频|图片目录，则在向量库中执行全文检索。
- 视频剪辑：
  - 文本（脚本）与视频相关度较低时（threshold<0.5），检索不到视频。可以尝试降低相关度阈值，或者提供更多相关性视频素材。
  - 文本（脚本）与图片，未作相关度阈值检测，简单返回top-N。
  - 当与文本匹配的视频片段或者图片不足时，系统使用空白（黑色）背景作为填充帧,补足时长。
  - 可以调整输出视频分辨率&帧率，默认分辨率为1080P（1920x1080，16：9），30fps

## install(安装)
### 安装miniconda 或者其他python管理工具

- 官网下载安装文件
- 配置更改下载源

### 创建python（建议3.9版本）环境

```shell
conda create -n tev python=3.9
```

### 安装torch

详情参考[pytorch](https://pytorch.org/)官方。

```shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 安装依赖库

```shell
pip install -r requirements.txt
```

## 安装ffmpeg

[ffmpeg官网](https://ffmpeg.org/download.html)下载，并将bin目录添加至系统环境变量PATH中。

## configuration（配置）

### 配置文件

- 首先加载./data/conf/config.json 目录配置项，加载系统配置。
- 如果未找到**./data/conf/config.json**，则加载./common/config-template.json 配置文件。

```python
{
    "version":"v0.1.0",
    "audio":"ms-tts",				# 音频模块，使用自己训练&部署模型，或者使用第三方接口，默认使用

    "clip_model_name":"ViT-H-14",	# cn_clip 库以及模型文件下载存储目录
    "download_root":"./clip_cn",
    "device":"cuda",

    "sentence_size": 256,               # 分词语句最大长度
    "negativate_class": "other",         
    
    "max_size": 224,                   	# 与cn_clip模型相关，ViT-H-14窗口大小为224，这里统一把图片/视频帧缩放至224
    		
    "img_top_n": 5,  					# 视频&图片检索默认设置，可由webui控制
    "video_top_n": 3,                        
    "threshold": 0.6,                  
    "d_value": 0.1,                    
    "max_lenght": 20,

    "image_vs":"./data/vs/image_vs/",	# 向量数据库目录，一个存储图片特征，一个存储视频特征
    "video_vs":"./data/vs/video_vs/",

    "voc_conf":"./data/conf/voc.json",		# 当audio选项为voc时，加载模型配置文件
    "voc_model":"G_54000.pth", 				# voc对应的自训练tts模型

    "sample_rate":44100,					# 音频采样率

    "cache_path":"./data/cache",			#缓存目录
    "output_path": "./data/output",			# 视频&音频 结果输出目录

    "mirror": "openi",						# 下载bert模型时，指定源，启智https://openi.org.cn/
    "openi_token": "",						# openi对应的token

    "port":7800
}
```

### bert 模型

voc模块，对应的语言模型，支持ZH、EN、JP三语种。当audio配置项为ms-tts时，不需要此模型。

### cn_clip 模型

中文clip模型，用于跨模态检索。更多信息可访问[github项目地址](https://github.com/OFA-Sys/Chinese-CLIP)。

CN-CLIPViT-H/14模型文件可以提前下载，并将模型文件放到项目目录下的clip_cn目录（注意与配置文件的download_root，目录一致）。 下载地址：[Download](https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/clip_cn_vit-h-14.pt)。

### voc 模型

自训练的语音模型，当audio配置项为ms-tts时，不需要此模型。
