import os,sys
import logging

current_file = os.path.abspath(__file__)
current_path = os.path.dirname(current_file)

sys.path.append(current_path)

# 日志
from common.log import get_logger
logger = get_logger()

from common.conf import get_conf
conf = get_conf()


import gradio as gr
import webbrowser

from collections import OrderedDict

resolutions = OrderedDict([('360p:640x360', [640,360, '16:9']), ('720p:1280x720', [1280,720, '16:9']),
                          ('1080p:1920x1080', [1920,1080, '16:9']), ('2k:2560x1440', [2560,1440, '16:9']),
                          ('4k:3840x2160', [3840,2160, '16:9']), 
                          ('720p:960x720', [960,720, '4:3']), ('1080p:1440x1080', [1440,1080, '4:3']), 
                          ('720p:720x1280', [720,1280, '9:16']), ('1080p:1080x1920', [1080,1920, '9:16'])])

def convert_path(paths):
    '''
    '''
    paths = paths.replace('\\', '/')
    paths = paths.strip()
    if paths:
        paths = paths.split('\n')
    else:
        paths = []
    
    return paths
    
def find(s_text, video_paths, image_paths, threshold, video_length, d_value, image_top_n, video_top_n):
    '''
    检索视频片段或图片
    '''
    # 检索图片 或者 图片数据库
    assert s_text, '检索字段不能为空'
    from worker.search import find_image, find_video
    image_paths = convert_path(image_paths)
    video_paths = convert_path(video_paths)
    
    # 检索参数设置
    kwargs = {}
    kwargs['threshold'] = threshold
    kwargs['lenght'] = video_length
    kwargs['dvalue'] = d_value
    kwargs['itopN'] = image_top_n
    kwargs['vtopN'] = video_top_n

    image_s= find_image(s_text, image_paths, **kwargs)
    
    video_s = find_video(s_text, video_paths, **kwargs)
    
    return image_s, video_s

def compose(scripts, video_paths, image_paths, resolution_rate, frame_rate, threshold, video_length, d_value, image_top_n, video_top_n):
    '''
    剪辑视频
    '''
    assert scripts, '脚本字段不能为空'
    image_paths = convert_path(image_paths)
    video_paths = convert_path(video_paths)
    
    width, height, wh_rate = resolutions[resolution_rate]       # 宽、高、宽高比
    
    
    from worker.compose import synthesis
    
    # 视频分辨率以及帧率设置
    kwargs = dict(width=width, height=height, whr=wh_rate, frate=frame_rate)
    # 检索参数设置
    kwargs['threshold'] = threshold
    kwargs['lenght'] = video_length
    kwargs['dvalue'] = d_value
    kwargs['itopN'] = image_top_n
    kwargs['vtopN'] = video_top_n
    
    # tts设置
    kwargs['speaker'] = 'fangqi'
    kwargs['language'] = 'ZH' 
    
    result = synthesis(scripts, video_paths, image_paths, **kwargs)
    
    return result

if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    with gr.Blocks() as app:
        with gr.Accordion("模型配置，展开编辑", open=False):
            with gr.Row():
                threshold = gr.Slider(
                    minimum=0, maximum=1, value=0.5, step=0.1, label="置信度，图片与目标语句的相识度。数值越高越相似，数值太高，可能不返回视频片段。"
                )
            with gr.Row():
                with gr.Column():
                    video_lenght = gr.Slider(
                        minimum=0, maximum=60, value=10, step=1, label="最大视频片段长度的一半，单位秒"
                    )
                    d_value = gr.Slider(
                        minimum=0, maximum=1, value=0.1, step=0.1, label="前后帧相似度与关键帧相似度差值，在范围内的才会采纳，否则终止。"
                    )               
                with gr.Column():   
                    image_top_n = gr.Slider(
                        minimum=0, maximum=10, value=3, step=1, label="最相近的N张图片"
                    )   
                    video_top_n = gr.Slider(
                        minimum=0, maximum=10, value=3, step=1, label="最相近的N个视频片段"
                    ) 
        with gr.Row():
            gr.Markdown(value='''## 素材目录
                        - 系统支持素材特征缓存，当文件被处理过一次后，就不会再处理相同文件名的素材
                        - 支持多个目录下的视频或图片
                        - 当前仅支持mp4格式视频，jpg格式图片（未来拓展其他类型支持）
                        - 图片|视频分辨率要一致，目前还未做分辨率对齐（以后拓展）
                        ''')
            with gr.Column():
                video_paths = gr.Textbox(placeholder='请输入视频文件目录,支持多个目录', lines=2, label='视频文件目录')
                video_ext = gr.Text(value='*.mp4', label='视频文件后缀')
            with gr.Column():            
                image_paths = gr.Textbox(placeholder='请输入图片文件目录，支持多个目录', lines=2, label='图片文件目录')
                image_ext = gr.Text(value='*.jpg', label='图片文件后缀')
        with gr.TabItem("多媒体资源检索"):
            gr.Markdown(value="""**根据给定关键词，检索指定目录下的素材资源库（图片&视频）**""")
            with gr.Row():
                s_text = gr.Textbox(placeholder='请输入文本（检索对象）', lines=3, max_lines=20, label='支持一次检索多条语句，语句之间用换行符分割；')
                btn_search = gr.Button(value='搜索')
            with gr.Row():
                video_r = gr.JSON(label='检索到的视频片段')    
                img_r = gr.JSON(label='检索到的图片')
        with gr.TabItem("视频剪辑"):
            with gr.Row():   
                gr.Markdown(value="""
                            ## 根据给定脚本，搜索给定素材库，智能剪辑视频
                            - 文案与视频素材尽可能匹配
                            - 当给定文本匹配素材不足时，采用黑色帧填充
                            - 文案目录未给定时，则检索匹配数据库中的全部图片&视频""")
            with gr.Row():  
                with gr.Column(): 
                    scripts = gr.Textbox(placeholder='请输入脚本（文本）', lines=3, max_lines=20, label='用于视频剪辑的脚本')
                with gr.Column():
                    resolution_rate = gr.Dropdown(choices=list(resolutions.keys()), value='720p:1280x720',label='视频分辨率')
                    frame_rate = gr.Slider(minimum=24, maximum=60, value=30, step=1, label="视频帧率")
                with gr.Column():
                    btn_compose = gr.Button(value='剪辑视频')
            with gr.Row():   
                composed_video = gr.Video() 

            
        btn_search.click(find, 
                         inputs=[s_text, video_paths, image_paths, threshold, video_lenght, d_value, image_top_n, video_top_n], 
                         outputs=[img_r, video_r])
        
        btn_compose.click(compose, 
                          inputs=[scripts, video_paths, image_paths, resolution_rate, frame_rate, threshold, video_lenght, d_value, image_top_n, video_top_n], 
                          outputs=[composed_video])

    webbrowser.open(f"http://127.0.0.1:{conf.port}")
    logger.info('app starting ...')
    app.launch(server_port=conf.port)
