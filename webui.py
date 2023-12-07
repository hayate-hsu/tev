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

from worker.search import load_chinese_clip, find_image, find_video
from db import dao


def find(s_text, video_path, image_path, threshold, video_length, d_value, image_top_n, video_top_n):
    assert s_text, '检索对象不能为空'
    
    # 加载向量数据库
    db_images, db_videos = dao.load_db()
    
    # 加载模型文件
    model, preprocess = load_chinese_clip(conf.clip_model_name, conf.download_root)
    
    # 检索图片 或者 图片数据库
    image_s= find_image(db_images, model, preprocess, s_text, image_path, threshold, image_top_n)
    
    video_s = find_video(db_videos, model, preprocess, s_text, video_path, threshold, d_value, video_top_n, video_length)
    
    return image_s, video_s

if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    with gr.Blocks() as app:
        with gr.Row():
            with gr.Column():
                video_path = gr.Textbox(placeholder='请输入视频文件目录', label='视频文件目录')
                video_ext = gr.Text(value='*.mp4', label='视频文件后缀')
            with gr.Column():            
                image_path = gr.Textbox(placeholder='请输入图片文件目录', label='图片文件目录')
                image_ext = gr.Text(value='*.jpg', label='图片文件后缀')
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
            s_text = gr.Textbox(placeholder='请输入文本（检索对象）', lines=3, max_lines=20, label='支持一次检索多条语句，语句之间用换行符分割；')
            btn_search = gr.Button(value='搜索')
        with gr.Row():
            pass
        with gr.Row():
            video_r = gr.JSON(label='检索到的视频片段')    
            img_r = gr.JSON(label='检索到的图片')

            
        btn_search.click(find, 
                         inputs=[s_text, video_path, image_path, threshold, video_lenght, d_value, image_top_n, video_top_n], 
                         outputs=[img_r, video_r])

    webbrowser.open(f"http://127.0.0.1:{conf.port}")
    logger.info('app starting ...')
    app.launch(server_port=conf.port)
