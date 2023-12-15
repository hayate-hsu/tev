from docarray import BaseDoc, DocList
from docarray.typing import ImageTensor, ImageUrl, NdArray



class ImageFeature(BaseDoc):
    uid:int
    url:str             # 文件
    folder:str          # 文件夹，因db不支持 $regex 正则操作，多存储目录，用于更为详细的过滤
    embedding:NdArray[1024]