from docarray import BaseDoc, DocList
from docarray.typing import ImageTensor, ImageUrl, NdArray



class ImageFeature(BaseDoc):
    uid:int
    url:str
    embedding:NdArray[1024]