import functools

from docarray.index import HnswDocumentIndex
    
    # 日志
from common.log import get_logger
logger = get_logger()

from common.conf import get_conf
conf = get_conf()

from document import ImageFeature
    
db_images = HnswDocumentIndex[ImageFeature](
    work_dir=conf.image_vs
)

@functools.lru_cache
def load_db():
    '''
    加载视频 & 图片数据库
    '''
    db_images = HnswDocumentIndex[ImageFeature](
        work_dir=conf.image_vs
    )
    
    db_videos = HnswDocumentIndex[ImageFeature](
        work_dir=conf.video_vs
    )
    
    return db_images, db_videos
    
def filter_one(db, queries):
    '''
        filter one elements
    '''
    results = db.filter(queries)
    element = None
    if len(results) == 0:
        pass
    else:
        element = results[0]
    
    logger.debug('exec filter_one {}, result: {}\n'.format(queries, element))
    return element

def filter_all(db, queries, limit=-1):
    '''
        filter all elements
        filter, limit, 默认限制10个
    '''
    results = db.filter(queries, limit=limit)

    if len(results) == 0:
        results = None
    
    logger.debug('exec filter_all {}, result: {}\n'.format(queries, results))
    return results


def update(db, docs):
    '''
    '''
    db.index(docs)
    logger.debug('exec update, docs size: {}\n'.format(len(docs)))    
    
    