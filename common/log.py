import logging.config
import functools

logger_conf = {
    'version': 1, 
    'disable_existing_loggers': False, 
    'formatters': 
        {
            'simple': {'format': '【%(asctime)s】 【%(name)s】 【%(levelname)s】 【%(message)s】'}, 
            'verbose': {'format': '[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s'}
        }, 
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler', 
            'level': 'DEBUG', 
            'formatter': 'verbose'
        }, 
        'info_file_handler': {
            'class': 'logging.handlers.TimedRotatingFileHandler', 
            'level': 'INFO', 
            'formatter': 'simple', 
            'filename': 'info.log', 
            'when': 'W0', 
            'backupCount': 7, 
            'encoding': 'utf8'
        }, 
        'error_file_handler': {
            'class': 'logging.handlers.TimedRotatingFileHandler', 
            'level': 'ERROR', 
            'formatter': 'verbose', 
            'filename': 'errors.log', 
            'when': 'W0', 
            'backupCount': 7, 
            'encoding': 'utf8'
        }
    }, 
    'logger': {
        'my_module': {
            'level': 'ERROR', 
            'handlers': ['info_file_handler'], 
            'propagate': 'no'
        }, 
        'app': {
            'level': 'INFO', 
            'handlers': ['console', 'info_file_handler', 'error_file_handler'], 
            'propagate': 'no'
        }
    }, 
    'root': {
        'level': 'INFO', 
        'handlers': ['console', 'info_file_handler', 'error_file_handler'], 
        'propagate': 'no'
    }
}

def setup_logging(conf={}, logger_conf=logger_conf):
    if conf:
        # 更新日志保存目录，以及更新逻辑
        logger_conf['handlers']['info_file_handler']['filename'] = conf.get('info_log_path', 'info.log')
        logger_conf['handlers']['error_file_handler']['filename'] = conf.get('error_log_path', 'error.log')
        
        logger_conf['handlers']['info_file_handler']['when'] = conf.get('when', 'W0')
        logger_conf['handlers']['error_file_handler']['when'] = conf.get('when', 'W0')        
    
    logging.config.dictConfig(logger_conf)
    

@functools.lru_cache
def get_logger(name='app'):
    from common.conf import get_conf
    conf = get_conf()
    setup_logging(conf)
    
    return logging.getLogger(name)
    
    
def func():
    logger = get_logger()
    logger.info("start func")
    
    logger.info("exec func")
    
    logger.debug('hello world!')
    
    logger.info("end func")
    
    
if __name__ == '__main__':
    setup_logging()
    func()