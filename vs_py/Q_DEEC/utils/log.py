# utils/log.py
import logging
import sys

def setup_logger(log_level=logging.INFO, log_file=None):
    """
    配置并返回一个logger实例。
    """
    logger = logging.getLogger("WSN_Simulation")
    logger.setLevel(log_level)

    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 清除已有的handlers，防止重复添加 (特别是在Jupyter等环境中)
    if logger.hasHandlers():
        logger.handlers.clear()

    # 控制台处理器
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # 文件处理器 (可选)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# 在其他模块中通过 from utils.log import logger 来使用
# logger = setup_logger() 
# 如果希望在导入时就配置好，可以直接调用，或者让 main.py 来调用并传递实例
# 为简单起见，这里直接创建一个默认logger供其他模块导入
logger = setup_logger(log_file="simulation.log") # 日志会输出到控制台和 simulation.log 文件