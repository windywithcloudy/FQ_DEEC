# utils/log.py
import logging
import sys
from logging.handlers import RotatingFileHandler # 导入
from pathlib import Path

def setup_logger(log_level=logging.INFO, log_file=None,max_bytes=5*1024*1024, backup_count=5):
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
        # 当日志文件达到 max_bytes 时，会重命名为 log_file.1, log_file.2 ...
        # 最多保留 backup_count 个备份文件。当创建新文件时，最旧的备份会被删除。
        file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, 
                                           backupCount=backup_count, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# 在其他模块中通过 from utils.log import logger 来使用
# logger = setup_logger() 
# 如果希望在导入时就配置好，可以直接调用，或者让 main.py 来调用并传递实例
# 为简单起见，这里直接创建一个默认logger供其他模块导入
SIMULATION_LOG_PATH = Path(__file__).resolve().parent.parent / "simulation.log"
logger = setup_logger(log_file=str(SIMULATION_LOG_PATH), 
                      max_bytes=2*1024*1024, # 例如，每个日志文件最大2MB
                      backup_count=3)         # 保留最近3个备份 + 当前日志文件