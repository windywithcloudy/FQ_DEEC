# utils/log.py
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_logger(log_level=logging.INFO, log_file_path=None, max_bytes=5*1024*1024, backup_count=5):
    """
    配置并返回一个名为 "WSN_Simulation" 的logger实例。
    """
    # 获取名为 "WSN_Simulation" 的 logger，这是我们整个项目的统一logger名称
    logger = logging.getLogger("WSN_Simulation")
    logger.setLevel(log_level)

    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 清除已有的handlers，防止在多进程或重复调用时出现问题
    if logger.hasHandlers():
        logger.handlers.clear()

    # 控制台处理器 (所有进程共享，打印到主控制台)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # 文件处理器 (如果提供了文件路径)
    if log_file_path:
        # 确保日志目录存在
        log_dir = Path(log_file_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建一个特定于该进程的文件处理器
        file_handler = RotatingFileHandler(log_file_path, maxBytes=max_bytes, 
                                           backupCount=backup_count, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# --- [核心修改] 不再在这里创建全局logger实例 ---
# 其他模块将通过 logging.getLogger("WSN_Simulation") 来获取logger
# 而这个logger的配置工作，将由主程序或每个进程的入口来完成。