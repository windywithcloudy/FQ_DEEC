# src/main.py

import multiprocessing
import psutil
import os
from pathlib import Path
# 1. 只从runner导入核心运行函数
from runner import run_simulation 
# 2. 只从log导入setup_logger，用于配置主进程的日志
from utils.log import setup_logger 

# --- [核心修正] 在主进程的入口处，正确地配置主进程的logger ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
# 确保日志目录存在
LOG_DIR.mkdir(parents=True, exist_ok=True) 
# 使用正确的参数名 log_file_path，并传递完整路径
logger = setup_logger(log_file_path=str(LOG_DIR / "main_launcher.log"))

def worker(config_file_path_str, output_dir, algorithm_name, core_id):
    """
    每个进程要执行的工作函数。
    接收的第一个参数是配置文件的字符串路径。
    """
    try:
        p = psutil.Process(os.getpid())
        available_cores = list(range(multiprocessing.cpu_count()))
        if core_id in available_cores:
            p.cpu_affinity([core_id])
            # 这个print语句在多进程中是安全的
            print(f"进程 for {algorithm_name} (PID: {os.getpid()}) 已成功绑定到 CPU核心 {core_id}")
        else:
            print(f"警告：CPU核心 {core_id} 不可用。进程 for {algorithm_name} 将由操作系统自动调度。")
    except Exception as e:
        print(f"警告：为 {algorithm_name} 绑定CPU核心 {core_id} 失败: {e}")
    
    # 调用核心仿真逻辑
    run_simulation(config_file_path_str, output_dir, algorithm_name)

if __name__ == "__main__":
    # --- 仿真配置 ---
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    CONFIG_FILE_PATH = PROJECT_ROOT / "config" / "config.yml"

    # 检查基础配置文件是否存在
    if not CONFIG_FILE_PATH.exists():
        logger.error(f"基础配置文件不存在: {CONFIG_FILE_PATH}")
        logger.error("请确保 'config/config.yml' 文件存在。")
        exit() # 如果基础配置不存在，则直接退出

    # 定义要并行运行的实验
    # 格式: (算法名, 输出目录名, 分配的CPU核心ID)
    experiments = [
        ('FQ-DEEC', 'reports_fq_deec', 0),
        ('DEEC',    'reports_deec',    1),
        ('Q-DEEC',  'reports_q_deec',  2), # 如果你已经创建了 env_q_deec.
        ('HEED',    'reports_heed',    3),
    ]

    # --- 创建并启动进程 ---
    processes = []
    logger.info("="*20)
    logger.info("开始分发并行仿真任务...")
    logger.info(f"将要运行 {len(experiments)} 个实验。")
    logger.info("="*20)

    for algo, out_dir, core in experiments:
        # --- [核心修改] 将Path对象转换为字符串再传递 ---
        process = multiprocessing.Process(
            target=worker, 
            args=(str(CONFIG_FILE_PATH), out_dir, algo, core)
        )
        processes.append(process)
        process.start()
        logger.info(f"已启动进程 for {algo}, 目标核心: {core}, PID: {process.pid}")

    # --- 等待所有进程完成 ---
    for process in processes:
        process.join()

    logger.info("="*20)
    logger.info("所有仿真任务已完成！请检查 'reports' 和 'logs' 目录。")
    logger.info("="*20)