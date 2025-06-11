# src/main.py
from pathlib import Path
import sys
import multiprocessing
import psutil
import os
from pathlib import Path
from runner import run_simulation # 从我们新创建的 runner.py 导入函数
import logging # 导入标准的logging库
logger = logging.getLogger("WSN_Simulation")
sys.path.append(str(Path(__file__).parent.parent))
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = PROJECT_ROOT / "config" / "config.yml"

def worker(config_file, output_dir, algorithm_name, core_id):
    """
    每个进程要执行的工作函数。
    """
    try:
        p = psutil.Process(os.getpid())
        # 确保核心ID在有效范围内
        available_cores = list(range(multiprocessing.cpu_count()))
        if core_id in available_cores:
            p.cpu_affinity([core_id])
            print(f"进程 for {algorithm_name} (PID: {os.getpid()}) 已成功绑定到 CPU核心 {core_id}")
        else:
            print(f"警告：CPU核心 {core_id} 不可用。进程 for {algorithm_name} 将由操作系统自动调度。")
    except Exception as e:
        print(f"警告：为 {algorithm_name} 绑定CPU核心 {core_id} 失败: {e}")
    
    # 调用核心仿真逻辑
    run_simulation(config_file, output_dir, algorithm_name)

if __name__ == "__main__":
    # --- 仿真配置 ---
    # 所有算法共享一个基础配置文件
    
    # 定义要并行运行的实验
    # 格式: (算法名, 输出目录名, 分配的CPU核心ID)
    # 算法名必须与 get_env_class 函数中的名称匹配
    experiments = [
        ('FQ-DEEC', 'reports_fq_deec', 0),  # 你的完整算法，在核心0上运行
        ('DEEC',    'reports_deec',    1),  # 经典DEEC，在核心1上运行
        # ('Q-DEEC',  'reports_q_deec',  2),  # 如果实现了，可以取消这行注释
    ]

    # --- 创建并启动进程 ---
    processes = []
    logger.info("开始分发并行仿真任务...")
    for algo, out_dir, core in experiments:
        process = multiprocessing.Process(
            target=worker, 
            args=(CONFIG_FILE, out_dir, algo, core)
        )
        processes.append(process)
        process.start()
        logger.info(f"已启动进程 for {algo}, 目标核心: {core}")

    # --- 等待所有进程完成 ---
    for process in processes:
        process.join()

    logger.info("所有仿真任务已完成！请检查 'reports' 目录下的各个子目录。")