# src/runner.py

import importlib
import logging
import os
import shutil
import sys
from pathlib import Path

# 1. 将项目根目录添加到Python的模块搜索路径中
# 这确保了无论从哪里运行，都能正确找到 env, utils 等模块
sys.path.append(str(Path(__file__).resolve().parent.parent))

# 2. 导入需要的函数和类
from utils.log import setup_logger  # 只导入日志配置函数
from utils.virtual import visualize_network


def get_env_class(algorithm_name):
    """
    根据算法名称，动态导入并返回对应的Env类。
    这是实现算法模块化和可插拔的关键。
    """
    try:
        if algorithm_name == 'FQ-DEEC':
            # 你的核心算法
            module = importlib.import_module('env')
            return getattr(module, 'WSNEnv')
        elif algorithm_name == 'DEEC':
            # 经典的DEEC对比算法
            module = importlib.import_module('env_deec')
            return getattr(module, 'WSNEnvDEEC')
        # elif algorithm_name == 'Q-DEEC':
        #     # 如果你未来创建了env_q_deec.py，在这里添加即可
        #     module = importlib.import_module('env_q_deec')
        #     return getattr(module, 'WSNEnvQDEEC')
        else:
            # 如果传入未知的算法名，则抛出错误
            raise ValueError(f"未知的算法名称: {algorithm_name}")
    except ImportError as e:
        # 如果找不到对应的文件 (例如 env_deec.py 不存在)
        logging.error(f"无法导入模块 for {algorithm_name}: {e}. 请确保文件存在。")
        raise
    except AttributeError as e:
        # 如果文件存在，但文件中没有对应的类 (例如 WSNEnvDEEC 写错了)
        logging.error(f"在模块中找不到对应的Env类 for {algorithm_name}: {e}。")
        raise


def run_simulation(config_file, output_dir_name, algorithm_name):
    """
    一个独立的、通用的仿真运行函数。
    它接收算法名称，动态加载环境，并执行完整的仿真流程。
    """
    # --- 3. 为当前进程设置独立的日志文件 ---
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    LOG_DIR = PROJECT_ROOT / "logs"
    log_file_path = LOG_DIR / f"simulation_{algorithm_name.lower()}.log"

    
    # 调用setup_logger来配置这个进程的日志系统
    logger = setup_logger(log_file_path=str(log_file_path))
    
    # 为日志格式化器添加算法名称和PID，便于在控制台区分并行日志
    for handler in logger.handlers:
        handler.setFormatter(logging.Formatter(f'%(asctime)s - {algorithm_name}(PID:{os.getpid()}) - %(levelname)s - %(message)s'))
    
    logger.info(f"--- 开始运行仿真: {algorithm_name} (日志文件: {log_file_path}) ---")
    
    # --- 4. 设置独立的输出目录 ---
    REPORTS_DIR = PROJECT_ROOT / "reports" / output_dir_name
    if REPORTS_DIR.exists():
        shutil.rmtree(REPORTS_DIR)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- 5. 初始化性能日志文件 ---
    PERFORMANCE_LOG_FILE = REPORTS_DIR / "performance_log.csv"
    CH_BEhavior_LOG_FILE = REPORTS_DIR / "ch_behavior_log.csv"
    
    with open(PERFORMANCE_LOG_FILE, "w", encoding="utf-8") as f:
        f.write("round,alive_nodes,total_energy,avg_energy,num_ch,avg_ch_energy,avg_members,ch_load_variance,packets_generated,packets_to_bs,avg_delay\n")
    with open(CH_BEhavior_LOG_FILE, "w", encoding="utf-8") as f:
        f.write("epoch,ch_id,energy_at_election,time_since_last_ch,neighbors_at_election,dist_to_bs,final_members\n")

    try:
        # 6. 动态获取并实例化对应的Env类
        EnvClass = get_env_class(algorithm_name)
        environment = EnvClass(config_path=config_file,performance_log_path=PERFORMANCE_LOG_FILE,ch_behavior_log_path=CH_BEhavior_LOG_FILE)
        if algorithm_name == 'FQ-DEEC':
            # 检查配置文件中是否启用了预训练
            if environment.config.get('pre_training', {}).get('enabled', False):
                # 调用我们新创建的预训练主函数
                environment.run_pretraining()
            else:
                logger.info("配置文件中未启用预训练，跳过此阶段。")

        total_rounds = environment.config.get('simulation', {}).get('total_rounds', 5000)
        viz_interval = environment.config.get('visualization', {}).get('update_interval', 20)

        # 7. 运行仿真主循环
        for r in range(total_rounds):
            if not environment.step(r):
                logger.info(f"仿真在第 {r} 轮提前结束。")
                break
            
            # 定期可视化
            if (r + 1) % viz_interval == 0 or r == 0 or r == total_rounds - 1:
                output_image_file = REPORTS_DIR / f"network_round_{r+1:04d}.png"
                visualize_network(
                    nodes_data=environment.nodes,
                    base_station_pos=environment.config['network']['base_position'],
                    area_dims=environment.config['network']['area_size'],
                    filename=str(output_image_file),
                    current_round=r + 1,
                    confirmed_chs_in_epoch=environment.confirmed_cluster_heads_for_epoch,
                    config=environment.config
                )
        
        # 8. 仿真结束，记录最终结果
        overall_pdr = environment.sim_packets_delivered_bs_total / environment.sim_packets_generated_total if environment.sim_packets_generated_total > 0 else 0
        logger.info(f"--- 仿真结束: {algorithm_name} | 总轮次: {environment.current_round + 1} | 最终PDR: {overall_pdr:.4f} ---")

    except Exception as e:
        logger.error(f"仿真 {algorithm_name} 发生严重错误: {e}", exc_info=True)