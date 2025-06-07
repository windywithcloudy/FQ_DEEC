# src/main.py

from pathlib import Path
import sys
import shutil
import os # 导入 os 模块

sys.path.append(str(Path(__file__).parent.parent))
from env import WSNEnv # 从同级目录的 env.py 导入 WSNEnv
from utils.virtual import visualize_network # 从 utils 包导入可视化函数
from utils.log import logger # 从 utils 包导入 logger

# --- 配置常量 ---
# 如果 CONFIG_FILE 的路径需要更灵活，可以考虑使用 argparse 等获取命令行参数
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = PROJECT_ROOT / "config" / "config.yml"
OUTPUT_IMAGE_FILE = PROJECT_ROOT / "reports" / "network_topology.png" # 假设输出到 reports 文件夹
# 输出图像将带有轮次信息
REPORTS_DIR = PROJECT_ROOT / "reports" / "deec_topology" 
RAW_STATE_LOG_FILE_MAIN = PROJECT_ROOT / "raw_state_log.csv" # 定义日志文件路径
PERFORMANCE_LOG_FILE = PROJECT_ROOT / "reports" / "performance_log.csv"
CH_BEHAVIOR_LOG_FILE = PROJECT_ROOT / "reports" / "ch_behavior_log.csv"
#NODE_ENERGY_LOG_FILE = PROJECT_ROOT / "reports" / "node_energy_log.csv" # 如果需要

def main():
    logger.info("主程序开始执行...")
    
    # --- 清理操作开始 ---
    if REPORTS_DIR.exists(): # 检查目录是否存在
        try:
            shutil.rmtree(REPORTS_DIR) # 删除整个目录及其内容
            #.info(f"已删除旧的报告目录: {REPORTS_DIR}")
        except OSError as e: # 处理可能的删除错误，例如文件被占用
            logger.error(f"删除目录 {REPORTS_DIR} 失败: {e}")
            # 可以选择在这里退出或继续（但后续保存可能会失败或写入旧目录）
            # return 
    # --- 清理操作结束 ---

    REPORTS_DIR.mkdir(parents=True, exist_ok=True) # 重新创建报告目录
    #logger.info(f"已创建报告目录: {REPORTS_DIR}")

    try:
        if RAW_STATE_LOG_FILE_MAIN.exists():
            RAW_STATE_LOG_FILE_MAIN.unlink() # 删除文件
            #logger.info(f"已删除旧的原始状态日志文件: {RAW_STATE_LOG_FILE_MAIN}")
        # 创建一个新的空文件并写入表头 (如果 env.py 中的写入逻辑依赖表头已存在)
        # 或者让 env.py 中的首次写入逻辑自己处理表头
        # with open(RAW_STATE_LOG_FILE_MAIN, "w", encoding="utf-8") as f:
        #     f.write("round,node_id,e_self_raw,t_last_ch_raw,n_neighbor_raw,e_neighbor_avg_raw,n_ch_nearby_raw,d_bs_normalized_raw\n")
        # logger.info(f"已创建新的空的原始状态日志文件: {RAW_STATE_LOG_FILE_MAIN} (带表头)")
        # 注意：如果env.py中的写入逻辑是追加模式("a")并且会自己检查是否写表头，则这里不需要主动创建带表头的文件。
        # 你的env.py中的写入逻辑是：
        # with open("raw_state_log.csv", "a") as f:
        #    if f.tell() == 0: #写入表头
        # 所以，这里只需要删除文件即可，env.py会处理新文件的表头。
    except OSError as e:
        logger.error(f"删除或创建原始状态日志文件 {RAW_STATE_LOG_FILE_MAIN} 失败: {e}")
    
    log_files_to_initialize = {
        PERFORMANCE_LOG_FILE: "round,alive_nodes,total_energy,avg_energy,num_ch,avg_ch_energy,avg_members,ch_load_variance,packets_generated,packets_to_bs,avg_delay\n",
        CH_BEHAVIOR_LOG_FILE: "epoch,ch_id,energy_at_election,time_since_last_ch,neighbors_at_election,dist_to_bs,final_members\n",
        #NODE_ENERGY_LOG_FILE: "round,node_id,role,energy_consumed_tx,energy_consumed_rx,energy_consumed_base,total_consumed_this_round,remaining_energy\n" # 更详细的能耗记录
    }

    for log_file_path, header in log_files_to_initialize.items():
        try:
            log_file_path.parent.mkdir(parents=True, exist_ok=True) # 确保父目录存在
            if log_file_path.exists():
                log_file_path.unlink()
            with open(log_file_path, "w", encoding="utf-8") as f:
                f.write(header)
            logger.info(f"已初始化日志文件: {log_file_path}")
        except OSError as e:
            logger.error(f"初始化日志文件 {log_file_path} 失败: {e}")

    try:
        # 1. 初始化环境 (这将自动加载配置并初始化节点)
        logger.info(f"使用配置文件: {CONFIG_FILE}")
        environment = WSNEnv(config_path=CONFIG_FILE)
        
        total_rounds = environment.config.get('simulation', {}).get('total_rounds', 10) # 运行少量轮次演示
        viz_interval = environment.config.get('visualization', {}).get('update_interval', 1)
        # 2. 获取生成的节点数据和配置参数用于可视化
            # 可视化当前轮次的分簇结果
        for r in range(total_rounds):
            if not environment.step(r): # 执行一轮仿真，包括DEEC选举和分配
                logger.info(f"仿真在第 {r} 轮提前结束。")
                break
            
            # 可视化当前轮次的分簇结果
            if (r + 1) % viz_interval == 0 or r == 0 or r == total_rounds -1 : 
                output_image_file = REPORTS_DIR / f"network_round_{r+1:04d}.png"
                visualize_network(
                    nodes_data=environment.nodes,
                    base_station_pos=environment.config['network']['base_position'],
                    area_dims=environment.config['network']['area_size'],
                    filename=str(output_image_file),
                    current_round=r + 1,
                    # 传递本epoch确认的CH列表
                    confirmed_chs_in_epoch=environment.confirmed_cluster_heads_for_epoch, 
                    config=environment.config 
                )
        
        logger.info("DEEC分簇仿真演示完成。")
        overall_pdr = environment.sim_packets_delivered_bs_total / environment.sim_packets_generated_total if environment.sim_packets_generated_total > 0 else 0
        logger.info(f"仿真结束。总生成数据包: {environment.sim_packets_generated_total}, 总送达数据包: {environment.sim_packets_delivered_bs_total}")
        logger.info(f"最终全局数据包投递率 (Overall PDR): {overall_pdr:.4f}")
        
        # --- 在这里可以继续您的其他仿真逻辑 ---
        # 例如:
        # if environment.nodes:
        #     neighbors = environment.get_node_neighbors(0, max_distance=100) # 获取节点0在100米内的邻居
        #     logger.info(f"节点0的邻居 (100m内): {neighbors}")
        #     if neighbors:
        #         # 模拟节点0向其第一个邻居发送数据
        #         first_neighbor_id = neighbors[0]
        #         distance_to_neighbor = environment.calculate_distance(0, first_neighbor_id)
        #         logger.info(f"节点0向节点 {first_neighbor_id} 发送数据，距离: {distance_to_neighbor:.2f}m")
        #         logger.info(f"节点0发送前能量: {environment.nodes[0]['energy']:.4e} J")
        #         environment.update_energy(node_id=0, distance=distance_to_neighbor, is_tx=True)
        #         logger.info(f"节点0发送后能量: {environment.nodes[0]['energy']:.4e} J")
                
        #         logger.info(f"节点{first_neighbor_id}接收前能量: {environment.nodes[first_neighbor_id]['energy']:.4e} J")
        #         environment.update_energy(node_id=first_neighbor_id, distance=0, is_tx=False) #接收距离参数通常不用于计算能量消耗
        #         logger.info(f"节点{first_neighbor_id}接收后能量: {environment.nodes[first_neighbor_id]['energy']:.4e} J")


    except FileNotFoundError:
        logger.error(f"错误：配置文件 {CONFIG_FILE} 未找到。请确保文件路径正确。")
    except KeyError as e:
        logger.error(f"错误：配置文件 {CONFIG_FILE} 中缺少必要的键: {e}。请检查配置文件格式。")
    except Exception as e:
        logger.error(f"主执行过程中发生未捕获的错误: {e}", exc_info=True)

if __name__ == "__main__":
    # (可选) 如果希望在运行main.py时，如果config.yml不存在，则创建一个演示用的
    # 但通常配置文件应该由用户预先准备好
    if not CONFIG_FILE.exists():
        logger.warning(f"配置文件 {CONFIG_FILE} 不存在。请创建它。")
        # 示例创建逻辑 (如果需要)
        # demo_config_content = """...""" 
        # CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        # with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        #     f.write(demo_config_content)
        # logger.info(f"已创建示例配置文件于 {CONFIG_FILE}")
    
    main()