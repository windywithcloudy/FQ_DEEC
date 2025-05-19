# src/main.py

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from env import WSNEnv # 从同级目录的 env.py 导入 WSNEnv
from utils.virtual import visualize_network # 从 utils 包导入可视化函数
from utils.log import logger # 从 utils 包导入 logger

# --- 配置常量 ---
# 如果 CONFIG_FILE 的路径需要更灵活，可以考虑使用 argparse 等获取命令行参数
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = PROJECT_ROOT / "config" / "config.yml"
OUTPUT_IMAGE_FILE = PROJECT_ROOT / "reports" / "network_topology.png" # 假设输出到 reports 文件夹

def main():
    logger.info("主程序开始执行...")
    
    # 确保输出图像的目录存在
    OUTPUT_IMAGE_FILE.parent.mkdir(parents=True, exist_ok=True)

    try:
        # 1. 初始化环境 (这将自动加载配置并初始化节点)
        logger.info(f"使用配置文件: {CONFIG_FILE}")
        environment = WSNEnv(config_path=CONFIG_FILE)
        
        # 2. 获取生成的节点数据和配置参数用于可视化
        nodes_for_viz = environment.nodes # WSNEnv.nodes 存储了节点列表
        base_station_pos = environment.config['network']['base_position']
        area_dims = environment.config['network']['area_size']
        
        # 3. 可视化网络
        if nodes_for_viz: # 仅当有节点时才可视化
            visualize_network(
                nodes_data=nodes_for_viz,
                base_station_pos=base_station_pos,
                area_dims=area_dims,
                filename=str(OUTPUT_IMAGE_FILE) # 确保是字符串路径
            )
        else:
            logger.info("没有节点可供可视化。")

        logger.info("网络拓扑生成和可视化完成。")
        
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