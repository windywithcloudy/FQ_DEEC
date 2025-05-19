import torch
import time
import numpy as np
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

# 监控GPU内存使用
def print_gpu_usage():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU Memory Used: {info.used//1024**2} MB")

# 参数配置
class Config:
    num_nodes = 100       # 节点数量
    num_rounds = 5000      # 仿真轮次
    init_energy = 100.0   # 初始能量（J）
    epsilon = 0.1         # 探索率
    alpha = 0.1           # 学习率
    gamma = 0.9           # 折扣因子
    tx_cost = 0.1         # 传输能耗系数（J/m²）
    agg_cost = 0.05       # 数据聚合能耗（J）
    comm_radius = 50      # 通信半径（m）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Q-DEEC协议核心实现
class QDEEC:
    def __init__(self, config):
        self.cfg = config
        self.nodes = self._init_nodes()
        
        # Q表初始化（状态维度：剩余能量(5档) x 传输成本(5档) x 历史负载(3档)）
        self.q_table = torch.rand((5, 5, 3, 3), device=self.cfg.device)  # [E, C, L, A]

    def _init_nodes(self):
        # 随机生成节点位置（模拟2D网络）
        pos = torch.rand((self.cfg.num_nodes, 2), device=self.cfg.device) * 100
        
        # 节点属性矩阵 [能量, 历史负载, x坐标, y坐标]
        nodes = torch.zeros((self.cfg.num_nodes, 4), device=self.cfg.device)
        nodes[:, 0] = self.cfg.init_energy  # 初始能量
        nodes[:, 2:] = pos                  # 位置信息
        return nodes

    def _get_state(self, node_idx):
        # 状态离散化
        energy = self.nodes[node_idx, 0] / self.cfg.init_energy
        e_bin = torch.clamp((energy * 5).int(), 0, 4)
        
        # 计算传输成本（到基站距离）
        dx = self.nodes[node_idx, 2] - 50  # 基站位于(50,50)
        dy = self.nodes[node_idx, 3] - 50
        distance = torch.sqrt(dx**2 + dy**2)
        c_bin = torch.clamp((distance / 20).int(), 0, 4)  # 每20m一档
        
        # 历史负载
        l_bin = torch.clamp(self.nodes[node_idx, 1].int(), 0, 2)
        return e_bin, c_bin, l_bin

    def run(self):
        start_time = time.time()
        
        for round in range(self.cfg.num_rounds):
            # 1. 选择动作（ε-greedy）
            rand_mask = torch.rand(self.cfg.num_nodes, device=self.cfg.device) < self.cfg.epsilon
            actions = torch.zeros(self.cfg.num_nodes, dtype=torch.long, device=self.cfg.device)
            
            # 贪婪动作选择
            for i in range(self.cfg.num_nodes):
                e, c, l = self._get_state(i)
                if not rand_mask[i]:
                    actions[i] = self.q_table[e, c, l].argmax()
            
            # 2. 执行簇头选举（简化逻辑）
            ch_mask = actions == 2  # 动作2表示提高竞选概率
            ch_indices = torch.where(ch_mask)[0]
            
            # 3. 计算能耗
            energy_cost = torch.zeros_like(self.nodes[:, 0])
            for i in ch_indices:
                dx = self.nodes[i, 2] - 50
                dy = self.nodes[i, 3] - 50
                distance = torch.sqrt(dx**2 + dy**2)
                energy_cost[i] = self.cfg.agg_cost + self.cfg.tx_cost * distance**2
            
            # 4. 更新能量和负载
            self.nodes[:, 0] -= energy_cost
            self.nodes[ch_indices, 1] += 1  # 历史负载+1
            
            # 5. 计算奖励并更新Q表
            with torch.no_grad():
                for i in range(self.cfg.num_nodes):
                    e_prev, c_prev, l_prev = self._get_state(i)
                    current_q = self.q_table[e_prev, c_prev, l_prev, actions[i]]
                    
                    # 奖励计算（简化版）
                    reward = (self.nodes[i, 0] / self.cfg.init_energy) * 5  # 剩余能量奖励
                    if ch_mask[i]: 
                        reward -= 0.5 * self.nodes[i, 1]**2  # 负载惩罚
                    
                    # 下一状态
                    e_new, c_new, l_new = self._get_state(i)
                    next_max_q = self.q_table[e_new, c_new, l_new].max()
                    
                    # Q表更新
                    new_q = current_q + self.cfg.alpha * (reward + self.cfg.gamma * next_max_q - current_q)
                    self.q_table[e_prev, c_prev, l_prev, actions[i]] = new_q

            # 每50轮打印状态
            if round % 50 == 0:
                alive_nodes = torch.sum(self.nodes[:, 0] > 0).item()
                print(f"Round {round}: Alive Nodes={alive_nodes}")
                print_gpu_usage()

        # 性能统计
        total_time = time.time() - start_time
        print(f"\nTotal Time: {total_time:.2f}s")
        print(f"Final Alive Nodes: {torch.sum(self.nodes[:, 0] > 0).item()}")
        print_gpu_usage()

if __name__ == "__main__":
    config = Config()
    qdeec = QDEEC(config)
    qdeec.run()