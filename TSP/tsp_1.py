# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 15:50:12 2024

@author: ThinkPad X1
"""

import pandas as pd
import numpy as np
import torch
import math
import folium
import random
import time

# 设置使用4号GPU
device = torch.device('cuda:4')
torch.cuda.set_device(4)

start_time = time.time()

# 地球半径（千米）
EARTH_RADIUS = 6378.137


def rad(d):
    """将角度转换为弧度"""
    return d * math.pi / 180.0


def get_distance(lng1, lat1, lng2, lat2):
    """计算两个坐标点之间的距离"""
    radLat1 = rad(lat1)
    radLat2 = rad(lat2)
    a = radLat1 - radLat2
    b = rad(lng1) - rad(lng2)
    s = 2 * torch.asin(torch.sqrt(torch.pow(torch.sin(a / 2), 2) +
                                  torch.cos(radLat1) * torch.cos(radLat2) *
                                  torch.pow(torch.sin(b / 2), 2)))
    s = s * EARTH_RADIUS
    s = torch.round(s * 10000) / 10000
    return s


def create_haversine_distance_matrix(data):
    """创建哈弗赛因公式的距离矩阵"""
    num_locations = len(data)
    distance_matrix = torch.zeros((num_locations, num_locations), device=device)
    for i in range(num_locations):
        for j in range(num_locations):
            if i != j:
                distance_matrix[i, j] = get_distance(data[i, 0], data[i, 1], data[j, 0], data[j, 1])
            else:
                distance_matrix[i, j] = float('inf')
    return distance_matrix


def calculate_total_distance(tour, distance_matrix):
    """计算总路径长度"""
    total_distance = 0
    for i in range(len(tour) - 1):
        total_distance += distance_matrix[tour[i]][tour[i + 1]]
    total_distance += distance_matrix[tour[-1]][tour[0]]  # 回到起点的距离
    return total_distance


# 加载数据
file_dir = '/root/nfs/gaobin/wt/Datasets/TSP/scan_parcel.xlsx'
data = pd.read_excel(file_dir)

# 添加新的起始点
new_start = pd.DataFrame({'lng': [144.8419271], 'lat': [-37.8015836]})
data = pd.concat([new_start, data], ignore_index=True)

# 提取经纬度数据并转换为GPU上的张量
data_gpu = torch.tensor(data[['lng', 'lat']].values, dtype=torch.float32, device=device)

# 创建距离矩阵
distance_matrix = create_haversine_distance_matrix(data_gpu)

# 多次运行带有评分惩罚的贪心算法和模拟退火算法，从新的起点出发
best_tour = None
best_distance = float('inf')
start_index = 0  # 新的起点是数据中的第一个点


def apply_score_penalty(tour, score_penalty, start_index):
    """根据评分系统施加惩罚"""
    num_delivered = len(tour)
    for i in range(num_delivered):
        if i < num_delivered // 4:
            score_penalty[tour[i]] *= (1 + i / (num_delivered // 4))
        elif i > 3 * num_delivered // 4:
            score_penalty[tour[i]] *= (1 + (num_delivered - i) / (num_delivered // 4))
        else:
            score_penalty[tour[i]] = 1
    return score_penalty


def dijkstra(distance_matrix, start_index):
    """迪杰斯特拉算法进行局部优化"""
    num_locations = len(distance_matrix)
    visited = torch.zeros(num_locations, dtype=torch.bool, device=device)
    dist = torch.full((num_locations,), float('inf'), device=device)
    dist[start_index] = 0
    parent = torch.full((num_locations,), -1, dtype=torch.int, device=device)

    for _ in range(num_locations):
        min_distance = float('inf')
        u = -1
        for i in range(num_locations):
            if not visited[i] and dist[i] < min_distance:
                min_distance = dist[i]
                u = i

        if u == -1:
            break

        visited[u] = True

        for v in range(num_locations):
            if not visited[v] and distance_matrix[u][v] != float('inf') and dist[u] + distance_matrix[u][v] < dist[v]:
                dist[v] = dist[u] + distance_matrix[u][v]
                parent[v] = u

    return parent


def reconstruct_path(parent, start_index, num_locations):
    """根据父节点重建路径"""
    path = []
    current_index = start_index
    while current_index != -1:
        path.append(current_index)
        current_index = parent[current_index]
    path.reverse()
    return path


def greedy_with_penalty(distance_matrix, start_index=0):
    """带有评分惩罚的贪心算法"""
    num_locations = len(distance_matrix)
    unvisited = set(range(num_locations))
    unvisited.remove(start_index)
    tour = [start_index]
    current_index = start_index
    score_penalty = torch.ones(num_locations, device=device)

    while unvisited:
        score_penalty = apply_score_penalty(tour, score_penalty, start_index)
        next_index = min(unvisited, key=lambda x: distance_matrix[current_index, x] * score_penalty[x])
        tour.append(next_index)
        unvisited.remove(next_index)
        current_index = next_index

    tour.append(start_index)  # 返回到起点
    return tour


def simulated_annealing(tour, distance_matrix, initial_temp=10000, cooling_rate=0.995, min_temp=1):
    """模拟退火算法"""

    def swap_2opt(tour):
        """2-opt交换"""
        new_tour = tour[:]
        i, j = sorted(random.sample(range(1, len(tour) - 1), 2))
        new_tour[i:j] = new_tour[i:j][::-1]
        return new_tour

    current_temp = initial_temp
    best_tour = tour
    best_distance = calculate_total_distance(tour, distance_matrix)

    while current_temp > min_temp:
        new_tour = swap_2opt(best_tour)
        new_distance = calculate_total_distance(new_tour, distance_matrix)

        if torch.isnan(new_distance):
            print(f"Invalid distance encountered: {new_distance}")
            continue

        if new_distance < best_distance or random.random() < math.exp((best_distance - new_distance) / current_temp):
            best_tour = new_tour
            best_distance = new_distance
        current_temp *= cooling_rate

    return best_tour


def plot_tsp_paths(data, tour):
    """使用folium绘制TSP路径，避免形成闭环，不绘制到最后一个节点的路径"""
    m = folium.Map(location=[data.iloc[tour[0]]['lat'], data.iloc[tour[0]]['lng']], zoom_start=12)

    # 绘制路径，但不包括最后一个节点
    for i in range(len(tour) - 2):  # 改变循环条件，避免包括最后一个节点
        start_point = tour[i]
        end_point = tour[i + 1]
        start_coords = [data.iloc[start_point]['lat'], data.iloc[start_point]['lng']]
        end_coords = [data.iloc[end_point]['lat'], data.iloc[end_point]['lng']]
        folium.PolyLine(locations=[start_coords, end_coords], color='blue', weight=2).add_to(m)

        folium.CircleMarker(start_coords, radius=10, color='red', fill=True, fill_color='red', fill_opacity=1).add_to(m)
        folium.Marker(
            start_coords,
            icon=folium.DivIcon(html=f'<div style="font-size: 12pt; color: white; text-align: center;">{i}</div>')
        ).add_to(m)

    # 绘制倒数第二个点，但不绘制其到最后一个点的路径
    second_last_coords = [data.iloc[tour[-2]]['lat'], data.iloc[tour[-2]]['lng']]
    folium.CircleMarker(second_last_coords, radius=12, color='green', fill=True, fill_color='green',
                        fill_opacity=1).add_to(m)
    folium.Marker(
        second_last_coords,
        icon=folium.DivIcon(html=f'<div style="font-size: 14pt; color: white; text-align: center;">Final</div>')
    ).add_to(m)

    # 添加最后一个点的标记，但不连接
    last_coords = [data.iloc[tour[-1]]['lat'], data.iloc[tour[-1]]['lng']]
    folium.CircleMarker(last_coords, radius=12, color='black', fill=True, fill_color='blue', fill_opacity=1).add_to(m)
    folium.Marker(
        last_coords,
        icon=folium.DivIcon(html=f'<div style="font-size: 14pt; color: white; text-align: center;">Start</div>')
    ).add_to(m)

    return m


initial_tour = greedy_with_penalty(distance_matrix, start_index)
if calculate_total_distance(initial_tour, distance_matrix) < float('inf'):
    parent = dijkstra(distance_matrix, start_index)
    optimized_tour = reconstruct_path(parent, start_index, len(data))
    optimized_tour = simulated_annealing(optimized_tour, distance_matrix)
    tour_distance = calculate_total_distance(optimized_tour, distance_matrix)
    if tour_distance < best_distance:
        best_distance = tour_distance
        best_tour = optimized_tour

# 保存路径A
A_tour = best_tour
A_distance = best_distance

# 绘制当前最优路径A
map_result_A = plot_tsp_paths(data, A_tour if A_tour else initial_tour)
map_result_A.save('/root/nfs/gaobin/wt/Datasets/TSP/tsp_result_map_A.html')  # 保存结果到HTML文件

print(f"路径A的最佳路径长度: {A_distance}")

# 使用强化学习优化路径A生成路径B
# Q-learning 参数
alpha = 0.1  # 学习率
gamma = 0.9  # 折扣因子
epsilon = 0.1  # 探索概率
num_episodes = 5

num_locations = len(data)
Q = torch.zeros((num_locations, num_locations), device=device)


def select_action(state, epsilon):
    """选择动作"""
    if random.uniform(0, 1) < epsilon:
        return random.choice(range(num_locations))
    else:
        return torch.argmin(Q[state]).item()


def update_q_table(state, action, reward, next_state, alpha, gamma):
    """更新Q表"""
    best_next_action = torch.argmin(Q[next_state]).item()
    td_target = reward + gamma * Q[next_state, best_next_action]
    Q[state, action] += alpha * (td_target - Q[state, action])


# 训练Q-learning
for episode in range(num_episodes):
    state = random.choice(range(num_locations))
    tour = [state]
    total_distance = 0

    while len(tour) < num_locations:
        action = select_action(state, epsilon)
        if action not in tour:
            next_state = action
            reward = -distance_matrix[state, next_state]
            update_q_table(state, action, reward, next_state, alpha, gamma)
            total_distance += distance_matrix[state, next_state]
            state = next_state
            tour.append(state)

    # 回到起点
    total_distance += distance_matrix[tour[-1], tour[0]]
    reward = -distance_matrix[tour[-1], tour[0]]
    update_q_table(state, tour[0], reward, tour[0], alpha, gamma)


# 使用训练好的Q表来选择最佳路径
def get_best_tour(Q):
    state = random.choice(range(num_locations))
    tour = [state]
    while len(tour) < num_locations:
        action = torch.argmin(Q[state]).item()
        if action not in tour:
            state = action
            tour.append(state)
    return tour


# 生成路径B
B_tour = get_best_tour(Q)
B_distance = calculate_total_distance(B_tour, distance_matrix)

# 绘制路径B
map_result_B = plot_tsp_paths(data, B_tour)
map_result_B.save('/root/nfs/gaobin/wt/Datasets/TSP/tsp_result_map_B.html')  # 保存结果到HTML文件

print(f"路径B的最佳路径长度: {B_distance}")

end_time = time.time()
execution_time = end_time - start_time
print(f"总执行时间: {execution_time} 秒")
