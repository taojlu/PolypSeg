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

# 设置使用多块GPU
devices = [torch.device(f'cuda:{i}') for i in range(3)]
torch.cuda.set_device(devices[0])

start_time = time.time()

# 地球半径（千米）
EARTH_RADIUS = 6378.137


def rad(d):
    """将角度转换为弧度"""
    return d * math.pi / 180.0


def get_distance_matrix(data, device):
    """创建哈弗赛因公式的距离矩阵"""
    num_locations = len(data)
    data_rad = rad(data)
    lat = data_rad[:, 1]
    lng = data_rad[:, 0]

    lat1 = lat.unsqueeze(1).expand(num_locations, num_locations)
    lat2 = lat.unsqueeze(0).expand(num_locations, num_locations)
    lng1 = lng.unsqueeze(1).expand(num_locations, num_locations)
    lng2 = lng.unsqueeze(0).expand(num_locations, num_locations)

    a = lat1 - lat2
    b = lng1 - lng2

    sin_a2 = torch.sin(a / 2) ** 2
    sin_b2 = torch.sin(b / 2) ** 2
    cos_lat1 = torch.cos(lat1)
    cos_lat2 = torch.cos(lat2)

    d = 2 * torch.asin(torch.sqrt(sin_a2 + cos_lat1 * cos_lat2 * sin_b2))
    d = d * EARTH_RADIUS
    distance_matrix = torch.round(d * 10000) / 10000
    distance_matrix.fill_diagonal_(float('inf'))
    return distance_matrix.to(device)


def calculate_total_distance(tour, distance_matrix):
    """计算总路径长度"""
    total_distance = 0
    for i in range(len(tour) - 1):
        total_distance += distance_matrix[tour[i], tour[i + 1]]
    total_distance += distance_matrix[tour[-1], tour[0]]  # 回到起点的距离
    return total_distance


# 加载数据
file_dir = '/root/nfs/gaobin/wt/Datasets/TSP/scan_parcel.xlsx'
data = pd.read_excel(file_dir)

# 添加新的起始点
new_start = pd.DataFrame({'lng': [144.8419271], 'lat': [-37.8015836]})
data = pd.concat([new_start, data], ignore_index=True)

# 提取经纬度数据并转换为GPU上的张量
data_gpu = torch.tensor(data[['lng', 'lat']].values, dtype=torch.float32)

# 创建距离矩阵并分配到多块GPU上
distance_matrix = get_distance_matrix(data_gpu, devices[0])

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


def dijkstra(distance_matrix, start_index, device):
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


def greedy_with_penalty(distance_matrix, start_index=0, device=None):
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


def ant_colony_optimization(tour, distance_matrix, num_ants=10, alpha=1, beta=2, rho=0.5, Q=100, iterations=100):
    """蚁群算法进行局部优化"""
    num_locations = len(tour)
    pheromone = torch.ones((num_locations, num_locations), device=devices[0])
    best_tour = tour
    best_distance = calculate_total_distance(tour, distance_matrix)

    for iteration in range(iterations):
        all_tours = []
        all_distances = []

        for _ in range(num_ants):
            ant_tour = [tour[0]]
            for i in range(1, len(tour)):
                current_city = ant_tour[-1]
                probabilities = pheromone[current_city] ** alpha * ((1 / distance_matrix[current_city]) ** beta)
                probabilities[ant_tour] = 0  # Avoid revisiting cities already in the tour
                next_city = torch.multinomial(probabilities, 1).item()
                ant_tour.append(next_city)
            all_tours.append(ant_tour)
            all_distances.append(calculate_total_distance(ant_tour, distance_matrix))

        # 更新信息素
        pheromone *= (1 - rho)
        for i in range(num_ants):
            for j in range(len(all_tours[i]) - 1):
                pheromone[all_tours[i][j], all_tours[i][j + 1]] += Q / all_distances[i]
            pheromone[all_tours[i][-1], all_tours[i][0]] += Q / all_distances[i]

        # 更新最优路径
        min_distance = min(all_distances)
        if min_distance < best_distance:
            best_distance = min_distance
            best_tour = all_tours[all_distances.index(min_distance)]

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


initial_tour = greedy_with_penalty(distance_matrix, start_index, device=devices[0])
tour_distance = calculate_total_distance(initial_tour, distance_matrix)

if tour_distance == float('inf'):
    best_tour = initial_tour
    best_distance = tour_distance
else:
    parent = dijkstra(distance_matrix, start_index, device=devices[0])
    optimized_tour = reconstruct_path(parent, start_index, len(data))
    optimized_tour = ant_colony_optimization(optimized_tour, distance_matrix)
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
epsilon = 0.05  # 探索概率，降低探索概率
num_episodes = 100  # 减少训练的总迭代次数

num_locations = len(data)
Q = torch.zeros((num_locations, num_locations), device=devices[0])


def select_action(state, epsilon):
    """选择动作"""
    if random.uniform(0, 1) < epsilon:
        return random.choice(A_tour)
    else:
        return A_tour[torch.argmin(Q[state, A_tour]).item()]


def update_q_table(state, action, reward, next_state, alpha, gamma):
    """更新Q表"""
    best_next_action = torch.argmin(Q[next_state, A_tour]).item()
    td_target = reward + gamma * Q[next_state, best_next_action]
    Q[state, action] += alpha * (td_target - Q[state, action])


# 训练Q-learning在路径A上进行优化
for episode in range(num_episodes):
    state = A_tour[0]
    tour = [state]
    total_distance = 0

    while len(tour) < len(A_tour):
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
    state = A_tour[0]
    tour = [state]
    while len(tour) < len(A_tour):
        action = A_tour[torch.argmin(Q[state, A_tour]).item()]
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
