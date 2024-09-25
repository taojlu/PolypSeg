import pandas as pd
import torch
import math
import folium
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

def get_distance(lng1, lat1, lng2, lat2):
    """计算两个坐标点之间的距离"""
    radLat1 = rad(lat1)
    radLat2 = rad(lat2)
    a = radLat1 - radLat2
    b = rad(lng1) - rad(lng2)
    s = 2 * math.asin(math.sqrt(math.pow(math.sin(a / 2), 2) +
                                math.cos(radLat1) * math.cos(radLat2) *
                                math.pow(math.sin(b / 2), 2)))
    s = s * EARTH_RADIUS
    s = round(s * 10000) / 10000
    return s

def get_distance_matrix(data, device):
    """创建距离矩阵"""
    num_locations = len(data)
    distance_matrix = torch.zeros((num_locations, num_locations), device=device)

    for i in range(num_locations):
        for j in range(i + 1, num_locations):
            lng1, lat1 = data.iloc[i]['lng'], data.iloc[i]['lat']
            lng2, lat2 = data.iloc[j]['lng'], data.iloc[j]['lat']
            distance = get_distance(lng1, lat1, lng2, lat2)
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    large_value = 1e6  # 设置一个较大的常数
    distance_matrix.fill_diagonal_(large_value)  # 确保对角线为一个较大的常数，避免访问自身
    return distance_matrix

def calculate_total_distance(tour, data):
    """根据节点的坐标重新计算路径的总长度"""
    total_distance = 0
    for i in range(len(tour) - 1):
        lng1, lat1 = data.iloc[tour[i]]['lng'], data.iloc[tour[i]]['lat']
        lng2, lat2 = data.iloc[tour[i + 1]]['lng'], data.iloc[tour[i + 1]]['lat']
        distance = get_distance(lng1, lat1, lng2, lat2)
        total_distance += distance
    # 计算最后一个节点回到起点的距离
    lng1, lat1 = data.iloc[tour[-1]]['lng'], data.iloc[tour[-1]]['lat']
    lng2, lat2 = data.iloc[tour[0]]['lng'], data.iloc[tour[0]]['lat']
    distance = get_distance(lng1, lat1, lng2, lat2)
    total_distance += distance
    return total_distance

# 加载数据
file_dir = '/root/nfs/gaobin/wt/Datasets/TSP/scan_parcel.xlsx'
data = pd.read_excel(file_dir)


# 检查数据中的经纬度值
# print("节点2, 3, 4的经纬度值：")
# print(data.iloc[2][['lng', 'lat']])
# print(data.iloc[3][['lng', 'lat']])
# print(data.iloc[4][['lng', 'lat']])

# 添加新的起始点
new_start = pd.DataFrame({'lng': [144.8419271], 'lat': [-37.8015836]})
data = pd.concat([new_start, data], ignore_index=True)

# 提取经纬度数据并转换为GPU上的张量
data_gpu = torch.tensor(data[['lng', 'lat']].values, dtype=torch.float32)

# 创建距离矩阵并分配到多块GPU上
distance_matrix = get_distance_matrix(data, devices[0])

# 打印距离矩阵，检查是否正确
print("距离矩阵：")
# print(distance_matrix.cpu().numpy())

# 多次运行带有评分惩罚的贪心算法和模拟退火算法，从新的起点出发
start_index = 0  # 新的起点是数据中的第一个点

def apply_score_penalty(tour, score_penalty, start_index):
    """根据路径中的节点位置应用评分惩罚"""
    num_delivered = len(tour)
    for i in range(num_delivered):
        if i < num_delivered // 4:
            # 路径前1/4位置的节点，逐渐增加评分系数
            score_penalty[tour[i]] *= (1 + i / (num_delivered // 4))
        elif i > 3 * num_delivered // 4:
            # 路径后1/4位置的节点，逐渐增加评分系数
            score_penalty[tour[i]] *= (1 + (num_delivered - i) / (num_delivered // 4))
        else:
            # 中间位置的节点，保持评分系数为1
            score_penalty[tour[i]] = 1
    return score_penalty

def greedy_with_penalty(distance_matrix, start_index=0, device=None, k=6):
    """带有评分惩罚的贪心算法，优先考虑当前节点周围的k个最近邻节点"""
    num_locations = len(distance_matrix)
    unvisited = set(range(num_locations))
    unvisited.remove(start_index)
    tour = [start_index]
    current_index = start_index
    score_penalty = torch.ones(num_locations, device=device)

    while unvisited:
        score_penalty = apply_score_penalty(tour, score_penalty, start_index)
        neighbors = torch.argsort(distance_matrix[current_index])[:k]  # 获取当前节点的k个最近邻节点
        valid_neighbors = [n for n in neighbors.tolist() if n in unvisited]  # 过滤出未访问的邻居节点
        if not valid_neighbors:  # 如果没有未访问的邻居节点，选择任意未访问的节点
            next_index = min(unvisited, key=lambda x: distance_matrix[current_index, x] * score_penalty[x])
        else:
            next_index = min(valid_neighbors, key=lambda x: distance_matrix[current_index, x] * score_penalty[x])
        tour.append(next_index)
        unvisited.remove(next_index)
        current_index = next_index

    tour.append(start_index)  # 返回到起点
    return tour

def print_tour_coordinates(tour, data):
    print("路径上的节点及其坐标：")
    for idx in tour:
        print(f"节点索引: {idx}, 坐标: (lng: {data.iloc[idx]['lng']}, lat: {data.iloc[idx]['lat']})")


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

# 初始贪心算法路径
initial_tour = greedy_with_penalty(distance_matrix, start_index, device=devices[0], k=6)

# 打印路径坐标
print_tour_coordinates(initial_tour, data)

# 重新计算路径长度
tour_distance = calculate_total_distance(initial_tour, data)
# print(f"重新计算的路径总长度: {tour_distance}")

# 绘制路径并保存
map_result_A = plot_tsp_paths(data, initial_tour)
map_result_A.save('/root/nfs/gaobin/wt/Datasets/TSP/tsp_result_map_A.html')  # 保存结果到HTML文件

end_time = time.time()
execution_time = end_time - start_time
print(f"总执行时间: {execution_time} 秒")


