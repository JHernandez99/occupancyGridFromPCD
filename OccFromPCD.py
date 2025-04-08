#Multiway Registration
#%%
import open3d as o3d
import numpy as np

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


#points = np.loadtxt(f"B:/MasterIE/TercerCuatri/Robotica Movil/robot_output_2/20250314-140315_densidad_maxima/vectors/1.csv",delimiter=',')
points = np.loadtxt(f"B:/MasterIE/TercerCuatri/Robotica Movil/robot_output_2/fused_pointcloud.csv",delimiter=',')
pcd2 = o3d.geometry.PointCloud()
#pcd2.points = o3d.utility.Vector3dVector(points)
#o3d.visualization.draw_geometries([pcd2])

#z_min = 0.05
#r_min=3
#filtered_points = points[points[:, 1] < z_min]
#norms = np.linalg.norm(filtered_points[:,:3], axis=1)
#filtered_points = filtered_points[norms <= r_min]


# Convertir a PointCloud
pcd = o3d.geometry.PointCloud()

pcd.points = o3d.utility.Vector3dVector(points)

# Calcular normales antes de usar ICP
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))


# Downsample (si es necesario)
pcd_down = pcd.voxel_down_sample(voxel_size=0.02)

# Visualizar
o3d.visualization.draw_geometries([pcd_down])#,
                                  #zoom=0.3412,
                                  #front=[0.4257, -0.2125, -0.8795],
                                  #lookat=[2.6172, 2.0475, 1.532],
                                  #up=[-0.0694, -0.9768, 0.2024])

#create mesh from
#mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_down, alpha=0.1)
#o3d.visualization.draw_geometries([mesh])

points = np.asarray(pcd_down.points)

xy_pts = points[:,[0,2]]
#res
grid_size = 0.1
x_min, y_min = np.min(xy_pts, axis=0)
x_max, y_max = np.max(xy_pts, axis=0)

# Calcular dimensiones del grid
grid_width = int((x_max - x_min) / grid_size) + 1
grid_height = int((y_max - y_min) / grid_size) + 1

# Crear un mapa vacío (0 = libre)
occupancy_grid = np.zeros((grid_height, grid_width))

# Mapear los puntos a celdas del grid
for x, y in xy_pts:
    i = int((y - y_min) / grid_size)  # Índice en Y
    j = int((x - x_min) / grid_size)  # Índice en X
    occupancy_grid[i, j] = 1  # Marcar como ocupado

# Visualizar el mapa de ocupación
plt.imshow(occupancy_grid, cmap="gray_r", origin="lower")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Mapa de Ocupación")
plt.grid(True)
plt.show()

#%% ADD SAVE FUNCTION

#%%  ROUTE PLANNING
'''
 ALGORITMO A*
 ALGORITMO RTT*
'''

import heapq

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(occupancy_grid, start, goal):
    rows, cols = occupancy_grid.shape
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start))
    came_from = {}
    g_score = {start: 0}
    while open_set:
        _, current_cost, current = heapq.heappop(open_set)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if occupancy_grid[neighbor] == 1:
                    continue
                tentative_g = current_cost + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))
                    came_from[neighbor] = current
    return None

def plot_grid_with_path(grid, path, start, goal):
    fig, ax = plt.subplots()
    cmap = plt.cm.gray_r
    ax.imshow(grid, cmap=cmap, origin="lower")

    if path:
        y_coords, x_coords = zip(*path)
        #print("Ruta encontrada:", path)

        ax.plot(x_coords, y_coords, color='blue', linewidth=2, label="Ruta A*")

    ax.plot(start[1], start[0], "go", label="Inicio")
    ax.plot(goal[1], goal[0], "ro", label="Meta")
    ax.legend()
    if path:
        dist = sum(np.linalg.norm(np.array(path[i]) - np.array(path[i - 1])) for i in range(1, len(path)))
        plt.title(f"Ruta A* en Occupancy Grid - Distancia: {dist:.2f}")
    else:
        plt.title("Ruta A* en Occupancy Grid - No se encontró ruta")
    plt.grid(True)
    plt.show()

import time

#USE
inicio = time.time()
start = (30,13)
goal = (5,5)
path = a_star(occupancy_grid, start, goal)
plot_grid_with_path(occupancy_grid, path, start, goal)
fin = time.time()
print("Tiempo ruta 1 A* {}".format(fin-inicio))


inicio = time.time()
start = (30,13)
goal = (12,28)
path = a_star(occupancy_grid, start, goal)
plot_grid_with_path(occupancy_grid, path, start, goal)
fin = time.time()
print("Tiempo ruta 2 A* {}".format(fin-inicio))

inicio = time.time()
start = (30,13)
goal = (15,2)
path = a_star(occupancy_grid, start, goal)
plot_grid_with_path(occupancy_grid, path, start, goal)
fin = time.time()
print("Tiempo ruta 3 A* {}".format(fin-inicio))

#%% APLICANDO ALGORITMO RTT*
from scipy.spatial import KDTree
import random

class Node:
    def __init__(self, pos):
        self.pos = pos
        self.parent = None
        self.cost = 0

def distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def is_free(grid, point):
    x, y = point
    h, w = grid.shape
    return 0 <= x < h and 0 <= y < w and grid[int(x), int(y)] == 0

def steer(from_node, to_pos, step_size):
    direction = np.array(to_pos) - np.array(from_node.pos)
    length = np.linalg.norm(direction)
    if length == 0:
        return from_node.pos
    direction = direction / length
    new_pos = np.array(from_node.pos) + direction * min(step_size, length)
    return tuple(map(int, new_pos))

def rrt_star(grid, start, goal, max_iter=500, step_size=5, goal_radius=5, neighbor_radius=10):
    start_node = Node(start)
    goal_node = Node(goal)
    nodes = [start_node]

    for _ in range(max_iter):
        rand_pos = (random.randint(0, grid.shape[0]-1), random.randint(0, grid.shape[1]-1))
        if not is_free(grid, rand_pos):
            continue

        # Nearest node
        dlist = [distance(n.pos, rand_pos) for n in nodes]
        nearest_node = nodes[np.argmin(dlist)]

        new_pos = steer(nearest_node, rand_pos, step_size)
        if not is_free(grid, new_pos):
            continue

        new_node = Node(new_pos)
        new_node.parent = nearest_node
        new_node.cost = nearest_node.cost + distance(nearest_node.pos, new_pos)

        # Rewire
        neighbor_idxs = [i for i, n in enumerate(nodes) if distance(n.pos, new_node.pos) < neighbor_radius]
        for i in neighbor_idxs:
            neighbor = nodes[i]
            temp_cost = new_node.cost + distance(new_node.pos, neighbor.pos)
            if temp_cost < neighbor.cost and is_free(grid, neighbor.pos):
                neighbor.parent = new_node
                neighbor.cost = temp_cost

        nodes.append(new_node)

        # ¿Llegamos al objetivo?
        if distance(new_node.pos, goal_node.pos) < goal_radius:
            goal_node.parent = new_node
            goal_node.cost = new_node.cost
            nodes.append(goal_node)
            break

    # Reconstruir ruta
    path = []
    node = goal_node
    while node is not None:
        path.append(node.pos)
        node = node.parent
    path.reverse()

    return path, nodes

def plot_rrt(grid, path, nodes, start, goal):
    fig, ax = plt.subplots()
    ax.imshow(grid, cmap=plt.cm.gray_r, origin='lower')

    # dibujar árbol
    for node in nodes:
        if node.parent:
            x = [node.pos[1], node.parent.pos[1]]
            y = [node.pos[0], node.parent.pos[0]]
            ax.plot(x, y, color='lightblue', linewidth=0.5)

    # ruta final
    if path and len(path) > 1:
        y_coords, x_coords = zip(*path)
        ax.plot(x_coords, y_coords, color='blue', linewidth=2, label="Ruta RRT*")

    ax.plot(start[1], start[0], "go", label="Inicio")
    ax.plot(goal[1], goal[0], "ro", label="Meta")
    ax.legend()
    if path and len(path) > 1:
        dist = sum(np.linalg.norm(np.array(path[i]) - np.array(path[i - 1])) for i in range(1, len(path)))
        title = f"Ruta RRT* en Occupancy Grid - Distancia: {dist:.2f}"
    else:
        title = "Ruta RRT* en Occupancy Grid - No se encontró ruta"

    plt.title(title)
    plt.grid(True)
    plt.show()

#USE
inicio = time.time()
start = (30,13)
goal = (5,5)
path, nodes = rrt_star(occupancy_grid, start, goal)
plot_rrt(occupancy_grid, path, nodes, start, goal)
fin = time.time()
print("Tiempo ruta 1 RTT* {}".format(fin-inicio))

inicio = time.time()
start = (30,13)
goal = (12,28)
path, nodes = rrt_star(occupancy_grid, start, goal)
plot_rrt(occupancy_grid, path, nodes, start, goal)
fin = time.time()
print("Tiempo ruta 2 RRT* {}".format(fin-inicio))

inicio = time.time()
start = (30,13)
goal = (15,2)
path, nodes = rrt_star(occupancy_grid, start, goal)
plot_rrt(occupancy_grid, path, nodes, start, goal)
fin = time.time()
print("Tiempo ruta 3 RRT* {}".format(fin-inicio))