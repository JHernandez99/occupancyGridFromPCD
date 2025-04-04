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

