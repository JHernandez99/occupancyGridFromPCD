# -*- coding: utf-8 -*-

import open3d as o3d
import pandas as pd
import numpy as np
import os

# Función para cargar las nubes de puntos
def cargar_nubes(archivo_csv):
    df = pd.read_csv(archivo_csv, header=None)
    puntos = df.iloc[:, :3].to_numpy()
    nube = o3d.geometry.PointCloud()
    nube.points = o3d.utility.Vector3dVector(puntos)
    return nube

# Función para realizar registro ICP con refinamiento
def registrar_nubes(source, target, trans_init=np.identity(4), threshold=0.02):
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance=threshold, init=trans_init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return reg_p2p.transformation

# Cambiar el directorio de a la carpeta donde está el script de Python
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Cargar la nube de puntos con densidad máxima sin imágenes
ruta = "./robot_output_2/20250314-140315_densidad_maxima/vectors/"

# Ruta de salida para guardar las nubes de puntos procesadas
ruta_salida = "./robot_output_2/20250314-140315_densidad_maxima/processed/"
os.makedirs(ruta_salida, exist_ok=True)

# Matrices de rotación de 180 grados
matriz_rotacion_z = np.array([[-1, 0, 0],
                              [0, -1, 0],
                              [0, 0, 1]])

matriz_rotacion_y = np.array([[-1, 0, 0],
                              [0, 1, 0],
                              [0, 0, -1]])

matriz_rotacion = np.dot(matriz_rotacion_y, matriz_rotacion_z)

# Lista para almacenar las nubes de puntos y sus transformaciones
nubes = []
transformaciones = [np.identity(4)]  # Matriz de transformación global acumulada

# Se crea el objeto de nube de puntos fusionada en Open3D
total_nube = o3d.geometry.PointCloud()

# Bucle para procesar las nubes de puntos y mostrarlas
for i in range(106):
    
    archivo = str(i) + ".csv"
    archivo_csv = ruta + archivo
    df = pd.read_csv(archivo_csv, header=None)

    # Se toman las tres primeras columnas como las coordenadas X, Y, Z
    puntos = df.iloc[:, :3].to_numpy()
    
    # Se calculan las distancias euclidianas desde el origen (Norma L2)
    distancias = np.linalg.norm(puntos, axis=1)
    
    # Se filtran los puntos cuya distancia sea menor o igual a 2 metros
    puntos = puntos[distancias <= 2]
    
    # Eliminación del suelo 
    puntos = puntos[puntos[:, 1] <= 0.19] # Se modifica el eje Y
    
    # Se crea el objeto de nube de puntos en Open3D
    nube_puntos = o3d.geometry.PointCloud()
    nube_puntos.points = o3d.utility.Vector3dVector(puntos)

    # Se crea el frame de coordenadas (ejes X, Y, Z)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, 
                                                              origin=[0, 0, 0])
    
    # Eliminación del ruido usando el filtrado estadístico 
    nube_puntos, ind = nube_puntos.remove_statistical_outlier(nb_neighbors=1000, 
                                                              std_ratio=3)
    
    # Se aplica la rotación a la nube de puntos
    nube_puntos.rotate(matriz_rotacion, center=(0, 0, 0))
    
    # Guardar las nubes de puntos procesadas en un archivo CSV
    # puntos_procesados = np.asarray(nube_puntos.points)
    # archivo_salida = os.path.join(ruta_salida, f"{i}.csv")
    # pd.DataFrame(puntos_procesados).to_csv(archivo_salida, 
    #                                        header=False, 
    #                                        index=False)

    # Visualizar las nubes de puntos
    #o3d.visualization.draw_geometries([nube_puntos, frame],
    #                                   window_name = 'Nube de Puntos',
    #                                   width = 640,
    #                                   height = 480,
    #                                   mesh_show_wireframe = True,
    #                                   mesh_show_back_face = True,
    #                                   lookat=[0.0, 0.0, 0.0],
    #                                   zoom = 0.1)
    
    nubes.append(nube_puntos)
    
# Aplicación de Multiway Registration

# Construcción del grafo de poses
for i in range(1, len(nubes)):
    trans_icp = registrar_nubes(nubes[i], nubes[i - 1])
    trans_global = np.dot(transformaciones[i - 1], trans_icp)
    transformaciones.append(trans_global)

# Aplicar las transformaciones acumuladas y fusionar las nubes
for i in range(len(nubes)):
    nubes[i].transform(transformaciones[i])
    total_nube += nubes[i]

# Eliminación del ruido usando el filtrado estadístico 
total_nube, ind = total_nube.remove_statistical_outlier(nb_neighbors=1000, std_ratio=3)

# Exportar la nube de puntos fusionada a un archivo CSV
# output_csv = ruta + "fused.csv"
# puntos_totales = np.asarray(total_nube.points)
# pd.DataFrame().to_csv(output_csv, header=False, index=False)
# print(f"Nube de puntos fusionada guardada en {output_csv}")

# Visualizar la nube de puntos fusionada
o3d.visualization.draw_geometries([total_nube, frame],
                                  window_name = 'Nube de Puntos Fusionada',
                                  width = 640,
                                  height = 480,
                                  mesh_show_wireframe = True,
                                  mesh_show_back_face = True,
                                  zoom = 0.1)