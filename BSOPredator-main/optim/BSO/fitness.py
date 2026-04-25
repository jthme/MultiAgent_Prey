import numpy as np
from scipy.spatial import ConvexHull
from matplotlib.path import Path

def fitness_repel(robots, idx, pos, Dmin=1):
    distances = np.linalg.norm(robots - pos, axis=1)
    distances[idx] = np.inf
    nnd = np.min(distances)
    return np.exp(2 * (Dmin - nnd)) if nnd < Dmin else 1

def is_point_on_segment(point, p1, p2):
    v1 = point - p1
    v2 = point - p2
    if np.cross(np.append(v1, [0]), np.append(v2, [0]))[-1] == 0:
        if v1[0] * v2[0] <= 0 and v1[1] * v2[1] <= 0:
            return True
    return False
   
def fitness_closure(robots, idx, pos, prey):
    robots[idx] = pos
    try:
        hull = ConvexHull(robots)
    except Exception:
        return 1
    hull_vertices = robots[hull.vertices]
    hull_path = Path(hull_vertices)
    if hull_path.contains_point(prey):
        return 0
    for i in range(len(hull_vertices)):
        p1 = hull_vertices[i]
        p2 = hull_vertices[(i+1) % len(hull_vertices)]
        if is_point_on_segment(prey, p1, p2):
            return 0.5
    return 1

def fitness_expanse(robots, idx, pos, prey):
    # robots[idx] = pos
    # return np.sum(np.linalg.norm(robots - prey, axis=1)) / len(robots)
    return np.linalg.norm(pos - prey)

def fitness_uniformity(robots, idx, pos, prey):
    robots[idx] = pos
    mask11 = (robots[:, 0] < prey[0]) & (robots[:, 1] > prey[1])
    mask12 = (robots[:, 0] == prey[0]) & (robots[:, 1] > prey[1])
    mask13 = (robots[:, 0] > prey[0]) & (robots[:, 1] > prey[1])
    mask21 = (robots[:, 0] < prey[0]) & (robots[:, 1] == prey[1])
    mask22 = (robots[:, 0] == prey[0]) & (robots[:, 1] == prey[1])
    mask23 = (robots[:, 0] > prey[0]) & (robots[:, 1] == prey[1])
    mask31 = (robots[:, 0] < prey[0]) & (robots[:, 1] < prey[1])
    mask32 = (robots[:, 0] == prey[0]) & (robots[:, 1] < prey[1])
    mask33 = (robots[:, 0] > prey[0]) & (robots[:, 1] < prey[1])
    std = np.std(np.array([sum(mask11), sum(mask12), sum(mask13), sum(mask21), sum(mask22), sum(mask23), sum(mask31), sum(mask32), sum(mask33)]))
    return std

def fitness_edge(robots, idx, pos, prey, grid):
    robots[idx] = pos
    corners = [(0, 0), (0, grid), (grid, 0), (grid, grid)]

    distances = [np.linalg.norm(np.array(prey) - np.array(corner)) for corner in corners]
    prey_min_to_edge = min(distances)
    # if prey_min_to_edge <= (grid/5):
    #     return 0
    
    closest_corner = corners[distances.index(prey_min_to_edge)]
    valid_points = get_valid_points(closest_corner, grid)
    dist1 = np.linalg.norm(robots[idx] - valid_points[0])
    dist2 = np.linalg.norm(robots[idx] - valid_points[1])
    predator_min = min(dist1, dist2)
    
    return 1*np.exp(1*prey_min_to_edge/(np.sqrt(2)*grid)) + 1*np.exp(1*predator_min/(np.sqrt(2)*grid))

def fitness(robots, idx, pos, prey, grid, Dmin=1):
    f_repel = fitness_repel(robots.copy(), idx, pos, Dmin)
    f_closure = fitness_closure(robots.copy(), idx, pos, prey)
    f_expanse = fitness_expanse(robots.copy(), idx, pos, prey)
    f_uniformity = fitness_uniformity(robots.copy(), idx, pos, prey)
    f_edge = fitness_edge(robots.copy(), idx, pos, prey, grid)
    # return f_repel * (f_closure + f_expanse + f_uniformity) / (np.linalg.norm(pos - robots[idx]) + 1)
    return f_repel * (f_closure + f_expanse + f_uniformity + f_edge) / (np.linalg.norm(pos - robots[idx]) + 1)



def get_valid_points(corner, grid):
    valid = []
    
    if corner == (0,0):
        valid.append((0,1))
        valid.append((1,0))
    
    if corner == (0,grid):
        valid.append((0,grid-1))
        valid.append((1,grid))
    
    if corner == (grid,0):
        valid.append((grid,1))
        valid.append((grid-1,0))

    if corner == (grid,grid):
        valid.append((grid-1,grid))
        valid.append((grid,grid-1))
    
    return valid