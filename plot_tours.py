instance_file    = "tsp_50.csv"
final_tour_file  = "tsp_50_tour.csv"

import numpy as np
import matplotlib.pyplot as plt

def distance_matrix(coords):
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))

def nearest_neighbor(dist, start=0):
    n = dist.shape[0]
    unvisited = set(range(n))
    tour = [start]
    unvisited.remove(start)
    while unvisited:
        last = tour[-1]
        next_city = min(unvisited, key=lambda j: dist[last, j])
        tour.append(next_city)
        unvisited.remove(next_city)
    return tour

def plot_tour(coords, tour, title):
    pts = coords[tour + [tour[0]]]
    plt.figure()
    plt.plot(pts[:, 0], pts[:, 1], marker="o")
    plt.title(title)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("x"); plt.ylabel("y"); plt.tight_layout()

coords = np.loadtxt(instance_file, delimiter=",")
tour_final = np.loadtxt(final_tour_file, delimiter=",", dtype=int).tolist()

dist = distance_matrix(coords)
tour_init = nearest_neighbor(dist)

plot_tour(coords, tour_init,  "Initial Tour – Nearest Neighbour")
plot_tour(coords, tour_final, "Final Tour – Tabu Search Best")
plt.show()
