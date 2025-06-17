from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

def generate_random_instance(n: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n, 2)) * 100.0


def save_instance_csv(coords: np.ndarray, path: Path) -> None:
    with path.open("w", newline="") as f:
        csv.writer(f).writerows(coords.tolist())


def load_instance_csv(path: Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",")
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("CSV must have exactly two columns(x,y)")
    return data.astype(float)


def distance_matrix(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def nearest_neighbor(dist: np.ndarray, start: int = 0) -> List[int]:
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


def tour_length(tour: List[int], dist: np.ndarray) -> float:
    return sum(dist[tour[i], tour[(i + 1) % len(tour)]] for i in range(len(tour)))


def two_opt_swap(tour: List[int], i: int, k: int) -> List[int]:
    return tour[:i] + list(reversed(tour[i : k + 1])) + tour[k + 1 :]

# Tabu Search
def tabu_search(
    dist: np.ndarray,
    initial_tour: List[int],
    *,
    tabu_size: int = 10,
    max_iter: int = 5_000,
    time_limit: float = 60.0,
    neighbourhood_sample: int = 100,
    rng: random.Random | None = None,
) -> Tuple[List[int], float]:

    if rng is None:
        rng = random.Random()

    n = dist.shape[0]
    current = initial_tour[:] # working tour
    best = current[:] # best tour found so far
    best_length = tour_length(best, dist) # length of the best tour

    tabu: dict[Tuple[int, int], int] = {}
    start_time = time.time()

    #search loop
    for iteration in range(1, max_iter + 1):
        if (time.time() - start_time) >= time_limit:
            break

       #sample neighbourhood
        candidates: List[Tuple[int, int]] = []
        for _ in range(neighbourhood_sample):
            i, k = sorted(rng.sample(range(1, n), 2)) #choose 2opt slice
            if i == k:
                continue
            candidates.append((i, k))

        best_cand: Tuple[int, int] | None = None
        best_cand_len = math.inf
        best_cand_tour: List[int] | None = None

        #candidate evaluation
        for i, k in candidates:
            new_tour = two_opt_swap(current, i, k)
            new_len = tour_length(new_tour, dist)

            edge1 = tuple(sorted((current[i - 1], current[i])))
            edge2 = tuple(sorted((current[k], current[(k + 1) % n])))

            tabu_hit = (edge1 in tabu) or (edge2 in tabu) #tabu check
            aspiration = new_len < best_length

            if not tabu_hit or aspiration:
                if new_len < best_cand_len:
                    best_cand = (i, k)
                    best_cand_len = new_len
                    best_cand_tour = new_tour

        if best_cand is None or best_cand_tour is None:
            break

        #update tabu list
        i, k = best_cand
        swapped_edges = [
            tuple(sorted((current[i - 1], current[i]))),
            tuple(sorted((current[k], current[(k + 1) % n]))),
        ]
        expiry_iter = iteration + tabu_size
        for e in swapped_edges:
            tabu[e] = expiry_iter #mark as tabu

        tabu = {e: exp for e, exp in tabu.items() if exp > iteration} #drop expired

        current = best_cand_tour

        #record global best result
        if best_cand_len < best_length:
            best = current[:]
            best_length = best_cand_len

    return best, best_length #best tour and its distance

def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tabu Search for TSP (2‑opt)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--instance", type=Path, help="Path to an existing CSV of coordinates")
    g.add_argument("--random", type=int, metavar="N", help="Generate random instance with N cities")
    p.add_argument("--seed", type=int, default=0, help="Random seed (0 = system default)")
    p.add_argument("--iter", type=int, default=5_000, help="Max iterations (default 5000)")
    p.add_argument("--time", type=float, default=60.0, help="Time limit in seconds (default 60)")
    return p.parse_args(argv)

def main(argv: List[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)

    alg_rng = random.Random(args.seed if args.seed else None)

    if args.random is not None:
        coords = generate_random_instance(args.random, args.seed if args.seed else None)
        inst_path = Path(f"tsp_{args.random}.csv")
        save_instance_csv(coords, inst_path)
        print(f"[INFO] Random instance saved to {inst_path}")
    else:
        coords = load_instance_csv(args.instance)
        inst_path = args.instance

    n = coords.shape[0]
    print(f"[INFO] Solving TSP with {n} cities …")

    dist = distance_matrix(coords)

    init_tour = nearest_neighbor(dist)
    init_len = tour_length(init_tour, dist)
    print(f"[INFO] Nearest‑Neighbour length: {init_len:.2f}")

    best_tour, best_len = tabu_search(
        dist,
        init_tour,
        tabu_size=10,
        max_iter=args.iter,
        time_limit=args.time,
        rng=alg_rng,
    )

    improvement = 100.0 * (init_len - best_len) / init_len if init_len else 0.0
    print(f"[RESULT] Best length: {best_len:.2f} (improved {improvement:.2f}% over initial)")

    tour_path = inst_path.with_name(inst_path.stem + "_tour.csv")
    with tour_path.open("w", newline="") as f:
        csv.writer(f).writerow(best_tour)
    print(f"[INFO] Tour saved to {tour_path}")


if __name__ == "__main__":
    main()
