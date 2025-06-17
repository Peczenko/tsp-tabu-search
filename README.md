
# Tabu‑Search TSP 
## 1 Overview
We solve the **Travelling Salesperson Problem (TSP)** with the meta‑heuristic  
**Tabu Search** and a 2‑opt neighbourhood.

Steps performed by `tsp_tabu.py`  
1. **Load and/or generate** city coordinates.  
2. **Initial tour** via nearest‑neighbour.  
3. **Improve** with Tabu Search (tabu list + aspiration). 
4. **Save** best tour, print stats.

---
## 2 Running the solver

### Install
```bash
pip install numpy
```

### Command‑line
```bash
python tsp_tabu.py (--random N | --instance FILE)
                   [--seed INT] [--iter INT] [--time SEC]
```

Example:
```bash
python tsp_tabu.py --random 50 --seed 42
```
| Flag              | Default | Very short note              |
| ----------------- | ------- | ---------------------------- |
| `--random N`      | —       | generate **N**-city instance |
| `--instance FILE` | —       | load coords from CSV         |
| `--seed`          | `0`     | fix RNG for repeat runs      |
| `--iter`          | `5000`  | max loops of Tabu Search     |
| `--time`          | `60 s`  | wall-clock stop              |


Paramters for the code:
| Var                    | Default | Note                        |
| ---------------------- | ------- | --------------------------- |
| `tabu_size`            | `10`    | edge stays tabu this long   |
| `neighbourhood_sample` | `100`   | 2-opt moves tested per loop |


Outputs  
* `tsp_50.csv` – instance (if generated)  
* `tsp_50_tour.csv` – best tour order  
* Console: start length, best length, % improvement

---

## 3 Visualising tours (plot_tours.py)

`plot_tours.py` draws **two separate charts**:

* *Initial* tour (nearest‑neighbour)  
* *Final* tour (Tabu‑Search best)

Usage:
```bash
python plot_tours.py --instance tsp_50.csv --tour tsp_50_tour.csv
```
