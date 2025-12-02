import os
import copy
import time
import pickle
import csv
import pandas as pd  # <--- ADICIONADO (O erro estava aqui)
import numpy as np
import numpy.random as rnd
from pathlib import Path
from tqdm import tqdm

# --- IMPORTS ML ---
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from kneed import KneeLocator

# --- IMPORTS CVRP ---
from routing.cvrp.alns_cvrp import cvrp_helper_functions
from routing.cvrp.alns_cvrp.cvrp_env import cvrpEnv
from routing.cvrp.alns_cvrp.initial_solution import compute_initial_solution

# --- OPERADORES ---
from routing.cvrp.alns_cvrp.destroy_operators import random_removal, relatedness_removal, neighbor_graph_removal
from routing.cvrp.alns_cvrp.repair_operators import regret_insertion
# SEUS OPERADORES
from routing.cvrp.alns_cvrp.cluster_destroy_operator import cluster_representative_removal
from routing.cvrp.alns_cvrp.cluster_repair_operator import cluster_priority_repair

# --- CONFIGURAÇÕES ---
DATA_FILE = "routing/cvrp/data/cvrp_100_10000.pkl"
N_TEST_INSTANCES = 100 
ITERATIONS = 10000
SEED = 12345

# Parâmetros ALNS
R_DECAY = 0.8   
W_SCORES = [5, 3, 1, 0] 
START_TEMP = 5.0

def find_optimal_k_elbow_cvrp(X, Y, demands, random_state, max_k=15):
    data = np.column_stack((X, Y, demands))
    if len(data) < 3: return 2
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    inertias = []
    limit_k = min(max_k + 1, len(X_pca))
    K_range = range(2, limit_k)
    if len(K_range) == 0: return 2
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(X_pca)
        inertias.append(km.inertia_)
    kl = KneeLocator(K_range, inertias, curve="convex", direction="decreasing")
    return kl.elbow if kl.elbow else 3

class PureALNS_CVRP:
    def __init__(self, seed=12345):
        self.rnd = rnd.RandomState(seed)
        self.destroy_ops = [random_removal, relatedness_removal, neighbor_graph_removal, cluster_representative_removal]
        self.repair_ops = [regret_insertion, cluster_priority_repair]
        self.d_weights = np.ones(len(self.destroy_ops))
        self.r_weights = np.ones(len(self.repair_ops))

    def select_operator(self, weights):
        probs = weights / np.sum(weights)
        return self.rnd.choice(range(len(weights)), p=probs)

    def solve(self, instance_data, instance_id, iterations):
        nb_cust = instance_data['nb_customers']
        cap = instance_data['truck_capacity']
        dist_mat = instance_data['dist_matrix']
        dist_depot = instance_data['dist_depot']
        demands = instance_data['demands']
        cust_x = instance_data['x']
        cust_y = instance_data['y']

        k_optimal = find_optimal_k_elbow_cvrp(cust_x, cust_y, demands, self.rnd)
        
        state = cvrpEnv([], nb_cust, cap, dist_mat, dist_depot, demands, cust_x, cust_y, instance_id, self.rnd.randint(0, 10000))
        curr_sol = compute_initial_solution(state, self.rnd)
        best_sol = copy.deepcopy(curr_sol)
        
        T = START_TEMP
        
        for it in range(iterations):
            d_idx = self.select_operator(self.d_weights)
            r_idx = self.select_operator(self.r_weights)
            
            degree = self.rnd.uniform(0.1, 0.4)
            nr_remove = max(1, round(degree * curr_sol.nb_customers))
            
            destroyed = self.destroy_ops[d_idx](curr_sol, self.rnd, nr_nodes_to_remove=nr_remove, k_optimal=k_optimal)
            candidate = self.repair_ops[r_idx](destroyed, self.rnd)
            
            cand_cost = candidate.objective()
            curr_cost = curr_sol.objective()
            best_cost = best_sol.objective()
            
            score_type = 3
            if cand_cost < best_cost:
                best_sol = copy.deepcopy(candidate)
                curr_sol = candidate
                score_type = 0
            elif cand_cost < curr_cost:
                curr_sol = candidate
                score_type = 1
            else:
                delta = cand_cost - curr_cost
                if self.rnd.random() < np.exp(-delta / T):
                    curr_sol = candidate
                    score_type = 2
            
            reward = W_SCORES[score_type]
            self.d_weights[d_idx] = R_DECAY * self.d_weights[d_idx] + (1 - R_DECAY) * reward
            self.r_weights[r_idx] = R_DECAY * self.r_weights[r_idx] + (1 - R_DECAY) * reward
            
            if hasattr(candidate, 'routes'):
                 curr_sol.graph = best_sol.graph = cvrp_helper_functions.update_neighbor_graph(candidate, candidate.routes, candidate.objective())
            
            T = T * 0.999
            
        return best_sol.objective()

def load_dataset_manual(filepath):
    base_path = Path(__file__).resolve().parent
    full_path = base_path / filepath
    print(f"Carregando: {full_path}")
    with open(full_path, 'rb') as f:
        return pickle.load(f)

def run_benchmark():
    print(f"\n{'='*60}")
    print(f"BENCHMARK CVRP 100 NÓS: LONG RUN ({ITERATIONS} ITERAÇÕES)")
    print(f"{'='*60}")
    
    try:
        dataset = load_dataset_manual(DATA_FILE)
    except FileNotFoundError:
        print("ERRO: Arquivo de dados não encontrado.")
        return

    test_data = dataset[-N_TEST_INSTANCES:] 
    alns = PureALNS_CVRP(seed=SEED)
    
    results = []
    
    print(f"\n>>> Iniciando processamento de {N_TEST_INSTANCES} instâncias...")
    
    for i, _ in tqdm(enumerate(test_data), total=len(test_data)):
        nb_cust, cap, dist_mat, dist_depot, demands, cust_x, cust_y = cvrp_helper_functions.read_input_cvrp(str(Path(__file__).resolve().parent / DATA_FILE), 9000+i)
        
        instance_data = {
            'nb_customers': nb_cust, 'truck_capacity': cap,
            'dist_matrix': dist_mat, 'dist_depot': dist_depot,
            'demands': demands, 'x': cust_x, 'y': cust_y
        }
        
        start_time = time.time()
        cost = alns.solve(instance_data, 9000+i, iterations=ITERATIONS)
        duration = time.time() - start_time
        
        results.append({
            "ID": 9000+i,
            "Cost": cost,
            "Time": duration
        })
        
        # Salva parcial a cada 10 instâncias (COM PROTEÇÃO AGORA)
        if (i+1) % 10 == 0:
            try:
                pd.DataFrame(results).to_csv("cvrp_10k_results_partial.csv", index=False)
            except Exception as e:
                print(f"\n[Aviso] Erro ao salvar parcial: {e}. Continuando...")

    df = pd.DataFrame(results)
    avg_cost = df["Cost"].mean()
    
    print(f"\n{'='*40}")
    print(f"MÉDIA FINAL (10k): {avg_cost:.2f}")
    print(f"Target Paper (DR-ALNS 10k): ~16.38")
    print(f"{'='*40}")
    
    df.to_csv("cvrp_10k_results_final.csv", index=False)

if __name__ == "__main__":
    run_benchmark()