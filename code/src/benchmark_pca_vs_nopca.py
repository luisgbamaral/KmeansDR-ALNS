import os
import copy
import time
import numpy as np
import pandas as pd
import numpy.random as rnd
from pathlib import Path
from tqdm import tqdm

# --- IMPORTS DO PROBLEMA ---
from orienteering.ai4tsp.alns_ai4tsp import ai4tsp_helper_functions
from orienteering.ai4tsp.alns_ai4tsp.ai4tsp_env import ai4tspEnv
from orienteering.ai4tsp.alns_ai4tsp.initial_solution import empty_route

# --- OPERADORES PADRÃO ---
from orienteering.ai4tsp.alns_ai4tsp.destroy_operators import random_removal, relatedness_removal, neighbor_graph_removal
from orienteering.ai4tsp.alns_ai4tsp.repair_operators import random_best_distance_repair, random_best_prize_repair, random_best_ratio_repair

# --- SEUS OPERADORES (COM PCA) ---
from orienteering.ai4tsp.alns_ai4tsp.cluster_operators_op import cluster_representative_removal_op, cluster_priority_repair_op, get_node_features_3d

# --- IMPORTS ML ---
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from kneed import KneeLocator

# --- CONFIGURAÇÕES ---
# Paper: "stopping criteria... to 200 for 100 customers"
ITERATIONS_MAP = {20: 100, 50: 100, 100: 200}
N_INSTANCES = 250 # Exatamente como no paper

BASE_PATH = Path(__file__).resolve().parent
INSTANCES_DIR = BASE_PATH / 'orienteering/ai4tsp/data/test/instances'
ADJ_DIR = BASE_PATH / 'orienteering/ai4tsp/data/test/adjs'

# ==============================================================================
# 1. DEFINIÇÃO DAS VARIANTES (COM E SEM PCA)
# ==============================================================================

def calculate_k_elbow(x_matrix, random_state, use_pca=True, max_k=15):
    """Calcula K com ou sem PCA"""
    data = x_matrix[:, [0, 1, -2]]
    if len(data) < 3: return 2
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)
    
    if use_pca:
        pca = PCA(n_components=2)
        data_to_cluster = pca.fit_transform(X_scaled)
    else:
        data_to_cluster = X_scaled # Usa os dados 3D brutos normalizados
        
    inertias = []
    limit_k = min(max_k + 1, len(data_to_cluster))
    K_range = range(2, limit_k)
    
    if len(K_range) == 0: return 2
    
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(data_to_cluster)
        inertias.append(km.inertia_)
        
    kl = KneeLocator(K_range, inertias, curve="convex", direction="decreasing")
    return kl.elbow if kl.elbow else 3

def cluster_destroy_no_pca(current, random_state, degree_of_destruction=None, **kwargs):
    """
    VARIANTE: Clusterização DIRETA (Sem PCA).
    Usa as features normalizadas (X, Y, Prize) diretamente no K-Means.
    """
    if len(current.route) <= 2: return current
    destroyed_solution = copy.deepcopy(current)
    served_nodes = list(set(destroyed_solution.route))
    if 1 in served_nodes: served_nodes.remove(1)
        
    k_optimal = kwargs.get('k_optimal', 5)
    if len(served_nodes) < k_optimal:
        return random_removal(current, random_state, degree_of_destruction=degree_of_destruction)

    # --- DIFERENÇA AQUI: SEM PCA ---
    df_nodes = get_node_features_3d(current, served_nodes)
    features = df_nodes[['x', 'y', 'prize']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features) 
    # Pula PCA, vai direto pro KMeans com dados 3D
    
    current_k = min(k_optimal, len(served_nodes))
    if current_k <= 1: return destroyed_solution

    kmeans = KMeans(n_clusters=current_k, random_state=random_state, n_init=1)
    df_nodes['cluster'] = kmeans.fit_predict(X_scaled)
    # -------------------------------
    
    ids_to_remove_list = []
    deg = degree_of_destruction if degree_of_destruction is not None else 0.3

    for i in range(current_k):
        cluster_mask = (df_nodes['cluster'] == i)
        nodes_in_cluster = df_nodes[cluster_mask].copy()
        if nodes_in_cluster.empty: continue
        
        indices = nodes_in_cluster.index
        if i >= len(kmeans.cluster_centers_): continue
        centroid = kmeans.cluster_centers_[i]
        
        # Distância no espaço 3D (X, Y, Prize)
        dists = np.linalg.norm(X_scaled[indices] - centroid, axis=1)
        nodes_in_cluster['dist_centroid'] = dists
        
        n_total = len(nodes_in_cluster)
        n_remove = int(np.round(n_total * deg))
        n_remove = max(1, min(n_remove, n_total - 1))
        n_keep = n_total - n_remove
        
        remove_df = nodes_in_cluster.sort_values('dist_centroid', ascending=True).iloc[n_keep:]
        ids_to_remove_list.append(remove_df[['id', 'dist_centroid']])

    if not ids_to_remove_list: return destroyed_solution

    df_outliers = pd.concat(ids_to_remove_list)
    df_outliers = df_outliers.sort_values('dist_centroid', ascending=True)
    priority_list = df_outliers['id'].tolist()
    destroyed_solution.priority_list = priority_list
    ids_set = set(priority_list)
    destroyed_solution.route = [n for n in destroyed_solution.route if n not in ids_set]
    return destroyed_solution

# ==============================================================================
# 2. CLASSE DE ALNS PURO (CONFIGURÁVEL)
# ==============================================================================

class PureALNS:
    def __init__(self, use_pca=True, seed=12345):
        self.rnd = rnd.RandomState(seed)
        self.use_pca = use_pca
        
        # Seleciona o operador baseado na config
        if self.use_pca:
            self.cluster_op = cluster_representative_removal_op # O original (Com PCA)
        else:
            self.cluster_op = cluster_destroy_no_pca # O novo (Sem PCA)

        self.destroy_ops = [random_removal, relatedness_removal, neighbor_graph_removal, self.cluster_op]
        self.repair_ops = [random_best_distance_repair, random_best_prize_repair, random_best_ratio_repair, cluster_priority_repair_op]
        
        self.d_weights = np.ones(len(self.destroy_ops))
        self.r_weights = np.ones(len(self.repair_ops))
        self.scores = [5, 3, 1, 0]
        self.decay = 0.8

    def solve(self, instance_path, iterations):
        name = instance_path.stem
        adj_path = ADJ_DIR / f"adj-{name}.csv"
        try:
            x, adj, _ = ai4tsp_helper_functions.read_instance(str(instance_path), str(adj_path))
        except: return 0

        # Calcula K (Com ou Sem PCA dependendo da config)
        k_optimal = calculate_k_elbow(x, self.rnd, use_pca=self.use_pca)
        
        nodes = [(i+1) for i in range(len(x))]
        env = ai4tspEnv(nodes, [], x, adj, name, self.rnd.randint(0, 10000))
        curr_sol = empty_route(env, 1)
        best_sol = copy.deepcopy(curr_sol)
        T = 5.0
        
        for _ in range(iterations):
            # Roleta
            d_probs = self.d_weights / np.sum(self.d_weights)
            r_probs = self.r_weights / np.sum(self.r_weights)
            d_idx = self.rnd.choice(range(len(self.destroy_ops)), p=d_probs)
            r_idx = self.rnd.choice(range(len(self.repair_ops)), p=r_probs)
            
            degree = self.rnd.uniform(0.1, 0.4)
            
            # Executa
            destroyed = self.destroy_ops[d_idx](curr_sol, self.rnd, degree_of_destruction=degree, k_optimal=k_optimal)
            candidate = self.repair_ops[r_idx](destroyed, self.rnd)
            
            # Avalia
            cand_score = abs(candidate.objective())
            curr_score = abs(curr_sol.objective())
            best_score = abs(best_sol.objective())
            
            score_type = 3
            if cand_score > best_score:
                best_sol = copy.deepcopy(candidate)
                curr_sol = candidate
                score_type = 0
            elif cand_score > curr_score:
                curr_sol = candidate
                score_type = 1
            elif self.rnd.random() < np.exp((cand_score - curr_score) / T):
                curr_sol = candidate
                score_type = 2
            
            # Atualiza pesos
            self.d_weights[d_idx] = self.decay * self.d_weights[d_idx] + (1-self.decay) * self.scores[score_type]
            self.r_weights[r_idx] = self.decay * self.r_weights[r_idx] + (1-self.decay) * self.scores[score_type]
            T *= 0.99
            
        return abs(best_sol.objective())

# ==============================================================================
# 3. EXECUÇÃO DO BENCHMARK
# ==============================================================================

def get_valid_instances(size):
    """Filtro robusto de 250 instâncias"""
    print(f"   -> Buscando instâncias de {size} nós...")
    all_files = list(INSTANCES_DIR.glob('*.csv'))
    valid = []
    for f in all_files:
        try:
            if len(pd.read_csv(f)) == size: valid.append(f)
        except: continue
        if len(valid) >= N_INSTANCES: break
    
    if len(valid) < N_INSTANCES:
        print(f"   AVISO: Só encontrei {len(valid)} instâncias. Usando o que tem.")
    return valid

def run_comparison():
    results = []
    
    for size in [20, 50, 100]:
        print(f"\n{'='*60}")
        print(f"COMPARANDO: TAMANHO {size} | ITERAÇÕES: {ITERATIONS_MAP[size]}")
        print(f"{'='*60}")
        
        instances = get_valid_instances(size)
        if not instances: continue
        
        # --- RODADA 1: COM PCA (Standard) ---
        print(f"1. Rodando Cluster COM PCA (Standard)...")
        alns_pca = PureALNS(use_pca=True)
        scores_pca = []
        start = time.time()
        for inst in tqdm(instances, desc="   With PCA"):
            scores_pca.append(alns_pca.solve(inst, ITERATIONS_MAP[size]))
        time_pca = time.time() - start
        avg_pca = np.mean(scores_pca)
        
        # --- RODADA 2: SEM PCA (Raw Features) ---
        print(f"2. Rodando Cluster SEM PCA (Raw 3D)...")
        alns_nopca = PureALNS(use_pca=False)
        scores_nopca = []
        start = time.time()
        for inst in tqdm(instances, desc="   No PCA  "):
            scores_nopca.append(alns_nopca.solve(inst, ITERATIONS_MAP[size]))
        time_nopca = time.time() - start
        avg_nopca = np.mean(scores_nopca)
        
        # --- RELATÓRIO ---
        gap = ((avg_pca - avg_nopca) / avg_nopca) * 100
        print(f"\n>>> RESULTADO PARCIAL ({size} NÓS) <<<")
        print(f"   Com PCA: {avg_pca:.2f} (Tempo: {time_pca:.1f}s)")
        print(f"   Sem PCA: {avg_nopca:.2f} (Tempo: {time_nopca:.1f}s)")
        print(f"   Impacto do PCA: {gap:+.2f}%")
        
        results.append({
            "Size": size,
            "With PCA": avg_pca,
            "No PCA": avg_nopca,
            "Gap (%)": f"{gap:+.2f}%",
            "Time PCA (s)": f"{time_pca:.0f}",
            "Time NoPCA (s)": f"{time_nopca:.0f}"
        })

    print("\n\n==================================================")
    print("           RELATÓRIO FINAL: IMPACTO DO PCA        ")
    print("==================================================")
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    df.to_csv("pca_impact_benchmark.csv", index=False)

if __name__ == "__main__":
    run_comparison()