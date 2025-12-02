import os
import copy
import time
import pickle
import numpy as np
import numpy.random as rnd
from pathlib import Path
from tqdm import tqdm

# --- IMPORTS ML (Para o seu Cluster) ---
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from kneed import KneeLocator

# --- IMPORTS DO CVRP ---
from routing.cvrp.alns_cvrp import cvrp_helper_functions
from routing.cvrp.alns_cvrp.cvrp_env import cvrpEnv
from routing.cvrp.alns_cvrp.initial_solution import compute_initial_solution

# --- OPERADORES PADRÃO (Paper) ---
from routing.cvrp.alns_cvrp.destroy_operators import random_removal, relatedness_removal, neighbor_graph_removal
from routing.cvrp.alns_cvrp.repair_operators import regret_insertion

# --- SEUS OPERADORES (Cluster) ---
from routing.cvrp.alns_cvrp.cluster_destroy_operator import cluster_representative_removal
from routing.cvrp.alns_cvrp.cluster_repair_operator import cluster_priority_repair

# --- CONFIGURAÇÕES ---
DATA_FILE = "routing/cvrp/data/cvrp_100_10000.pkl" # Arquivo de 100 nós
N_TEST_INSTANCES = 100 # Paper usa 5000, mas 100 é suficiente para estatística rápida
SEED = 12345

# Parâmetros do ALNS Vanilla (Ropke & Pisinger / Paper)
R_DECAY = 0.8   
W_SCORES = [5, 3, 1, 0] # [Best, Better, Accepted, Rejected]
START_TEMP = 5.0

def find_optimal_k_elbow_cvrp(X, Y, demands, random_state, max_k=15):
    """Calcula K-Ótimo para CVRP usando (X, Y, Demanda)"""
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
        
        # --- ESPAÇO DE AÇÃO (Igual ao seu Env RL) ---
        self.destroy_ops = [
            random_removal,
            relatedness_removal,
            neighbor_graph_removal,
            cluster_representative_removal # Seu Operador
        ]
        # No paper original era só Regret. Adicionamos o seu Cluster.
        self.repair_ops = [
            regret_insertion,
            cluster_priority_repair # Seu Operador
        ]
        
        self.d_weights = np.ones(len(self.destroy_ops))
        self.r_weights = np.ones(len(self.repair_ops))
        
        # Stats
        self.d_counts = np.zeros(len(self.destroy_ops))
        self.r_counts = np.zeros(len(self.repair_ops))

    def select_operator(self, weights):
        probs = weights / np.sum(weights)
        return self.rnd.choice(range(len(weights)), p=probs)

    def solve(self, instance_data, instance_id, iterations):
        # Desempacota dados (Lógica do read_input_cvrp manual para não ler arquivo toda vez)
        nb_cust = instance_data['nb_customers']
        cap = instance_data['truck_capacity']
        dist_mat = instance_data['dist_matrix']
        dist_depot = instance_data['dist_depot']
        demands = instance_data['demands']
        cust_x = instance_data['x']
        cust_y = instance_data['y']

        # 1. Cálculo K-Elbow (Uma vez no início)
        k_optimal = find_optimal_k_elbow_cvrp(cust_x, cust_y, demands, self.rnd)
        
        # 2. Inicialização
        state = cvrpEnv([], nb_cust, cap, dist_mat, dist_depot, demands, cust_x, cust_y, instance_id, self.rnd.randint(0, 10000))
        curr_sol = compute_initial_solution(state, self.rnd)
        best_sol = copy.deepcopy(curr_sol)
        
        T = START_TEMP
        
        for it in range(iterations):
            # Seleção
            d_idx = self.select_operator(self.d_weights)
            r_idx = self.select_operator(self.r_weights)
            
            # Intensidade Aleatória (A grande vantagem do ALNS Puro sobre políticas rígidas)
            degree = self.rnd.uniform(0.1, 0.4)
            factors = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4, 4: 0.5, 5: 0.6, 6: 0.7, 7: 0.8, 8: 0.9, 9: 1.0}
            # Simula a discretização do RL ou usa contínuo? Vamos usar contínuo para dar poder total.
            nr_remove = max(1, round(degree * curr_sol.nb_customers))
            
            # Execução
            # Note: cluster operator usa k_optimal, outros usam nr_nodes_to_remove
            destroyed = self.destroy_ops[d_idx](
                curr_sol, 
                self.rnd, 
                nr_nodes_to_remove=nr_remove, 
                k_optimal=k_optimal
            )
            
            candidate = self.repair_ops[r_idx](destroyed, self.rnd)
            
            # Avaliação (CVRP = MINIMIZAÇÃO)
            cand_cost = candidate.objective()
            curr_cost = curr_sol.objective()
            best_cost = best_sol.objective()
            
            score_type = 3 # Rejected
            
            if cand_cost < best_cost: # Melhor Global
                best_sol = copy.deepcopy(candidate)
                curr_sol = candidate
                score_type = 0
            elif cand_cost < curr_cost: # Melhor Local
                curr_sol = candidate
                score_type = 1
            else:
                # Aceitação por Simulated Annealing
                delta = cand_cost - curr_cost # Positivo (piora)
                prob = np.exp(-delta / T)
                if self.rnd.random() < prob:
                    curr_sol = candidate
                    score_type = 2
            
            # Atualiza Pesos
            reward = W_SCORES[score_type]
            self.d_weights[d_idx] = R_DECAY * self.d_weights[d_idx] + (1 - R_DECAY) * reward
            self.r_weights[r_idx] = R_DECAY * self.r_weights[r_idx] + (1 - R_DECAY) * reward
            
            self.d_counts[d_idx] += 1
            self.r_counts[r_idx] += 1
            
            # Atualiza Grafo de Vizinhança (usado pelo neighbor_removal)
            if hasattr(candidate, 'routes'):
                 curr_sol.graph = best_sol.graph = cvrp_helper_functions.update_neighbor_graph(candidate, candidate.routes, candidate.objective())
            
            T = T * 0.99
            
        return best_sol.objective()

def load_dataset_manual(filepath):
    """Carrega o pickle manualmente para controle total"""
    base_path = Path(__file__).resolve().parent
    full_path = base_path / filepath
    print(f"Carregando dataset: {full_path}")
    with open(full_path, 'rb') as f:
        data = pickle.load(f)
    return data

def run_benchmark():
    print(f"\n{'='*60}")
    print(f"BENCHMARK CVRP 100 NÓS: PURE ALNS + CLUSTER (SINGLE MODEL)")
    print(f"{'='*60}")
    
    try:
        # O arquivo pickle contém uma lista de tuplas/objetos
        dataset = load_dataset_manual(DATA_FILE)
    except FileNotFoundError:
        print("ERRO: Arquivo de dados não encontrado. Verifique o caminho.")
        return

    # Usamos as últimas N instâncias para teste (padrão comum para não viciar se usasse no treino)
    # O paper usa 5000. Vamos usar 50 ou 100 para ser rápido.
    test_data = dataset[-N_TEST_INSTANCES:] 
    
    alns = PureALNS_CVRP(seed=SEED)
    
    # Bateria 1: 1.000 Iterações
    print(f"\n>>> Rodando 1.000 Iterações (Fast) em {N_TEST_INSTANCES} instâncias...")
    scores_1k = []
    for i, data_tuple in tqdm(enumerate(test_data), total=len(test_data)):
        # O pickle do CVRP geralmente vem como tupla (x, y, demands...) ou objeto.
        # Vamos adaptar para o formato esperado pelo solve
        # Assumindo formato padrão do VRP-Model-Pytorch/POMO que esse repo usa
        
        # Formatação dos dados (Desempacotando a tupla do pickle)
        # Formato usual: (depot, loc, demand, capacity, ...)
        # Vou usar a função helper do repo para garantir
        
        # TRUQUE: Salvar num arquivo temporário para usar o helper original se precisar
        # Mas para ser rápido, vamos estruturar o dicionário direto:
        
        # O formato do pickle deste repo (cvrp_100_10000.pkl) geralmente é:
        # Uma lista de 10000 itens. Cada item é (input, ...).
        # Vamos usar o helper read_input_cvrp como referência.
        
        # MODO SEGURO: Usar o helper para processar uma "instância falsa" baseada no índice
        # O helper lê do arquivo baseado no ID.
        nb_cust, cap, dist_mat, dist_depot, demands, cust_x, cust_y = cvrp_helper_functions.read_input_cvrp(str(Path(__file__).resolve().parent / DATA_FILE), 9000+i)
        
        instance_data = {
            'nb_customers': nb_cust,
            'truck_capacity': cap,
            'dist_matrix': dist_mat,
            'dist_depot': dist_depot,
            'demands': demands,
            'x': cust_x,
            'y': cust_y
        }
        
        cost = alns.solve(instance_data, 9000+i, iterations=1000)
        scores_1k.append(cost)

    avg_1k = np.mean(scores_1k)
    print(f"Média Custo (1k): {avg_1k:.2f}")
    print(f"Ref Paper (Vanilla 1k): ~17.84 | Ref Paper (DRL 1k): ~16.80")

    # Bateria 2: 10.000 Iterações (Opcional se demorar muito)
    # print(f"\n>>> Rodando 10.000 Iterações (Full)...")
    # scores_10k = []
    # for i, _ in tqdm(enumerate(test_data), total=len(test_data)):
    #    ... (mesma lógica com iterations=10000)

if __name__ == "__main__":
    run_benchmark()