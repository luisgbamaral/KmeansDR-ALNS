import copy
import numpy as np
import pandas as pd
import numpy.random as rnd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Importa a função de inserção gulosa original do problema (Necessária para o Repair)
# Certifique-se que o caminho 'orienteering.ai4tsp...' está acessível no seu PYTHONPATH
from orienteering.ai4tsp.alns_ai4tsp.repair_operators import get_best_prize_insertion_for_node
from orienteering.ai4tsp.alns_ai4tsp.destroy_operators import random_removal # Fallback

def get_node_features_3d(current, nodes_list: list):
    """
    Extrai as features (X, Y, Prize) dos nós para alimentar o PCA.
    No OPSWTW, 'current.x' é uma matriz numpy com todas as infos estáticas.
    """
    features = []
    all_features = current.x
    
    for node_id in nodes_list:
        idx = node_id - 1 # O ID do nó começa em 1, índice do array em 0
        x_coord = all_features[idx][0]
        y_coord = all_features[idx][1]
        prize = all_features[idx][-2] # A penúltima coluna é o Prêmio
        features.append([node_id, x_coord, y_coord, prize])
        
    return pd.DataFrame(features, columns=['id', 'x', 'y', 'prize'])

def cluster_representative_removal_op(current, random_state, degree_of_destruction=None, **kwargs):
    """
    Operador de Destruição Baseado em Cluster Inteligente.
    1. Reduz dimensão (X, Y, Prize) -> 2D Latente via PCA.
    2. Clusteriza usando K-Means (K ótimo calculado no reset do ambiente).
    3. Remove 'Outliers' (nós periféricos) respeitando o grau de destruição do RL.
    """
    # 1. Proteção: Se a rota é vazia ou só tem depósito, não faz nada
    if len(current.route) <= 2:
        return current

    destroyed_solution = copy.deepcopy(current)
    
    # Lista de nós visitados (excluindo depósito '1')
    served_nodes = list(set(destroyed_solution.route))
    if 1 in served_nodes: 
        served_nodes.remove(1)
        
    # Recupera o K calculado no ambiente (passado via kwargs)
    k_optimal = kwargs.get('k_optimal', 5)
    
    # 2. Fallback: Se a rota for menor que K, não dá para clusterizar.
    # Usamos Random Removal para não desperdiçar a ação do agente.
    if len(served_nodes) < k_optimal:
        return random_removal(current, random_state, degree_of_destruction=degree_of_destruction)

    # 3. Pipeline de ML: Extração -> Scale -> PCA
    df_nodes = get_node_features_3d(current, served_nodes)
    features = df_nodes[['x', 'y', 'prize']].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Ajuste de segurança para o K-Means
    current_k = min(k_optimal, len(served_nodes))
    if current_k <= 1: 
        return destroyed_solution

    # 4. Clustering
    kmeans = KMeans(n_clusters=current_k, random_state=random_state, n_init=1)
    df_nodes['cluster'] = kmeans.fit_predict(X_pca)
    
    ids_to_remove_list = []
    
    # Define grau de destruição (Se vier None, usa 0.3 como padrão seguro)
    deg = degree_of_destruction if degree_of_destruction is not None else 0.3

    # 5. Seleção de Outliers por Cluster
    for i in range(current_k):
        cluster_mask = (df_nodes['cluster'] == i)
        nodes_in_cluster = df_nodes[cluster_mask].copy()
        
        if nodes_in_cluster.empty: continue
        
        # Centroide
        indices = nodes_in_cluster.index
        if i >= len(kmeans.cluster_centers_): continue
        centroid = kmeans.cluster_centers_[i]
        
        # Distância Euclidiana no Espaço Latente (PCA)
        X_cluster = X_pca[indices]
        nodes_in_cluster['dist_centroid'] = np.linalg.norm(X_cluster - centroid, axis=1)
        
        # --- LÓGICA CORRIGIDA (CRUCIAL PARA 100 NÓS) ---
        # Calcula quantos remover baseado na decisão do RL (deg)
        n_total = len(nodes_in_cluster)
        n_remove = int(np.round(n_total * deg))
        
        # Garante limites lógicos (não remover 0 nem todos, se possível)
        n_remove = max(1, min(n_remove, n_total - 1))
        n_keep = n_total - n_remove
        
        # Ordena: Mais próximos (keep) -> Mais distantes (remove)
        remove_df = nodes_in_cluster.sort_values('dist_centroid', ascending=True).iloc[n_keep:]
        
        ids_to_remove_list.append(remove_df[['id', 'dist_centroid']])

    if not ids_to_remove_list:
        return destroyed_solution

    # 6. Priorização Global e Remoção
    df_outliers = pd.concat(ids_to_remove_list)
    # Ordena globalmente: Reparo tentará inserir os "quase centrais" primeiro
    df_outliers = df_outliers.sort_values('dist_centroid', ascending=True)
    
    priority_list = df_outliers['id'].tolist()
    
    # Anexa lista para o operador de Reparo usar
    destroyed_solution.priority_list = priority_list
    
    # Remove os nós da rota atual
    ids_set = set(priority_list)
    destroyed_solution.route = [n for n in destroyed_solution.route if n not in ids_set]
            
    return destroyed_solution

def cluster_priority_repair_op(current, random_state, **kwargs):
    """
    Operador de Reparo Inteligente.
    Usa a 'priority_list' gerada na destruição para guiar a reinserção.
    """
    # Se não houver lista (ex: outro destroy foi usado), usa o padrão Prize Repair
    if not hasattr(current, 'priority_list'):
        from orienteering.ai4tsp.alns_ai4tsp.repair_operators import random_best_prize_repair
        return random_best_prize_repair(current, random_state, **kwargs)

    # Consome a lista de prioridade
    unvisited_ordered = current.priority_list
    repaired = copy.deepcopy(current)
    delattr(repaired, 'priority_list')
    
    curr_score = repaired.objective()
    pool = kwargs.get('pool', None)

    # Tenta inserir cada nó na melhor posição possível
    for node in unvisited_ordered:
        new_route = get_best_prize_insertion_for_node(
            node, 
            repaired.nodes, 
            repaired.route, 
            curr_score, 
            repaired.adj, 
            repaired.x, 
            pool
        )
        
        # Se a rota mudou, atualiza o score e segue
        if new_route != repaired.route:
            repaired.route = new_route
            curr_score = repaired.objective()
            
    return repaired