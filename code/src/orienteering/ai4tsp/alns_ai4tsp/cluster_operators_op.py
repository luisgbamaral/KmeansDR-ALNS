import copy
import numpy as np
import pandas as pd
import numpy.random as rnd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Importa a função de inserção gulosa original do problema
from orienteering.ai4tsp.alns_ai4tsp.repair_operators import get_best_prize_insertion_for_node
from orienteering.ai4tsp.alns_ai4tsp.destroy_operators import random_removal

def get_node_features_3d(current, nodes_list: list):
    """Extrai features (X, Y, Prize) para o PCA."""
    features = []
    all_features = current.x
    for node_id in nodes_list:
        idx = node_id - 1
        x_coord = all_features[idx][0]
        y_coord = all_features[idx][1]
        prize = all_features[idx][-2]
        features.append([node_id, x_coord, y_coord, prize])
    return pd.DataFrame(features, columns=['id', 'x', 'y', 'prize'])

def cluster_representative_removal_op(current, random_state, degree_of_destruction=None, **kwargs):
    """
    Destruição por Cluster Dinâmica.
    """
    # Proteção para rotas vazias ou só com depot
    if len(current.route) <= 2:
        return current

    destroyed_solution = copy.deepcopy(current)
    
    # Prepara lista de nós servidos (sem depot)
    served_nodes = list(set(destroyed_solution.route))
    if 1 in served_nodes: served_nodes.remove(1)
        
    k_optimal = kwargs.get('k_optimal', 5)
    
    # Fallback: Se a rota for pequena demais, usa Random Removal
    if len(served_nodes) < k_optimal:
        return random_removal(current, random_state, degree_of_destruction=degree_of_destruction)

    # Pipeline PCA + K-Means
    try:
        df_nodes = get_node_features_3d(current, served_nodes)
        features = df_nodes[['x', 'y', 'prize']].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        current_k = min(k_optimal, len(served_nodes))
        if current_k <= 1: return destroyed_solution

        kmeans = KMeans(n_clusters=current_k, random_state=random_state, n_init=1)
        df_nodes['cluster'] = kmeans.fit_predict(X_pca)
    except Exception:
        # Se o PCA/KMeans falhar por qualquer motivo numérico, fallback
        return random_removal(current, random_state, degree_of_destruction=degree_of_destruction)
    
    ids_to_remove_list = []
    
    # Define grau de destruição
    deg = degree_of_destruction if degree_of_destruction is not None else 0.3

    for i in range(current_k):
        cluster_mask = (df_nodes['cluster'] == i)
        nodes_in_cluster = df_nodes[cluster_mask].copy()
        if nodes_in_cluster.empty: continue
        
        # Centroide
        indices = nodes_in_cluster.index
        if i >= len(kmeans.cluster_centers_): continue
        centroid = kmeans.cluster_centers_[i]
        
        X_cluster = X_pca[indices]
        nodes_in_cluster['dist_centroid'] = np.linalg.norm(X_cluster - centroid, axis=1)
        
        # Lógica de Intensidade
        n_total = len(nodes_in_cluster)
        n_remove = int(np.round(n_total * deg))
        n_remove = max(1, min(n_remove, n_total - 1)) # Garante remover e manter pelo menos 1
        n_keep = n_total - n_remove
        
        # Ordena e corta
        remove_df = nodes_in_cluster.sort_values('dist_centroid', ascending=True).iloc[n_keep:]
        ids_to_remove_list.append(remove_df[['id', 'dist_centroid']])

    if not ids_to_remove_list:
        return destroyed_solution

    # Priorização
    df_outliers = pd.concat(ids_to_remove_list)
    df_outliers = df_outliers.sort_values('dist_centroid', ascending=True)
    
    priority_list = df_outliers['id'].tolist()
    destroyed_solution.priority_list = priority_list
    
    # Efetiva a remoção da rota
    ids_set = set(priority_list)
    destroyed_solution.route = [n for n in destroyed_solution.route if n not in ids_set]
            
    return destroyed_solution

def cluster_priority_repair_op(current, random_state, **kwargs):
    """
    Reparo Priorizado com verificação de segurança contra duplicatas.
    """
    if not hasattr(current, 'priority_list'):
        from orienteering.ai4tsp.alns_ai4tsp.repair_operators import random_best_prize_repair
        return random_best_prize_repair(current, random_state, **kwargs)

    unvisited_ordered = current.priority_list
    repaired = copy.deepcopy(current)
    delattr(repaired, 'priority_list')
    
    # Score inicial (usado pelo helper de inserção)
    # Nota: objective() retorna Custo negativo ou Reward positivo dependendo da impl.
    # Vamos confiar que o get_best_prize_insertion sabe lidar com o valor retornado.
    curr_score = repaired.objective()
    
    pool = kwargs.get('pool', None)

    for node in unvisited_ordered:
        # --- SEGURANÇA CRÍTICA (CORREÇÃO DO ERRO) ---
        # Se o nó já estiver na rota (por falha no destroy ou duplicação na lista),
        # ignoramos para evitar quebra do ambiente (AssertionError len(sol)).
        if node in repaired.route:
            continue
        # ---------------------------------------------

        new_route = get_best_prize_insertion_for_node(
            node, 
            repaired.nodes, 
            repaired.route, 
            curr_score, 
            repaired.adj, 
            repaired.x, 
            pool
        )
        
        if new_route != repaired.route:
            repaired.route = new_route
            curr_score = repaired.objective()
            
    return repaired