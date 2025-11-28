import copy
import numpy as np
import pandas as pd
import numpy.random as rnd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Importamos a função de inserção "gulosa" nativa do problema OP
# Ela avalia a melhor posição de inserção considerando Prêmios e Penalidades
from orienteering.ai4tsp.alns_ai4tsp.repair_operators import get_best_prize_insertion_for_node

def get_node_features_3d(current, nodes_list: list):
    """
    Extrai X, Y e Prize (análogo a Demanda no CVRP) dos nós servidos.
    No AI4TSP, current.x é uma matriz numpy onde:
    coluna 0: X
    coluna 1: Y
    coluna -2: Prize
    """
    features = []
    all_features = current.x # Matriz estática do problema
    
    for node_id in nodes_list:
        idx = node_id - 1 # Ajuste de índice (nós começam em 1, array em 0)
        x_coord = all_features[idx][0]
        y_coord = all_features[idx][1]
        prize = all_features[idx][-2] # O "Peso" agora é o Prêmio
        features.append([node_id, x_coord, y_coord, prize])
        
    return pd.DataFrame(features, columns=['id', 'x', 'y', 'prize'])

def cluster_representative_removal_op(current, random_state, degree_of_destruction=None, **kwargs):
    """
    Operador de Destruição Inteligente (Lógica CVRP portada para OP):
    1. Clusteriza clientes atendidos por (X, Y, Prize) usando PCA + K-Means.
    2. Identifica 25% "representantes" (próximos ao centroide) e mantém.
    3. Destrói 75% "outliers", guardando numa lista prioritária ordenada pela distância ao centroide.
    """
    # Proteção: Se a rota for vazia ou só tiver depósito [1, 1]
    if len(current.route) <= 2:
        return current

    destroyed_solution = copy.deepcopy(current)
    
    # Identifica nós servidos (excluindo depósito '1')
    served_nodes = list(set(destroyed_solution.route))
    if 1 in served_nodes: 
        served_nodes.remove(1)
        
    # Pega o K ótimo calculado no reset do ambiente
    k_optimal = kwargs.get('k_optimal', 5) 
    
    # Se tivermos poucos nós para clusterizar, retorna sem fazer nada
    if len(served_nodes) < k_optimal:
        return destroyed_solution

    # 1. Extração de Features (X, Y, Prize)
    df_nodes = get_node_features_3d(current, served_nodes)
    features = df_nodes[['x', 'y', 'prize']].values
    
    # 2. Pipeline: Padronização -> PCA -> K-Means
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    n_samples = X_pca.shape[0]
    current_k = min(k_optimal, n_samples)
    
    if current_k <= 1:
        return destroyed_solution

    # Executa K-Means no espaço latente
    kmeans = KMeans(n_clusters=current_k, random_state=random_state, n_init=1)
    df_nodes['cluster'] = kmeans.fit_predict(X_pca)
    
    ids_to_remove_list = []
    
    # 3. Seleção de Representantes vs Outliers
    for i in range(current_k):
        cluster_mask = (df_nodes['cluster'] == i)
        nodes_in_cluster = df_nodes[cluster_mask].copy()
        
        if nodes_in_cluster.empty: continue
        
        indices = nodes_in_cluster.index
        if i >= len(kmeans.cluster_centers_): continue
        centroid = kmeans.cluster_centers_[i]
        
        # Calcula distância Euclidiana no espaço PCA (2D)
        X_cluster = X_pca[indices]
        dists = np.linalg.norm(X_cluster - centroid, axis=1)
        nodes_in_cluster['dist_centroid'] = dists
        
        # Lógica idêntica ao CVRP:
        # Mantém os 25% mais próximos (núcleo)
        # Remove os 75% mais distantes (outliers)
        n_keep = int(np.ceil(len(nodes_in_cluster) * 0.25))
        
        # Ordena: Menor dist -> Maior dist. Pega do n_keep até o fim.
        remove_df = nodes_in_cluster.sort_values('dist_centroid', ascending=True).iloc[n_keep:]
        
        # Guarda ID e Distância para priorização
        ids_to_remove_list.append(remove_df[['id', 'dist_centroid']])

    if not ids_to_remove_list:
        return destroyed_solution

    # Concatena todos os removidos
    df_outliers_priorizado = pd.concat(ids_to_remove_list)
    
    # 4. Ordenação da Lista de Prioridade
    # Exatamente como no CVRP: Ordenamos de forma crescente pela distância ao centroide.
    # Isso significa que o Repair tentará reinserir primeiro os nós que são "quase centrais"
    # e por último os "totalmente periféricos".
    df_outliers_priorizado = df_outliers_priorizado.sort_values('dist_centroid', ascending=True)
    
    priority_list = df_outliers_priorizado['id'].tolist()
    
    # Anexa a lista à solução para o operador de Reparo usar
    destroyed_solution.priority_list = priority_list
    
    # Remove efetivamente os nós da rota
    ids_set = set(priority_list)
    destroyed_solution.route = [node for node in destroyed_solution.route if node not in ids_set]
            
    return destroyed_solution

def cluster_priority_repair_op(current, random_state, **kwargs):
    """
    Operador de Reparo Priorizado (Lógica CVRP portada para OP):
    1. Verifica se existe priority_list.
    2. Tenta inserir sequencialmente seguindo a ordem da lista.
    3. Usa inserção gulosa (Best Prize Insertion).
    """
    # Fallback se a solução não veio do cluster destroy
    if not hasattr(current, 'priority_list'):
        from orienteering.ai4tsp.alns_ai4tsp.repair_operators import random_best_prize_repair
        return random_best_prize_repair(current, random_state, **kwargs)

    # 1. Pega a lista e limpa o atributo
    unvisited_ordered = current.priority_list
    repaired = copy.deepcopy(current)
    delattr(repaired, 'priority_list') 
    
    curr_score = repaired.objective() 
    pool = kwargs.get('pool', None)

    # 2. Loop Priorizado
    for node in unvisited_ordered:
        # Tenta encontrar a melhor posição de inserção para este nó específico
        # A função get_best_prize_insertion_for_node avalia todas as posições
        # e retorna a nova rota se a inserção for benéfica e viável.
        new_route = get_best_prize_insertion_for_node(
            node, 
            repaired.nodes, 
            repaired.route, 
            curr_score, 
            repaired.adj, 
            repaired.x, 
            pool
        )
        
        # Se a rota mudou, significa que inserimos com sucesso
        if new_route != repaired.route:
            repaired.route = new_route
            curr_score = repaired.objective() # Atualiza score
            
    return repaired