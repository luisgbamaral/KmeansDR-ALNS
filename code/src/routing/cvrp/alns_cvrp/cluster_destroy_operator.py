import copy
import numpy as np
import pandas as pd
import numpy.random as rnd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
# from kneed import KneeLocator # Removido - Não é mais necessário aqui
from routing.cvrp.alns_cvrp.cvrp_env import cvrpEnv

def get_customer_features_3d(current: cvrpEnv, served_customers: list):
    """ Extrai X, Y, e Demanda dos clientes servidos. """
    features = []
    for customer_id in served_customers:
        idx = customer_id - 1
        x = current.customers_x[idx]
        y = current.customers_y[idx]
        demand = current.demands_data[idx]
        features.append([customer_id, x, y, demand])
    return pd.DataFrame(features, columns=['id', 'x', 'y', 'demand'])

# REMOVIDA: A função find_optimal_k_elbow foi removida.
# Ela será movida para o ficheiro do ambiente (cvrp_AlnsEnv_LSA1.py)
# para ser executada apenas uma vez por instância.

def cluster_representative_removal(current: cvrpEnv, random_state: rnd.RandomState, nr_nodes_to_remove=None, **kwargs):
    """
    Operador de Destruição (Sua Lógica de Amostragem) - VERSÃO RÁPIDA:
    1. Clusteriza clientes por (X, Y, Demanda) usando PCA + K-Means (com 'k' pré-calculado).
    2. Identifica os 25% "representantes" (mais próximos do centroide).
    3. Destrói os 75% "outliers", guardando-os numa lista prioritária.
    """
    destroyed_solution = current.copy()
    
    served_customers = [customer for route in destroyed_solution.routes for customer in route]
    
    # --- MODIFICAÇÃO: Recebe K em vez de calcular ---
    # Define um k padrão de 10 caso não seja passado (segurança)
    k_optimal = kwargs.get('k_optimal', 10) 
    # -----------------------------------------------

    if len(served_customers) < k_optimal: # Muito pequeno para clusterizar
        return destroyed_solution

    df_clientes = get_customer_features_3d(current, served_customers)
    features = df_clientes[['x', 'y', 'demand']].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # --- MODIFICAÇÃO: Garante que 'k' não é maior que o n. de amostras ---
    n_samples = X_pca.shape[0]
    current_k = min(k_optimal, n_samples)
    
    if current_k <= 1: # Não pode clusterizar com k=1 ou k=0
        return destroyed_solution
    # -----------------------------------------------------------------

    # K-Means (agora roda apenas UMA vez, com n_init=1 para velocidade)
    kmeans = KMeans(n_clusters=current_k, random_state=random_state, n_init=1) 
    df_clientes['cluster'] = kmeans.fit_predict(X_pca)
    
    ids_para_remover_df_list = []
    
    # --- MODIFICAÇÃO: Itera sobre 'current_k' (k real) ---
    for i in range(current_k):
        cluster_mask = (df_clientes['cluster'] == i)
        
        # Correção do SettingWithCopyWarning (você já tinha)
        clientes_no_cluster = df_clientes[cluster_mask].copy() 
        
        if clientes_no_cluster.empty:
            continue
            
        # --- MODIFICAÇÃO: Correção do IndexError ---
        # A forma segura de obter os dados do PCA é usando os índices do DataFrame
        # 'df_clientes' e 'X_pca' partilham os mesmos índices (0, 1, 2...)
        indices_cluster = clientes_no_cluster.index
        X_cluster = X_pca[indices_cluster]
        # ------------------------------------------
        
        # Lida com o caso de um cluster ter menos pontos que o k-means esperava
        if i >= len(kmeans.cluster_centers_):
            continue

        centroide = kmeans.cluster_centers_[i]
        
        distancias = np.linalg.norm(X_cluster - centroide, axis=1)
        
        clientes_no_cluster['dist_centroide'] = distancias
        
        n_representantes = int(np.ceil(len(clientes_no_cluster) * 0.25))
        
        clientes_a_remover_df = clientes_no_cluster.sort_values('dist_centroide', ascending=True).iloc[n_representantes:]
        
        ids_para_remover_df_list.append(clientes_a_remover_df[['id', 'dist_centroide']])

    if not ids_para_remover_df_list:
        return destroyed_solution 

    df_outliers_priorizado = pd.concat(ids_para_remover_df_list).sort_values('dist_centroide', ascending=True)
    
    ids_para_remover_set = set(df_outliers_priorizado['id'].tolist())
    
    # Guarda a lista priorizada para o operador de Reparo
    destroyed_solution.priority_list = df_outliers_priorizado['id'].tolist()
    
    # Remove os clientes (outliers)
    destroyed_solution.remove_clientes(ids_para_remover_set)
    
    return destroyed_solution