import copy
import numpy as np
import numpy.random as rnd
from routing.cvrp.alns_cvrp.cvrp_env import cvrpEnv
from routing.cvrp.alns_cvrp.repair_operators import regret_insertion # Importamos o original como fallback
from routing.cvrp.alns_cvrp.cvrp_helper_functions import compute_route_load

def get_best_greedy_insertion(route, customer, dist_matrix_data, dist_depot_data):
    """
    Encontra o melhor local de inserção "greedy" (menor custo) para um cliente numa rota.
    """
    best_cost = float('inf')
    best_idx = -1

    # --- MODIFICAÇÃO: Lidar com rota vazia ---
    if not route: # Se a rota estiver vazia
        # Custo = (Depósito -> Cliente) + (Cliente -> Depósito)
        cost = dist_depot_data[customer - 1] + dist_depot_data[customer - 1]
        return cost, 0 # Insere na posição 0
    # -----------------------------------------

    for i in range(len(route) + 1):
        if i == 0: # Início da rota
            cost = dist_depot_data[customer - 1] + dist_matrix_data[customer - 1][route[0] - 1] - dist_depot_data[route[0] - 1]
        elif i == len(route): # Fim da rota
            cost = dist_matrix_data[route[-1] - 1][customer - 1] + dist_depot_data[customer - 1] - dist_depot_data[route[-1] - 1]
        else: # Meio da rota
            cost = dist_matrix_data[route[i-1] - 1][customer - 1] + dist_matrix_data[customer - 1][route[i] - 1] - dist_matrix_data[route[i-1] - 1][route[i] - 1]
        
        if cost < best_cost:
            best_cost = cost
            best_idx = i
            
    return best_cost, best_idx

def cluster_priority_repair(current: cvrpEnv, random_state: rnd.RandomState, **kwargs):
    """
    Operador de Reparo (Sua Ideia de Prioridade):
    1. Verifica se foi o `cluster_representative_removal` que o chamou (procurando `priority_list`).
    2. Se SIM: Insere os clientes na ordem de prioridade (mais próximo do centroide primeiro),
       usando uma inserção Greedy (melhor local).
    3. Se NÃO: Executa o `regret_insertion` padrão como fallback.
    """
    
    # --- MODIFICAÇÃO: Corrigir o AttributeError ---
    # Verifica se o 'current' (o estado destruído) tem a lista.
    if hasattr(current, 'priority_list'):
        
        # 1. Pega a lista ANTES de copiar
        unvisited_customers = current.priority_list 
        
        # 2. Agora faz a cópia
        repaired = current.copy()
        
        # 3. Limpa o atributo do estado 'repaired' para não causar problemas noutras iterações
        if hasattr(repaired, 'priority_list'):
             delattr(repaired, 'priority_list')
    # ----------------------------------------------
        
        # Loop de inserção Greedy Priorizada
        for customer_to_insert in unvisited_customers:
            best_overall_cost = float('inf')
            best_route_idx = -1
            best_insert_idx = -1
            
            cust_demand = repaired.demands_data[customer_to_insert - 1]

            # Encontra o melhor local em *todas* as rotas existentes
            for r_idx, route in enumerate(repaired.routes):
                route_load = compute_route_load(route, repaired.demands_data)
                
                # Verifica capacidade
                if route_load + cust_demand <= repaired.truck_capacity:
                    cost, insert_idx = get_best_greedy_insertion(route, customer_to_insert, 
                                                                 repaired.dist_matrix_data, 
                                                                 repaired.dist_depot_data)
                    if cost < best_overall_cost:
                        best_overall_cost = cost
                        best_route_idx = r_idx
                        best_insert_idx = insert_idx
            
            # Se achou um local, insere.
            if best_route_idx != -1:
                repaired.routes[best_route_idx].insert(best_insert_idx, customer_to_insert)
            else:
                # Não coube em nenhuma rota existente, cria uma nova
                if cust_demand <= repaired.truck_capacity:
                    repaired.routes.append([customer_to_insert])
        
        return repaired

    else:
        # Fallback: se outro 'destroy' foi usado, executa o 'regret_insertion'
        return regret_insertion(current, random_state, **kwargs)