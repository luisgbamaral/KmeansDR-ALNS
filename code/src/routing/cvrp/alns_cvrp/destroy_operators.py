import copy
import random
from routing.cvrp.alns_cvrp.cvrp_helper_functions import determine_nr_nodes_to_remove, NormalizeData

#TODO: put nr_nodes_to_remove in kwargs statement

# --- random removal ---
def random_removal(current, random_state, nr_nodes_to_remove=None, **kwargs):
    destroyed_solution = copy.deepcopy(current)
    visited_customers = [customer for route in destroyed_solution.routes for customer in route]

    if nr_nodes_to_remove is None:
        nr_nodes_to_remove = determine_nr_nodes_to_remove(destroyed_solution.nb_customers)

    # --- CORREÇÃO ADICIONADA (para ser seguro como o neighbor_graph) ---
    if not visited_customers or len(visited_customers) < nr_nodes_to_remove:
         # Não há clientes suficientes para remover, retorna o estado original
         return destroyed_solution
    # ------------------------------------------------------------------

    nodes_to_remove = random.sample(visited_customers, nr_nodes_to_remove)
    for node in nodes_to_remove:
        for route in destroyed_solution.routes:
            while node in route:
                route.remove(node)
                visited_customers.remove(node)
    destroyed_solution.routes = [route for route in destroyed_solution.routes if route != []]

    return destroyed_solution


# --- relatedness destroy method ---

# see: Shaw - Using Constraint Programming and Local Search Methods to Solve Vehicle Routing Problems
# see: Santini, Ropke - A comparison of acceptance criteria for the adaptive large neighbourhood search metaheuristic


def relatedness_removal(current, random_state, nr_nodes_to_remove=None, prob=5, **kwargs):
    destroyed_solution = copy.deepcopy(current)
    visited_customers = [customer for route in destroyed_solution.routes for customer in route]

    if nr_nodes_to_remove is None:
        nr_nodes_to_remove = determine_nr_nodes_to_remove(destroyed_solution.nb_customers)

    # --- CORREÇÃO ADICIONADA (para ser seguro como o neighbor_graph) ---
    if not visited_customers:
        return destroyed_solution # Não há clientes para remover
    # ------------------------------------------------------------------
    
    node_to_remove = random_state.choice(visited_customers)
    for route in destroyed_solution.routes:
        while node_to_remove in route:
            route.remove(node_to_remove)
            visited_customers.remove(node_to_remove)

    for i in range(nr_nodes_to_remove - 1):
        
        # --- CORREÇÃO ADICIONADA (para ser seguro) ---
        if not visited_customers: # Se remover o último cliente
            break
        # ---------------------------------------------
        
        related_nodes = []
        normalized_distances = NormalizeData(destroyed_solution.dist_matrix_data[node_to_remove - 1])
        route_node_to_remove = [route for route in current.routes if node_to_remove in route][0]
        for route in destroyed_solution.routes:
            for node in route:
                if node in route_node_to_remove:
                    related_nodes.append((node, normalized_distances[node - 1]))
                else:
                    related_nodes.append((node, normalized_distances[node - 1] + 1))

        if random_state.random() < 1 / prob:
            node_to_remove = random_state.choice(visited_customers)
        else:
            # --- CORREÇÃO ADICIONADA (para ser seguro) ---
            if not related_nodes: # Se não houver nós relacionados (ex: rota ficou vazia)
                node_to_remove = random_state.choice(visited_customers) # Escolhe um aleatório
            else:
                node_to_remove = min(related_nodes, key=lambda x: x[1])[0]
            # ---------------------------------------------

        for route in destroyed_solution.routes:
            while node_to_remove in route:
                route.remove(node_to_remove)
                visited_customers.remove(node_to_remove)
    destroyed_solution.routes = [route for route in destroyed_solution.routes if route != []]

    return destroyed_solution


# --- neighbor/history graph removal
# see: A unified heuristic for a large class of Vehicle Routing Problems with Backhauls
def neighbor_graph_removal(current, random_state, nr_nodes_to_remove=None, prob=5, **kwargs):
    destroyed_solution = copy.deepcopy(current)

    if nr_nodes_to_remove is None:
        nr_nodes_to_remove = determine_nr_nodes_to_remove(destroyed_solution.nb_customers)

    values = {}
    for route in destroyed_solution.routes:
        if len(route) == 1:
            values[route[0]] = current.graph.get_edge_weight(0, route[0]) + current.graph.get_edge_weight(route[0], 0)
        else:
            for i in range(len(route)):
                if i == 0:
                    values[route[i]] = current.graph.get_edge_weight(0, route[i]) + current.graph.get_edge_weight(
                        route[i], route[1])
                elif i == len(route) - 1:
                    values[route[i]] = current.graph.get_edge_weight(route[i - 1],
                                                                      route[i]) + current.graph.get_edge_weight(
                        route[i], 0)
                else:
                    values[route[i]] = current.graph.get_edge_weight(route[i - 1],
                                                                      route[i]) + current.graph.get_edge_weight(
                        route[i], route[i + 1])
    
    # --- MODIFICAÇÃO 1: VERIFICA SE HÁ ALGO PARA REMOVER ---
    if not values: # Se o dict 'values' estiver vazio, não há clientes
        return destroyed_solution
    # -----------------------------------------------------

    removed_nodes = []
    
    # --- MODIFICAÇÃO 2: Garante que não remove mais do que existe ---
    nodes_to_remove_count = min(nr_nodes_to_remove, len(values))
    # -------------------------------------------------------------

    while len(removed_nodes) < nodes_to_remove_count:
        # sort the nodes based on their neighbor graph scores in descending order
        sorted_nodes = sorted(values.items(), key=lambda x: x[1], reverse=True)
        
        # --- MODIFICAÇÃO 3: VERIFICA SE AINDA HÁ NÓS ---
        if not sorted_nodes: # Se 'sorted_nodes' ficar vazio no meio do loop
            break # Para de tentar remover
        # -----------------------------------------------

        # select the node to remove
        removal_option = 0
        while random_state.random() < 1 / prob and removal_option < len(sorted_nodes) - 1:
            removal_option += 1
        
        node_to_remove, score = sorted_nodes[removal_option] # Este era o ponto do IndexError

        # remove the node from its route
        for route in destroyed_solution.routes:
            if node_to_remove in route:
                route.remove(node_to_remove)
                removed_nodes.append(node_to_remove)

                values.pop(node_to_remove)
                if len(route) == 0:
                    destroyed_solution.routes.remove([])

                elif len(route) == 1:
                    values[route[0]] = current.graph.get_edge_weight(0, route[0]) + current.graph.get_edge_weight(
                        route[0], 0)
                else:
                    for i in range(len(route)):
                        if i == 0:
                            values[route[i]] = current.graph.get_edge_weight(0, route[
                                i]) + current.graph.get_edge_weight(route[i], route[1])
                        elif i == len(route) - 1:
                            values[route[i]] = current.graph.get_edge_weight(route[i - 1], route[
                                i]) + current.graph.get_edge_weight(route[i], 0)
                        else:
                            values[route[i]] = current.graph.get_edge_weight(route[i - 1], route[
                                i]) + current.graph.get_edge_weight(route[i], route[i + 1])

                break

    return destroyed_solution