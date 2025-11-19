import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import copy
import time
import gymnasium as gym
import random
from alns import ALNS
import numpy as np
import numpy.random as rnd
from pathlib import Path

# --- MODIFICAÇÃO 1: Novas Importações ---
# Importações necessárias para o cálculo do K-Ótimo (Elbow)
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from kneed import KneeLocator
# ---------------------------------------

from routing.cvrp.alns_cvrp import cvrp_helper_functions

from routing.cvrp.alns_cvrp.cvrp_env import cvrpEnv
from routing.cvrp.alns_cvrp.destroy_operators import neighbor_graph_removal, random_removal, relatedness_removal
from routing.cvrp.alns_cvrp.repair_operators import regret_insertion
from routing.cvrp.alns_cvrp.initial_solution import compute_initial_solution

# --- Importações dos seus novos operadores ---
from routing.cvrp.alns_cvrp.cluster_destroy_operator import cluster_representative_removal
from routing.cvrp.alns_cvrp.cluster_repair_operator import cluster_priority_repair
# -------------------------------------------


# --- MODIFICAÇÃO 2: Funções Helper (Para o "Passo Anterior") ---
# Estas funções são usadas para calcular o K-Ótimo uma vez por instância.

def get_all_customer_features_3d(nb_customers, customers_x, customers_y, demands_data):
    """ Extrai X, Y, e Demanda de TODOS os clientes da instância. """
    features = []
    for customer_id in range(1, nb_customers + 1):
        idx = customer_id - 1
        x = customers_x[idx]
        y = customers_y[idx]
        demand = demands_data[idx]
        features.append([customer_id, x, y, demand])
    return pd.DataFrame(features, columns=['id', 'x', 'y', 'demand'])

def find_optimal_k_elbow_for_instance(X_norm, random_state, max_k=15):
    """ Executa o Método Elbow (Cotovelo) para achar o k ideal. """
    inertias = []
    try:
        # Garante que k não seja maior que o número de pontos únicos
        num_unique_points = np.unique(X_norm, axis=0).shape[0]
    except Exception:
        num_unique_points = X_norm.shape[0]
    
    k_range = range(2, min(max_k + 1, num_unique_points))
    
    if len(k_range) == 0: return 2 # Fallback
        
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(X_norm)
        inertias.append(km.inertia_)

    if len(k_range) == 1: return k_range[0] # Só testou um k
    
    try:
        kneedle = KneeLocator(k_range, inertias, curve='convex', direction='decreasing')
        optimal_k = kneedle.elbow
    except Exception:
        optimal_k = k_range[len(k_range) // 2] # Fallback

    return optimal_k if optimal_k else 2
# --------------------------------------------------------------------------


class cvrpAlnsEnv_LSA1(gym.Env):
    def __init__(self, config, **kwargs):

        # Parameters
        self.config = config["environment"]
        self.rnd_state = rnd.RandomState()

        # Simulated annealing acceptance criteria
        self.max_temperature = 5
        self.temperature = 5

        # LOAD INSTANCE
        base_path = Path(__file__).resolve().parents[2]
        self.instance_file = str(base_path.joinpath(self.config["instance_file"]))

        self.instances = self.config["instance_nr"]
        self.instance = None
        self.best_routes = []

        self.initial_solution = None
        self.best_solution = None
        self.current_solution = None
        
        self.k_optimal_instance = 10 # Valor padrão

        self.improvement = None
        self.cost_difference_from_best = None
        self.current_updated = None
        self.current_improved = None

        # Gym-related part
        self.reward = 0  # Total episode reward
        self.done = False  # Termination
        self.episode = 0  # Episode number (one episode consists of ngen generations)
        self.iteration = 0  # Current gen in the episode
        self.max_iterations = self.config["iterations"]  # max number of generations in an episode

        # --- MODIFICAÇÃO 3: Atualizar o Action Space ---
        # Action space (4 destroy ops, 2 repair ops, 10 destroy factors, 100 temperatures)
        self.action_space = gym.spaces.MultiDiscrete([4, 2, 10, 100])
        # ---------------------------------------------
        self.observation_space = gym.spaces.Box(shape=(8,), low=0, high=100, dtype=np.float64)

    def make_observation(self):
        """
        Return the environment's current state
        """

        is_current_best = 0
        if self.current_solution.objective() == self.best_solution.objective():
            is_current_best = 1

        state = np.array(
            [self.improvement, self.cost_difference_from_best, is_current_best, self.temperature,
             self.stagcount, self.iteration / self.max_iterations, self.current_updated, self.current_improved],
            dtype=np.float64).squeeze()

        return state

    def reset(self, seed=None, options=None):
        """
        The reset method: returns the current state of the environment (first state after initialization/reset)
        """

        SEED = random.randint(0, 10000)
        random_state = rnd.RandomState(SEED) # Usar o SEED local

        # randomly select problem instance
        self.instance = random.choice(self.instances)

        # --- MODIFICAÇÃO 4: Calcular K-Ótimo (O seu "Passo Anterior") ---
        # Load instance e X, Y
        nb_customers, truck_capacity, dist_matrix_data, dist_depot_data, demands_data, customers_x, customers_y = cvrp_helper_functions.read_input_cvrp(self.instance_file, self.instance)

        print(f"Calculando K-Ótimo para instância {self.instance}...")
        df_full = get_all_customer_features_3d(nb_customers, customers_x, customers_y, demands_data)
        X_full = df_full[['x', 'y', 'demand']].values
        X_scaled = StandardScaler().fit_transform(X_full)
        X_pca = PCA(n_components=2).fit_transform(X_scaled)
        
        # Armazena o K ideal para esta instância
        self.k_optimal_instance = find_optimal_k_elbow_for_instance(X_pca, random_state, max_k=20)
        print(f"K-Ótimo encontrado: {self.k_optimal_instance}")
        # ------------------------------------------------------------------
        
        # Passa X, Y para o construtor do cvrpEnv
        state = cvrpEnv([], nb_customers, truck_capacity, dist_matrix_data, dist_depot_data, demands_data, customers_x, customers_y, self.instance, SEED)

        self.initial_solution = compute_initial_solution(state, random_state)
        self.current_solution = copy.deepcopy(self.initial_solution)
        self.best_solution = copy.deepcopy(self.initial_solution)

        # --- MODIFICAÇÃO 5: Adicionar seus operadores ---
        self.dr_alns = ALNS(random_state)
        # Destroy
        self.dr_alns.add_destroy_operator(random_removal)
        self.dr_alns.add_destroy_operator(relatedness_removal)
        self.dr_alns.add_destroy_operator(neighbor_graph_removal)
        self.dr_alns.add_destroy_operator(cluster_representative_removal) # Seu (índice 3)
        # Repair
        self.dr_alns.add_repair_operator(regret_insertion)
        self.dr_alns.add_repair_operator(cluster_priority_repair)          # Seu (índice 1)
        # -----------------------------------------------

        # reset tracking values
        self.stagcount = 0
        self.current_improved = 0
        self.current_updated = 0
        self.episode += 1
        self.temperature = self.max_temperature
        self.improvement = 0
        self.cost_difference_from_best = 0

        self.iteration, self.reward = 0, 0
        self.done = False

        return self.make_observation(), {}

    def step(self, action, **kwargs):
        self.iteration += 1
        self.stagcount += 1
        self.current_updated = 0
        self.reward = 0
        self.improvement = 0
        self.cost_difference_from_best = 0
        self.current_improved = 0

        current = self.current_solution
        best = self.best_solution

        d_idx, r_idx = action[0], action[1]
        d_name, d_operator = self.dr_alns.destroy_operators[d_idx]

        factors = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4, 4: 0.5, 5: 0.6, 6: 0.7, 7: 0.8, 8: 0.9, 9: 1.0}
        nr_nodes_to_remove = round(factors[action[2]] * current.nb_customers)

        self.temperature = (1/(action[3]+1)) * self.max_temperature


        if nr_nodes_to_remove == current.nb_customers and current.nb_customers > 0: # Adicionado check
            nr_nodes_to_remove -= 1

        # --- MODIFICAÇÃO 6: Passar o K-Ótimo para o operador ---
        # Passa o 'k' pré-calculado para o seu operador de destruição
        kwargs_for_op = {'k_optimal': self.k_optimal_instance}
        destroyed = d_operator(current, self.rnd_state, nr_nodes_to_remove, **kwargs_for_op)
        # -------------------------------------------------------

        r_name, r_operator = self.dr_alns.repair_operators[r_idx]
        candidate = r_operator(destroyed, self.rnd_state)

        new_best, new_current = self.consider_candidate(best, current, candidate)

        if new_best != best and new_best is not None:
            # found new best solution
            self.best_solution = new_best
            self.current_solution = new_best
            self.current_updated = 1
            self.reward += 5
            self.stagcount = 0
            self.current_improved = 1

        elif new_current != current and new_current.objective() > current.objective():
            # solution accepted, because better than current, but not better than best
            self.current_solution = new_current
            self.current_updated = 1
            self.current_improved = 1
            # self.reward += 3

        elif new_current != current and new_current.objective() <= current.objective():
            # solution accepted
            self.current_solution = new_current
            self.current_updated = 1
            # self.reward += 1

        if new_current.objective() > current.objective():
            self.improvement = 1

        self.cost_difference_from_best = (self.current_solution.objective() / self.best_solution.objective()) * 100

        # update graph of current and best solutions
        self.current_solution.graph = self.best_solution.graph = cvrp_helper_functions.update_neighbor_graph(candidate, candidate.routes, candidate.objective())

        state = self.make_observation()
        self.best_routes.append(self.best_solution.objective())

        # Check if episode is finished (max ngen per episode)
        if self.iteration == self.max_iterations:
            self.done = True

            import random, string, csv, os
            # --- MODIFICAÇÃO 7: Corrigir caminho do log (apenas um exemplo, ajuste se necessário) ---
            # O caminho original /hpc/ era para um cluster. Mudei para um caminho relativo.
            directory_path = './output_trajectories_drl/'
            # -----------------------------------------------------------------
            
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)

            # Generate random file name
            file_name = ''.join(random.choices(string.ascii_letters + string.digits, k=100)) + '.csv'
            random_string = os.path.join(directory_path, file_name)

            # Write data to the file
            with open(random_string, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(self.best_routes)

        # Corrigido para 5 valores de retorno: (observação, recompensa, terminado, truncado, info)
        return state, self.reward, self.done, False, {}

    # --------------------------------------------------------------------------------------------------------------------

    def consider_candidate(self, best, curr, cand):
        # Simulated Annealing
        probability = np.exp((curr.objective() - cand.objective()) / self.temperature)

        # best:
        if cand.objective() < best.objective():
            return cand, cand

        # accepted:
        elif probability >= rnd.random():
            return None, cand

        else:
            return None, curr

    # --------------------------------------------------------------------------------------------------------------------

    # --- MODIFICAÇÃO 8: Corrigir o unpack do step (5 valores) ---
    def run(self, model, episodes=1):
        """
        Use a trained model to select actions
        """
        try:
            for episode in range(episodes):
                self.done = False
                state, info = self.reset() # Reset agora retorna 2 valores
                while not self.done:
                    action = model.predict(state)
                    state, reward, terminated, truncated, info = self.step(action[0])
                    self.done = terminated or truncated # Atualiza o done
                    # print(state, reward, self.iteration)
        except KeyboardInterrupt:
            pass


    def run_time_limit(self, model, episodes=1):
        """
        Use a trained model to select actions
        """
        try:
            for episode in range(episodes):
                start_time = time.time()
                time_done = False
                state, info = self.reset() # Reset agora retorna 2 valores
                while not time_done:
                    action = model.predict(state)
                    state, reward, terminated, truncated, info = self.step(action[0])
                    current_time = time.time() - start_time
                    print(current_time)
                    if current_time > 30:
                        time_done = True
                    # print(state, reward, self.iteration)
        except KeyboardInterrupt:
            pass

    def sample(self):
        """
        Sample random actions and run the environment
        """
        for episode in range(2):
            self.done = False
            state, info = self.reset() # Reset agora retorna 2 valores
            print("start episode: ", episode, " with start state: ", state)
            while not self.done:
                action = self.action_space.sample()
                state, reward, terminated, truncated, info = self.step(action)
                self.done = terminated or truncated # Atualiza o done
                print(
                    "step {}, action: {}, New state: {}, Reward: {:2.3f}".format(
                        self.iteration, action, state, reward
                    )
                )
    # -------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # --- MODIFICAÇÃO 9: Importações necessárias no __main__ ---
    from rl.baselines import get_parameters, Trainer
    # --------------------------------------------------------

    env = cvrpAlnsEnv_LSA1(get_parameters("cvrpAlnsEnv_LSA1"))
    # print("Sampling random actions...")
    # env.sample()

    print('Start training')
    model = Trainer("cvrpAlnsEnv_LSA1", "models").create_model()
    # model._tensorboard()
    model.train()
    print("Training done")
    input("Run trained model (Enter)")
    env.run(model)