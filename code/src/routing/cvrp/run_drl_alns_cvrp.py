import csv
from pathlib import Path
import time
import helper_functions  # Importação de Nível Superior (funcionará a partir de 'src')
from stable_baselines3 import PPO

# --- MODIFICAÇÃO 1: Corrigir Caminhos Relativos ---
# Obter o diretório ONDE ESTE SCRIPT ESTÁ
SCRIPT_DIR = Path(__file__).resolve().parent
# O config file está na subpasta 'configs'
PARAMETERS_FILE = SCRIPT_DIR / "configs/drl_alns_cvrp_debug.json"

# A pasta 'single_runs' deve estar no nível 'code' (três níveis acima)
# SCRIPT_DIR = .../src/routing/cvrp
# parents[0] = .../src/routing
# parents[1] = .../src
# parents[2] = .../code
DEFAULT_RESULTS_ROOT = SCRIPT_DIR.parents[2] / "single_runs/"
# ---------------------------------------------------

# Importa o ambiente (isto agora funciona porque o corremos a partir de 'src')
from rl.environments.cvrp_AlnsEnv_LSA1 import cvrpAlnsEnv_LSA1


def run_algo(folder, exp_name, client=None, **kwargs):
    instance_nr = kwargs['instance_nr']
    seed = kwargs['rseed']
    iterations = kwargs['iterations']

    # --- MODIFICAÇÃO 2: Corrigir Caminho Absoluto ---
    # O base_path é a pasta 'src'
    # SCRIPT_DIR.parents[1] aponta para .../code/src/
    base_path = SCRIPT_DIR.parents[1] 
    instance_file = str(base_path.joinpath(kwargs['instance_file']))
    
    # O model_path está relativo a 'src'
    model_path = base_path.joinpath(kwargs['model_directory']).joinpath('model')
    # --------------------------------------------------

    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)

    parameters = {'environment': {'iterations': iterations, 'instance_nr': [instance_nr], 'instance_file': instance_file}}
    env = cvrpAlnsEnv_LSA1(parameters)

    # --- MODIFICAÇÃO 3: Medir o Tempo de Execução ---
    print(f"Running inference on instance {instance_nr}...")
    start_time = time.time()
    env.run(model)
    elapsed_time = time.time() - start_time
    # ----------------------------------------------

    best_objective = env.best_solution.objective()
    print(f"Best objective found: {best_objective}")
    print(f"Execution time: {elapsed_time:.4f} seconds")

    Path(folder).mkdir(parents=True, exist_ok=True)
    csv_path = Path(folder) / (exp_name + ".csv")
    
    print(f"Writing results to: {csv_path}")
    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        
        # --- MODIFICAÇÃO 4: Adicionar tempo ao CSV ---
        writer.writerow(['problem_instance', 'rseed', 'iterations', 'solution', 'best_objective', 'execution_time', 'instance_file'])
        writer.writerow([instance_nr, seed, iterations, env.best_solution.routes, best_objective, elapsed_time, kwargs['instance_file']])
        # -------------------------------------------

    return [], best_objective


def main(param_file=PARAMETERS_FILE):
    # --- MODIFICAÇÃO 5: Usar o caminho absoluto para o JSON ---
    abs_param_file = param_file.resolve()
    parameters = helper_functions.readJSONFile(abs_param_file)
    # --------------------------------------------------------

    folder = DEFAULT_RESULTS_ROOT
    exp_name = 'drl_alns_' + str(parameters["instance_nr"]) + "_" + str(parameters["rseed"])

    best_objective = run_algo(folder, exp_name, **parameters)
    return best_objective


if __name__ == "__main__":
    main()