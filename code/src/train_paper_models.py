import os
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

# Importa o ambiente modificado que está em rl/environments/
from rl.environments.ai4tsp_AlnsEnv_LSA1 import ai4tspAlnsEnv_LSA1

# --- CONFIGURAÇÕES DO PAPER ---
# "The training process involved 300,000 steps with 100 search iterations."
TRAINING_STEPS = 300000 
SEARCH_ITERATIONS = 100 
N_ENVS = 10  # "ten parallel environments"
SEED = 12345

# Caminhos baseados na estrutura code/src/
BASE_PATH = Path(__file__).resolve().parent
INSTANCES_DIR = BASE_PATH / 'routing/orienteering/ai4tsp/data/test/instances'
# Onde os modelos serão salvos
MODELS_DIR = BASE_PATH / 'rl/trained_models/ai4tspAlnsEnv_LSA1/models'

def make_env(rank, size, instances_list):
    """Cria ambientes isolados para treino paralelo"""
    def _init():
        env_config = {
            'environment': {
                'iterations': SEARCH_ITERATIONS,
                'instances': instances_list
            }
        }
        env = ai4tspAlnsEnv_LSA1(env_config)
        # Garante seeds diferentes para diversidade
        env.reset(seed=SEED + rank)
        return env
    return _init

def get_instances_for_size(size):
    """Filtra arquivos de treino por tamanho (ex: instance_20_...)"""
    all_files = sorted([f.stem for f in INSTANCES_DIR.glob('*.csv')])
    filtered = [f for f in all_files if f'_{size}_' in f]
    
    if not filtered:
        print(f"AVISO: Não achei padrão '_{size}_'. Usando todas as {len(all_files)} instâncias.")
        return all_files
    
    # O paper usou 250 instâncias para treino. Limitamos para garantir consistência.
    return filtered[:250]

def train_model(n_nodes):
    print(f"\n{'='*50}")
    print(f"INICIANDO TREINAMENTO: TAMANHO {n_nodes}")
    print(f"{'='*50}")
    
    save_dir = MODELS_DIR / f"DR-ALNS_{n_nodes}"
    os.makedirs(save_dir, exist_ok=True)
    
    instances = get_instances_for_size(n_nodes)
    print(f"-> Carregadas {len(instances)} instâncias para treino.")

    # Criação dos ambientes (Multiprocessing se N_ENVS > 1)
    if N_ENVS > 1:
        env = SubprocVecEnv([make_env(i, n_nodes, instances) for i in range(N_ENVS)])
    else:
        env = DummyVecEnv([make_env(0, n_nodes, instances)])

    # Configuração do PPO (Padrão MlpPolicy é 2x64, similar ao paper)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=256,
        learning_rate=3e-4,
        batch_size=64,
        seed=SEED,
        device="auto"
    )
    
    print(f"-> Treinando por {TRAINING_STEPS} timesteps...")
    model.learn(total_timesteps=TRAINING_STEPS)
    
    # Salva o modelo final
    model_path = save_dir / "model"
    model.save(model_path)
    print(f"-> Modelo salvo com sucesso em: {model_path}.zip")
    
    env.close()

if __name__ == "__main__":
    set_random_seed(SEED)
    
    # Treina sequencialmente os 3 tamanhos
    for size in [20, 50, 100]:
        train_model(size)
        
    print("\n>>> TREINAMENTO FINALIZADO! <<<")
    print("Agora execute: python evaluate_paper_results.py")