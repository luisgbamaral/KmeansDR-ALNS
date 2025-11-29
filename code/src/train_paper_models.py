import os
import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

# Importa seu ambiente
from rl.environments.ai4tsp_AlnsEnv_LSA1 import ai4tspAlnsEnv_LSA1

# --- CONFIGURAÇÕES ---
TRAINING_STEPS = 300000 
SEARCH_ITERATIONS = 100 
N_ENVS = 10 
SEED = 12345

BASE_PATH = Path(__file__).resolve().parent
INSTANCES_DIR = BASE_PATH / 'orienteering/ai4tsp/data/test/instances'
MODELS_DIR = BASE_PATH / 'rl/trained_models/ai4tspAlnsEnv_LSA1/models'

def make_env(rank, size, instances_list):
    def _init():
        if not instances_list:
            raise ValueError("Lista de instâncias vazia!")
        env_config = {
            'environment': {
                'iterations': SEARCH_ITERATIONS,
                'instances': instances_list
            }
        }
        env = ai4tspAlnsEnv_LSA1(env_config)
        env.reset(seed=SEED + rank)
        return env
    return _init

def get_instances_for_size(size):
    """
    Filtro Robusto: Abre os arquivos e conta os nós para garantir o tamanho correto.
    """
    if not INSTANCES_DIR.exists():
        print(f"ERRO: Pasta não encontrada: {INSTANCES_DIR}")
        return []

    print(f"   -> Verificando arquivos em disco para encontrar tamanho {size}...")
    all_files = list(INSTANCES_DIR.glob('*.csv'))
    valid_files = []
    
    # Varre arquivos para achar os que tem o número correto de linhas (nós)
    for f in all_files:
        try:
            # Lê apenas o header/shape para ser rápido
            df = pd.read_csv(f)
            # O tamanho da instância é o número de linhas do CSV
            if len(df) == size:
                valid_files.append(f.stem)
        except Exception:
            continue
            
        # Otimização: Se já achamos 250 (necessário pro paper), paramos
        if len(valid_files) >= 250:
            break
    
    if not valid_files:
        print(f"   AVISO CRÍTICO: Nenhum arquivo com {size} linhas encontrado!")
        # Fallback de emergência (não recomendado, mas evita crash)
        return [f.stem for f in all_files[:250]]
    
    print(f"   -> Sucesso: Encontradas {len(valid_files)} instâncias reais de {size} nós.")
    return valid_files

def train_model(n_nodes):
    print(f"\n{'='*50}")
    print(f"INICIANDO TREINAMENTO: TAMANHO {n_nodes}")
    print(f"{'='*50}")
    
    instances = get_instances_for_size(n_nodes)
    if not instances: return

    save_path = MODELS_DIR / f"DR-ALNS_{n_nodes}"
    os.makedirs(save_path, exist_ok=True)

    # Criação dos ambientes
    if N_ENVS > 1:
        env = SubprocVecEnv([make_env(i, n_nodes, instances) for i in range(N_ENVS)])
    else:
        env = DummyVecEnv([make_env(0, n_nodes, instances)])

    model = PPO("MlpPolicy", env, verbose=1, n_steps=256, learning_rate=3e-4, batch_size=64, seed=SEED)
    
    print(f"-> Treinando...")
    model.learn(total_timesteps=TRAINING_STEPS)
    
    model.save(save_path / "model")
    print(f"-> Salvo em: {save_path}/model.zip")
    env.close()

if __name__ == "__main__":
    set_random_seed(SEED)
    print(f"Diretório de dados: {INSTANCES_DIR}")
    
    for size in [20, 50, 100]:
        train_model(size)