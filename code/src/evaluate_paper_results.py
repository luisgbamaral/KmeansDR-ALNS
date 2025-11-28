import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from stable_baselines3 import PPO
from tqdm import tqdm

# --- AJUSTE 1: Importação direta (sem 'src.') ---
# Como o arquivo está em code/src/, importamos direto de rl.environments
from rl.environments.ai4tsp_AlnsEnv_LSA1 import ai4tspAlnsEnv_LSA1

# --- CONFIGURAÇÕES DO PAPER ---
# --- AJUSTE 2: Caminhos relativos a partir de 'src' ---
EXPERIMENT_CONFIGS = {
    20: {'iterations': 100, 'model_path': 'rl/trained_models/ai4tspAlnsEnv_LSA1/models/DR-ALNS_20/model'},
    50: {'iterations': 100, 'model_path': 'rl/trained_models/ai4tspAlnsEnv_LSA1/models/DR-ALNS_50/model'},
    100: {'iterations': 200, 'model_path': 'rl/trained_models/ai4tspAlnsEnv_LSA1/models/DR-ALNS_100/model'}
}

# --- AJUSTE 3: Caminho base é o diretório atual ---
BASE_PATH = Path(__file__).resolve().parent 
INSTANCES_DIR = BASE_PATH / 'routing/orienteering/ai4tsp/data/test/instances'

def get_test_instances(size):
    """Filtra as instâncias pelo tamanho"""
    all_files = sorted([f.stem for f in INSTANCES_DIR.glob('*.csv')])
    
    # Tenta filtrar pelo padrão de nome (ex: instance_20_...)
    filtered = [f for f in all_files if f'_{size}_' in f]
    
    if not filtered:
        print(f"Aviso: Não achei padrão de nome para tamanho {size}. Usando as primeiras 250.")
        return all_files[:250]
        
    return filtered[:250]

def evaluate_size(n_nodes, config):
    print(f"\n--- Avaliando Tamanho: {n_nodes} (Iterações: {config['iterations']}) ---")
    
    model_path = config['model_path']
    if not os.path.exists(model_path + ".zip"):
        print(f"Erro: Modelo não encontrado em {model_path}. Verifique se treinou este tamanho.")
        return None
        
    model = PPO.load(model_path)
    
    instances = get_test_instances(n_nodes)
    env_config = {
        'environment': {
            'iterations': config['iterations'],
            'instances': instances
        }
    }
    
    # Instancia o ambiente (já com Cluster/Elbow automático no reset)
    env = ai4tspAlnsEnv_LSA1(env_config)
    
    results = []
    
    for instance_name in tqdm(instances, desc=f"Simulando {n_nodes} nós"):
        env.instances = [instance_name] 
        state, _ = env.reset()
        
        done = False
        while not done:
            action, _ = model.predict(state, deterministic=True)
            state, reward, done, _, _ = env.step(action)
            
        best_score = env.best_solution.objective()
        real_score = abs(best_score) 
        
        results.append({
            'instance': instance_name,
            'size': n_nodes,
            'best_score': real_score,
            'iterations': env.iteration
        })
        
    return pd.DataFrame(results)

def plot_results(all_results_df):
    if all_results_df.empty: return

    summary = all_results_df.groupby('size')['best_score'].agg(['mean', 'std']).reset_index()
    
    print("\n=== RESUMO DOS RESULTADOS (Tabela 3 Replica) ===")
    print(summary)
    
    all_results_df.to_csv("replication_results_full.csv", index=False)
    summary.to_csv("replication_results_summary.csv", index=False)

    plt.figure(figsize=(10, 6))
    plt.bar(summary['size'].astype(str), summary['mean'], yerr=summary['std'], capsize=5, color='skyblue', alpha=0.7)
    plt.xlabel('Tamanho da Instância (Nós)')
    plt.ylabel('Média do Melhor Score (Prêmio)')
    plt.title('Performance do DR-ALNS com Clustering')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, v in enumerate(summary['mean']):
        plt.text(i, v + 0.5, f"{v:.2f}", ha='center', fontweight='bold')
        
    plt.savefig("replication_performance_plot.png")
    print("\nGráfico salvo em 'replication_performance_plot.png'")

if __name__ == "__main__":
    all_data = []
    # Avalia os 3 tamanhos sequencialmente
    for size in [20, 50, 100]:
        df = evaluate_size(size, EXPERIMENT_CONFIGS[size])
        if df is not None:
            all_data.append(df)
            
    if all_data:
        full_df = pd.concat(all_data)
        plot_results(full_df)