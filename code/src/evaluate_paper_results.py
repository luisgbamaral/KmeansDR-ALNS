import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from stable_baselines3 import PPO
from tqdm import tqdm

# Importação direta (pois o script está na pasta src)
from rl.environments.ai4tsp_AlnsEnv_LSA1 import ai4tspAlnsEnv_LSA1

# --- CONFIGURAÇÕES DE TESTE (PAPER) ---
# Iterações aumentam para 200 no caso de 100 nós
EXPERIMENT_CONFIGS = {
    20:  {'iterations': 100, 'model_path': 'rl/trained_models/ai4tspAlnsEnv_LSA1/models/DR-ALNS_20/model'},
    50:  {'iterations': 100, 'model_path': 'rl/trained_models/ai4tspAlnsEnv_LSA1/models/DR-ALNS_50/model'},
    100: {'iterations': 200, 'model_path': 'rl/trained_models/ai4tspAlnsEnv_LSA1/models/DR-ALNS_100/model'}
}

BASE_PATH = Path(__file__).resolve().parent
INSTANCES_DIR = BASE_PATH / 'routing/orienteering/ai4tsp/data/test/instances'

def get_test_instances(size):
    """Mesma lógica do treino, mas agora para teste"""
    all_files = sorted([f.stem for f in INSTANCES_DIR.glob('*.csv')])
    filtered = [f for f in all_files if f'_{size}_' in f]
    
    # Fallback se nomes não tiverem padrão
    if not filtered:
        return all_files[:250]
    return filtered[:250]

def evaluate_size(n_nodes, config):
    print(f"\n>>> Avaliando {n_nodes} Nós ({config['iterations']} iterações) <<<")
    
    model_path = config['model_path']
    if not os.path.exists(model_path + ".zip"):
        print(f"ERRO CRÍTICO: Modelo não encontrado em {model_path}")
        print("Certifique-se de ter rodado 'train_paper_models.py' primeiro.")
        return None
        
    # Carrega modelo treinado
    model = PPO.load(model_path)
    instances = get_test_instances(n_nodes)
    
    # Configura ambiente
    env_config = {
        'environment': {
            'iterations': config['iterations'],
            'instances': instances
        }
    }
    env = ai4tspAlnsEnv_LSA1(env_config)
    
    results = []
    
    for instance_name in tqdm(instances, desc=f"Simulando {n_nodes}"):
        env.instances = [instance_name]
        
        # O reset recalcula o K-Elbow para esta instância específica
        state, _ = env.reset()
        
        done = False
        while not done:
            # Deterministic=True é padrão para avaliação final
            action, _ = model.predict(state, deterministic=True)
            state, _, done, _, _ = env.step(action)
            
        # O ambiente retorna reward negativo (custo).
        # Convertemos para positivo (Prêmio Coletado) para o gráfico.
        final_score = abs(env.best_solution.objective())
        
        results.append({
            'size': n_nodes,
            'instance': instance_name,
            'score': final_score
        })
        
    return pd.DataFrame(results)

def plot_and_save(all_results_df):
    if all_results_df.empty: return

    # Resumo estatístico
    summary = all_results_df.groupby('size')['score'].agg(['mean', 'std', 'max']).reset_index()
    print("\n=== RESULTADOS FINAIS DA REPLICAÇÃO ===")
    print(summary)
    
    # Salvar dados
    all_results_df.to_csv("results_full_clustering.csv", index=False)
    summary.to_csv("results_summary_clustering.csv", index=False)

    # Plotagem
    sizes = summary['size'].astype(str)
    means = summary['mean']
    stds = summary['std']
    
    plt.figure(figsize=(10, 6))
    
    # Barras principais
    bars = plt.bar(sizes, means, yerr=stds, capsize=10, color='#3498db', alpha=0.9, label='Nossa Abordagem (Cluster)')
    
    # Linhas de referência do Paper (Valores aproximados da Tabela 3)
    paper_means = {'20': 5.63, '50': 8.44, '100': 11.75}
    
    for i, size in enumerate(sizes):
        val = means[i]
        plt.text(i, val + 0.3, f"{val:.2f}", ha='center', color='black', fontweight='bold')
        
        # Adiciona linha vermelha do paper para comparação
        ref = paper_means.get(size, 0)
        plt.hlines(ref, i-0.4, i+0.4, colors='red', linestyles='dashed', linewidth=2)
        plt.text(i, ref + 0.5, f"Paper: {ref}", ha='center', color='red', fontsize=9, fontweight='bold')

    plt.xlabel('Tamanho da Instância (Nós)')
    plt.ylabel('Prêmio Total Coletado')
    plt.title('Comparação: DR-ALNS Original vs. DR-ALNS com Clustering')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.savefig("clustering_performance_plot.png")
    print("\nGráfico salvo em: clustering_performance_plot.png")

if __name__ == "__main__":
    dfs = []
    # Avalia na ordem
    for size in [20, 50, 100]:
        df = evaluate_size(size, EXPERIMENT_CONFIGS[size])
        if df is not None: dfs.append(df)
    
    if dfs:
        full_df = pd.concat(dfs)
        plot_and_save(full_df)