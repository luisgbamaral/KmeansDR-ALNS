import csv
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm  # Barra de progresso: pip install tqdm

# Importações locais (assumindo execução a partir de 'src')
import helper_functions
from stable_baselines3 import PPO
from rl.environments.cvrp_AlnsEnv_LSA1 import cvrpAlnsEnv_LSA1

# --- CONFIGURAÇÃO DE CAMINHOS E CONSTANTES ---

# Diretório onde este script está (src/routing/cvrp)
SCRIPT_DIR = Path(__file__).resolve().parent

# Diretório base 'src' (dois níveis acima)
BASE_PATH = SCRIPT_DIR.parents[1]

# Onde salvar os resultados (na raiz do projeto, fora de src)
RESULTS_ROOT = SCRIPT_DIR.parents[2] / "batch_runs/"

# Arquivo de configuração base
PARAMETERS_FILE = SCRIPT_DIR / "configs/drl_alns_cvrp_debug.json"

# --- CONFIGURAÇÃO DA BATERIA DE TESTES ---
NUM_INSTANCES = 5000        # Quantas instâncias rodar (0 a 4999)
NUM_CUSTOMERS = 100         # Tamanho do problema (para localizar o arquivo de dados)
ITERATION_SETS = [1000, 10000]  # Os dois cenários de iterações

def get_dataset_path(base_path, num_customers):
    """
    Retorna o caminho do arquivo .pkl que contém TODAS as instâncias.
    Padrão identificado: routing/cvrp/data/cvrp_{N}_10000.pkl
    """
    # Caminho relativo a partir da pasta 'src'
    data_path_relative = f"routing/cvrp/data/cvrp_{num_customers}_10000.pkl"
    full_path = base_path.joinpath(data_path_relative)
    return str(full_path)

def run_batch(model, iteration_count, num_instances, base_path, csv_writer, base_parameters):
    """
    Executa o loop principal de inferência para um conjunto de iterações.
    """
    objectives = []
    times = []

    # Define o arquivo de dados único para este tamanho de problema
    instance_file_path = get_dataset_path(base_path, NUM_CUSTOMERS)
    
    print(f"\n>>> Iniciando bateria: {iteration_count} iterações | {num_instances} instâncias.")
    print(f">>> Lendo dados de: {instance_file_path}")

    # tqdm cria uma barra de progresso visual no terminal
    for instance_nr in tqdm(range(num_instances), desc=f"Iter {iteration_count}"):
        
        # Preparar parâmetros para esta execução específica
        env_params = base_parameters.copy()
        
        # AQUI ESTÁ O TRUQUE:
        # 1. Sobrescrevemos as iterações
        # 2. Apontamos para o índice específico (instance_nr) dentro do arquivozão
        # 3. Apontamos para o arquivo de dados correto
        env_params['environment']['iterations'] = iteration_count
        env_params['environment']['instance_nr'] = [instance_nr]
        env_params['environment']['instance_file'] = instance_file_path

        try:
            # Inicializa o ambiente (K-means será calculado aqui para esta instância)
            env = cvrpAlnsEnv_LSA1(env_params)
            
            # Medição de tempo e execução
            start_time = time.time()
            env.run(model)  # Roda o ALNS guiado pelo PPO
            elapsed_time = time.time() - start_time

            # Coleta do melhor resultado
            best_obj = env.best_solution.objective()
            
            # Armazena estatísticas para média
            objectives.append(best_obj)
            times.append(elapsed_time)

            # Escreve resultado individual no CSV detalhado (backup)
            csv_writer.writerow([
                instance_nr, 
                iteration_count, 
                best_obj, 
                f"{elapsed_time:.4f}", 
                instance_file_path
            ])
            
        except FileNotFoundError:
            print(f"\n[Erro Crítico] Arquivo de dados não encontrado: {instance_file_path}")
            break # Se não achou o arquivo principal, não adianta continuar
        except Exception as e:
            print(f"\n[Aviso] Falha na instância {instance_nr}: {e}")
            # Continua para a próxima instância caso uma específica esteja corrompida
            continue

    return objectives, times

def main():
    # 1. Carregar Configurações do JSON
    abs_param_file = PARAMETERS_FILE.resolve()
    if not abs_param_file.exists():
        raise FileNotFoundError(f"Arquivo de config não encontrado: {abs_param_file}")
        
    json_parameters = helper_functions.readJSONFile(abs_param_file)
    
    # Estrutura base que o ambiente espera ({'environment': {params}})
    base_env_parameters = {'environment': json_parameters}

    # 2. Carregar Modelo PPO (Apenas uma vez para economizar tempo)
    model_dir_relative = json_parameters.get('model_directory')
    if not model_dir_relative:
        raise ValueError("O JSON deve conter o campo 'model_directory'.")

    model_path = BASE_PATH.joinpath(model_dir_relative).joinpath('model')
    print(f"Carregando modelo de: {model_path}")
    
    # Tratamento para extensão .zip (padrão do Stable Baselines 3)
    if not model_path.exists():
        model_path_zip = model_path.with_suffix('.zip')
        if model_path_zip.exists():
            model_path = model_path_zip
        else:
            raise FileNotFoundError(f"Modelo PPO não encontrado em: {model_path}")

    # Carrega o modelo na memória/GPU
    model = PPO.load(model_path)

    # 3. Preparar Arquivos de Saída
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    filename_base = f"results_{NUM_CUSTOMERS}cust_{timestamp}"
    detailed_csv = RESULTS_ROOT / f"detailed_{filename_base}.csv"
    summary_csv = RESULTS_ROOT / f"summary_{filename_base}.csv"

    print(f"Salvando resultados em: {RESULTS_ROOT}")

    # Abre o arquivo CSV para escrita detalhada
    with open(detailed_csv, "w", newline='') as f_detailed:
        writer_detailed = csv.writer(f_detailed)
        # Cabeçalho do CSV detalhado
        writer_detailed.writerow(['instance_nr', 'iterations', 'best_objective', 'execution_time_s', 'dataset_file'])

        summary_data = []

        # 4. Loop Principal (Executa para 1000 e depois para 10000 iterações)
        for n_iter in ITERATION_SETS:
            objs, exec_times = run_batch(
                model, 
                n_iter, 
                NUM_INSTANCES, 
                BASE_PATH, 
                writer_detailed, 
                base_env_parameters
            )
            
            if objs:
                # Cálculos estatísticos
                avg_obj = np.mean(objs)
                avg_time = np.mean(exec_times)
                min_obj = np.min(objs)
                max_obj = np.max(objs)
                
                print(f"\n--- Resumo Final ({n_iter} iterações) ---")
                print(f"Média Objetivo: {avg_obj:.2f}")
                print(f"Média Tempo:    {avg_time:.4f}s")
                
                summary_data.append([n_iter, NUM_INSTANCES, avg_obj, avg_time, min_obj, max_obj])
            else:
                print(f"\n[Aviso] Nenhum resultado coletado para {n_iter} iterações.")

    # 5. Salvar CSV de Resumo (Com as médias para a Tabela 6)
    if summary_data:
        with open(summary_csv, "w", newline='') as f_summary:
            writer_summary = csv.writer(f_summary)
            writer_summary.writerow(['iterations', 'num_instances', 'avg_objective', 'avg_time_s', 'min_obj', 'max_obj'])
            writer_summary.writerows(summary_data)
        print(f"\nArquivo de médias salvo em: {summary_csv}")

    print("\nExperimento finalizado com sucesso.")

if __name__ == "__main__":
    main()