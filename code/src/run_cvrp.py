from rl.baselines import get_parameters, Trainer
from rl.environments.cvrp_AlnsEnv_LSA1 import cvrpAlnsEnv_LSA1

if __name__ == "__main__":
    # --- MODIFICAÇÃO 9: Importações necessárias no __main__ ---

    # --------------------------------------------------------

    env = cvrpAlnsEnv_LSA1(get_parameters("cvrpAlnsEnv_LSA1"))
    # print("Sampling random actions...")
    # env.sample()

    print("Start training")
    model = Trainer("cvrpAlnsEnv_LSA1", "models").create_model()
    # model._tensorboard()
    model.train()
    print("Training done")
    input("Run trained model (Enter)")
    env.run(model)
