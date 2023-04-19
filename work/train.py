from stable_baselines3 import PPO
from utils import TensorboardCallback, create_env

save_interval = 50_000
save_path = "./models/new_reward_11"
log_dir = "./metrics/"
log_name = "new_reward_11"
maps = list(range(1, 250))

env = create_env(maps=maps)

model = PPO(
    "MultiInputPolicy",
    env,
    verbose=2,
    # n_steps=2048,
    # ent_coef=0.001,
    learning_rate=0.0001,
    # vf_coef=0.5,
    # max_grad_norm=0.5,
    # gae_lambda=0.95,
    # gamma=0.99,
    # n_epochs=10,
    # clip_range=0.2,  # Adjust this value as needed
    # use_sde=True,
    tensorboard_log=log_dir,
)

# model_path = "models/ppo_model_7000000.zip"
# model = PPO.load(model_path, env, tensorboard_log=log_dir)

combined_callback = TensorboardCallback(save_interval, save_path, verbose=1)
model.learn(
    total_timesteps=10000_000,
    callback=combined_callback,
    progress_bar=True,
    tb_log_name=log_name,
)

env.close()
