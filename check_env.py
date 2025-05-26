from EldenEnv import EldenEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import os
import tensorflow as tf  # parry_detector 호환을 위해 필요

logdir = r"C:\GitHub\EldenRingAI\log"
model_path = "ppo_episode_model"
env = EldenEnv(logdir)
max_models_to_keep = 10  # 최신 모델 10개 유지

# ✔️ 콜백: 매 에피소드마다 저장, 최근 10개만 유지
class SaveRecentEpisodesCallback(BaseCallback):
    def __init__(self, save_path, max_keep=10, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path
        self.max_keep = max_keep
        self.episode_num = 0
        self.saved_eps = []

    def _on_step(self) -> bool:
        if self.locals.get("dones", [False])[0]:
            self.episode_num += 1
            path = f"{self.save_path}_ep{self.episode_num}.zip"
            self.model.save(path)
            self.saved_eps.append(self.episode_num)
            if self.verbose > 0:
                print(f"[💾] 에피소드 {self.episode_num} 종료 후 모델 저장됨: {path}")

            # 최대 모델 개수 초과 시 삭제
            while len(self.saved_eps) > self.max_keep:
                old_ep = self.saved_eps.pop(0)
                old_path = f"{self.save_path}_ep{old_ep}.zip"
                if os.path.exists(old_path):
                    os.remove(old_path)
                    if self.verbose > 0:
                        print(f"[🗑️] 오래된 모델 삭제됨: {old_path}")
        return True

# ✔️ 모델 불러오기 또는 새로 만들기
if os.path.exists(model_path + "_ep1.zip"):
    latest_ep = 1
    while os.path.exists(f"{model_path}_ep{latest_ep + 1}.zip"):
        latest_ep += 1
    latest_model_path = f"{model_path}_ep{latest_ep}.zip"
    print(f"[✔] 에피소드 {latest_ep} 모델 불러오는 중...")
    model = PPO.load(latest_model_path, env=env)

    # 🔁 마지막 모델을 ep1로 덮어쓰기
    ep1_model_path = f"{model_path}_ep1.zip"
    model.save(ep1_model_path)
    print(f"[🔁] {latest_model_path} → {ep1_model_path} 로 초기화 저장 완료")

    # 🗑️ ep1 제외하고 모두 삭제
    for ep in range(2, latest_ep + 1):
        old_model_path = f"{model_path}_ep{ep}.zip"
        if os.path.exists(old_model_path):
            os.remove(old_model_path)
            print(f"[🗑️] 삭제됨: {old_model_path}")
else:
    print("[🆕] 새 PPO 모델 생성")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_logs")

# ✔️ 학습 시작
callback = SaveRecentEpisodesCallback(save_path=model_path, max_keep=max_models_to_keep)
model.learn(total_timesteps=100_000, callback=callback)

print("🎉 학습 완료!")