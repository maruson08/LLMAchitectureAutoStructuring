# # # import torch
# # # import torch.nn as nn
# # # import torch.nn.functional as F
# # # from torch.distributions import Categorical
# # # import torch.optim as optim
# # # import random
# # # import copy

# # # # ===== 정책/가치 신경망 (Policy Network) =====
# # # class PolicyNetwork(nn.Module):
# # #     def __init__(self, input_dim, hidden_dim, output_dim):
# # #         super(PolicyNetwork, self).__init__()
# # #         self.fc = nn.Linear(input_dim, hidden_dim)
# # #         self.action_head = nn.Linear(hidden_dim, output_dim)
# # #         self.value_head = nn.Linear(hidden_dim, 1)

# # #     def forward(self, x):
# # #         x = F.relu(self.fc(x))
# # #         action_probs = F.softmax(self.action_head(x), dim=-1)
# # #         state_value = self.value_head(x)
# # #         return action_probs, state_value


# # # # ===== NAS 서치 (간단 예시: 랜덤 탐색) =====
# # # def random_nas_search(search_space, num_samples):
# # #     sampled_archs = []
# # #     for _ in range(num_samples):
# # #         arch = {k: random.choice(v) for k, v in search_space.items()}
# # #         sampled_archs.append(arch)
# # #     return sampled_archs


# # # # ===== 모델 압축 (간단 예시: 가중치 프루닝) =====
# # # def model_pruning(model, prune_ratio=0.2):
# # #     with torch.no_grad():
# # #         for name, param in model.named_parameters():
# # #             if 'weight' in name:
# # #                 threshold = torch.quantile(param.abs(), prune_ratio)
# # #                 mask = (param.abs() > threshold).float()
# # #                 param.mul_(mask)
# # #     return model


# # # # ===== 평가 함수 (간단 정확도 평가 예시) =====
# # # def evaluate_model(model, data_loader, device):
# # #     model.eval()
# # #     correct = 0
# # #     total = 0
# # #     with torch.no_grad():
# # #         for inputs, targets in data_loader:
# # #             inputs = inputs.to(device)
# # #             targets = targets.to(device)
# # #             action_probs, _ = model(inputs)  # 여기서 두 값 분리
# # #             _, predicted = torch.max(action_probs, 1)
# # #             correct += (predicted == targets).sum().item()
# # #             total += targets.size(0)
# # #     model.train()
# # #     return correct / total if total > 0 else 0


# # # # ===== PPO 에이전트 =====
# # # class PPOAgent:
# # #     def __init__(self, input_dim, hidden_dim, action_dim, device):
# # #         self.device = device
# # #         self.policy = PolicyNetwork(input_dim, hidden_dim, action_dim).to(device)
# # #         self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
# # #         self.eps_clip = 0.2
# # #         self.gamma = 0.99

# # #     def select_action(self, state):
# # #         if not isinstance(state, torch.Tensor):
# # #             state = torch.FloatTensor(state)
# # #         state = state.unsqueeze(0).to(self.device)
# # #         probs, value = self.policy(state)
# # #         dist = Categorical(probs)
# # #         action = dist.sample()
# # #         return action.item(), dist.log_prob(action), value

# # #     def compute_returns(self, rewards, masks, values, next_value):
# # #         R = next_value
# # #         returns = []
# # #         for step in reversed(range(len(rewards))):
# # #             R = rewards[step] + self.gamma * R * masks[step]
# # #             returns.insert(0, R)
# # #         return returns

# # #     def update(self, memory):
# # #         states = torch.stack(memory.states).to(self.device)
# # #         actions = torch.tensor(memory.actions).to(self.device)
# # #         old_log_probs = torch.stack(memory.log_probs).to(self.device)
# # #         returns = torch.tensor(memory.returns).to(self.device)
# # #         values = torch.stack(memory.values).to(self.device).squeeze()

# # #         probs, state_values = self.policy(states)
# # #         dist = Categorical(probs)
# # #         log_probs = dist.log_prob(actions)
# # #         entropy = dist.entropy().mean()

# # #         ratios = torch.exp(log_probs - old_log_probs.detach())
# # #         advantages = returns - values.detach()

# # #         surr1 = ratios * advantages
# # #         surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

# # #         policy_loss = -torch.min(surr1, surr2).mean()
# # #         value_loss = F.mse_loss(state_values.squeeze(), returns)
# # #         loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

# # #         self.optimizer.zero_grad()
# # #         loss.backward()
# # #         self.optimizer.step()


# # # # ===== 메모리 클래스 =====
# # # class Memory:
# # #     def __init__(self):
# # #         self.states = []
# # #         self.actions = []
# # #         self.log_probs = []
# # #         self.rewards = []
# # #         self.masks = []
# # #         self.values = []
# # #         self.returns = []

# # #     def clear(self):
# # #         self.__init__()


# # # # ===== 메인 파이프라인 함수 =====
# # # def main_pipeline():
# # #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # #     # 하이퍼파라미터 및 상태 차원 설정
# # #     input_dim = 10
# # #     hidden_dim = 64
# # #     action_dim = 5  # 예시: NAS action 후보 수

# # #     # NAS 탐색 공간 정의 예시
# # #     search_space = {
# # #         'num_layers': [1, 2, 3, 4, 5, 20, 50, 100, 200, 300, 400],
# # #         'hidden_size': [32, 64, 128, 256],
# # #         'activation': ['relu', 'tanh']
# # #     }

# # #     # NAS 탐색 (랜덤 샘플링)
# # #     candidate_archs = random_nas_search(search_space, num_samples=10)

# # #     # PPO 에이전트 초기화
# # #     agent = PPOAgent(input_dim, hidden_dim, action_dim, device)
# # #     memory = Memory()

# # #     # 가상의 데이터 로더 (평가용), 실제로는 데이터셋으로 교체 필요
# # #     class DummyDataset(torch.utils.data.Dataset):
# # #         def __init__(self, size=100):
# # #             self.size = size
# # #             self.data = torch.randn(size, input_dim)
# # #             self.labels = torch.randint(0, action_dim, (size,))
# # #         def __len__(self):
# # #             return self.size
# # #         def __getitem__(self, idx):
# # #             return self.data[idx], self.labels[idx]

# # #     dummy_loader = torch.utils.data.DataLoader(DummyDataset(), batch_size=16)

# # #     # RL 탐색 루프 예시 (단순화)
# # #     for epoch in range(5):  # 실제 연구에서는 훨씬 더 긴 학습 수행
# # #         state = torch.randn(input_dim).numpy()
# # #         memory.clear()

# # #         for step in range(20):
# # #             action, log_prob, value = agent.select_action(state)

# # #             # NAS 후보 중 선택된 아키텍처 (단순 선택 예시)
# # #             selected_arch = candidate_archs[action % len(candidate_archs)]

# # #             # 선택된 아키텍처로 임시 모델 생성 (PolicyNetwork 활용, 실제론 NAS 아키텍처 반영)
# # #             model = PolicyNetwork(input_dim, hidden_dim, action_dim).to(device)

# # #             # 모델 압축 (프루닝 적용)
# # #             model = model_pruning(model, prune_ratio=0.2)

# # #             # 평가 (정확도)
# # #             accuracy = evaluate_model(model, dummy_loader, device)

# # #             # 보상은 정확도로 간단히 설정
# # #             reward = accuracy

# # #             done = (step == 19)
# # #             mask = 0 if done else 1

# # #             # 메모리에 저장
# # #             memory.states.append(torch.FloatTensor(state))
# # #             memory.actions.append(action)
# # #             memory.log_probs.append(log_prob)
# # #             memory.rewards.append(reward)
# # #             memory.masks.append(mask)
# # #             memory.values.append(value)

# # #             # 다음 상태 생성 (임의)
# # #             state = torch.randn(input_dim).numpy()

# # #             if done:
# # #                 break

# # #         # 마지막 상태 가치 추정
# # #         next_state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
# # #         with torch.no_grad():
# # #             _, next_value = agent.policy(next_state_tensor)
# # #         next_value = next_value.detach()

# # #         # 리턴 계산 및 업데이트
# # #         memory.returns = agent.compute_returns(memory.rewards, memory.masks, memory.values, next_value)
# # #         agent.update(memory)

# # #         print(f"Epoch {epoch+1} 완료, 마지막 reward: {memory.rewards[-1]:.4f}, 평균 reward: {sum(memory.rewards)/len(memory.rewards):.4f}")

# # # if __name__ == "__main__":
# # #     main_pipeline()
# # # # import torch
# # # # import torch.nn as nn
# # # # import torch.optim as optim
# # # # import torch.nn.functional as F
# # # # from torch.distributions import Categorical
# # # # import numpy as np

# # # # # Actor-Critic 정책 네트워크
# # # # class PolicyNetwork(nn.Module):
# # # #     def __init__(self, input_dim, hidden_dim, action_dim):
# # # #         super(PolicyNetwork, self).__init__()
# # # #         self.fc1 = nn.Linear(input_dim, hidden_dim)
# # # #         self.fc_actor = nn.Linear(hidden_dim, action_dim)
# # # #         self.fc_critic = nn.Linear(hidden_dim, 1)

# # # #     def forward(self, x):
# # # #         x = F.relu(self.fc1(x))
# # # #         action_logits = self.fc_actor(x)
# # # #         state_value = self.fc_critic(x)
# # # #         return action_logits, state_value

# # # # # PPO 에이전트
# # # # class PPOAgent:
# # # #     def __init__(self, input_dim, hidden_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2):
# # # #         self.policy = PolicyNetwork(input_dim, hidden_dim, action_dim).to(device)
# # # #         self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
# # # #         self.gamma = gamma
# # # #         self.eps_clip = eps_clip

# # # #     def select_action(self, state):
# # # #         state = torch.FloatTensor(state).to(device)
# # # #         logits, _ = self.policy(state)
        
# # # #         # logits 값이 NaN인지 확인
# # # #         if torch.isnan(logits).any():
# # # #             print("Warning: NaN detected in logits during select_action")
# # # #             logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
        
# # # #         dist = Categorical(logits=logits)
# # # #         action = dist.sample()
# # # #         return action.item(), dist.log_prob(action), dist.entropy()

# # # #     def evaluate(self, states, actions):
# # # #         logits, state_values = self.policy(states)
        
# # # #         # NaN 처리
# # # #         if torch.isnan(logits).any():
# # # #             print("Warning: NaN detected in logits during evaluate")
# # # #             logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
        
# # # #         dist = Categorical(logits=logits)
# # # #         action_logprobs = dist.log_prob(actions)
# # # #         dist_entropy = dist.entropy()
# # # #         return action_logprobs, torch.squeeze(state_values), dist_entropy


# # # #     def update(self, memory):
# # # #         # memory: dict with states, actions, logprobs, rewards, dones
# # # #         states = torch.FloatTensor(memory['states']).to(device)
# # # #         actions = torch.LongTensor(memory['actions']).to(device)
# # # #         old_logprobs = torch.FloatTensor(memory['logprobs']).to(device)
# # # #         rewards = memory['rewards']
# # # #         dones = memory['dones']

# # # #         # Calculate discounted rewards
# # # #         discounted_rewards = []
# # # #         discounted_reward = 0
# # # #         for reward, done in zip(reversed(rewards), reversed(dones)):
# # # #             if done:
# # # #                 discounted_reward = 0
# # # #             discounted_reward = reward + (self.gamma * discounted_reward)
# # # #             discounted_rewards.insert(0, discounted_reward)
# # # #         discounted_rewards = torch.FloatTensor(discounted_rewards).to(device)
# # # #         discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-7)

# # # #         # PPO update
# # # #         for _ in range(4):  # PPO epoch 수
# # # #             logprobs, state_values, dist_entropy = self.evaluate(states, actions)
# # # #             ratios = torch.exp(logprobs - old_logprobs.detach())

# # # #             advantages = discounted_rewards - state_values.detach()
# # # #             surr1 = ratios * advantages
# # # #             surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

# # # #             loss = -torch.min(surr1, surr2).mean() + \
# # # #                    0.5 * F.mse_loss(state_values, discounted_rewards) - \
# # # #                    0.01 * dist_entropy.mean()

# # # #             self.optimizer.zero_grad()
# # # #             loss.backward()
# # # #             self.optimizer.step()

# # # # # 환경 시뮬레이터 예시 (더미)
# # # # class DummyEnv:
# # # #     def __init__(self, state_dim, action_dim):
# # # #         self.state_dim = state_dim
# # # #         self.action_dim = action_dim
# # # #         self.state = np.random.rand(state_dim)

# # # #     def reset(self):
# # # #         self.state = np.random.rand(self.state_dim)
# # # #         return self.state

# # # #     def step(self, action):
# # # #         reward = np.random.rand()  # 랜덤 보상 (실제 환경에 맞게 변경)
# # # #         done = np.random.rand() > 0.95
# # # #         next_state = np.random.rand(self.state_dim)
# # # #         return next_state, reward, done, {}

# # # # # 평가 함수 (정확도 또는 reward 평균 등)
# # # # def evaluate_policy(agent, env, episodes=10):
# # # #     total_reward = 0
# # # #     for _ in range(episodes):
# # # #         state = env.reset()
# # # #         done = False
# # # #         while not done:
# # # #             action, _, _ = agent.select_action(state)
# # # #             state, reward, done, _ = env.step(action)
# # # #             total_reward += reward
# # # #     return total_reward / episodes

# # # # # 메인 훈련 루프
# # # # device = 'cpu'
# # # # # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # # input_dim = 10
# # # # hidden_dim = 64
# # # # action_dim = 10

# # # # env = DummyEnv(input_dim, action_dim)
# # # # agent = PPOAgent(input_dim, hidden_dim, action_dim)

# # # # max_epochs = 100
# # # # max_timesteps = 200

# # # # for epoch in range(max_epochs):
# # # #     memory = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'dones': []}
# # # #     state = env.reset()

# # # #     for t in range(max_timesteps):
# # # #         action, logprob, _ = agent.select_action(state)
# # # #         next_state, reward, done, _ = env.step(action)

# # # #         memory['states'].append(state)
# # # #         memory['actions'].append(action)
# # # #         memory['logprobs'].append(logprob.item())
# # # #         memory['rewards'].append(reward)
# # # #         memory['dones'].append(done)

# # # #         state = next_state

# # # #         if done:
# # # #             break

# # # #     agent.update(memory)

# # # #     avg_reward = evaluate_policy(agent, env, episodes=5)
# # # #     print(f"Epoch {epoch+1} 완료, 평균 reward: {avg_reward:.4f}")

# # import numpy as np
# # import matplotlib.pyplot as plt
# # import pandas as pd

# # # 데이터, 환경, 용도 프로파일 클래스
# # class DataProfile:
# #     def __init__(self, data_size, diversity, complexity, noise_level):
# #         self.data_size = data_size            # 토큰 수
# #         self.diversity = diversity            # 데이터 다양성 (0~1)
# #         self.complexity = complexity          # 데이터 복잡성 (0~1)
# #         self.noise_level = noise_level        # 잡음 수준 (0~1)

# # class EnvironmentProfile:
# #     def __init__(self, gpu_count, gpu_mem_gb, cpu_cores, max_training_time_hours, max_model_size_gb):
# #         self.gpu_count = gpu_count
# #         self.gpu_mem_gb = gpu_mem_gb
# #         self.cpu_cores = cpu_cores
# #         self.max_training_time_hours = max_training_time_hours
# #         self.max_model_size_gb = max_model_size_gb

# # class UseCaseProfile:
# #     def __init__(self, task_type, target_accuracy, input_seq_len, output_seq_len, latency_requirement_ms):
# #         self.task_type = task_type
# #         self.target_accuracy = target_accuracy
# #         self.input_seq_len = input_seq_len
# #         self.output_seq_len = output_seq_len
# #         self.latency_requirement_ms = latency_requirement_ms

# # # 자동 LLM 설계 클래스
# # class AutoLLMDesigner:
# #     def __init__(self):
# #         pass

# #     def predict_model_size(self, n_params):
# #         # 모델 파라미터 수에 따른 대략적인 저장 공간(MB) 계산
# #         return n_params * 4 / 1e6  # float32, 4 bytes per param

# #     def predict_loss(self, n_params, data_profile):
# #         # 확장 법칙 + 데이터 잡음/복잡성 반영 손실 예측 (가상의 수식)
# #         base_loss = 1.0 / (n_params ** 0.3)
# #         noise_factor = 1 + data_profile.noise_level * 2
# #         complexity_factor = 1 + data_profile.complexity
# #         diversity_factor = 1 / (data_profile.diversity + 0.1)
# #         loss = base_loss * noise_factor * complexity_factor * diversity_factor
# #         return loss

# #     def compression_ratio_estimate(self, model_size_mb):
# #         # 간단한 모델 압축 비율 예측 (50%로 고정)
# #         return 0.5

# #     def optimize_architecture(self, env_profile, usecase_profile, data_profile):
# #         # 모델 크기 (파라미터 수) 초기 추정: 환경+용도+데이터 기반 복합 수식 (가상)
# #         base_params = 5e7  # 50M params 기본
# #         scale_factor = (
# #             (data_profile.data_size / 1e9) ** 0.5 + 
# #             usecase_profile.target_accuracy * 2 + 
# #             (env_profile.gpu_count * env_profile.gpu_mem_gb / 64)
# #         )
# #         n_params = base_params * scale_factor
# #         n_params = min(n_params, env_profile.max_model_size_gb * 250e6)  # 1GB ≈ 250M params

# #         # 손실 예측
# #         predicted_loss = self.predict_loss(n_params, data_profile)

# #         # 모델 크기 예측
# #         model_size_mb = self.predict_model_size(n_params)

# #         # 압축 적용
# #         compression_ratio = self.compression_ratio_estimate(model_size_mb)
# #         compressed_size_mb = model_size_mb * compression_ratio

# #         # 환경 제약에 맞춰 조정 (압축 후 크기)
# #         if compressed_size_mb > env_profile.max_model_size_gb * 1024:
# #             compressed_size_mb = env_profile.max_model_size_gb * 1024
# #             # 파라미터 수도 비례 감소
# #             n_params = compressed_size_mb * 1e6 / 4

# #         # latency, training time 등 복합 요소는 연구 필요, 단순화함

# #         return {
# #             "n_params": n_params,
# #             "predicted_loss": predicted_loss,
# #             "original_model_size_mb": model_size_mb,
# #             "compressed_model_size_mb": compressed_size_mb,
# #             "compression_ratio": compression_ratio
# #         }

# #     def design(self, env_profile, usecase_profile, data_profile):
# #         # 사용자 데이터 반영 예: 데이터 특성 기반 최적화
# #         optimized_result = self.optimize_architecture(env_profile, usecase_profile, data_profile)
# #         return optimized_result

# # # 실험 결과 테이블 출력
# # def print_results_table(results):
# #     rows = []
# #     for i, r in enumerate(results):
# #         dp = r["data_profile"]
# #         rows.append({
# #             "Experiment": i+1,
# #             "Data Size (tokens)": int(dp.data_size),
# #             "Diversity": round(dp.diversity, 2),
# #             "Complexity": round(dp.complexity, 2),
# #             "Noise Level": round(dp.noise_level, 2),
# #             "Params (millions)": round(r["n_params"] / 1e6, 2),
# #             "Predicted Loss": round(r["predicted_loss"], 4),
# #             "Orig Model Size (MB)": round(r["original_model_size_mb"], 2),
# #             "Compressed Size (MB)": round(r["compressed_model_size_mb"], 2),
# #             "Compression Ratio": r["compression_ratio"]
# #         })
# #     df = pd.DataFrame(rows)
# #     print(df.to_markdown(index=False))
# #     return df

# # # 추가 시각화 함수
# # def additional_plots(results):
# #     data_sizes = [r["data_profile"].data_size / 1e9 for r in results]  # 단위: 10억 토큰
# #     noise_levels = [r["data_profile"].noise_level for r in results]
# #     params = [r["n_params"] / 1e6 for r in results]  # 백만 단위
# #     losses = [r["predicted_loss"] for r in results]

# #     fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# #     # 1) 데이터 크기 대비 파라미터 및 손실
# #     ax = axs[0]
# #     ax2 = ax.twinx()
# #     ax.plot(data_sizes, params, 'o-', color='tab:green', label='Parameters (M)')
# #     ax2.plot(data_sizes, losses, 's--', color='tab:orange', label='Predicted Loss')
# #     ax.set_xlabel("Data Size (B tokens)")
# #     ax.set_ylabel("Parameters (Million)", color='tab:green')
# #     ax2.set_ylabel("Predicted Loss", color='tab:orange')
# #     ax.set_title("Parameters and Loss vs Data Size")
# #     ax.legend(loc="upper left")
# #     ax2.legend(loc="upper right")

# #     # 2) 잡음 수준 대비 파라미터 및 손실
# #     ax = axs[1]
# #     ax2 = ax.twinx()
# #     ax.plot(noise_levels, params, 'o-', color='tab:purple', label='Parameters (M)')
# #     ax2.plot(noise_levels, losses, 's--', color='tab:red', label='Predicted Loss')
# #     ax.set_xlabel("Noise Level")
# #     ax.set_ylabel("Parameters (Million)", color='tab:purple')
# #     ax2.set_ylabel("Predicted Loss", color='tab:red')
# #     ax.set_title("Parameters and Loss vs Noise Level")
# #     ax.legend(loc="upper left")
# #     ax2.legend(loc="upper right")

# #     plt.tight_layout()
# #     plt.show()

# # # 메인 실행 함수
# # def run_experiment_with_table_and_plots():
# #     env = EnvironmentProfile(gpu_count=4, gpu_mem_gb=24, cpu_cores=32,
# #                              max_training_time_hours=72, max_model_size_gb=12)
# #     usecase = UseCaseProfile(task_type="text_generation", target_accuracy=0.85,
# #                              input_seq_len=1024, output_seq_len=256,
# #                              latency_requirement_ms=100)
# #     data_samples = [
# #         DataProfile(data_size=1e9, diversity=0.9, complexity=0.2, noise_level=0.05),
# #         DataProfile(data_size=5e8, diversity=0.7, complexity=0.5, noise_level=0.1),
# #         DataProfile(data_size=2e8, diversity=0.5, complexity=0.8, noise_level=0.2),
# #         DataProfile(data_size=1e8, diversity=0.4, complexity=0.9, noise_level=0.3),
# #     ]
    
# #     designer = AutoLLMDesigner()
# #     results = []
# #     for i, dp in enumerate(data_samples):
# #         res = designer.design(env, usecase, dp)
# #         res["data_profile"] = dp
# #         results.append(res)
# #         print(f"Experiment {i+1} complete.")
    
# #     df = print_results_table(results)
# #     additional_plots(results)
# #     return df

# # if __name__ == "__main__":
# #     run_experiment_with_table_and_plots()

# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # 데이터 프로파일링 모듈
# class DataProfile:
#     def __init__(self, data_size, diversity, complexity, noise_level, domain_specificity, imbalance_ratio):
#         self.data_size = data_size  # 예: 샘플 수
#         self.diversity = diversity  # 다양성 (0~1)
#         self.complexity = complexity  # 복잡도 (0~1)
#         self.noise_level = noise_level  # 노이즈 (0~1)
#         self.domain_specificity = domain_specificity  # 도메인 특이성 (0~1)
#         self.imbalance_ratio = imbalance_ratio  # 불균형 정도 (0~1)

# # 사용자 환경 프로파일
# class EnvironmentProfile:
#     def __init__(self, gpu_count, gpu_mem_gb, cpu_cores, max_training_time_hours, max_model_size_gb):
#         self.gpu_count = gpu_count
#         self.gpu_mem_gb = gpu_mem_gb
#         self.cpu_cores = cpu_cores
#         self.max_training_time_hours = max_training_time_hours
#         self.max_model_size_gb = max_model_size_gb

# # LLM 용도별 프로파일
# class UseCaseProfile:
#     def __init__(self, task_type, target_accuracy, max_seq_length, batch_size, latency_ms):
#         self.task_type = task_type
#         self.target_accuracy = target_accuracy
#         self.max_seq_length = max_seq_length
#         self.batch_size = batch_size
#         self.latency_ms = latency_ms

# # 자동 설계 핵심 클래스
# class AutoLLMDesigner:
#     def __init__(self):
#         pass

#     # 손실 예측 (데이터 프로파일 특성 반영)
#     def predict_loss(self, n_params, data_profile: DataProfile):
#         base_loss = 1.0 / np.sqrt(n_params / 1e6)
#         # 데이터 특성 영향 반영
#         complexity_factor = (data_profile.complexity + data_profile.domain_specificity) / 2
#         noise_factor = 1 + data_profile.noise_level * 0.5
#         imbalance_factor = 1 + data_profile.imbalance_ratio * 0.3
#         adjusted_loss = base_loss * complexity_factor * noise_factor * imbalance_factor
#         return adjusted_loss

#     # 모델 크기 예측 (매개변수 수 기반)
#     def predict_model_size(self, n_params):
#         # 파라미터당 4 bytes(32bit float) 가정, MB 단위 변환
#         size_mb = n_params * 4 / (1024**2)
#         return size_mb

#     # 압축 비율 추정
#     def compression_ratio_estimate(self, model_size_mb, compression_method="fixed", precision_bits=8):
#         if compression_method == "fixed":
#             return 0.5
#         elif compression_method == "quantization":
#             return precision_bits / 32
#         elif compression_method == "pruning":
#             return 0.7
#         else:
#             return 1.0

#     # 최적 구조 설계 (환경, 용도, 데이터 프로파일 고려)
#     def optimize_architecture(self, env_profile: EnvironmentProfile, usecase_profile: UseCaseProfile, 
#                               data_profile: DataProfile, compression_method="fixed", precision_bits=8):
#         base_params = 5e7

#         # 사용자 데이터 반영 스케일링 (manual + scaling law 응용)
#         data_factor = (data_profile.data_size / 1e9) ** 0.5
#         accuracy_factor = usecase_profile.target_accuracy * 2
#         env_factor = (env_profile.gpu_count * env_profile.gpu_mem_gb) / 64
#         seq_len_factor = np.log(usecase_profile.max_seq_length + 1) / np.log(1024)

#         scale_factor = data_factor + accuracy_factor + env_factor + seq_len_factor
#         n_params = base_params * scale_factor

#         # 환경 최대 모델 크기 제한 적용
#         max_params_by_env = env_profile.max_model_size_gb * 250e6
#         n_params = min(n_params, max_params_by_env)

#         predicted_loss = self.predict_loss(n_params, data_profile)
#         model_size_mb = self.predict_model_size(n_params)

#         compression_ratio = self.compression_ratio_estimate(model_size_mb, compression_method, precision_bits)
#         compressed_size_mb = model_size_mb * compression_ratio

#         # 압축 후 최대 용량 초과 시 조정
#         if compressed_size_mb > env_profile.max_model_size_gb * 1024:
#             compressed_size_mb = env_profile.max_model_size_gb * 1024
#             n_params = compressed_size_mb * 1e6 / 4  # 4 bytes per param

#         return {
#             "n_params": n_params,
#             "predicted_loss": predicted_loss,
#             "original_model_size_mb": model_size_mb,
#             "compressed_model_size_mb": compressed_size_mb,
#             "compression_ratio": compression_ratio,
#             "compression_method": compression_method,
#             "task_type": usecase_profile.task_type,
#             "gpu_count": env_profile.gpu_count,
#             "gpu_mem_gb": env_profile.gpu_mem_gb,
#             "target_accuracy": usecase_profile.target_accuracy
#         }

# # 실험 및 시각화 함수
# def run_experiments():
#     # 환경 변수 샘플 (4가지)
#     env_profiles = [
#         EnvironmentProfile(gpu_count=1, gpu_mem_gb=8, cpu_cores=16, max_training_time_hours=48, max_model_size_gb=6),
#         EnvironmentProfile(gpu_count=2, gpu_mem_gb=16, cpu_cores=32, max_training_time_hours=72, max_model_size_gb=12),
#         EnvironmentProfile(gpu_count=4, gpu_mem_gb=32, cpu_cores=64, max_training_time_hours=96, max_model_size_gb=24),
#         EnvironmentProfile(gpu_count=8, gpu_mem_gb=48, cpu_cores=96, max_training_time_hours=120, max_model_size_gb=48)
#     ]

#     # 용도별 샘플
#     usecases = [
#         UseCaseProfile("text_generation", 0.85, 1024, 256, 100),
#         UseCaseProfile("text_classification", 0.9, 512, 64, 50),
#         UseCaseProfile("summarization", 0.8, 768, 128, 150)
#     ]

#     # 데이터 프로파일 샘플
#     data_profiles = [
#         DataProfile(data_size=5e8, diversity=0.7, complexity=0.6, noise_level=0.1, domain_specificity=0.3, imbalance_ratio=0.2),
#         DataProfile(data_size=1e9, diversity=0.8, complexity=0.7, noise_level=0.05, domain_specificity=0.5, imbalance_ratio=0.1),
#         DataProfile(data_size=2e9, diversity=0.9, complexity=0.8, noise_level=0.02, domain_specificity=0.7, imbalance_ratio=0.05)
#     ]

#     designer = AutoLLMDesigner()

#     results = []
#     for env in env_profiles:
#         for usecase in usecases:
#             for data_profile in data_profiles:
#                 for compression_method in ["fixed", "quantization", "pruning"]:
#                     res = designer.optimize_architecture(
#                         env, usecase, data_profile,
#                         compression_method=compression_method,
#                         precision_bits=8 if compression_method == "quantization" else 32
#                     )
#                     res.update({
#                         "data_size": data_profile.data_size,
#                         "noise_level": data_profile.noise_level,
#                         "diversity": data_profile.diversity,
#                         "complexity": data_profile.complexity
#                     })
#                     results.append(res)

#     df = pd.DataFrame(results)
#     return df

# def plot_results(df):
#     # GPU 개수별 평균 압축 후 모델 크기 비교
#     plt.figure(figsize=(10, 6))
#     for method in df["compression_method"].unique():
#         subset = df[df["compression_method"] == method]
#         means = subset.groupby("gpu_count")["compressed_model_size_mb"].mean()
#         plt.plot(means.index, means.values, label=f"Compression: {method}")

#     plt.title("GPU Count vs Average Compressed Model Size (MB)")
#     plt.xlabel("GPU Count")
#     plt.ylabel("Compressed Model Size (MB)")
#     plt.legend()
#     plt.grid(True)
#     plt.show()

#     # 용도별 예측 손실 비교 (압축별)
#     plt.figure(figsize=(10, 6))
#     for method in df["compression_method"].unique():
#         subset = df[df["compression_method"] == method]
#         means = subset.groupby("task_type")["predicted_loss"].mean()
#         plt.bar(means.index + f" ({method})", means.values, alpha=0.7, label=f"Compression: {method}")

#     plt.title("Task Type vs Predicted Loss by Compression Method")
#     plt.xlabel("Task Type")
#     plt.ylabel("Predicted Loss")
#     plt.legend()
#     plt.show()

#     # 데이터 복잡도에 따른 모델 크기
#     plt.figure(figsize=(10, 6))
#     plt.scatter(df["complexity"], df["n_params"]/1e6, c=df["compressed_model_size_mb"], cmap='viridis', alpha=0.7)
#     plt.colorbar(label="Compressed Model Size (MB)")
#     plt.title("Data Complexity vs Number of Parameters (Million)")
#     plt.xlabel("Data Complexity")
#     plt.ylabel("Number of Parameters (Million)")
#     plt.show()

# def plot_and_save_results(df, save_dir="results"):
#     os.makedirs(save_dir, exist_ok=True)

#     # 1. GPU 개수별 평균 압축 후 모델 크기 비교
#     plt.figure(figsize=(10, 6))
#     for method in df["compression_method"].unique():
#         subset = df[df["compression_method"] == method]
#         means = subset.groupby("gpu_count")["compressed_model_size_mb"].mean()
#         plt.plot(means.index, means.values, label=f"Compression: {method}")

#     plt.title("GPU Count vs Average Compressed Model Size (MB)")
#     plt.xlabel("GPU Count")
#     plt.ylabel("Compressed Model Size (MB)")
#     plt.legend()
#     plt.grid(True)
#     gpu_model_size_path = os.path.join(save_dir, "gpu_vs_compressed_model_size.png")
#     plt.savefig(gpu_model_size_path)
#     plt.close()

#     # 2. 용도별 예측 손실 비교 (압축별)
#     plt.figure(figsize=(10, 6))
#     bar_width = 0.25
#     x = np.arange(len(df["task_type"].unique()))
#     task_types = sorted(df["task_type"].unique())
#     for i, method in enumerate(df["compression_method"].unique()):
#         subset = df[df["compression_method"] == method]
#         means = [subset[subset["task_type"] == t]["predicted_loss"].mean() for t in task_types]
#         plt.bar(x + i * bar_width, means, width=bar_width, alpha=0.7, label=f"Compression: {method}")

#     plt.title("Task Type vs Predicted Loss by Compression Method")
#     plt.xlabel("Task Type")
#     plt.ylabel("Predicted Loss")
#     plt.xticks(x + bar_width, task_types)
#     plt.legend()
#     plt.grid(True)
#     predicted_loss_path = os.path.join(save_dir, "task_vs_predicted_loss.png")
#     plt.savefig(predicted_loss_path)
#     plt.close()

#     # 3. 데이터 복잡도에 따른 모델 크기 산점도
#     plt.figure(figsize=(10, 6))
#     scatter = plt.scatter(df["complexity"], df["n_params"] / 1e6,
#                           c=df["compressed_model_size_mb"], cmap='viridis', alpha=0.7)
#     plt.colorbar(scatter, label="Compressed Model Size (MB)")
#     plt.title("Data Complexity vs Number of Parameters (Million)")
#     plt.xlabel("Data Complexity")
#     plt.ylabel("Number of Parameters (Million)")
#     complexity_scatter_path = os.path.join(save_dir, "complexity_vs_nparams.png")
#     plt.savefig(complexity_scatter_path)
#     plt.close()

#     return [gpu_model_size_path, predicted_loss_path, complexity_scatter_path]

# # 분석 및 결과 저장 함수
# def analyze_and_save(df, save_dir="results"):
#     os.makedirs(save_dir, exist_ok=True)
#     summary_path = os.path.join(save_dir, "summary_results.csv")
#     analysis_path = os.path.join(save_dir, "detailed_analysis.txt")

#     # 1. 요약 통계 테이블 저장
#     summary = df.groupby(["gpu_count", "task_type", "compression_method"]).agg({
#         "n_params": "mean",
#         "predicted_loss": "mean",
#         "compressed_model_size_mb": "mean"
#     }).round(3).reset_index()
#     summary.to_csv(summary_path, index=False)

#     # 2. 상세 분석 텍스트 파일 작성
#     with open(analysis_path, "w") as f:
#         f.write("=== LLM Architecture Design Automated Analysis ===\n\n")
#         f.write("1. Summary Statistics\n")
#         f.write(summary.to_string(index=False))
#         f.write("\n\n")

#         f.write("2. Correlation Matrix (Numerical Columns)\n")
#         corr = df[["n_params", "predicted_loss", "compressed_model_size_mb", "data_size", "noise_level", "complexity"]].corr()
#         f.write(corr.to_string())
#         f.write("\n\n")

#         f.write("3. Compression Method Impact on Predicted Loss\n")
#         comp_loss = df.groupby("compression_method")["predicted_loss"].mean().round(4)
#         f.write(comp_loss.to_string())
#         f.write("\n\n")

#         f.write("4. GPU Count Impact on Model Size\n")
#         gpu_size = df.groupby("gpu_count")["compressed_model_size_mb"].mean().round(2)
#         f.write(gpu_size.to_string())
#         f.write("\n\n")

#         f.write("5. Task Type Impact on Predicted Loss\n")
#         task_loss = df.groupby("task_type")["predicted_loss"].mean().round(4)
#         f.write(task_loss.to_string())
#         f.write("\n\n")

#     return summary_path, analysis_path


# if __name__ == "__main__":
#     df_results = run_experiments()

#     # 결과 저장 및 분석
#     save_folder = "llm_design_results"
#     image_paths = plot_and_save_results(df_results, save_dir=save_folder)
#     summary_file, analysis_file = analyze_and_save(df_results, save_dir=save_folder)

#     print(f"결과 이미지 저장 위치: {image_paths}")
#     print(f"요약 결과 CSV 저장 위치: {summary_file}")
#     print(f"상세 분석 파일 저장 위치: {analysis_file}")

import os
import time
import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from transformers import BertConfig, BertForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch.nn.utils.prune as prune
import numpy as np

# --- 1. 데이터 프로파일링 ---

def profile_dataset(dataset, text_field='sentence'):
    lengths = [len(x[text_field].split()) for x in dataset]
    profile = {
        'num_samples': len(dataset),
        'avg_length': np.mean(lengths),
        'max_length': np.max(lengths),
        'min_length': np.min(lengths),
        'length_std': np.std(lengths),
    }
    return profile

# --- 2. 자동 LLM 설계 ---

def design_transformer(target_n_params, vocab_size=30522, 
                       min_layers=2, max_layers=24, 
                       min_hidden=128, max_hidden=1024):
    best_config = None
    min_diff = float('inf')

    for num_layers in range(min_layers, max_layers + 1):
        low, high = min_hidden, max_hidden
        while low <= high:
            hidden_dim = (low + high) // 2
            ffn_dim = 4 * hidden_dim
            embed_params = vocab_size * hidden_dim
            layer_params = num_layers * (4 * hidden_dim ** 2 + 2 * hidden_dim * ffn_dim)
            total_params = embed_params + layer_params

            diff = total_params - target_n_params
            if abs(diff) < min_diff and total_params <= target_n_params:
                min_diff = abs(diff)
                best_config = {
                    'num_layers': num_layers,
                    'hidden_dim': hidden_dim,
                    'ffn_dim': ffn_dim,
                    'total_params': total_params
                }

            if diff > 0:
                high = hidden_dim - 1
            else:
                low = hidden_dim + 1

    return best_config

def create_model_from_config(config, num_labels=2, vocab_size=30522):
    bert_config = BertConfig(
        hidden_size=config['hidden_dim'],
        num_hidden_layers=config['num_layers'],
        intermediate_size=config['ffn_dim'],
        num_attention_heads=max(1, config['hidden_dim'] // 64),
        vocab_size=vocab_size,
    )
    model = BertForSequenceClassification(bert_config, num_labels=num_labels)
    return model

# --- 3. 데이터셋 준비 및 전처리 ---

def prepare_dataset(dataset_name="glue", subset="sst2", sample_ratio=0.01, text_field='sentence'):
    dataset = load_dataset(dataset_name, subset, split=f"train[:{int(sample_ratio*100)}%]")
    return dataset

def preprocess_dataset(dataset, tokenizer, text_field='sentence', max_length=128):
    def preprocess(batch):
        return tokenizer(batch[text_field], truncation=True, padding='max_length', max_length=max_length)
    dataset = dataset.map(preprocess, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    return dataset

# --- 4. 모델 압축 (Pruning + Quantization) ---

def apply_pruning(model, amount=0.2):
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parameters_to_prune.append((module, 'weight'))
    prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount)
    return model

def apply_quantization(model):
    model_quantized = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    return model_quantized

# --- 5. 학습 및 평가 함수 ---

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
            labels = batch["label"].to(device)
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0

# --- 6. 모델 크기 측정 ---

def get_model_size(model):
    tmp_path = "temp_model.pt"
    torch.save(model.state_dict(), tmp_path)
    size_mb = os.path.getsize(tmp_path) / (1024*1024)
    os.remove(tmp_path)
    return size_mb

# --- 7. 결과 저장 및 시각화 ---

def save_results(results, filepath="experiment_results.json"):
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)

def plot_results(results, save_path="results_plot.png"):
    epochs = list(range(1, len(results['train_loss']) + 1))
    plt.figure(figsize=(10,6))
    plt.plot(epochs, results['train_loss'], label="Train Loss")
    plt.plot(epochs, results['eval_accuracy'], label="Eval Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Training Loss and Evaluation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

# --- 8. 전체 파이프라인 통합 ---

def run_experiment(target_n_params=50_000_000, sample_ratio=0.01, pruning_amount=0.2, epochs=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("1) 데이터셋 로드 및 프로파일링")
    dataset = prepare_dataset(sample_ratio=sample_ratio)
    profile = profile_dataset(dataset)
    print("데이터 프로파일링 결과:", profile)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = preprocess_dataset(dataset, tokenizer)

    split_idx = int(len(dataset) * 0.8)
    train_dataset = dataset.select(range(split_idx))
    eval_dataset = dataset.select(range(split_idx, len(dataset)))
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=8)

    print("2) 모델 자동 설계 (목표 파라미터 수 기준)")
    config = design_transformer(target_n_params)
    print("설계된 모델 구성:", config)

    model = create_model_from_config(config)
    model.to(device)

    print("3) 모델 압축 적용")
    model = apply_pruning(model, amount=pruning_amount)
    model = apply_quantization(model)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()

    train_loss_history = []
    eval_acc_history = []

    print("4) 학습 시작")
    for epoch in range(epochs):
        start_time = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        eval_acc = evaluate(model, eval_loader, device)
        elapsed = time.time() - start_time

        train_loss_history.append(train_loss)
        eval_acc_history.append(eval_acc)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Eval Acc={eval_acc:.4f}, Time={elapsed:.1f}s")

    model_size = get_model_size(model)
    print(f"최종 모델 크기: {model_size:.2f} MB")

    results = {
        'data_profile': profile,
        'model_config': config,
        'train_loss': train_loss_history,
        'eval_accuracy': eval_acc_history,
        'model_size_MB': model_size,
        'target_n_params': target_n_params,
        'pruning_amount': pruning_amount,
        'epochs': epochs,
    }

    print("5) 결과 저장 및 시각화")
    save_results(results)
    plot_results(results)

    print("실험 완료. 결과는 'experiment_results.json'과 'results_plot.png'에 저장됨.")

if __name__ == "__main__":
    run_experiment()
