import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import random
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import interp1d
from scipy.stats import mode
import warnings
import gymnasium as gym
from gymnasium import spaces

warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
matplotlib.use('Agg')

# ========================= Global Configuration =========================
SOLVER_TIME_STEP = 1 / 30
DECISION_TIME_STEP = 3
SOLVER_STEPS_PER_DECISION = int(DECISION_TIME_STEP / SOLVER_TIME_STEP)
PLOT_X_LIMIT = 360
MAX_T = 360
TERMINATION_THRESHOLD = 1.2

# RL Hyperparameters
TRAIN_EPISODES = 3000
TARGET_UPDATE_FREQ = 100
BATCH_SIZE = 128
HIDDEN_SIZE = 256
MEMORY_SIZE = 100000
GAMMA = 0.9999
EPSILON = 0.99
EPSILON_DECAY = 0.9985
EPSILON_MIN = 0.01

# Reward Settings
REWARD_BASE = 1
RANGE_PUNISH_PER_STEP = -1.0
THRESHOLD_PUNISH = -5 * 200
SUCCESS_REWARD = 5 * 300

# Global English Plot Settings
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["font.size"] = 7
COLOR_REAL = "#1f77b4"
COLOR_RL = "#ff4500"

# Patient Data Path
REAL_DATA_FOLDER = r"C:\Users\22940\Desktop\prostate cancer clinical treatment data\Bruchovsky_et_al"
# Target Patient IDs
TARGET_PIDS = ["011", "012", "019", "036", "052", "054", "088", "099"]
# Parameter Columns for Boundary Extraction
PARAM_COLUMNS = ["r1", "r2", "r3", "r4",
                 "d1", "d2", "d3", "d4",
                 "a1", "a2", "K", "x10", "x20", "x30", "x40"]

# ========================= DQN 0-1 Two-Level Control =========================
DOSE_LEVELS = [0.0, 1.0]
N_DOSE = len(DOSE_LEVELS)
ACTION_SIZE = N_DOSE * N_DOSE  # 4 actions


def action_to_dose(action):
    d_idx = action // N_DOSE
    f_idx = action % N_DOSE
    return DOSE_LEVELS[d_idx], DOSE_LEVELS[f_idx]

# ========================= Gym Environment =========================
class CancerEnv(gym.Env):
    def __init__(self, patient_params):
        super().__init__()
        self.p = patient_params
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(ACTION_SIZE)
        self.reset()

    def _solver_step(self, D, F):
        x = self.x
        N = np.sum(x)
        done = False
        step_reward = 0.0

        if N >= TERMINATION_THRESHOLD:
            step_reward += THRESHOLD_PUNISH
            done = True
            return x.copy(), step_reward, done
        if self.t >= MAX_T:
            step_reward += SUCCESS_REWARD
            done = True
            return x.copy(), step_reward, done

        r1, r2, r3, r4 = self.p["r1"], self.p["r2"], self.p["r3"], self.p["r4"]
        d1, d2, d3, d4 = self.p["d1"], self.p["d2"], self.p["d3"], self.p["d4"]
        a1, a2, K = self.p["a1"], self.p["a2"], self.p["K"]

        dx1 = r1 * x[0] * (1 - N / K) * max(0, 1 - a1 * D - a2 * F) - d1 * x[0]
        dx2 = r2 * x[1] * (1 - N / K) * max(0, 1 - a2 * F) - d2 * x[1]
        dx3 = r3 * x[2] * (1 - N / K) * max(0, 1 - a1 * D) - d3 * x[2]
        dx4 = r4 * x[3] * (1 - N / K) - d4 * x[3]

        self.x += np.array([dx1, dx2, dx3, dx4]) * SOLVER_TIME_STEP
        self.x = np.maximum(self.x, 0.0)
        self.t += SOLVER_TIME_STEP

        a1_val = self.p["a1"]
        a2_val = self.p["a2"]
        sum_a = a1_val + a2_val + 1e-8
        step_reward += REWARD_BASE

        step_reward += 0.8 * (1 - D) * a1_val / sum_a
        step_reward += 0.8 * (1 - F) * a2_val / sum_a

        if N > 1.0 or N < 0.5:
            step_reward += RANGE_PUNISH_PER_STEP

        return self.x.copy(), step_reward, done

    def step(self, action):
        D, F = action_to_dose(action)
        total_reward = 0.0
        done = False
        records = []

        for _ in range(SOLVER_STEPS_PER_DECISION):
            if done:
                break
            x, r, done = self._solver_step(D, F)
            total_reward += r
            records.append({
                "solver_t": self.t,
                "N": np.sum(x),
                "done": done
            })

        info = {"solver_records": records}
        self.decision_step += 1
        return torch.tensor(self.x, dtype=torch.float32), total_reward, done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.x = np.array([self.p["x10"], self.p["x20"], self.p["x30"], self.p["x40"]], dtype=np.float32)
        self.t = 0.0
        self.decision_step = 0
        return torch.tensor(self.x, dtype=torch.float32), {}

# ========================= DQN Network and Agent =========================
class QNetwork(nn.Module):
    def __init__(self, state_size=4, action_size=ACTION_SIZE, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, action_size)

    def forward(self, s):
        x = F.relu(self.fc1(s))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class DQNAgent:
    def __init__(self):
        self.state_size = 4
        self.action_size = ACTION_SIZE
        self.gamma = GAMMA
        self.eps = EPSILON
        self.eps_decay = EPSILON_DECAY
        self.eps_min = EPSILON_MIN
        self.batch = BATCH_SIZE

        self.qnet = QNetwork()
        self.target_qnet = QNetwork()
        self.target_qnet.load_state_dict(self.qnet.state_dict())
        self.opt = torch.optim.Adam(self.qnet.parameters(), lr=1e-4)
        self.memory = deque(maxlen=MEMORY_SIZE)

    def act(self, s):
        if random.random() < self.eps:
            return random.choice(range(self.action_size))
        with torch.no_grad():
            return torch.argmax(self.qnet(s)).item()

    def remember(self, s, a, r, s2, done):
        self.memory.append((s.numpy(), a, r, s2.numpy(), done))

    def replay(self):
        if len(self.memory) < self.batch:
            return 0
        batch = random.sample(self.memory, self.batch)
        s, a, r, s2, done = zip(*batch)

        s = torch.tensor(np.array(s), dtype=torch.float32)
        a = torch.tensor(a, dtype=torch.int64).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32).unsqueeze(1)
        s2 = torch.tensor(np.array(s2), dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32).unsqueeze(1)

        with torch.no_grad():
            max_q = self.target_qnet(s2).max(1, keepdim=True)[0]
            tgt = r + (1 - done) * self.gamma * max_q

        cur = self.qnet(s).gather(1, a)
        loss = F.mse_loss(cur, tgt)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()

    def update_target(self):
        self.target_qnet.load_state_dict(self.qnet.state_dict())

    def decay_eps(self):
        if self.eps > self.eps_min:
            self.eps *= self.eps_decay

# ========================= Utility Functions =========================
def get_param_bounds(target_df):
    """Get parameter bounds from 8 target patients"""
    bounds = {}
    for col in PARAM_COLUMNS:
        vals = target_df[col].values
        bounds[col] = (np.min(vals), np.max(vals))
    return bounds

def generate_random_patients(bounds, num=100):
    """Generate random patient parameters within bounds"""
    patients = []
    for _ in range(num):
        p = {}
        for col in PARAM_COLUMNS:
            low, high = bounds[col]
            p[col] = np.random.uniform(low, high)
        patients.append(p)
    return patients

def load_real_N(pid):
    """Load real patient N (PSA) data"""
    file_path = os.path.join(REAL_DATA_FOLDER, f"patient{pid}.txt")
    if not os.path.exists(file_path):
        return np.array([]), np.array([])

    dataTable = pd.read_csv(file_path, header=None)
    dataTable = dataTable.apply(pd.to_numeric, errors='coerce')
    data = dataTable.iloc[:, 2:10].values
    N = data.shape[0]
    t = np.arange(1, N + 1)

    y = data[:, 2].astype(float)
    nan_idx = np.isnan(y)
    if np.any(nan_idx):
        x_idx = np.arange(len(y))
        f = interp1d(x_idx[~nan_idx], y[~nan_idx], kind='linear', fill_value='extrapolate')
        y = f(x_idx)

    # Normalization
    y = (y - np.min(y)) / (np.max(y) - np.min(y) + 1e-6)
    return t, y

def test_rl_on_patient(agent, patient_params):
    """Test trained RL agent on a single patient, return N trajectory"""
    env = CancerEnv(patient_params)
    s, _ = env.reset()
    done = False
    records = []

    while not done:
        a = torch.argmax(agent.qnet(s)).item()
        s2, r, done, _, info = env.step(a)
        s = s2
        records.extend(info["solver_records"])

    df = pd.DataFrame(records)
    return df

# ========================= Population RL Training =========================
def train_population_rl(random_patients):
    """Population DQN training: interact with a random patient each episode"""
    agent = DQNAgent()
    print("Start Population DQN Training (Random Patients)...")

    for ep in range(TRAIN_EPISODES):
        # Randomly select one patient
        patient = random.choice(random_patients)
        env = CancerEnv(patient)

        s, _ = env.reset()
        total_r = 0
        done = False

        while not done:
            a = agent.act(s)
            s2, r, done, _, info = env.step(a)
            agent.remember(s, a, r, s2, done)

            agent.replay()
            s = s2
            total_r += r

            if env.decision_step % TARGET_UPDATE_FREQ == 0:
                agent.update_target()

        agent.decay_eps()

        if (ep + 1) % 100 == 0:
            print(f"Episode {ep+1:4d} | Reward: {total_r:6.1f} | Eps: {agent.eps:.3f}")

    print("Population Training Finished!")
    return agent

# ========================= 3×3 Plot =========================
def plot_3x3_results(target_df, agent):
    fig = plt.figure(figsize=(9, 9))
    fig.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.08, wspace=0.3, hspace=0.4)
    fig.text(0.5, 0.02, 'Time / Month', ha='center', fontsize=11)
    fig.text(0.02, 0.5, 'Normalized Tumor Size (N)', va='center', rotation=90, fontsize=11)

    # Subplot 1-8: 8 patients
    for idx, pid in enumerate(TARGET_PIDS):
        ax = fig.add_subplot(3, 3, idx + 1)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 1.5)

        # 1. Plot real patient N
        t_real, y_real = load_real_N(pid)
        ax.plot(t_real, y_real, color=COLOR_REAL, lw=1.2, label='Real N')

        # 2. Plot RL controlled N
        row = target_df[target_df["PatientID"].astype(str).str.zfill(3) == pid].iloc[0]
        rl_df = test_rl_on_patient(agent, row.to_dict())
        rl_df = rl_df[rl_df.solver_t <= 200]
        ax.plot(rl_df.solver_t, rl_df.N, color=COLOR_RL, lw=1.2, label='RL N')

        # Threshold line
        ax.axhline(TERMINATION_THRESHOLD, color='c', ls='--', lw=0.8, alpha=0.7)
        ax.set_title(f"Patient {pid}", fontsize=9)
        ax.tick_params(labelsize=7)

    # Subplot 9: Legend
    ax_leg = fig.add_subplot(3, 3, 9)
    ax_leg.axis('off')
    line1 = ax_leg.plot([], [], color=COLOR_REAL, lw=2)[0]
    line2 = ax_leg.plot([], [], color=COLOR_RL, lw=2)[0]
    line3 = ax_leg.plot([], [], color='c', ls='--', lw=1)[0]
    ax_leg.legend([line1, line2, line3],
                 ['Real Patient N', 'Population RL Control N', 'Termination Threshold'],
                 loc='center', fontsize=10)

    # Save figure
    save_path = os.path.join(os.path.expanduser("~"), "Desktop", "Population_RL_3x3_Result.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path

# ========================= Main Function =========================
def main():
    print("=" * 60)
    print("  Population DQN for Prostate Cancer Treatment")
    print("  Patients: 11, 12, 19, 36, 52, 54, 88, 99")
    print("  Generate 100 random patients for training")
    print("  Plot: 3×3 subplots (8 patients + legend)")
    print("=" * 60)

    # 1. Load parameter table
    excel_path = r"C:\Users\22940\Desktop\Patient Fitted Parameter Results Table.xlsx"
    if not os.path.exists(excel_path):
        print("ERROR: Parameter Excel file not found!")
        return
    params_df = pd.read_excel(excel_path)
    params_df["PatientID"] = params_df["PatientID"].astype(str).str.zfill(3)

    # 2. Filter target patients and get parameter bounds
    target_df = params_df[params_df["PatientID"].isin(TARGET_PIDS)].reset_index(drop=True)
    param_bounds = get_param_bounds(target_df)
    print(f"Loaded {len(target_df)} target patients")
    print("Generated parameter bounds successfully")

    # 3. Generate 100 random patients
    random_patients = generate_random_patients(param_bounds, num=100)
    print(f"Generated {len(random_patients)} random patients for training")

    # 4. Population RL training
    trained_agent = train_population_rl(random_patients)

    # 5. Plot 3×3 results
    fig_path = plot_3x3_results(target_df, trained_agent)
    print(f"\nResult figure saved to: {fig_path}")

if __name__ == "__main__":
    main()