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
PLOT_X_LIMIT = 200
MAX_T = 200
TERMINATION_THRESHOLD = 1.2

# RL Hyperparameters
TRAIN_EPISODES = 3000
TARGET_UPDATE_FREQ = 100
BATCH_SIZE = 128
HIDDEN_SIZE = 256
MEMORY_SIZE = 50000
GAMMA = 0.9999
EPSILON = 0.99
EPSILON_DECAY = 0.998
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

# Fixed Color Scheme for Four Subtypes
COLOR_X1 = "#1f77b4"
COLOR_X2 = "#2ca02c"
COLOR_X3 = "#ff7f0e"
COLOR_X4 = "#d62728"

# Patient Data Path
REAL_DATA_FOLDER = r"C:\Users\22940\Desktop\prostate cancer clinical treatment data\Bruchovsky_et_al"

# ========================= DQN with 0-1 Binary Control =========================
DOSE_LEVELS = [0.0, 1.0]
N_DOSE = len(DOSE_LEVELS)
ACTION_SIZE = N_DOSE * N_DOSE  # 2×2=4 actions


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
            if done: break
            x, r, done = self._solver_step(D, F)
            total_reward += r
            records.append({
                "solver_t": self.t,
                "x1": x[0], "x2": x[1], "x3": x[2], "x4": x[3],
                "N": np.sum(x), "D": D, "F": F, "done": done
            })

        info = {"solver_records": records, "D": D, "F": F}
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


# ========================= AT50 Strategy =========================
def generate_at50_data_for_patient(patient_params):
    class SimpleModel:
        def __init__(self, p):
            self.p = p
            self.x = np.array([p["x10"], p["x20"], p["x30"], p["x40"]], dtype=np.float32)
            self.t = 0.0
            self.done = False

        def step(self, D, F):
            p = self.p
            N = np.sum(self.x)
            dx1 = p["r1"] * self.x[0] * (1 - N / p["K"]) * max(0, 1 - p["a1"] * D - p["a2"] * F) - p["d1"] * self.x[0]
            dx2 = p["r2"] * self.x[1] * (1 - N / p["K"]) * max(0, 1 - p["a2"] * F) - p["d2"] * self.x[1]
            dx3 = p["r3"] * self.x[2] * (1 - N / p["K"]) * max(0, 1 - p["a1"] * D) - p["d3"] * self.x[2]
            dx4 = p["r4"] * self.x[3] * (1 - N / p["K"]) - p["d4"] * self.x[3]
            self.x += np.array([dx1, dx2, dx3, dx4]) * SOLVER_TIME_STEP
            self.x = np.maximum(self.x, 0)
            self.t += SOLVER_TIME_STEP
            if N >= TERMINATION_THRESHOLD or self.t >= MAX_T:
                self.done = True
            return self.x.copy(), np.sum(self.x)

    model = SimpleModel(patient_params)
    traj = []
    D, F = 0.0, 0.0
    while not model.done:
        x, N = model.x, np.sum(model.x)
        traj.append({"solver_t": model.t, "x1": x[0], "x2": x[1], "x3": x[2], "x4": x[3], "N": N, "D": D, "F": F})
        if N >= 1.0:
            D, F = 1, 1
        elif N < 0.5:
            D, F = 0, 0
        model.step(D, F)
    return pd.DataFrame(traj)


# ========================= DQN Training =========================
def generate_rl_data_for_patient(patient_params):
    env = CancerEnv(patient_params)
    agent = DQNAgent()
    best_reward = -np.inf
    best_records = []
    episode_rewards = []
    episode_losses = []

    print("Start DQN Training...")
    for ep in range(TRAIN_EPISODES):
        s, _ = env.reset()
        total_r = 0.0
        total_loss = 0.0
        loss_count = 0
        ep_records = []
        done = False

        while not done:
            a = agent.act(s)
            s2, r, done, _, info = env.step(a)
            agent.remember(s, a, r, s2, done)

            loss = agent.replay()
            if loss > 0:
                total_loss += loss
                loss_count += 1

            s = s2
            total_r += r
            ep_records.extend(info["solver_records"])

            if env.decision_step % TARGET_UPDATE_FREQ == 0:
                agent.update_target()

        agent.decay_eps()
        episode_rewards.append(total_r)
        avg_loss = total_loss / loss_count if loss_count > 0 else 0.0
        episode_losses.append(avg_loss)

        if total_r > best_reward:
            best_reward = total_r
            best_records = ep_records.copy()
        if (ep + 1) % 50 == 0:
            print(
                f"Episode {ep + 1:3d} | Reward: {total_r:6.1f} | Loss: {avg_loss:.4f} | Best: {best_reward:6.1f} | eps: {agent.eps:.3f}")

    return pd.DataFrame(best_records), episode_rewards, episode_losses


# ========================= Plot Functions (Full English) =========================
def plot_real_patient(ax, pid):
    file_path = os.path.join(REAL_DATA_FOLDER, f"patient{pid}.txt")
    if not os.path.exists(file_path):
        ax.text(0.5, 0.5, "No Real Data", ha='center', va='center', transform=ax.transAxes)
        return

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
        data[:, 2] = f(x_idx)

    target_cols = [0, 1]
    treat_col = 5
    for col in target_cols:
        for i in range(N):
            val = data[i, col]
            if not np.isnan(val):
                continue
            treat = data[i, treat_col]
            if treat == 0:
                data[i, col] = 0
            else:
                s = max(0, i - 3)
                e = min(N - 1, i + 3)
                win = data[s:e + 1, col]
                valid = win[~np.isnan(win) & (win != 0)]
                data[i, col] = mode(valid)[0] if len(valid) > 0 else 0

    def norm_fun(x):
        x = x.astype(float)
        return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + 1e-6)

    data[:, 0] = norm_fun(data[:, 0])
    data[:, 1] = norm_fun(data[:, 1])
    data[:, 2] = norm_fun(data[:, 2])

    y_data = data[:, 2]
    D = data[:, 0]
    F = data[:, 1]

    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.55)
    ax.set_xlim(0, 200)
    ax.plot(t, y_data, 'bo-', lw=1.0, ms=2)

    for i in range(N):
        xi = t[i]
        if xi > 200: continue
        if D[i] > 1e-3:
            ax.fill_between([xi - 0.5, xi + 0.5], 1.2, 1.35, color='b', alpha=D[i], lw=0)
        if F[i] > 1e-3:
            ax.fill_between([xi - 0.5, xi + 0.5], 1.35, 1.5, color='r', alpha=F[i], lw=0)

    cumD = np.cumsum(D)
    D_total_val = np.sum(D)
    temp_idx = np.where(cumD == D_total_val)[0]
    et = temp_idx[-1] + 1 if len(temp_idx) > 0 else N
    if 1 <= et <= N:
        ax.axvline(et, color='k', ls='--', lw=1.0)
        ax.text(et, 1.42, f'{et}', fontsize=6, ha='center', bbox=dict(facecolor='w', pad=1))

    ax.set_title(f"ID {pid} | Real Clinical Data", fontsize=7)


def plot_at50_patient(ax, params, pid):
    df = generate_at50_data_for_patient(params)
    df = df[df.solver_t <= 200]
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.55)
    ax.set_xlim(0, 200)
    ax.plot(df.solver_t, df.x1, COLOR_X1, lw=0.8)
    ax.plot(df.solver_t, df.x2, COLOR_X2, lw=0.8)
    ax.plot(df.solver_t, df.x3, COLOR_X3, lw=0.8)
    ax.plot(df.solver_t, df.x4, COLOR_X4, lw=0.8)
    ax.plot(df.solver_t, df.N, 'k-', lw=1.0)
    for _, row in df.iterrows():
        if row.D > 0.5:
            ax.fill_between([row.solver_t - 0.5, row.solver_t + 0.5], 1.2, 1.35, color='b', alpha=0.6, lw=0)
        if row.F > 0.5:
            ax.fill_between([row.solver_t - 0.5, row.solver_t + 0.5], 1.35, 1.5, color='r', alpha=0.6, lw=0)
    ax.axhline(TERMINATION_THRESHOLD, color='c', ls='--', lw=0.6)
    ax.set_title(f"ID {pid} | AT50 Strategy", fontsize=7)


def plot_rl_patient(ax, params, pid):
    df, episode_rewards, episode_losses = generate_rl_data_for_patient(params)
    df = df[df.solver_t <= 200]
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.55)
    ax.set_xlim(0, 200)
    ax.plot(df.solver_t, df.x1, COLOR_X1, lw=0.8)
    ax.plot(df.solver_t, df.x2, COLOR_X2, lw=0.8)
    ax.plot(df.solver_t, df.x3, COLOR_X3, lw=0.8)
    ax.plot(df.solver_t, df.x4, COLOR_X4, lw=0.8)
    ax.plot(df.solver_t, df.N, 'k-', lw=1.0)
    for _, row in df.iterrows():
        if row.D > 0.05:
            ax.fill_between([row.solver_t - 0.4, row.solver_t + 0.4], 1.22, 1.38, color='blue', alpha=row.D * 0.55,
                            lw=0)
        if row.F > 0.05:
            ax.fill_between([row.solver_t - 0.4, row.solver_t + 0.4], 1.38, 1.52, color='red', alpha=row.F * 0.55, lw=0)
    ax.axhline(TERMINATION_THRESHOLD, color='c', ls='--', lw=0.6)
    ax.set_title(f"ID {pid} | DQN 0-1 Control", fontsize=7)
    return episode_rewards, episode_losses


def plot_training_curve(ax, rewards, pid):
    episodes = np.arange(1, len(rewards) + 1)
    ax.grid(True, alpha=0.3)
    ax.plot(episodes, rewards, color='#9400D3', lw=1.0)
    ax.set_xlabel("Episode", fontsize=7)
    ax.set_ylabel("Total Reward", fontsize=7)
    ax.set_title(f"ID {pid} | DQN Reward Curve", fontsize=7)
    ax.tick_params(labelsize=6)


# ========================= Main Plot: 4×2 Full English =========================
def plot_4row_2col(params_df):
    fig = plt.figure(figsize=(6.4, 7.5))
    fig.subplots_adjust(left=0.08, right=0.94, top=0.95, bottom=0.08, wspace=0.3, hspace=0.4)
    fig.text(0.5, 0.02, 'Time / Month', ha='center', fontsize=10)
    fig.text(0.02, 0.5, 'Tumor Size / PSA', va='center', rotation=90, fontsize=10)

    params_df["PatientID"] = params_df["PatientID"].astype(str).str.zfill(3)
    row_011 = params_df[params_df.PatientID == "011"].iloc[0]
    row_019 = params_df[params_df.PatientID == "019"].iloc[0]

    # Left Column: Patient 011
    ax1 = fig.add_subplot(4, 2, 1)
    plot_real_patient(ax1, "011")

    ax2 = fig.add_subplot(4, 2, 3)
    plot_at50_patient(ax2, row_011.to_dict(), "011")

    ax3 = fig.add_subplot(4, 2, 5)
    reward_011, _ = plot_rl_patient(ax3, row_011.to_dict(), "011")

    ax4 = fig.add_subplot(4, 2, 7)
    plot_training_curve(ax4, reward_011, "011")

    # Right Column: Patient 019
    ax5 = fig.add_subplot(4, 2, 2)
    plot_real_patient(ax5, "019")

    ax6 = fig.add_subplot(4, 2, 4)
    plot_at50_patient(ax6, row_019.to_dict(), "019")

    ax7 = fig.add_subplot(4, 2, 6)
    reward_019, _ = plot_rl_patient(ax7, row_019.to_dict(), "019")

    ax8 = fig.add_subplot(4, 2, 8)
    plot_training_curve(ax8, reward_019, "019")

    # English Legend
    ax_leg = fig.add_subplot(4, 2, 1)
    ax_leg.axis('off')
    ax_leg.legend([
        ax_leg.plot([], [], 'bo-', lw=1)[0],
        ax_leg.plot([], [], 'k-', lw=1)[0],
        ax_leg.plot([], [], color=COLOR_X1, lw=1)[0],
        ax_leg.plot([], [], color=COLOR_X2, lw=1)[0],
        ax_leg.plot([], [], color=COLOR_X3, lw=1)[0],
        ax_leg.plot([], [], color=COLOR_X4, lw=1)[0]
    ], ['Real PSA', 'Total N', 'x1', 'x2', 'x3', 'x4'], loc='upper right', fontsize=7)

    save_path = os.path.join(os.path.expanduser("~"), "Desktop", "ID011_ID019_4x2_DQN_01_Control.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path


# ========================= Main Function =========================
def main():
    print("=" * 60)
    print("  DQN with 0-1 Binary Control - Full English Version")
    print("  All labels, comments and outputs are in English")
    print("  Layout: 4×2 | Left: Patient 011 | Right: Patient 019")
    print("=" * 60)

    excel_path = r"C:\Users\22940\Desktop\Patient Fitted Parameter Results Table.xlsx"
    if not os.path.exists(excel_path):
        print("Error: Parameter file not found!")
        return

    params = pd.read_excel(excel_path)
    saved_file = plot_4row_2col(params)
    print(f"Figure saved successfully to: {saved_file}")


if __name__ == "__main__":
    main()