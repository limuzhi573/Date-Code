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

# Plot Settings
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["font.size"] = 8
COLOR_X1 = "#1f77b4"
COLOR_X2 = "#2ca02c"
COLOR_X3 = "#ff7f0e"
COLOR_X4 = "#d62728"

# Patient Data Path
REAL_DATA_FOLDER = r"C:\Users\22940\Desktop\prostate cancer clinical treatment data\Bruchovsky_et_al"

# ========================= DQN Discrete Dose Definition =========================
DOSE_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]
N_DOSE = len(DOSE_LEVELS)
ACTION_SIZE = N_DOSE * N_DOSE


def action_to_dose(action):
    d_idx = action // N_DOSE
    f_idx = action % N_DOSE
    return DOSE_LEVELS[d_idx], DOSE_LEVELS[f_idx]


# ========================= Clinical Data Processing =========================
def get_processed_clinical_data(pid):
    file_path = os.path.join(REAL_DATA_FOLDER, f"patient{pid}.txt")
    dataTable = pd.read_csv(file_path, header=None).apply(pd.to_numeric, errors='coerce')
    data = dataTable.iloc[:, 2:10].values
    N = data.shape[0]
    t_clin = np.arange(1, N + 1)

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

    D_clin = data[:, 0]
    F_clin = data[:, 1]

    D_total = np.sum(D_clin)
    cumD = np.cumsum(D_clin)
    temp_idx = np.where(cumD == D_total)[0]
    end_clin_t = temp_idx[-1] + 1 if len(temp_idx) > 0 else N

    return t_clin, D_clin, F_clin, end_clin_t


# ========================= Gym Environment =========================
class CancerEnv(gym.Env):
    def __init__(self, patient_params):
        super().__init__()
        self.p = patient_params
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(ACTION_SIZE)

    def set_init_state(self, x0, t0):
        self.x = x0.copy()
        self.t = t0
        self.decision_step = 0

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
                "solver_t": self.t, "x1": x[0], "x2": x[1], "x3": x[2], "x4": x[3],
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


# ========================= DQN Network & Agent =========================
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


# ========================= Clinical Simulation to Midpoint =========================
def simulate_clinic_get_mid_state(patient_params, pid):
    t_clin, D_clin, F_clin, end_clin_t = get_processed_clinical_data(pid)
    mid_clin_t = end_clin_t / 2
    f_D = interp1d(t_clin, D_clin, kind='nearest', fill_value='extrapolate')
    f_F = interp1d(t_clin, F_clin, kind='nearest', fill_value='extrapolate')

    class SimpleModel:
        def __init__(self, p):
            self.p = p
            self.x = np.array([p["x10"], p["x20"], p["x30"], p["x40"]], dtype=np.float32)
            self.t = 0.0

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
            return self.x.copy()

    model = SimpleModel(patient_params)
    traj = []
    while model.t < mid_clin_t:
        D = float(f_D(model.t))
        F = float(f_F(model.t))
        x = model.x
        traj.append(
            {"solver_t": model.t, "x1": x[0], "x2": x[1], "x3": x[2], "x4": x[3], "N": np.sum(x), "D": D, "F": F})
        model.step(D, F)
    final_x = model.x.copy()
    final_t = model.t
    return traj, final_x, final_t, mid_clin_t


# ========================= Revised: Clinical Midpoint + AT50 =========================
def generate_clinic_then_at50(patient_params, pid):
    clinic_traj, x_end, t_end, switch_t = simulate_clinic_get_mid_state(patient_params, pid)
    p = patient_params

    class M:
        def __init__(self):
            self.x = x_end.copy()
            self.t = t_end
            self.done = False

        def step(self, D, F):
            N = np.sum(self.x)
            if N >= TERMINATION_THRESHOLD or self.t >= MAX_T:
                self.done = True
                return self.x.copy()
            dx1 = p["r1"] * self.x[0] * (1 - N / p["K"]) * max(0, 1 - p["a1"] * D - p["a2"] * F) - p["d1"] * self.x[0]
            dx2 = p["r2"] * self.x[1] * (1 - N / p["K"]) * max(0, 1 - p["a2"] * F) - p["d2"] * self.x[1]
            dx3 = p["r3"] * self.x[2] * (1 - N / p["K"]) * max(0, 1 - p["a1"] * D) - p["d3"] * self.x[2]
            dx4 = p["r4"] * self.x[3] * (1 - N / p["K"]) - p["d4"] * self.x[3]
            self.x += np.array([dx1, dx2, dx3, dx4]) * SOLVER_TIME_STEP
            self.x = np.maximum(self.x, 0)
            self.t += SOLVER_TIME_STEP
            return self.x.copy()

    m = M()
    at50_traj = []
    D, F = 0.0, 0.0
    while not m.done:
        x = m.x
        N = np.sum(x)
        at50_traj.append({"solver_t": m.t, "x1": x[0], "x2": x[1], "x3": x[2], "x4": x[3], "N": N, "D": D, "F": F})
        if N >= 1.0:
            D, F = 1, 1
        elif N < 0.5:
            D, F = 0, 0
        m.step(D, F)

    df = pd.DataFrame(clinic_traj + at50_traj)
    return df, switch_t


# ========================= Revised: Clinical Midpoint + DQN =========================
def generate_clinic_then_rl(patient_params, pid):
    clinic_traj, x_end, t_end, switch_t = simulate_clinic_get_mid_state(patient_params, pid)
    env = CancerEnv(patient_params)
    agent = DQNAgent()
    best_reward = -np.inf
    best_post = []
    episode_rewards = []

    for ep in range(TRAIN_EPISODES):
        env.set_init_state(x_end, t_end)
        s = torch.tensor(env.x, dtype=torch.float32)
        total_r = 0
        total_loss = 0
        loss_cnt = 0
        post = []
        done = False

        while not done:
            a = agent.act(s)
            s2, r, done, _, info = env.step(a)
            agent.remember(s, a, r, s2, done)
            loss = agent.replay()
            if loss > 0:
                total_loss += loss
                loss_cnt += 1
            s = s2
            total_r += r
            post.extend(info["solver_records"])
            if env.decision_step % TARGET_UPDATE_FREQ == 0:
                agent.update_target()

        agent.decay_eps()
        episode_rewards.append(total_r)

        if total_r > best_reward:
            best_reward = total_r
            best_post = post.copy()

        if (ep + 1) % 50 == 0:
            print(f"PID {pid} | Ep {ep + 1:4d} | R:{total_r:6.1f} | Best:{best_reward:6.1f}")

    df = pd.DataFrame(clinic_traj + best_post)
    return df, switch_t, episode_rewards


# ========================= Plotting Functions =========================
def plot_real_patient(ax, pid):
    t_clin, D_clin, F_clin, end_clin_t = get_processed_clinical_data(pid)
    file_path = os.path.join(REAL_DATA_FOLDER, f"patient{pid}.txt")
    dataTable = pd.read_csv(file_path, header=None).apply(pd.to_numeric, errors='coerce')
    data = dataTable.iloc[:, 2:10].values
    y = data[:, 2].astype(float)
    nan_idx = np.isnan(y)
    if np.any(nan_idx):
        f = interp1d(np.arange(len(y))[~nan_idx], y[~nan_idx], kind='linear', fill_value='extrapolate')
        data[:, 2] = f(np.arange(len(y)))

    def norm_fun(x):
        return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + 1e-6)

    y_data = norm_fun(data[:, 2])
    ax.grid(True)
    ax.set_ylim(0, 1.55)
    ax.set_xlim(0, 200)
    ax.plot(t_clin, y_data, 'bo-', lw=1.2, ms=3)
    for i in range(len(t_clin)):
        xi = t_clin[i]
        if D_clin[i] > 1e-3:
            ax.fill_between([xi - 0.5, xi + 0.5], 1.2, 1.35, color='b', alpha=D_clin[i], lw=0)
        if F_clin[i] > 1e-3:
            ax.fill_between([xi - 0.5, xi + 0.5], 1.35, 1.5, color='r', alpha=F_clin[i], lw=0)
    ax.set_title(f"ID {pid} | Real Clinical Data", fontsize=8)


def plot_clinic_at50(ax, params, pid):
    df, switch_t = generate_clinic_then_at50(params, pid)
    df = df[df.solver_t <= 200]
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.55)
    ax.set_xlim(0, 200)

    ax.plot(df.solver_t, df.x1, COLOR_X1, lw=1)
    ax.plot(df.solver_t, df.x2, COLOR_X2, lw=1)
    ax.plot(df.solver_t, df.x3, COLOR_X3, lw=1)
    ax.plot(df.solver_t, df.x4, COLOR_X4, lw=1)
    ax.plot(df.solver_t, df.N, 'k-', lw=1.5)

    t_clin, D_clin, F_clin, _ = get_processed_clinical_data(pid)
    for i in range(len(t_clin)):
        xi = t_clin[i]
        if xi > 200 or xi > switch_t: continue
        if D_clin[i] > 1e-3:
            ax.fill_between([xi - 0.5, xi + 0.5], 1.2, 1.35, color='b', alpha=D_clin[i], lw=0)
        if F_clin[i] > 1e-3:
            ax.fill_between([xi - 0.5, xi + 0.5], 1.35, 1.5, color='r', alpha=F_clin[i], lw=0)

    for _, row in df[df.solver_t > switch_t].iterrows():
        if row.D > 0.5: ax.fill_between([row.solver_t - 0.5, row.solver_t + 0.5], 1.2, 1.35, color='b', alpha=0.6, lw=0)
        if row.F > 0.5: ax.fill_between([row.solver_t - 0.5, row.solver_t + 0.5], 1.35, 1.5, color='r', alpha=0.6, lw=0)

    ax.axvline(switch_t, color='k', ls='--', lw=1.5)
    ax.text(switch_t, 1.42, f"{switch_t:.1f}", fontsize=7, ha='center', bbox=dict(facecolor='w', pad=1))
    ax.axhline(TERMINATION_THRESHOLD, color='c', ls='--', lw=0.8)
    ax.set_title(f"ID {pid} | Clinic Mid → AT50", fontsize=8)


def plot_clinic_rl(ax, params, pid):
    df, switch_t, rewards = generate_clinic_then_rl(params, pid)
    df = df[df.solver_t <= 200]
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.55)
    ax.set_xlim(0, 200)
    ax.plot(df.solver_t, df.x1, COLOR_X1, lw=1)
    ax.plot(df.solver_t, df.x2, COLOR_X2, lw=1)
    ax.plot(df.solver_t, df.x3, COLOR_X3, lw=1)
    ax.plot(df.solver_t, df.x4, COLOR_X4, lw=1)
    ax.plot(df.solver_t, df.N, 'k-', lw=1.3)

    t_clin, D_clin, F_clin, _ = get_processed_clinical_data(pid)
    for i in range(len(t_clin)):
        xi = t_clin[i]
        if xi > 200 or xi > switch_t: continue
        if D_clin[i] > 1e-3:
            ax.fill_between([xi - 0.5, xi + 0.5], 1.2, 1.35, color='b', alpha=D_clin[i], lw=0)
        if F_clin[i] > 1e-3:
            ax.fill_between([xi - 0.5, xi + 0.5], 1.35, 1.5, color='r', alpha=F_clin[i], lw=0)

    for _, row in df[df.solver_t > switch_t].iterrows():
        if row.D > 0.05: ax.fill_between([row.solver_t - 0.4, row.solver_t + 0.4], 1.22, 1.38, color='blue',
                                         alpha=row.D * 0.55, lw=0)
        if row.F > 0.05: ax.fill_between([row.solver_t - 0.4, row.solver_t + 0.4], 1.38, 1.52, color='red',
                                         alpha=row.F * 0.55, lw=0)

    ax.axvline(switch_t, color='k', ls='--', lw=1.5)
    ax.text(switch_t, 1.42, f"{switch_t:.1f}", fontsize=7, ha='center', bbox=dict(facecolor='w', pad=1))
    ax.axhline(TERMINATION_THRESHOLD, color='c', ls='--', lw=0.7)
    ax.set_title(f"ID {pid} | Clinic Mid → DQN", fontsize=8)
    return rewards


def plot_training_curve(ax, rewards, pid):
    ax.grid(True, alpha=0.3)
    ax.plot(range(1, len(rewards) + 1), rewards, color='#9400D3', lw=1.5)
    ax.set_xlabel("Episode", fontsize=8)
    ax.set_ylabel("Total Reward", fontsize=8)
    ax.set_title(f"ID {pid} | DQN Training Curve", fontsize=8)


# ========================= 4x2 Plot Layout =========================
def plot_4row_2col(params_df):
    fig = plt.figure(figsize=(6.4, 7.5))
    fig.subplots_adjust(left=0.08, right=0.94, top=0.95, bottom=0.08, wspace=0.3, hspace=0.4)
    fig.text(0.5, 0.02, 'Time / Month', ha='center', fontsize=10)
    fig.text(0.03, 0.5, 'Tumor Size / PSA', va='center', rotation=90, fontsize=10)

    params_df["PatientID"] = params_df["PatientID"].astype(str).str.zfill(3)
    row_011 = params_df[params_df.PatientID == "011"].iloc[0]
    row_019 = params_df[params_df.PatientID == "019"].iloc[0]

    ax1_11 = fig.add_subplot(4, 2, 1)
    plot_real_patient(ax1_11, "011")

    ax2_11 = fig.add_subplot(4, 2, 3)
    plot_clinic_at50(ax2_11, row_011.to_dict(), "011")

    ax3_11 = fig.add_subplot(4, 2, 5)
    r11 = plot_clinic_rl(ax3_11, row_011.to_dict(), "011")

    ax4_11 = fig.add_subplot(4, 2, 7)
    plot_training_curve(ax4_11, r11, "011")

    ax1_19 = fig.add_subplot(4, 2, 2)
    plot_real_patient(ax1_19, "019")

    ax2_19 = fig.add_subplot(4, 2, 4)
    plot_clinic_at50(ax2_19, row_019.to_dict(), "019")

    ax3_19 = fig.add_subplot(4, 2, 6)
    r19 = plot_clinic_rl(ax3_19, row_019.to_dict(), "019")

    ax4_19 = fig.add_subplot(4, 2, 8)
    plot_training_curve(ax4_19, r19, "019")

    ax_leg = fig.add_subplot(4, 2, 1)
    ax_leg.axis('off')
    ax_leg.legend([
        ax_leg.plot([], [], 'bo-', lw=1.2)[0],
        ax_leg.plot([], [], 'k-', lw=1.5)[0],
        ax_leg.plot([], [], color=COLOR_X1, lw=1.5)[0],
        ax_leg.plot([], [], color=COLOR_X2, lw=1.5)[0],
        ax_leg.plot([], [], color=COLOR_X3, lw=1.5)[0],
        ax_leg.plot([], [], color=COLOR_X4, lw=1.5)[0]
    ], ['Real PSA', 'Total Tumor N', 'x1', 'x2', 'x3', 'x4'], loc='upper right', fontsize=9)

    save_path = os.path.join(os.path.expanduser("~"), "Desktop", "011_019_4x2_Clinic_Mid_AT50_DQN.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path


# ========================= Main Function =========================
def main():
    print("🚀 Running: Dual Patients 011 + 019 | Clinical Midpoint → AT50 / DQN (4x2 Plot)")
    excel = r"C:\Users\22940\Desktop\Patient Fitted Parameter Results Table.xlsx"
    if not os.path.exists(excel):
        print("❌ Parameter file not found")
        return
    params = pd.read_excel(excel)
    path = plot_4row_2col(params)
    print(f"✅ Saved successfully: {path}")


if __name__ == "__main__":
    main()