import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import mode
from scipy.integrate import solve_ivp
import warnings
import os

warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ===================== Global Font Settings =====================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Patient list
patients = ['011', '012', '019', '036', '052', '054', '088', '099']
num_patients = len(patients)

D_total = np.zeros(num_patients)
D_sum_times = [0] * num_patients
# ===================== New: Store fitting error for each patient =====================
patient_errors = []

# ===================== File Paths =====================
REAL_DATA_FOLDER = r"C:\Users\22940\Desktop\prostate cancer clinical treatment data\Bruchovsky_et_al"
param_file = r"C:\Users\22940\Desktop\Patient Fitted Parameter Results Table.xlsx"

# ===================== Initialize Figure =====================
nrows = 3
ncols = 3
fig = plt.figure(figsize=(18, 11))

# Subplot scaling factor: shrink to 0.8x for margins
scale_factor = 0.8
ax_width = (1.0 / ncols) * scale_factor
ax_height = (1.0 / nrows) * scale_factor
start_x = (1 - ncols * ax_width) / 2
start_y = (1 - nrows * ax_height) / 2

axes = []
for i in range(nrows):
    for j in range(ncols):
        left = start_x + j * ax_width
        bottom = 1 - (start_y + (i + 1) * ax_height)
        ax = fig.add_axes([left, bottom, ax_width, ax_height])
        axes.append(ax)
axes = np.array(axes)


# ===================== ODE Model =====================
def model(t, y, r1, r2, r3, r4, d1, d2, d3, d4, a1, a2, K, D_seq, F_seq):
    x1, x2, x3, x4 = y
    idx = int(round(t))
    idx = max(0, min(idx, len(D_seq) - 1))
    D_t = D_seq[idx]
    F_t = F_seq[idx]

    dydt = np.zeros(4)
    x_sum = max(x1 + x2 + x3 + x4, 1e-6)

    dydt[0] = r1 * x1 * (1 - x_sum / K) * max(1 - a1 * D_t - a2 * F_t, 0) - d1 * x1
    dydt[1] = r2 * x2 * (1 - x_sum / K) * max(1 - a2 * F_t, 0) - d2 * x2
    dydt[2] = r3 * x3 * (1 - x_sum / K) * max(1 - a1 * D_t, 0) - d3 * x3
    dydt[3] = r4 * x4 * (1 - x_sum / K) - d4 * x4

    return dydt


# ===================== Load Parameters =====================
param_df = pd.read_excel(param_file, dtype={'PatientID': str})
param_df["PatientID"] = param_df["PatientID"].astype(str).str.zfill(3)

# ===================== Loop Plotting =====================
for idx in range(num_patients):
    pid = patients[idx]
    ax = axes[idx]
    print(f"\nPlotting patient: {pid}")

    file_path = os.path.join(REAL_DATA_FOLDER, f"patient{pid}.txt")
    if not os.path.exists(file_path):
        print(f"File not found: {pid}")
        continue

    dataTable = pd.read_csv(file_path, header=None)
    dataTable = dataTable.apply(pd.to_numeric, errors='coerce')
    data = dataTable.iloc[:, 2:10].values
    N = data.shape[0]
    t = np.arange(1, N + 1)

    # PSA interpolation
    y = data[:, 2].astype(float)
    nan_idx = np.isnan(y)
    if np.any(nan_idx):
        x_idx = np.arange(len(y))
        f = interp1d(x_idx[~nan_idx], y[~nan_idx], kind='linear', fill_value='extrapolate')
        data[:, 2] = f(x_idx)

    # Fill missing drug data
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

    # Normalization
    def norm_fun(x):
        x = x.astype(float)
        return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + 1e-6)

    data[:, 0] = norm_fun(data[:, 0])
    data[:, 1] = norm_fun(data[:, 1])
    data[:, 2] = norm_fun(data[:, 2])

    y_data = data[:, 2]
    D = data[:, 0]
    F = data[:, 1]
    D_total[idx] = np.sum(D)

    # Treatment termination time
    cumD = np.cumsum(D)
    temp_idx = np.where(cumD == D_total[idx])[0]
    D_sum_times[idx] = temp_idx[-1] + 1 if len(temp_idx) > 0 else N

    # Plot settings
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.55)

    # Plot real data
    ax.plot(t, y_data, 'k+', linewidth=3.9, markersize=8)

    # Drug concentration blocks
    for i in range(N):
        xi = t[i]
        if D[i] > 1e-3:
            ax.fill_between([xi - 0.5, xi + 0.5], 1.2, 1.35, color='b', alpha=D[i], lw=0)
        if F[i] > 1e-3:
            ax.fill_between([xi - 0.5, xi + 0.5], 1.35, 1.5, color='r', alpha=F[i], lw=0)

    # Fitting curve
    try:
        param_row = param_df[param_df['PatientID'] == pid].iloc[0]
        best_params = param_row.iloc[1:].values
        r1, r2, r3, r4 = best_params[0:4]
        d1, d2, d3, d4 = best_params[4:8]
        a1, a2, K = best_params[8:11]
        y0 = best_params[11:15]

        sol = solve_ivp(
            lambda tt, yy: model(tt, yy, r1, r2, r3, r4, d1, d2, d3, d4, a1, a2, K, D, F),
            t_span=(t[0], t[-1]), y0=y0, method='RK45', t_eval=t,
            rtol=1e-6, atol=1e-8, max_step=1
        )
        y_fit = np.sum(sol.y, axis=0)
        ax.plot(t, y_fit, 'b-', linewidth=5.4)

        # Calculate fitting errors
        mse = np.mean((y_data - y_fit) ** 2)
        mae = np.mean(np.abs(y_data - y_fit))
        patient_errors.append([pid, mse, mae])
        print(f"Patient {pid}  MSE: {mse:.4f}  MAE: {mae:.4f}")

    except Exception as e:
        print(f"Patient {pid} fitting failed: {e}")
        patient_errors.append([pid, np.nan, np.nan])

    # Termination dashed line
    et = D_sum_times[idx]
    if 1 <= et <= N:
        ax.axvline(et, color='k', ls='--', lw=1.5)
        ax.text(et, 1.42, str(et), fontsize=9, ha='center', bbox=dict(facecolor='w', pad=0.5))

    # Subplot title (top-left inside)
    ax.text(0.05, 0.92, f'ID {pid}', transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top', ha='left',
            zorder=999,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))

    # Hide y-tick labels for non-left columns
    if idx % ncols != 0:
        ax.set_yticklabels([])

# ===================== 9th subplot: Legend =====================
ax_leg = axes[8]
ax_leg.axis('off')
h1, = ax_leg.plot([], [], 'k+', linewidth=6.0, markersize=16)
h2, = ax_leg.plot([], [], 'b-', linewidth=10.8)
h3 = ax_leg.fill_between([], [], [], color='b', alpha=0.8)
h4 = ax_leg.fill_between([], [], [], color='r', alpha=0.8)

ax_leg.legend([h1, h2, h3, h4],
              ['Real Tumor Size', 'Fitted Curve', 'CPA Drug', 'LEU Drug'],
              loc='center',
              fontsize=20,
              handletextpad=1.0,
              columnspacing=2.0)

# ===================== Global Labels and Title =====================
fig.suptitle('Clinical Patient Data Fitting', fontsize=24, fontweight='extra bold', color='black', y=0.97)
fig.text(0.5, 0.03, 'Month', ha='center', fontsize=18, fontweight='bold')
fig.text(0.04, 0.5, 'Tumor Size', va='center', ha='center', rotation=90, fontsize=18, fontweight='bold')

# ===================== Treatment Summary Table =====================
D_table = pd.DataFrame({
    "PatientID": patients,
    "D_total": D_total,
    "Termination_Time": D_sum_times
})
print("\n==================== Patient Treatment Summary ====================")
print(D_table)

# ===================== Save Fitting Figure =====================
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "Clinical_Patient_Fitting.png")
plt.savefig(desktop_path, dpi=300, bbox_inches='tight', pad_inches=0)
print(f"\n✅ Saved to desktop: {desktop_path}")

plt.show()

# ===================== Plot Fitting Error Bar Chart =====================
error_df = pd.DataFrame(patient_errors, columns=['PatientID', 'MSE', 'MAE'])
print("\n==================== Patient Fitting Errors ====================")
print(error_df)

plt.figure(figsize=(12, 7))
x = np.arange(len(patients))
width = 0.35

plt.bar(x - width / 2, error_df['MSE'], width, label='Mean Squared Error (MSE)', color='#2E86AB', alpha=0.8)
plt.bar(x + width / 2, error_df['MAE'], width, label='Mean Absolute Error (MAE)', color='#A23B72', alpha=0.8)

plt.xlabel('Patient ID', fontsize=14, fontweight='bold')
plt.ylabel('Error Value', fontsize=14, fontweight='bold')
plt.title('Fitting Error Comparison for Tumor Size', fontsize=18, fontweight='bold', pad=20)
plt.xticks(x, error_df['PatientID'], fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12, loc='upper right')
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()

# Save error figure
error_save_path = os.path.join(os.path.expanduser("~"), "Desktop", "Patient_Fitting_Errors.png")
plt.savefig(error_save_path, dpi=300, bbox_inches='tight')
print(f"\n✅ Error bar chart saved to desktop: {error_save_path}")

plt.show()
