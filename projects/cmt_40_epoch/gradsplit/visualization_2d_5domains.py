import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, HTMLWriter, writers
import os

rounds = 10
interval = 900
trail_alpha = 0.18
eps = 1e-12
SAVE_GIF = False
GIF_PATH = "/home/jolle/mmdet/visualisations/gradsplit/visualization_2d_5domains.gif"
GIF_FPS = 2
SAVE_MP4 = False
MP4_PATH = "/home/jolle/mmdet/visualisations/gradsplit/visualization_2d_5domains.mp4"
MP4_FPS = 2
SAVE_HTML = False
HTML_PATH = "/home/jolle/mmdet/visualisations/gradsplit/visualization_2d_5domains.html"
HTML_FPS = 2

dim = 500
num_domains = 5
xy_idx = (3, 5)

# take random to visisualize
xy_idx = (np.random.randint(0, dim), np.random.randint(0, dim))

domain_names = ["A", "B", "C", "D", "E"]
domain_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

start = np.array([0.0, -1.0, 0.4, -0.2, 0.3, -0.1], dtype=float)



domain_optima = [
    np.array([0.15, 0.05, 0.20, -0.10, 0.00, 0.10], dtype=float),
    np.array([0.45, -0.05, 0.10, 0.20, -0.10, 0.00], dtype=float),
    np.array([0.75, 0.10, -0.05, 0.15, 0.20, -0.15], dtype=float),
    np.array([1.05, -0.08, 0.00, -0.05, 0.15, 0.20], dtype=float),
    np.array([1.30, 0.02, 0.15, 0.05, -0.05, 0.25], dtype=float),
]

# Generate N-dimensional domain optima
N = dim  # Set desired number of dimensions
np.random.seed(42)
domain_optima = [
    np.random.uniform(-0.5, 1.5, N).astype(float)
    for _ in range(num_domains)
]
start = np.random.uniform(-0.5, 0.5, N).astype(float)

SIZE_A = 265
SIZE_B = 125
SIZE_C = 226
SIZE_D = 71
SIZE_E = 13

LR = 0.01

informativeness = np.array([
    SIZE_A / (SIZE_A + SIZE_B + SIZE_C + SIZE_D + SIZE_E),
    SIZE_B / (SIZE_A + SIZE_B + SIZE_C + SIZE_D + SIZE_E),
    SIZE_C / (SIZE_A + SIZE_B + SIZE_C + SIZE_D + SIZE_E),
    SIZE_D / (SIZE_A + SIZE_B + SIZE_C + SIZE_D + SIZE_E),
    SIZE_E / (SIZE_A + SIZE_B + SIZE_C + SIZE_D + SIZE_E),
], dtype=float)

global_mix = np.array([0.15, 0.20, 0.25, 0.20, 0.20], dtype=float)
global_mix = global_mix / global_mix.sum()
global_opt = sum(w * p for w, p in zip(global_mix, domain_optima))

step_sizes = []
total_steps = rounds
peak_step = int(total_steps * 2 / 5)
start_lr = 0.002 # nice figure
start_lr = 0.005 # also ncie, shows overshooting
peak_lr = start_lr * 4.0
end_lr = start_lr * 0.5

for size in [SIZE_A, SIZE_B, SIZE_C, SIZE_D, SIZE_E]:
    lr_schedule = np.zeros(rounds)
    # Ascending phase: 0.0 to peak
    for i in range(peak_step):
        lr_schedule[i] = start_lr + (peak_lr - start_lr) * (i / peak_step)
    # Descending phase: peak to end
    for i in range(peak_step, rounds):
        lr_schedule[i] = peak_lr - (peak_lr - end_lr) * ((i - peak_step) / (rounds - peak_step))
    step_sizes.append(lr_schedule * size)

assert len(domain_optima) == num_domains
assert len(informativeness) == num_domains
assert len(step_sizes) == num_domains
assert len(start) == dim
assert len(global_mix) == num_domains
for arr in domain_optima:
    assert len(arr) == dim
for arr in step_sizes:
    assert len(arr) >= rounds

def xy(v):
    return np.array([v[xy_idx[0]], v[xy_idx[1]]])

def normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n

def proj_onto_dir(v, unit_dir):
    return np.dot(v, unit_dir) * unit_dir

shared_points = [start.copy()]
memory_points = [np.zeros(dim) for _ in range(num_domains)]
memory_history = [[np.zeros(dim)] for _ in range(num_domains)]

base_points_history = [[] for _ in range(num_domains)]
end_points_history = [[] for _ in range(num_domains)]
raw_updates_history = [[] for _ in range(num_domains)]
weighted_updates_history = [[] for _ in range(num_domains)]
proj_updates_history = [[] for _ in range(num_domains)]
residual_updates_history = [[] for _ in range(num_domains)]

shared_dirs = []
shared_updates = []

shared = start.copy()
memories = [np.zeros(dim) for _ in range(num_domains)]
prev_shared_dir = normalize(global_opt - start)

for r in range(rounds):
    base_points = []
    raw_updates = []
    weighted_updates = []

    for d in range(num_domains):
        base_d = shared + memories[d]
        step_d = step_sizes[d][r]
        raw_update_d = step_d * (domain_optima[d] - base_d)
        weighted_update_d = informativeness[d] * raw_update_d

        base_points.append(base_d.copy())
        raw_updates.append(raw_update_d.copy())
        weighted_updates.append(weighted_update_d.copy())

        base_points_history[d].append(base_d.copy())
        raw_updates_history[d].append(raw_update_d.copy())
        weighted_updates_history[d].append(weighted_update_d.copy())
        end_points_history[d].append((base_d + weighted_update_d).copy())

    dir_candidate = np.sum(weighted_updates, axis=0)
    if np.linalg.norm(dir_candidate) < eps:
        shared_dir = prev_shared_dir.copy()
    else:
        shared_dir = normalize(dir_candidate)
        prev_shared_dir = shared_dir.copy()

    shared_dirs.append(shared_dir.copy())

    proj_updates = []
    residual_updates = []
    for d in range(num_domains):
        proj_d = proj_onto_dir(weighted_updates[d], shared_dir)
        resid_d = weighted_updates[d] - proj_d

        proj_updates.append(proj_d.copy())
        residual_updates.append(resid_d.copy())

        proj_updates_history[d].append(proj_d.copy())
        residual_updates_history[d].append(resid_d.copy())

    shared_update = np.sum(proj_updates, axis=0) / num_domains
    shared_updates.append(shared_update.copy())

    shared = shared + shared_update
    shared_points.append(shared.copy())

    for d in range(num_domains):
        memories[d] = memories[d] + residual_updates[d]
        memory_history[d].append(memories[d].copy())

all_points_2d = [xy(start), xy(global_opt)]
for p in domain_optima:
    all_points_2d.append(xy(p))
for p in shared_points:
    all_points_2d.append(xy(p))
for d in range(num_domains):
    for r in range(len(memory_history[d])):
        all_points_2d.append(xy(shared_points[r] + memory_history[d][r]))
    for p in end_points_history[d]:
        all_points_2d.append(xy(p))

all_points_2d = np.vstack(all_points_2d)
pad = 1.2
xmin, ymin = all_points_2d.min(axis=0) - pad
xmax, ymax = all_points_2d.max(axis=0) + pad

fig, ax = plt.subplots(figsize=(10, 10))

def draw_arrow(p, v, color, lw=2.2, alpha=1.0, z=3, linestyle="-"):
    p2 = xy(p)
    v2 = xy(v)
    ax.annotate(
        "",
        xy=p2 + v2,
        xytext=p2,
        arrowprops=dict(arrowstyle="->", lw=lw, color=color, alpha=alpha, linestyle=linestyle),
        zorder=z
    )

def redraw(frame):
    ax.clear()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)

    ax.scatter(*xy(start), s=120, color="black", marker="s", zorder=6)
    ax.text(*(xy(start) + np.array([0.10, 0.08])), "start", fontsize=10)

    ax.scatter(*xy(global_opt), s=180, color="black", marker="*", zorder=7)
    ax.text(*(xy(global_opt) + np.array([0.10, 0.08])), "global optimum", fontsize=10)

    for d in range(num_domains):
        ax.scatter(*xy(domain_optima[d]), s=100, color=domain_colors[d], zorder=6)
        ax.text(*(xy(domain_optima[d]) + np.array([0.10, 0.08])), f"{domain_names[d]} optimum", fontsize=10, color=domain_colors[d])

    completed_rounds = frame // 3
    phase = frame % 3

    shared_hist = np.array([xy(p) for p in shared_points[:completed_rounds + 1]])
    if len(shared_hist) > 1:
        ax.plot(shared_hist[:, 0], shared_hist[:, 1], color="black", lw=2.5, alpha=0.9, zorder=2)

    for d in range(num_domains):
        branch_hist = np.array([
            xy(shared_points[i] + memory_history[d][i])
            for i in range(min(completed_rounds + 1, len(memory_history[d])))
        ])
        if len(branch_hist) > 1:
            ax.plot(branch_hist[:, 0], branch_hist[:, 1], color=domain_colors[d], lw=1.6, alpha=0.9, zorder=2)

    for r in range(completed_rounds):
        s = shared_points[r]

        draw_arrow(s, shared_updates[r], "black", lw=2.0, alpha=trail_alpha, z=2)

        for d in range(num_domains):
            base_d = base_points_history[d][r]
            resid_d = residual_updates_history[d][r]

            ax.plot([xy(s)[0], xy(base_d)[0]], [xy(s)[1], xy(base_d)[1]], color=domain_colors[d], alpha=trail_alpha, lw=1.2)
            draw_arrow(base_d, weighted_updates_history[d][r], domain_colors[d], lw=1.4, alpha=trail_alpha, z=2)
            draw_arrow(s, resid_d, domain_colors[d], lw=1.0, alpha=trail_alpha, z=2, linestyle="--")

    current_idx = min(completed_rounds, rounds - 1)

    if completed_rounds < rounds:
        s = shared_points[current_idx]
        ax.scatter(*xy(s), s=90, color="black", zorder=8)

        for d in range(num_domains):
            base_d = base_points_history[d][current_idx]
            end_d = end_points_history[d][current_idx]

            ax.scatter(*xy(base_d), s=55, color=domain_colors[d], zorder=8)
            ax.scatter(*xy(end_d), s=65, color=domain_colors[d], marker="x", zorder=9)

            ax.plot([xy(s)[0], xy(base_d)[0]], [xy(s)[1], xy(base_d)[1]], color=domain_colors[d], alpha=0.55, lw=1.5)

        if phase == 0:
            for d in range(num_domains):
                draw_arrow(
                    base_points_history[d][current_idx],
                    raw_updates_history[d][current_idx],
                    domain_colors[d],
                    lw=2.5,
                    alpha=1.0,
                    z=9
                )
            ax.set_title(f"Round {current_idx + 1}: raw domain updates", fontsize=14)

        elif phase == 1:
            for d in range(num_domains):
                draw_arrow(
                    base_points_history[d][current_idx],
                    weighted_updates_history[d][current_idx],
                    domain_colors[d],
                    lw=2.6,
                    alpha=1.0,
                    z=9
                )
            ax.set_title(f"Round {current_idx + 1}: raw updates × informativeness", fontsize=14)

        else:
            draw_arrow(shared_points[current_idx], shared_updates[current_idx], "black", lw=3.0, alpha=1.0, z=10)

            for d in range(num_domains):
                draw_arrow(
                    base_points_history[d][current_idx],
                    proj_updates_history[d][current_idx],
                    "black",
                    lw=2.3,
                    alpha=1.0,
                    z=9
                )
                draw_arrow(
                    shared_points[current_idx],
                    residual_updates_history[d][current_idx],
                    domain_colors[d],
                    lw=2.0,
                    alpha=1.0,
                    z=9,
                    linestyle="--"
                )

            ax.set_title(f"Round {current_idx + 1}: shared projection + private residual memory", fontsize=14)

    else:
        final_shared = shared_points[-1]
        ax.scatter(*xy(final_shared), s=110, color="black", zorder=10)

        for d in range(num_domains):
            final_branch = shared_points[-1] + memory_history[d][-1]
            ax.scatter(*xy(final_branch), s=70, color=domain_colors[d], zorder=10)

        ax.set_title("Finished: one full run", fontsize=14)

    legend_lines = [
        f"{domain_names[d]}: step[0]={step_sizes[d][0]:.2f}, info={informativeness[d]:.2f}"
        for d in range(num_domains)
    ]
    ax.text(
        0.02,
        0.98,
        "\n".join(legend_lines),
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

ani = FuncAnimation(fig, redraw, frames=rounds * 3 + 1, interval=interval, repeat=False)

if SAVE_GIF:
    os.makedirs(os.path.dirname(GIF_PATH), exist_ok=True)
    ani.save(GIF_PATH, writer="pillow", fps=GIF_FPS)
    print(f"Saved GIF to {GIF_PATH}")

if SAVE_MP4:
    if writers.is_available("ffmpeg"):
        os.makedirs(os.path.dirname(MP4_PATH), exist_ok=True)
        ani.save(MP4_PATH, writer="ffmpeg", fps=MP4_FPS)
        print(f"Saved MP4 to {MP4_PATH}")
    else:
        print("Skipped MP4: ffmpeg not available in this environment.")

if SAVE_HTML:
    os.makedirs(os.path.dirname(HTML_PATH), exist_ok=True)
    ani.save(HTML_PATH, writer=HTMLWriter(fps=HTML_FPS))
    print(f"Saved HTML animation to {HTML_PATH}")

plt.show()