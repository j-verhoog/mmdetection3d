import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, HTMLWriter, writers
import os


SAVE_HTML = True
HTML_PATH = "/home/jolle/mmdet/visualisations/gradsplit/visualization_2d_2domains.html"
HTML_FPS = 2
SAVE_GIF = True
GIF_PATH = "/home/jolle/mmdet/visualisations/gradsplit/visualization_2d_2domains.gif"
GIF_FPS = 2



rounds = 10
global_between = 0.5
interval = 900
trail_alpha = 0.22

OVERSHOOT=False
if OVERSHOOT:
    step_start = 1.35
    step_end = 0.05
else:
    step_start = 0.5
    step_end = 0.3

STEP_CONSTANT = True
if STEP_CONSTANT:
    step_sizes = np.geomspace(step_start, step_end, rounds)
else:
    step_sizes = np.linspace(step_start, step_end, rounds)


# first set
a_opt = np.array([0.0, 0.2, 0.3, -0.1])
b_opt = np.array([1.2, -0.1, 0.1, 0.2])
start = np.array([-6.0, -4.0, 0.5, -0.3])

# Very different domains, global optimum still in the middle
a_opt = np.array([0.0, 2.0, 0.3, -0.1])
b_opt = np.array([2.0, 0.0, 0.1, 0.2])
start = np.array([0.0, 0.0, 0.5, -0.3])



# Very different domains, global optimum still in the middle
a_opt = np.array([1.0, 2.0, 0.3, -0.1])
b_opt = np.array([2.0, 0.0, 0.1, 0.2])
start = np.array([0.0, 0.0, 0.5, -0.3])









global_opt = (1 - global_between) * a_opt + global_between * b_opt

# # Very different domains, global optimum not in the middle
# a_opt = np.array([0.0, 2.0, 0.3, -0.1])
# b_opt = np.array([2.0, 0.0, 0.1, 0.2])
# start = np.array([0.0, 0.0, 0.5, -0.3])
# global_opt = (0.3) * a_opt + (0.9) * b_opt







dim = len(start)
assert len(a_opt) == dim and len(b_opt) == dim and len(global_opt) == dim

def normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n

def proj_onto_dir(v, unit_dir):
    return np.dot(v, unit_dir) * unit_dir

shared_points = [start.copy()]
memory_a = [np.zeros(dim)]
memory_b = [np.zeros(dim)]

raw_a_vectors = []
raw_b_vectors = []
shared_dirs = []
proj_a_vectors = []
proj_b_vectors = []
residual_a_vectors = []
residual_b_vectors = []
shared_updates = []
base_a_points = []
base_b_points = []

shared = start.copy()
mem_a = np.zeros(dim)
mem_b = np.zeros(dim)
prev_shared_dir = np.zeros(dim)
end_a_points = []
end_b_points = []

for _ in range(rounds):
    base_a = shared + mem_a
    base_b = shared + mem_b

    step_size = step_sizes[_]
    vec_a = step_size * (a_opt - base_a)
    vec_b = step_size * (b_opt - base_b)

    dir_candidate = vec_a + vec_b
    if np.linalg.norm(dir_candidate) < 1e-12:
        shared_dir = prev_shared_dir.copy()
    else:
        shared_dir = normalize(dir_candidate)
        prev_shared_dir = shared_dir.copy()

    proj_a = proj_onto_dir(vec_a, shared_dir)
    proj_b = proj_onto_dir(vec_b, shared_dir)

    res_a = vec_a - proj_a
    res_b = vec_b - proj_b

    shared_update = 0.5 * (proj_a + proj_b)

    raw_a_vectors.append(vec_a.copy())
    raw_b_vectors.append(vec_b.copy())
    shared_dirs.append(shared_dir.copy())
    proj_a_vectors.append(proj_a.copy())
    proj_b_vectors.append(proj_b.copy())
    residual_a_vectors.append(res_a.copy())
    residual_b_vectors.append(res_b.copy())
    shared_updates.append(shared_update.copy())
    base_a_points.append(base_a.copy())
    base_b_points.append(base_b.copy())
    end_a_points.append((base_a + vec_a).copy())
    end_b_points.append((base_b + vec_b).copy())

    mem_a = mem_a + res_a
    mem_b = mem_b + res_b
    shared = shared + shared_update

    memory_a.append(mem_a.copy())
    memory_b.append(mem_b.copy())
    shared_points.append(shared.copy())


def xy(v):
    return v[:2]

all_points_2d = np.vstack(
    [xy(a_opt), xy(b_opt), xy(global_opt), xy(start)]
    + [xy(p) for p in shared_points]
    + [xy(shared_points[i] + memory_a[i]) for i in range(len(shared_points))]
    + [xy(shared_points[i] + memory_b[i]) for i in range(len(shared_points))]
)

pad = 1.2
xmin, ymin = all_points_2d.min(axis=0) - pad
xmax, ymax = all_points_2d.max(axis=0) + pad

fig, ax = plt.subplots(figsize=(9, 9))

def draw_arrow(p, v, color, lw=2.5, alpha=1.0, z=3):
    p2 = xy(p)
    v2 = xy(v)
    ax.annotate(
        "",
        xy=p2 + v2,
        xytext=p2,
        arrowprops=dict(arrowstyle="->", lw=lw, color=color, alpha=alpha),
        zorder=z
    )

def redraw(frame):
    ax.clear()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)

    ax.scatter(*xy(a_opt), s=110, color="tab:blue", zorder=5)
    ax.scatter(*xy(b_opt), s=110, color="tab:orange", zorder=5)
    ax.scatter(*xy(global_opt), s=140, color="black", marker="*", zorder=6)
    ax.scatter(*xy(start), s=110, color="tab:green", zorder=5)

    ax.text(*(xy(a_opt) + np.array([0.12, 0.08])), "A optimal", fontsize=11)
    ax.text(*(xy(b_opt) + np.array([0.12, 0.08])), "B optimal", fontsize=11)
    ax.text(*(xy(global_opt) + np.array([0.12, 0.08])), "global optimal", fontsize=11)
    ax.text(*(xy(start) + np.array([0.12, 0.08])), "start", fontsize=11)

    completed_rounds = frame // 3
    phase = frame % 3

    shared_hist = np.array([xy(p) for p in shared_points[:completed_rounds + 1]])
    if len(shared_hist) > 1:
        ax.plot(shared_hist[:, 0], shared_hist[:, 1], color="black", lw=2, alpha=0.9, zorder=2)

    branch_a_hist = np.array([
        xy(shared_points[i] + memory_a[i])
        for i in range(min(completed_rounds + 1, len(shared_points)))
    ])
    branch_b_hist = np.array([
        xy(shared_points[i] + memory_b[i])
        for i in range(min(completed_rounds + 1, len(shared_points)))
    ])

    if len(branch_a_hist) > 1:
        ax.plot(branch_a_hist[:, 0], branch_a_hist[:, 1], color="tab:blue", lw=1.8, alpha=0.9, zorder=2)
    if len(branch_b_hist) > 1:
        ax.plot(branch_b_hist[:, 0], branch_b_hist[:, 1], color="tab:orange", lw=1.8, alpha=0.9, zorder=2)

    for i in range(completed_rounds):
        s = shared_points[i]
        ba = base_a_points[i]
        bb = base_b_points[i]

        ax.plot([xy(s)[0], xy(ba)[0]], [xy(s)[1], xy(ba)[1]], color="tab:blue", alpha=trail_alpha, lw=1.5)
        ax.plot([xy(s)[0], xy(bb)[0]], [xy(s)[1], xy(bb)[1]], color="tab:orange", alpha=trail_alpha, lw=1.5)

        draw_arrow(ba, raw_a_vectors[i], "tab:blue", lw=1.6, alpha=trail_alpha)
        draw_arrow(bb, raw_b_vectors[i], "tab:orange", lw=1.6, alpha=trail_alpha)

        draw_arrow(s, shared_updates[i], "black", lw=2.0, alpha=trail_alpha)
        draw_arrow(s, residual_a_vectors[i], "tab:blue", lw=1.3, alpha=trail_alpha)
        draw_arrow(s, residual_b_vectors[i], "tab:orange", lw=1.3, alpha=trail_alpha)

    if completed_rounds < rounds:
        i = completed_rounds
        shared_now = shared_points[i]
        base_a = base_a_points[i]
        base_b = base_b_points[i]

        ax.scatter(*xy(shared_now), s=70, color="black", zorder=6)
        ax.scatter(*xy(base_a), s=55, color="tab:blue", zorder=6)
        ax.scatter(*xy(base_b), s=55, color="tab:orange", zorder=6)

        ax.plot([xy(shared_now)[0], xy(base_a)[0]], [xy(shared_now)[1], xy(base_a)[1]], color="tab:blue", alpha=0.55, lw=1.8)
        ax.plot([xy(shared_now)[0], xy(base_b)[0]], [xy(shared_now)[1], xy(base_b)[1]], color="tab:orange", alpha=0.55, lw=1.8)

        if phase == 0:
            draw_arrow(base_a, raw_a_vectors[i], "tab:blue", lw=2.8, alpha=1.0, z=8)
            draw_arrow(base_b, raw_b_vectors[i], "tab:orange", lw=2.8, alpha=1.0, z=8)
            ax.set_title(f"Round {i+1}: raw updates toward A and B", fontsize=14)

        elif phase == 1:
            draw_arrow(base_a, proj_a_vectors[i], "black", lw=2.8, alpha=1.0, z=9)
            draw_arrow(base_b, proj_b_vectors[i], "black", lw=2.8, alpha=1.0, z=9)
            draw_arrow(base_a, residual_a_vectors[i], "tab:blue", lw=2.2, alpha=1.0, z=9)
            draw_arrow(base_b, residual_b_vectors[i], "tab:orange", lw=2.2, alpha=1.0, z=9)
            ax.set_title(f"Round {i+1}: project onto shared invariant direction", fontsize=14)

        else:
            next_shared = shared_points[i + 1]
            next_a = next_shared + memory_a[i + 1]
            next_b = next_shared + memory_b[i + 1]

            ax.scatter(*xy(next_shared), s=80, color="black", zorder=8)
            ax.scatter(*xy(next_a), s=60, color="tab:blue", zorder=8)
            ax.scatter(*xy(next_b), s=60, color="tab:orange", zorder=8)

            ax.plot([xy(shared_now)[0], xy(next_shared)[0]], [xy(shared_now)[1], xy(next_shared)[1]], color="black", lw=2.5)
            ax.plot([xy(next_shared)[0], xy(next_a)[0]], [xy(next_shared)[1], xy(next_a)[1]], color="tab:blue", lw=2.0)
            ax.plot([xy(next_shared)[0], xy(next_b)[0]], [xy(next_shared)[1], xy(next_b)[1]], color="tab:orange", lw=2.0)

            ax.set_title(f"Round {i+1}: residual memory carried into next round", fontsize=14)
    else:
        ax.set_title("Finished: one full run", fontsize=14)
        current_done = min(completed_rounds, rounds - 1)
        ax.scatter(*xy(end_a_points[current_done]), s=70, color="tab:blue", marker="x", zorder=7)
        ax.scatter(*xy(end_b_points[current_done]), s=70, color="tab:orange", marker="x", zorder=7)
        ax.text(*(xy(end_a_points[current_done]) + np.array([0.08, 0.08])), "A end", fontsize=10, color="black")
        ax.text(*(xy(end_b_points[current_done]) + np.array([0.08, 0.08])), "B end", fontsize=10, color="black")



ani = FuncAnimation(fig, redraw, frames=rounds * 3 + 1, interval=interval, repeat=False)

if SAVE_GIF:
    os.makedirs(os.path.dirname(GIF_PATH), exist_ok=True)
    ani.save(GIF_PATH, writer="pillow", fps=GIF_FPS)
    print(f"Saved GIF to {GIF_PATH}")


if SAVE_HTML:
    os.makedirs(os.path.dirname(HTML_PATH), exist_ok=True)
    ani.save(HTML_PATH, writer=HTMLWriter(fps=HTML_FPS))
    print(f"Saved HTML animation to {HTML_PATH}")


plt.show()
