import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


x = np.linspace(-6, 6, 2000)
mask_neg = x <= 0
mask_pos = x >= 0


L_neg = 0.25 * x + 0.5
U_neg = np.full_like(x, 0.5)
L_pos = np.full_like(x, 0.5)
U_pos = 0.25 * x + 0.5


COL_FN = "#c28e00"  # σ(x)
COL_BLUE_D = "#2b6cb0"  # L/U = 0.25x + 0.5
COL_FILL_L = "#f8ebca"  # shade
COL_FILL_R = "#e7f5f8"  # shade
COL_U_LEFT = "#1ca39a"  # U=0.5
COL_L_RIGHT = "#e0a500"  #  L=0.5
COL_AXIS0 = "#e0a500"  # x=0

plt.figure(figsize=(7, 5))

# shade
plt.fill_between(
    x[mask_neg],
    L_neg[mask_neg],
    U_neg[mask_neg],
    color=COL_FILL_L,
    alpha=0.8,
    label="Bounds (x ≤ 0)",
)
plt.fill_between(
    x[mask_pos],
    L_pos[mask_pos],
    U_pos[mask_pos],
    color=COL_FILL_R,
    alpha=0.8,
    label="Bounds (x ≥ 0)",
)


plt.plot(x, sigmoid(x), color=COL_FN, linewidth=2, label="σ(x)")

plt.plot(
    x[mask_neg],
    L_neg[mask_neg],
    "--",
    color=COL_BLUE_D,
    linewidth=1.3,
    label="Lower (x ≤ 0): y=0.25x+0.5",
)
plt.plot(
    x[mask_pos],
    U_pos[mask_pos],
    "--",
    color=COL_BLUE_D,
    linewidth=1.3,
    label="Upper (x ≥ 0): y=0.25x+0.5",
)

# U=0.5 L=0.5
plt.plot(
    x[mask_neg],
    U_neg[mask_neg],
    ":",
    color=COL_U_LEFT,
    linewidth=1.6,
    label="Upper (x ≤ 0): y=0.5",
)
plt.plot(
    x[mask_pos],
    L_pos[mask_pos],
    ":",
    color=COL_L_RIGHT,
    linewidth=1.6,
    label="Lower (x ≥ 0): y=0.5",
)

# x=0
plt.axvline(0, color=COL_AXIS0, linestyle="--", linewidth=1)


plt.text(
    -5.8,
    0.93,
    "Mixed case (prev[l] < 0 < prev[u]):\n Lower: y=0, Upper: y=1.",
    fontsize=10,
    color="black",
)

# grid
plt.xlabel("prev (x)")
plt.ylabel("curr")
# plt.title("Sigmoid with Piecewise DeepPoly Bounds")
plt.legend(loc="lower right", fontsize=9)
plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
plt.xlim(-6, 6)
plt.ylim(-0.05, 1.05)
plt.tight_layout()


plt.savefig("rq3_sigmoid.pdf")
plt.show()
