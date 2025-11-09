from math import erf

import matplotlib.pyplot as plt
import numpy as np


def gelu(x):
    return 0.5 * x * (1.0 + np.vectorize(erf)(x / np.sqrt(2.0)))


x = np.linspace(-6, 6, 2000)

# DeepPoly
lower = 0.5 * x
upper_neg = np.zeros_like(x)
upper_pos = x

mask_neg = x <= 0
mask_pos = x >= 0


col_func = "#c28e00"  # GELU(x)
col_lower = "#2b6cb0"  # L
col_upper = "#cc8800"  # U
col_axis0 = "#e0a500"  # x=0
col_fill_neg = "#f8ebca"  # shade x<=0
col_fill_pos = "#e7f5f8"  # shade x>=0


plt.figure(figsize=(7, 5))

# shade
plt.fill_between(
    x[mask_neg],
    lower[mask_neg],
    upper_neg[mask_neg],
    color=col_fill_neg,
    alpha=0.8,
    label="Bounds (x ≤ 0)",
)
plt.fill_between(
    x[mask_pos],
    lower[mask_pos],
    upper_pos[mask_pos],
    color=col_fill_pos,
    alpha=0.8,
    label="Bounds (x ≥ 0)",
)

# GELU
plt.plot(x, gelu(x), color=col_func, linewidth=2, label="GELU(x)")

# U, L
plt.plot(x, lower, "--", color=col_lower, linewidth=1.3, label="Lower: y=0.5x")
plt.plot(
    x[mask_neg],
    upper_neg[mask_neg],
    ":",
    color=col_upper,
    linewidth=1.2,
    label="Upper (x ≤ 0): y=0",
)
plt.plot(
    x[mask_pos],
    upper_pos[mask_pos],
    "--",
    color=col_upper,
    linewidth=1.3,
    label="Upper (x ≥ 0: y=x",
)

# x=0
plt.axvline(0, color=col_axis0, linestyle="--", linewidth=1)

plt.text(
    -5.8,
    4.95,
    "Mixed case (prev[l] < 0 < prev[u]):\nUpper: sec((l, gelu(l)),(u, gelu(u)).\nLower: y=0.5x.",
    fontsize=10,
    color="black",
)


# grid
plt.xlabel("prev (x)")
plt.ylabel("curr")
# plt.title("DeepPoly Transformer for GELU (Sigmoid-style Colors)")
plt.legend(loc="lower right", fontsize=9)
plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
plt.xlim(-6, 6)
plt.ylim(-2, 6)
plt.tight_layout()

# 保存
plt.savefig("rq3_gelu.pdf")
plt.show()
