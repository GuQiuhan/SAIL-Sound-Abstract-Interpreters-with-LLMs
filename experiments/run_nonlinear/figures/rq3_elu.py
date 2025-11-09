import matplotlib.pyplot as plt
import numpy as np


def elu(x, alpha=1.0):
    """ELU(x) = x (x > 0); alpha * (exp(x) - 1) (x <= 0)"""
    x = np.array(x, dtype=float)
    y = np.copy(x)
    y[x < 0] = alpha * (np.exp(x[x < 0]) - 1.0)
    return y if y.size > 1 else float(y)


x = np.linspace(-6, 6, 2000)
y = elu(x)
l, u = -4.0, 3.0
yl, yu = elu(l), elu(u)

# DeepPoly
lower_neg = -1.0
slope = (yu - yl) / (u - l)
intercept = yl - slope * l
sec = slope * x + intercept


x_left = (-1.0 - intercept) / slope

x_right = intercept / (1.0 - slope)
y0 = intercept  # sec(0)


col_fn = "#c28e00"
col_low = "#2b6cb0"
col_up = "#f07f13"
col_axis0 = "#e0a500"
col_fill_L = "#f8ebca"
col_fill_R = "#e7f5f8"

plt.figure(figsize=(7, 5))

# shade
plt.fill(
    [0.0, 0.0, x_left],
    [-1.0, y0, -1.0],
    color=col_fill_L,
    alpha=0.9,
    label="Bounds (x ≤ 0)",
)

plt.fill(
    [0.0, 0.0, x_right],
    [0.0, y0, x_right],
    color=col_fill_R,
    alpha=0.9,
    label="Bounds (x ≥ 0)",
)


plt.plot(x, y, color=col_fn, linewidth=2, label="ELU(x)")
plt.plot(
    x[x < 0],
    np.full_like(x[x < 0], lower_neg),
    "--",
    color=col_low,
    linewidth=1.3,
    label="Lower (x<0): y=-1",
)
plt.plot(
    x[x >= 0], x[x >= 0], "--", color=col_low, linewidth=1.3, label="Lower (x≥0): y=x"
)
plt.plot(x, sec, "--", color=col_up, linewidth=1.5, label="Upper: sec(l,u)")


plt.scatter([l, u], [yl, yu], color=col_up, zorder=5)
plt.text(l - 0.3, yl - 0.3, "(l, ELU(l))", fontsize=10)
plt.text(u + 0.15, yu - 0.2, "(u, ELU(u))", fontsize=10)

# x=0
plt.axvline(0, color=col_axis0, linestyle="--", linewidth=1)

plt.text(
    -5.8,
    4.1,
    "Mixed case (prev[l] < 0 < prev[u]):\nUpper: sec((l, elu(l)),(u, elu(u)).\nLower: y=x.",
    fontsize=10,
    color="black",
)

# gird
plt.xlabel("prev (x)")
plt.ylabel("curr")
plt.legend(loc="lower right", fontsize=9)
plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
plt.xlim(-6, 6)
plt.ylim(-2, 5)
plt.tight_layout()

plt.savefig("rq3_elu.pdf")
plt.show()
