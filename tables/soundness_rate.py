import matplotlib.pyplot as plt

models = ["DeepSeek-V2-Lite", "Llama-3.3-70B-Instruct", "Gpt-4o"]
x = list(range(len(models)))

# Generation Time (seconds)
generation_times = [1325, 163, 0]  

# Soundness Rate
soundness_rates = [8/14, 7/14, 4/14]  
soundness_rates = [round(float(rate), 2) for rate in soundness_rates]




fig, ax1 = plt.subplots(figsize=(9, 5))


ax2 = ax1.twinx()
ax2.set_zorder(1)  
ax1.set_zorder(2) 
ax1.patch.set_visible(False) 

bar_colors = ["#f5b2b5", "#f28485", "#eb5a5e"] # https://colordrop.io/palette/34293
bars = ax2.bar(x, generation_times, color=bar_colors, width=0.4, alpha=0.8, label="Generation Time")
ax2.set_ylabel("Total Generation Time (s)", fontsize=12, color="#1d3557")
ax2.tick_params(axis='y', labelcolor="#1d3557")

for i, bar in enumerate(bars):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, height + 3, f"{height}s", ha='center', fontsize=10, color="#1d3557")


line_color = "#550e1e" # https://colordrop.io/palette/34293
ax1.plot(x, soundness_rates, linestyle='-', color=line_color, linewidth=2, label="Soundness Rate",zorder=10)
ax1.scatter(x, soundness_rates, color='black', s=60, zorder=5)
ax1.set_ylabel("Soundness Rate", fontsize=12, color=line_color)
ax1.set_ylim(0, 1)
ax1.tick_params(axis='y', labelcolor=line_color)
for i in range(len(models)):
    ax1.text(x[i], soundness_rates[i] + 0.03, f"{soundness_rates[i]:.2f}", ha='center', fontsize=10, color=line_color)


plt.xticks(x, models, fontsize=11)
ax1.set_title("Model Comparison: Soundness Rate & Generation Time", fontsize=14)
ax1.grid(axis='y', linestyle='--', alpha=0.6)
fig.tight_layout()

plt.savefig("pics/soundness_rate.png", dpi=300)
plt.show()
