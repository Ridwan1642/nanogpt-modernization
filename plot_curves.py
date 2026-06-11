"""Parse results.md and plot validation loss curves for all ablation runs.

Usage: python plot_curves.py [path/to/results.md]
Writes: results/val_curves.png
"""

import re
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else 'results.md'
with open(path, encoding='utf-8') as f:
    text = f.read()

runs = {}   # name -> (steps, val_losses)
current = None
for line in text.splitlines():
    m = re.match(r'### run: (\w+)', line)
    if m:
        current = m.group(1)
        runs[current] = ([], [])
        continue
    m = re.match(r'\s*step (\d+): train [\d.]+, val ([\d.]+)', line)
    if m and current:
        runs[current][0].append(int(m.group(1)))
        runs[current][1].append(float(m.group(2)))

colors = {'baseline': '#888888', 'rmsnorm': '#1f77b4', 'swiglu': '#d62728',
          'rope': '#2ca02c', 'all': '#9467bd'}

fig, ax = plt.subplots(figsize=(8, 5))
for name, (steps, vals) in runs.items():
    ax.plot(steps, vals, marker='o', markersize=3.5, linewidth=1.8,
            label=name, color=colors.get(name))
    # mark each run's best val
    b = min(range(len(vals)), key=lambda i: vals[i])
    ax.scatter([steps[b]], [vals[b]], s=70, facecolors='none',
               edgecolors=colors.get(name), linewidths=1.8, zorder=5)

ax.set_xlabel('training step')
ax.set_ylabel('validation loss')
ax.set_title('Validation loss by variant (circles = best val per run)')
ax.set_ylim(1.40, 2.00)   # crops the step-0 point (~4.2) to show the structure
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()

os.makedirs('results', exist_ok=True)
out = 'results/val_curves.png'
fig.savefig(out, dpi=150)
print(f'wrote {out} ({len(runs)} runs)')
