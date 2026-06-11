import numpy as np
import matplotlib.pyplot as plt

data   = np.loadtxt('rosenstein_ada')
i_vals = data[:, 0]
S      = data[:, 1]

# i=0 je artefakt — vynecháme
i_ = i_vals[1:]
S_ = S[1:]

fits = {
    'i=1..4'  : (slice(0, 4),  'red'),
    'i=1..7'  : (slice(0, 7),  'orange'),
    'i=1..10' : (slice(0, 10), 'green'),
}

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(i_vals, S, 'o-', color='purple', label='S(i)', zorder=5)

for label, (sl, color) in fits.items():
    x = i_[sl]
    y = S_[sl]
    slope, intercept = np.polyfit(x, y, 1)
    ax.plot(x, slope * x + intercept, '--', color=color,
            label=f'{label}: λ₁ ≈ {slope:.4f}')
    print(f"{label}: slope = {slope:.4f}")

ax.set_xlabel('Časový krok i')
ax.set_ylabel('<ln(divergence)>')
ax.set_title('Rosenstein – ADAUSD (i=0 vyloučen jako artefakt)')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig('rosenstein_ada_fits.png', dpi=150)
plt.show()
