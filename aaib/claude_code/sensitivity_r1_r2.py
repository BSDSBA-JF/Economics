"""
Sensitivity Analysis: r1 (live agent) x r2 (chatbot) Routing Policy
=====================================================================
Sweeps r1 and r2 across a grid and extracts the outsourcing threshold x*
for both policy classes:
  - Omega_e : a priori  (routing decision at call arrival)
  - Omega_l : a posteriori (routing decision at FIL epoch)

For each (r1, r2) pair, the table shows:
  - "always accept"    : live agent always preferred, regardless of queue length
  - "always outsource" : chatbot always preferred
  - "accept x < x*, outsource x >= x*" : threshold state where policy flips

Output: printed tables + saved PNG figure.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings("ignore")

# ── Fixed parameters ────────────────────────────────────────────────────────
lam   = 0.30   # class-1 arrival rate
s     = 2      # number of servers
mu    = 0.35   # per-server service rate
gamma = 0.20   # FIL epoch rate
L     = 2.0    # outsourcing cost (Lagrange multiplier)
omega = 0.30   # waiting-time penalty weight

# ── Sweep grid ───────────────────────────────────────────────────────────────
R1_VALS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]   # live agent reward
R2_VALS = [10, 8, 6, 4, 2, 1]               # chatbot reward  (high -> low for table rows)

# ── VI settings ──────────────────────────────────────────────────────────────
X_MAX   = 12     # queue truncation
EPS     = 1e-3   # convergence tolerance
MAX_K   = 300    # max iterations


# ══════════════════════════════════════════════════════════════════════════
#  Omega_e  —  A PRIORI value iteration
# ══════════════════════════════════════════════════════════════════════════
def vi_omega_e(lam, s, mu, r1, r2, L, omega,
               X_MAX=X_MAX, EPS=EPS, MAX_K=MAX_K):
    """
    Value iteration for Omega_e (equation 3).

    Returns
    -------
    policy : dict  {state x -> 'accept' | 'outsource'}
    thresh : int or None
        Lowest x >= 0 where action == 'outsource'.
        None  => always accept.
        0     => always outsource.
    V      : np.ndarray  final value function
    """
    states = list(range(-s, X_MAX + 1))
    N = len(states)

    # Uniformization: lambda + s*mu = 1
    scale = lam + s * mu
    le, me = lam / scale, mu / scale

    V = np.zeros(N)

    for _ in range(MAX_K):
        Vn = np.zeros(N)
        for i, x in enumerate(states):
            busy   = min(s, s + x)
            mu_eff = busy * me

            # U_k: class-1 arrival decision
            if x < 0:                          # vacant server -> accept, earn r1
                Uk = V[i] + r1
            else:                              # all busy: queue vs outsource
                v_queue   = V[i + 1] if i + 1 < N else V[i]
                Uk        = max(v_queue, V[i] - L)

            # W_k(x-1): class-2 initiation at previous state
            if x - 1 >= -s:
                j = i - 1
                if states[j] < 0:             # vacant: idle or initiate class-2
                    Wk_prev = max(V[j], (V[j + 1] if j + 1 < N else V[j]) + r2)
                else:
                    Wk_prev = V[j]
            else:
                Wk_prev = 0.0

            # Waiting-adjusted r1 for class-1 served from queue state x > 0
            r1_adj = r1 * (1.0 - omega * x / (s * me)) if x > 0 else 0.0

            # W_k(x): class-2 initiation at current state
            if x < 0:
                Wk = max(V[i], (V[i + 1] if i + 1 < N else V[i]) + r2)
            else:
                Wk = V[i]

            Vn[i] = (le * Uk
                     + mu_eff * (Wk_prev + r1_adj)
                     + (1.0 - le - mu_eff) * Wk)

        delta = np.max(np.abs(Vn - V))
        V = Vn
        if delta < EPS:
            break

    # Extract policy
    policy = {}
    for i, x in enumerate(states):
        if x < 0:
            policy[x] = 'accept'
        else:
            v_queue = V[i + 1] if i + 1 < N else V[i]
            policy[x] = 'accept' if v_queue >= V[i] - L else 'outsource'

    thresh = next((x for x in range(X_MAX + 1) if policy[x] == 'outsource'), None)
    return policy, thresh, V


# ══════════════════════════════════════════════════════════════════════════
#  Omega_l  —  A POSTERIORI value iteration
# ══════════════════════════════════════════════════════════════════════════
def F_op(x, V, s):
    """FIL operator: one outsourcing event removes one call."""
    xi = x + s
    if x <= 0:
        return V[xi]
    return V[xi - 1] if xi - 1 >= 0 else V[xi]


def vi_omega_l(lam, s, mu, gamma, r1, r2, L, omega,
               X_MAX=X_MAX, EPS=EPS, MAX_K=MAX_K):
    """
    Value iteration for Omega_l (equation 4).

    Returns
    -------
    policy : dict  {state x -> 'accept' | 'outsource'}
    thresh : int or None
    V      : np.ndarray
    """
    states = list(range(-s, X_MAX + 1))
    N = len(states)

    # Uniformization: lambda + s*mu + gamma = 1
    scale = lam + s * mu + gamma
    ll, ml, gl = lam / scale, mu / scale, gamma / scale

    V = np.zeros(N)

    for _ in range(MAX_K):
        Vn = np.zeros(N)
        for i, x in enumerate(states):
            if x <= 0:
                # Class-1 arrival decision
                if x < 0:
                    Uk = V[i] + r1
                else:
                    v_queue = V[i + 1] if i + 1 < N else V[i]
                    Uk = max(v_queue, V[i] - L)

                mu_eff = (s + x) * ml

                if x - 1 >= -s:
                    j = i - 1
                    if states[j] < 0:
                        Wk_prev = max(V[j], (V[j + 1] if j + 1 < N else V[j]) + r2)
                    else:
                        Wk_prev = V[j]
                else:
                    Wk_prev = 0.0

                if x < 0:
                    Wk = max(V[i], (V[i + 1] if i + 1 < N else V[i]) + r2)
                else:
                    Wk = V[i]

                Vn[i] = ll * Uk + mu_eff * Wk_prev + (1.0 - ll - mu_eff) * Wk

            else:
                # FIL epoch decision (equation 4)
                FV    = F_op(x, V, s)
                v_keep = V[i + 1] if i + 1 < N else V[i]
                Uk    = max(v_keep, FV - L)
                r1_adj = r1 * (1.0 - omega * x / max(gl, 1e-9))
                Vn[i] = (gl * Uk
                         + s * ml * (FV + r1_adj)
                         + (1.0 - gl - s * ml) * V[i])

        delta = np.max(np.abs(Vn - V))
        V = Vn
        if delta < EPS:
            break

    # Extract policy
    policy = {}
    for i, x in enumerate(states):
        if x < 0:
            policy[x] = 'accept'
        elif x == 0:
            v_queue = V[i + 1] if i + 1 < N else V[i]
            policy[x] = 'accept' if v_queue >= V[i] - L else 'outsource'
        else:
            FV     = F_op(x, V, s)
            v_keep = V[i + 1] if i + 1 < N else V[i]
            policy[x] = 'accept' if v_keep >= FV - L else 'outsource'

    thresh = next((x for x in range(X_MAX + 1) if policy[x] == 'outsource'), None)
    return policy, thresh, V


# ══════════════════════════════════════════════════════════════════════════
#  Run the sweep
# ══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("Sensitivity Analysis: r1 (live agent) x r2 (chatbot) Routing Policy")
print("=" * 70)
print(f"  Fixed: lambda={lam}, s={s}, mu={mu}, gamma={gamma}, L={L}, omega={omega}")
print(f"  r1 sweep: {R1_VALS}")
print(f"  r2 sweep: {R2_VALS}")
print()

results_e = {}   # results_e[r2][r1] = thresh
results_l = {}

for r2 in R2_VALS:
    results_e[r2] = {}
    results_l[r2] = {}
    for r1 in R1_VALS:
        _, te, _ = vi_omega_e(lam, s, mu, r1, r2, L, omega)
        _, tl, _ = vi_omega_l(lam, s, mu, gamma, r1, r2, L, omega)
        results_e[r2][r1] = te
        results_l[r2][r1] = tl


def thresh_label(t):
    """Convert threshold to readable string."""
    if t is None:
        return "always accept"
    if t == 0:
        return "always outsource"
    return f"x* = {t}"


# ── Print tables ─────────────────────────────────────────────────────────────
for label, results in [("Omega_e — a priori (decision at arrival)", results_e),
                        ("Omega_l — a posteriori (decision at FIL)", results_l)]:
    print(f"{'─'*70}")
    print(f"  {label}")
    print(f"  Threshold x*: queue depth where policy switches from ACCEPT to OUTSOURCE")
    print(f"{'─'*70}")

    # Header
    hdr = f"{'r2\\r1':>8}" + "".join(f"{'r1='+str(r1):>14}" for r1 in R1_VALS)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for r2 in R2_VALS:
        row = f"  r2={r2:>2} "
        for r1 in R1_VALS:
            t = results[r2][r1]
            lbl = thresh_label(t)
            row += f"{lbl:>14}"
        print(row)
    print()


# ══════════════════════════════════════════════════════════════════════════
#  Figure
# ══════════════════════════════════════════════════════════════════════════
def thresh_to_num(t, X_MAX=X_MAX):
    """Map threshold to a numeric value for color mapping."""
    if t is None:
        return X_MAX + 2    # always accept -> high value
    if t == 0:
        return -1           # always outsource -> low value
    return t


fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    "Sensitivity Analysis — Routing Policy vs r₁ (live agent) × r₂ (chatbot)\n"
    f"λ={lam}, s={s}, μ={mu}, γ={gamma}, L={L}, ω={omega}",
    fontsize=11, y=1.01
)

# Color scheme: red=always outsource, amber=threshold, green=always accept
COLORS = {
    'always_out':    '#FCEBEB',   # light red
    'low_thresh':    '#FAEEDA',   # light amber
    'mid_thresh':    '#FFF3CD',   # pale yellow
    'high_thresh':   '#EAF3DE',   # light green
    'always_acc':    '#C0DD97',   # medium green
}

def cell_color(t, X_MAX=X_MAX):
    if t == 0:          return COLORS['always_out']
    if t is None:       return COLORS['always_acc']
    frac = t / X_MAX
    if frac < 0.33:     return COLORS['low_thresh']
    if frac < 0.66:     return COLORS['mid_thresh']
    return COLORS['high_thresh']

def cell_text_color(t):
    if t == 0:    return '#791F1F'
    if t is None: return '#27500A'
    return '#633806'

for ax, (results, title) in zip(axes, [
    (results_e, "Ω_e — a priori\n(live agent routes at arrival)"),
    (results_l, "Ω_l — a posteriori\n(chatbot routes at FIL epoch)")
]):
    ax.set_xlim(0, len(R1_VALS))
    ax.set_ylim(0, len(R2_VALS))
    ax.set_title(title, fontsize=10, pad=8)
    ax.set_xlabel("r₁ — live agent reward →", fontsize=9)
    ax.set_ylabel("r₂ — chatbot reward →", fontsize=9)

    for j, r2 in enumerate(reversed(R2_VALS)):
        for i, r1 in enumerate(R1_VALS):
            t = results[r2][r1]
            color = cell_color(t)
            tcolor = cell_text_color(t)

            rect = mpatches.FancyBboxPatch(
                (i + 0.04, j + 0.04), 0.92, 0.92,
                boxstyle="round,pad=0.02",
                facecolor=color, edgecolor='white', linewidth=1.5
            )
            ax.add_patch(rect)

            if t is None:
                line1 = "always"
                line2 = "accept"
            elif t == 0:
                line1 = "always"
                line2 = "outsource"
            else:
                line1 = f"x* = {t}"
                line2 = f"acc x<{t}"

            ax.text(i + 0.5, j + 0.58, line1,
                    ha='center', va='center', fontsize=7.5,
                    fontweight='bold', color=tcolor)
            ax.text(i + 0.5, j + 0.32, line2,
                    ha='center', va='center', fontsize=6.5, color=tcolor)

    ax.set_xticks([i + 0.5 for i in range(len(R1_VALS))])
    ax.set_xticklabels([str(r) for r in R1_VALS], fontsize=8)
    ax.set_yticks([j + 0.5 for j in range(len(R2_VALS))])
    ax.set_yticklabels([str(r) for r in reversed(R2_VALS)], fontsize=8)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

# Legend
legend_patches = [
    mpatches.Patch(facecolor=COLORS['always_acc'],   label='Always accept (live agent always preferred)'),
    mpatches.Patch(facecolor=COLORS['high_thresh'],  label='High threshold x* (long queue tolerated)'),
    mpatches.Patch(facecolor=COLORS['mid_thresh'],   label='Mid threshold x*'),
    mpatches.Patch(facecolor=COLORS['low_thresh'],   label='Low threshold x* (outsource early)'),
    mpatches.Patch(facecolor=COLORS['always_out'],   label='Always outsource (chatbot always preferred)'),
]
fig.legend(handles=legend_patches, loc='lower center', ncol=3,
           fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.08))

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/sensitivity_r1_r2.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved to sensitivity_r1_r2.png")
