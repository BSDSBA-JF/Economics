"""
Value Iteration for Two-Class Queueing MDP with Outsourcing
============================================================
Implements the value functions and value iteration for both policy classes:
  - Omega_e : a priori  (decisions at class-1 arrivals)
  - Omega_l : a posteriori (decisions at FIL epochs)

Based on equations (3) and (4) from the paper.

Parameters
----------
lam   : float  -- class-1 arrival rate  (lambda)
s     : int    -- number of servers
mu    : float  -- per-server service rate
gamma : float  -- FIL (First-In-Line) epoch rate
r1    : float  -- reward per class-1 call served
r2    : float  -- reward per class-2 call initiated
L     : float  -- cost per outsourced call (Lagrange multiplier)
omega : float  -- waiting-time penalty weight
X_MAX : int    -- maximum queue length to truncate state space
eps   : float  -- convergence threshold for max |Delta V|
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Parameters ─────────────────────────────────────────────────────────────
lam   = 0.30   # arrival rate
s     = 2      # number of servers
mu    = 0.35   # per-server service rate
gamma = 0.20   # FIL epoch rate
r1    = 5.0    # class-1 reward
r2    = 3.0    # class-2 reward
L     = 2.0    # outsourcing cost (Lagrange multiplier)
omega = 0.30   # waiting-time penalty weight

X_MAX   = 15    # truncate queue at this length
EPS     = 1e-4  # convergence tolerance
MAX_ITER = 500  # safety cap on iterations

# ── State space ─────────────────────────────────────────────────────────────
# States: x in {-s, -s+1, ..., 0, 1, ..., X_MAX}
#   x < 0  : |x| vacant servers
#   x = 0  : all servers busy, empty queue
#   x > 0  : all servers busy, x calls in queue
states = list(range(-s, X_MAX + 1))
N      = len(states)

def idx(x):
    """Map state x to array index."""
    return x + s


# ══════════════════════════════════════════════════════════════════════════
#  Omega_e  —  A PRIORI policy  (equation 3)
# ══════════════════════════════════════════════════════════════════════════
# Uniformization: lambda + s*mu = 1  =>  scale rates
scale_e = lam + s * mu
lam_e   = lam / scale_e
mu_e    = mu  / scale_e


def U_k_e(x, V):
    """
    Decision operator at a class-1 arrival (Omega_e).

    For x in [-s, -1]: vacant server, accept immediately, earn r1, no wait.
    For x >= 0        : all servers busy; choose max(V(x+1), V(x) - L).
                        V(x+1) = queue the call; V(x) - L = outsource it.
    """
    i = idx(x)
    if x < 0:                        # vacant server -> always accept
        return V[i] + r1
    else:                            # all busy: queue vs outsource
        v_queue   = V[i + 1] if i + 1 < N else V[i]
        v_outside = V[i] - L
        return max(v_queue, v_outside)


def W_k_e(x, V):
    """
    Class-2 initiation operator (Omega_e).

    For x in [-s, -1]: vacant server; agent may idle or initiate class-2.
                        Initiating moves queue to x+1 and earns r2.
    For x >= 0        : all servers busy; cannot initiate class-2.
    """
    i = idx(x)
    if x < 0:
        v_idle   = V[i]
        v_init   = (V[i + 1] if i + 1 < N else V[i]) + r2
        return max(v_idle, v_init)
    else:
        return V[i]


def bellman_e(V):
    """
    One Bellman update for Omega_e — equation (3).

    V_{k+1}(x) = lam * U_k(x)
               + min(s, s+x) * mu * [W_k(x-1) + r1*(1 - omega*x/(s*mu_e)) * 1_{x>0}]
               + (1 - lam - min(s,s+x)*mu) * W_k(x)
    """
    Vn = np.zeros(N)
    for i, x in enumerate(states):
        servers_busy = min(s, s + x)   # min(s, s+x)  [= max(0, s+x) clipped to s]
        mu_eff       = servers_busy * mu_e

        Uk = U_k_e(x, V)

        # W_k(x-1): previous state's class-2 operator value
        if x - 1 >= -s:
            Wk_prev = W_k_e(x - 1, V)
        else:
            Wk_prev = 0.0

        # Waiting-adjusted reward for class-1 served from state x>0
        r1_adj = r1 * (1.0 - omega * x / (s * mu_e)) if x > 0 else 0.0

        Wk = W_k_e(x, V)

        Vn[i] = (lam_e * Uk
                 + mu_eff * (Wk_prev + r1_adj)
                 + (1.0 - lam_e - mu_eff) * Wk)
    return Vn


def value_iteration_e():
    """Run value iteration for Omega_e until convergence."""
    V        = np.zeros(N)
    conv     = []
    snapshots = {}
    snap_iters = {0, 5, 20, 100, 250}

    for k in range(MAX_ITER):
        if k in snap_iters:
            snapshots[k] = V.copy()

        Vn    = bellman_e(V)
        delta = np.max(np.abs(Vn - V))
        conv.append(delta)
        V = Vn

        if delta < EPS:
            print(f"[Omega_e] converged at k={k+1}, max|DeltaV|={delta:.2e}")
            break
    else:
        print(f"[Omega_e] reached MAX_ITER={MAX_ITER}, max|DeltaV|={conv[-1]:.2e}")

    snapshots['final'] = V.copy()
    return V, conv, snapshots


def extract_policy_e(V):
    """
    Extract the optimal policy for Omega_e.

    Returns a dict: state x -> action string
      'accept'    -- class-1 call is accepted (x < 0)
      'queue'     -- class-1 call is queued   (x >= 0, queue dominates)
      'outsource' -- class-1 call is outsourced (x >= 0, outsource dominates)
    Also returns class-2 action at each vacant-server state.
    """
    policy = {}
    for i, x in enumerate(states):
        if x < 0:
            # Class-1 action
            policy[x] = {'class1': 'accept (vacant server)'}
            # Class-2 action
            v_idle = V[i]
            v_init = (V[i + 1] if i + 1 < N else V[i]) + r2
            policy[x]['class2'] = 'initiate' if v_init >= v_idle else 'idle'
        else:
            v_queue   = V[i + 1] if i + 1 < N else V[i]
            v_outside = V[i] - L
            if v_queue >= v_outside:
                policy[x] = {'class1': 'queue', 'class2': 'N/A (all busy)'}
            else:
                policy[x] = {'class1': 'outsource', 'class2': 'N/A (all busy)'}
    return policy


# ══════════════════════════════════════════════════════════════════════════
#  Omega_l  —  A POSTERIORI policy  (equation 4)
# ══════════════════════════════════════════════════════════════════════════
# Uniformization: lambda + s*mu + gamma = 1
scale_l = lam + s * mu + gamma
lam_l   = lam   / scale_l
mu_l    = mu    / scale_l
gam_l   = gamma / scale_l


def F_operator(x, V):
    """
    FIL operator F applied to V at state x.

    F(f)(x) = sum_{h=0}^{x} q_{x,x-h} * f(x-h)   for x > 0
    F(f)(x) = f(x)                                  for x <= 0

    Simplified model: q_{x,x-1} = 1 (one outsourcing per FIL event).
    """
    if x <= 0:
        return V[idx(x)]
    else:
        xi = idx(x - 1)
        return V[xi] if xi >= 0 else V[idx(x)]


def U_k_l(x, V):
    """
    Decision operator at a FIL epoch (Omega_l).

    At a FIL event the agent may outsource the head-of-line call or keep it.
    Outsource: F(V)(x) - L
    Keep:      V(x+1)  [one more in queue after FIL is served elsewhere]
    """
    FV      = F_operator(x, V)
    v_keep  = V[idx(x) + 1] if idx(x) + 1 < N else V[idx(x)]
    v_out   = FV - L
    return max(v_keep, v_out)


def W_k_l(x, V):
    """
    Class-2 initiation operator (Omega_l).

    For x in [-s, 0]: vacant server possible; same logic as Omega_e.
    For x > 0:        all servers busy; no initiation.
    """
    i = idx(x)
    if x < 0:
        v_idle = V[i]
        v_init = (V[i + 1] if i + 1 < N else V[i]) + r2
        return max(v_idle, v_init)
    else:
        return V[i]


def bellman_l(V):
    """
    One Bellman update for Omega_l — equations from Section 3.1.3.

    For -s <= x <= 0:
        V_{k+1}(x) = lam * U_k(x) + (s+x)*mu * W_k(x-1)
                   + (1 - lam - (s+x)*mu) * W_k(x)

    For x > 0:
        V_{k+1}(x) = gam * U_k(x)
                   + s*mu * [F(W_k)(x) + r1*(1 - omega*x/gamma)]
                   + (1 - gam - s*mu) * W_k(x)
    """
    Vn = np.zeros(N)
    for i, x in enumerate(states):
        if x <= 0:
            # --- negative / boundary states ---
            if x < 0:
                Uk = V[i] + r1           # vacant: accept class-1
            else:                         # x == 0
                v_keep  = V[i + 1] if i + 1 < N else V[i]
                Uk      = max(v_keep, V[i] - L)

            mu_eff = (s + x) * mu_l      # effective service rate

            if x - 1 >= -s:
                Wk_prev = W_k_l(x - 1, V)
            else:
                Wk_prev = 0.0

            Wk = W_k_l(x, V)

            Vn[i] = (lam_l * Uk
                     + mu_eff * Wk_prev
                     + (1.0 - lam_l - mu_eff) * Wk)

        else:
            # --- positive queue states (equation 4) ---
            Uk   = U_k_l(x, V)
            FWk  = F_operator(x, V)      # F applied to W_k (approx V here)
            r1_adj = r1 * (1.0 - omega * x / max(gam_l, 1e-9))
            Wk   = W_k_l(x, V)

            Vn[i] = (gam_l * Uk
                     + s * mu_l * (FWk + r1_adj)
                     + (1.0 - gam_l - s * mu_l) * Wk)
    return Vn


def value_iteration_l():
    """Run value iteration for Omega_l until convergence."""
    V         = np.zeros(N)
    conv      = []
    snapshots = {}
    snap_iters = {0, 5, 20, 100, 250}

    for k in range(MAX_ITER):
        if k in snap_iters:
            snapshots[k] = V.copy()

        Vn    = bellman_l(V)
        delta = np.max(np.abs(Vn - V))
        conv.append(delta)
        V = Vn

        if delta < EPS:
            print(f"[Omega_l] converged at k={k+1}, max|DeltaV|={delta:.2e}")
            break
    else:
        print(f"[Omega_l] reached MAX_ITER={MAX_ITER}, max|DeltaV|={conv[-1]:.2e}")

    snapshots['final'] = V.copy()
    return V, conv, snapshots


def extract_policy_l(V):
    """
    Extract the optimal policy for Omega_l.

    At FIL epochs (x > 0): compare outsource vs keep.
    At arrival   (x <= 0): same as Omega_e for class-1; idle/initiate for class-2.
    """
    policy = {}
    for i, x in enumerate(states):
        if x < 0:
            policy[x] = {'class1': 'accept (vacant server)'}
            v_idle = V[i]
            v_init = (V[i + 1] if i + 1 < N else V[i]) + r2
            policy[x]['class2'] = 'initiate' if v_init >= v_idle else 'idle'
        elif x == 0:
            v_keep  = V[i + 1] if i + 1 < N else V[i]
            v_out   = V[i] - L
            action  = 'keep' if v_keep >= v_out else 'outsource at FIL'
            policy[x] = {'class1': action, 'class2': 'N/A'}
        else:
            FV     = F_operator(x, V)
            v_keep = V[idx(x) + 1] if idx(x) + 1 < N else V[idx(x)]
            v_out  = FV - L
            action = 'keep in queue' if v_keep >= v_out else 'outsource at FIL'
            policy[x] = {'class1': action, 'class2': 'N/A (all busy)'}
    return policy


# ══════════════════════════════════════════════════════════════════════════
#  Run both
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Value Iteration — Two-Class Queueing MDP")
print("=" * 60)
print(f"  lambda={lam}, s={s}, mu={mu}, gamma={gamma}")
print(f"  r1={r1}, r2={r2}, L={L}, omega={omega}")
print(f"  States: {states[0]} to {states[-1]}  (N={N})")
print("=" * 60)

Ve, conv_e, snaps_e = value_iteration_e()
Vl, conv_l, snaps_l = value_iteration_l()

policy_e = extract_policy_e(Ve)
policy_l = extract_policy_l(Vl)

# ── Print policies ──────────────────────────────────────────────────────────
print("\n--- Optimal Policy: Omega_e (a priori) ---")
print(f"{'State':>7}  {'Region':25}  {'Class-1 action':20}  {'Class-2 action'}")
print("-" * 75)
for x, p in policy_e.items():
    region = "vacant server" if x < 0 else ("boundary" if x == 0 else f"queue depth {x}")
    print(f"  x={x:3d}  {region:25}  {p['class1']:20}  {p['class2']}")

print("\n--- Optimal Policy: Omega_l (a posteriori) ---")
print(f"{'State':>7}  {'Region':25}  {'Action at FIL / arrival':25}  {'Class-2 action'}")
print("-" * 80)
for x, p in policy_l.items():
    region = "vacant" if x < 0 else ("boundary" if x == 0 else f"queue depth {x}")
    print(f"  x={x:3d}  {region:25}  {p['class1']:25}  {p['class2']}")


# ══════════════════════════════════════════════════════════════════════════
#  Plots
# ══════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(14, 10))
fig.suptitle(
    "Value Iteration — Two-Class Queueing MDP with Outsourcing\n"
    f"λ={lam}, s={s}, μ={mu}, γ={gamma}, r₁={r1}, r₂={r2}, L={L}, ω={omega}",
    fontsize=11, y=0.98
)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

x_arr   = np.array(states)
x_ticks = x_arr[::2]

# ── Panel 1: Convergence Omega_e ────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.semilogy(range(1, len(conv_e) + 1), conv_e, color='#378ADD', linewidth=1.5)
ax1.axhline(EPS, color='gray', linestyle='--', linewidth=0.8, label=f'ε={EPS}')
ax1.set_title('Convergence — Ω_e (a priori)', fontsize=10)
ax1.set_xlabel('Iteration k', fontsize=9)
ax1.set_ylabel('max |ΔV|', fontsize=9)
ax1.legend(fontsize=8)
ax1.grid(True, which='both', alpha=0.3)

# ── Panel 2: Convergence Omega_l ────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.semilogy(range(1, len(conv_l) + 1), conv_l, color='#1D9E75', linewidth=1.5)
ax2.axhline(EPS, color='gray', linestyle='--', linewidth=0.8, label=f'ε={EPS}')
ax2.set_title('Convergence — Ω_l (a posteriori)', fontsize=10)
ax2.set_xlabel('Iteration k', fontsize=9)
ax2.set_ylabel('max |ΔV|', fontsize=9)
ax2.legend(fontsize=8)
ax2.grid(True, which='both', alpha=0.3)

# ── Panel 3: Value function snapshots Omega_e ───────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
snap_colors_e = ['#D3D1C7', '#B5D4F4', '#85B7EB', '#378ADD', '#0C447C']
snap_keys_e   = sorted([k for k in snaps_e if k != 'final']) + ['final']
for j, k in enumerate(snap_keys_e):
    lw  = 2.0 if k == 'final' else 1.0
    lbl = f'k={k}' if k != 'final' else 'final'
    ax3.plot(x_arr, snaps_e[k], color=snap_colors_e[j % len(snap_colors_e)],
             linewidth=lw, label=lbl)
ax3.set_title('Value function V_k(x) — Ω_e', fontsize=10)
ax3.set_xlabel('State x', fontsize=9)
ax3.set_ylabel('V_k(x)', fontsize=9)
ax3.set_xticks(x_ticks)
ax3.legend(fontsize=7, ncol=2)
ax3.axvline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
ax3.grid(True, alpha=0.3)

# ── Panel 4: Value function snapshots Omega_l ───────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
snap_colors_l = ['#9FE1CB', '#5DCAA5', '#1D9E75', '#0F6E56', '#085041']
snap_keys_l   = sorted([k for k in snaps_l if k != 'final']) + ['final']
for j, k in enumerate(snap_keys_l):
    lw  = 2.0 if k == 'final' else 1.0
    lbl = f'k={k}' if k != 'final' else 'final'
    ax4.plot(x_arr, snaps_l[k], color=snap_colors_l[j % len(snap_colors_l)],
             linewidth=lw, label=lbl)
ax4.set_title('Value function V_k(x) — Ω_l', fontsize=10)
ax4.set_xlabel('State x', fontsize=9)
ax4.set_ylabel('V_k(x)', fontsize=9)
ax4.set_xticks(x_ticks)
ax4.legend(fontsize=7, ncol=2)
ax4.axvline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
ax4.grid(True, alpha=0.3)

plt.savefig('/mnt/user-data/outputs/value_iteration_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved to value_iteration_plots.png")
