"""
Reproduction of Figure 1 from:
Legros, Jouini, Koole (2021) – "Should We Wait Before Outsourcing?
Analysis of a Revenue-Generating Blended Contact Center"
Manufacturing & Service Operations Management, Vol. 23, No. 5, pp. 1118–1138

Figure 1 parameters: s=10, μ=1, r1=3, r2=1, ω=1,
                      C_outs/(λ·P̄_S)=1/2,  P̄_S=30%

Model overview
──────────────
• Two classes of calls: inbound (class-1, Poisson rate λ) and outbound (class-2).
• s homogeneous agents with exponential service rate μ.
• Revenue from a class-1 call served after waiting W_S:  r1·(1 − ω·W_S)
• Revenue from class-2 calls per unit time:               r2 · E(T)
• Two outsourcing policies
    – π*_e  (a priori):  outsource a call at arrival if queue length ≥ n.
    – π*_l  (a posteriori): outsource if call has waited ≥ τ in queue.
• Reservation threshold c: keep c agents idle for inbound when queue is empty.

Performance measures (Section 3.2 + Table 1)
─────────────────────────────────────────────
  Σ    = (s−1)!/a^{c−1} · Σ_{x=0}^{c−1} a^x/(s−c+x)!
  P_S  = [1 + (λ−sμ)·J] / (Σ + λ·J)
  E(W) = λ·J_H / (Σ + λ·J)
  E(T) = λ·(s−1)!/(a^{c−1}·(s−c−1)!) / (Σ + λ·J)   [c < s]
  E(W_S) = (sμ·J_1 − J) / (sμ·J − 1)
  E(G)  = r2·E(T) + r1·λ·(1−P_S)·(1−ω·E(W_S)) − C_outs

Building blocks (Table 1) differ between the two policies.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import factorial, exp


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation constant Σ  (Section 3.2)
# ─────────────────────────────────────────────────────────────────────────────

def Sigma(s, c, a):
    """
    Σ = (s-1)! / a^{c-1} * sum_{x=0}^{c-1} a^x / (s-c+x)!
    For c=0 the sum is empty; by convention the whole normalisation block
    contributed by the 'non-queuing' states collapses to 1
    (checked against the Erlang-C formula for c=0, n→∞).
    """
    if c == 0:
        return 1.0
    inner = sum(a**x / factorial(s - c + x) for x in range(c))
    return factorial(s - 1) / (a ** (c - 1)) * inner


def ET_prefactor(s, c, a):
    """
    Pre-factor for E(T): (s-1)! / (a^{c-1} * (s-c-1)!)
    Valid for c < s-1.  For c=s-1 use (s-1)!/a^{s-2}.
    For c>=s, E(T)=0 (all agents reserved, no outbound).
    """
    if c >= s:
        return 0.0
    # (s-c-1)! with s-c-1 >= 0 requires c <= s-1
    return factorial(s - 1) / (a ** (c - 1) * factorial(s - c - 1))


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks – A PRIORI  (Table 1, left column)
# ─────────────────────────────────────────────────────────────────────────────

def blocks_e(s, a, mu, n):
    """J, J1, JH for a priori policy with queue threshold n (integer ≥ 0)."""
    rho = a / s
    if abs(rho - 1.0) < 1e-10:
        return None, None, None

    smui = 1.0 / (s * mu)
    r1m  = 1.0 - rho

    J  = smui * (1.0 - rho**(n + 1)) / r1m
    J1 = smui**2 * (1.0 - (n + 2)*rho**(n+1) + (n + 1)*rho**(n+2)) / r1m**2
    JH = smui**2 * (1.0 - (n + 1)*rho**n + n*rho**(n+1)) / r1m**2

    return J, J1, JH


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks – A POSTERIORI  (Table 1, right column)
# ─────────────────────────────────────────────────────────────────────────────

def blocks_l(s, a, mu, lam, tau):
    """J, J1, JH for a posteriori policy with waiting threshold τ ≥ 0."""
    rho   = a / s
    delta = s * mu - lam   # sμ − λ  (> 0 for stability)

    if abs(rho - 1.0) < 1e-10 or delta <= 1e-12:
        return None, None, None

    smui  = 1.0 / (s * mu)
    r1m   = 1.0 - rho
    eterm = exp(-tau * delta)

    J  = smui * (1.0 - rho * eterm) / r1m
    J1 = smui**2 * (
        1.0 - (1.0 + r1m * (1.0 + s * mu * tau)) * rho * eterm
    ) / r1m**2
    JH = smui**2 * (1.0 - (1.0 + rho * tau * delta) * eterm) / r1m**2

    return J, J1, JH


# ─────────────────────────────────────────────────────────────────────────────
# Performance measures given building blocks
# ─────────────────────────────────────────────────────────────────────────────

def metrics(s, c, a, mu, lam, r1, r2, omega, Couts, J, J1, JH):
    if J is None or not np.isfinite(J):
        return None

    Sig   = Sigma(s, c, a)
    denom = Sig + lam * J
    if abs(denom) < 1e-15:
        return None

    PS  = (1.0 + (lam - s * mu) * J) / denom
    EW  = lam * JH / denom

    smuJ1 = s * mu * J - 1.0
    EWS   = (s * mu * J1 - J) / smuJ1 if abs(smuJ1) > 1e-15 else 0.0

    ETf = ET_prefactor(s, c, a)
    ET  = lam * ETf / denom

    EG  = r2 * ET + r1 * lam * (1.0 - PS) * (1.0 - omega * EWS) - Couts

    return dict(EG=EG, PS=PS, EW=EW, EWS=EWS)


# ─────────────────────────────────────────────────────────────────────────────
# Optimal policy search
# ─────────────────────────────────────────────────────────────────────────────

def optimise_apriori(s, lam, mu, r1, r2, omega, PS_bar, Couts_rate):
    """Grid search over (c, n) for best a priori policy."""
    Couts = Couts_rate * lam * PS_bar
    a     = lam / mu

    best_EG = -1e30
    best    = dict(EG=np.nan, PS=np.nan, EW=np.nan, EWS=np.nan)

    for c in range(s + 1):
        for n in range(0, 500):
            J, J1, JH = blocks_e(s, a, mu, n)
            res = metrics(s, c, a, mu, lam, r1, r2, omega, Couts, J, J1, JH)
            if res is None:
                break
            if not (0.0 <= res['PS'] <= 1.0):
                continue
            # Stability: λ(1−P_S) < sμ
            if lam * (1.0 - res['PS']) >= s * mu - 1e-9:
                continue
            # Contract constraint
            if res['PS'] > PS_bar + 1e-6:
                continue
            if res['EG'] > best_EG:
                best_EG = res['EG']
                best = res.copy()

    return best


def optimise_aposteriori(s, lam, mu, r1, r2, omega, PS_bar, Couts_rate):
    """Grid search over (c, τ) for best a posteriori policy."""
    Couts = Couts_rate * lam * PS_bar
    a     = lam / mu

    if s * mu - lam <= 1e-9:
        return dict(EG=np.nan, PS=np.nan, EW=np.nan, EWS=np.nan)

    # τ grid: 0 (outsource all) to large (serve all in-house)
    tau_grid = np.concatenate([[0.0], np.logspace(-3, 2.3, 700)])

    best_EG = -1e30
    best    = dict(EG=np.nan, PS=np.nan, EW=np.nan, EWS=np.nan)

    for c in range(s + 1):
        for tau in tau_grid:
            J, J1, JH = blocks_l(s, a, mu, lam, tau)
            res = metrics(s, c, a, mu, lam, r1, r2, omega, Couts, J, J1, JH)
            if res is None:
                continue
            if not (0.0 <= res['PS'] <= 1.0):
                continue
            if lam * (1.0 - res['PS']) >= s * mu - 1e-9:
                continue
            if res['PS'] > PS_bar + 1e-6:
                continue
            if res['EG'] > best_EG:
                best_EG = res['EG']
                best = res.copy()

    return best


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def generate_figure1():
    s          = 10
    mu         = 1.0
    r1         = 3.0
    r2         = 1.0
    omega      = 1.0
    PS_bar     = 0.30
    Couts_rate = 0.50

    lam_values = np.linspace(0.5, 13.5, 40)

    res_e = [optimise_apriori(s, l, mu, r1, r2, omega, PS_bar, Couts_rate)
             for l in lam_values]
    res_l = [optimise_aposteriori(s, l, mu, r1, r2, omega, PS_bar, Couts_rate)
             for l in lam_values]

    def series(results, key):
        arr = np.array([r[key] for r in results], dtype=float)
        arr[~np.isfinite(arr)] = np.nan
        return arr

    lam = lam_values
    EG_e  = series(res_e, 'EG');   EG_l  = series(res_l, 'EG')
    PS_e  = series(res_e, 'PS')*100; PS_l = series(res_l, 'PS')*100
    EW_e  = series(res_e, 'EW');   EW_l  = series(res_l, 'EW')
    EWS_e = series(res_e, 'EWS');  EWS_l = series(res_l, 'EWS')

    diff_EG  = np.abs(EG_l  - EG_e)
    diff_EW  = np.abs(EW_l  - EW_e)
    diff_EWS = np.abs(EWS_l - EWS_e)

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        r'Figure 1 — Comparison Between the Two Policy Classes'
        '\n'
        r'$s=10,\ \mu=1,\ r_1=3,\ r_2=1,\ \omega=1,\ '
        r'C_{outs}/(\lambda\bar{P}_S)=1/2,\ \bar{P}_S=30\%$',
        fontsize=11
    )

    kw_e   = dict(ls='--', color='tab:blue',  lw=1.6, label='a priori')
    kw_l   = dict(ls='-',  color='tab:orange', lw=1.6, label='a posteriori')
    kw_d   = dict(ls='-.',  color='tab:green',  lw=1.2, label='absolute difference')
    xlim   = [0, 14]

    def subplot(ax, y_e, y_l, y_d, ylabel, title, show_diff=True, pct=False):
        ax.plot(lam, y_e, **kw_e)
        ax.plot(lam, y_l, **kw_l)
        if show_diff:
            ax.plot(lam, y_d, **kw_d)
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xlabel(r'$\lambda$', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, loc='left')
        ax.set_xlim(xlim)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if pct:
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))

    subplot(axes[0,0], EG_e,  EG_l,  diff_EG,  r'$E(G)$',   '(a)')
    # Panel (b): P_S without absolute difference line (paper style)
    ax = axes[0,1]
    ax.plot(lam, PS_e, **kw_e)
    ax.plot(lam, PS_l, **kw_l)
    ax.axhline(30, color='k', lw=0.5, ls=':')
    ax.set_xlabel(r'$\lambda$', fontsize=11)
    ax.set_ylabel(r'$P_S$', fontsize=11)
    ax.set_title('(b)', loc='left')
    ax.set_xlim(xlim); ax.set_ylim([0, 35])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    subplot(axes[1,0], EW_e,  EW_l,  diff_EW,  r'$E(W)$',   '(c)')
    subplot(axes[1,1], EWS_e, EWS_l, diff_EWS, r'$E(W_S)$', '(d)')

    plt.tight_layout()
    out = '/mnt/user-data/outputs/figure1_legros2021.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Figure saved → {out}")

    import shutil
    shutil.copy('/home/claude/figure1.py', '/mnt/user-data/outputs/figure1_legros2021.py')
    print("Script saved.")


if __name__ == '__main__':
    generate_figure1()
