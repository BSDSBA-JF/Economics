"""
Chatbot Optimization Model
==========================
Finds optimal depth S* that maximizes expected value V_bot(S).

Model components:
  - rho_x     : step-level success probability (given, not optimized)
  - p_succ(S) : overall chatbot success probability up to depth S
  - tau_x     : time cost per step
  - c_bot(S)  : training/development cost (convex in p_succ)
  - V_bot(S)  : expected value as function of S
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


# ─────────────────────────────────────────────
# 1. Model Parameters
# ─────────────────────────────────────────────

def default_params() -> dict:
    """
    Returns default model parameters.

    rho   : list of step-level success probs (one per max step)
    tau   : list of time costs per step (seconds or normalized units)
    r     : reward if chatbot resolves the issue
    c_c   : cold transfer penalty (cost when chatbot fails and transfers)
    V_agent: value of human agent handling the issue
    R_star: time cost rate (value lost per unit time)
    a_bot : training cost scale parameter
    b_bot : training cost convexity parameter
    """
    return {
        "rho":     [0.40, 0.30, 0.20, 0.15, 0.10, 0.07, 0.05, 0.03, 0.02, 0.01],
        "tau":     [1.0,  1.2,  1.5,  1.8,  2.0,  2.2,  2.5,  2.8,  3.0,  3.5],
        "r":       100.0,
        "c_c":     10.0,
        "V_agent": 60.0,
        "R_star":  2.0,
        "a_bot":   0.05,
        "b_bot":   1.8,
    }


# ─────────────────────────────────────────────
# 2. Core Model Functions
# ─────────────────────────────────────────────

def p_succ(S: int, rho: List[float]) -> float:
    """
    Overall chatbot success probability for depth S.

    p_succ(S) = 1 - prod_{x=1}^{S} (1 - rho_x)

    Args:
        S   : depth (number of steps attempted, 1-indexed)
        rho : list of step-level success probabilities

    Returns:
        Scalar probability in [0, 1]
    """
    if S == 0:
        return 0.0
    failure_prob = 1.0
    for x in range(S):
        failure_prob *= (1.0 - rho[x])
    return 1.0 - failure_prob


def total_time(S: int, tau: List[float]) -> float:
    """
    Total time cost for running S steps.

    T(S) = sum_{x=1}^{S} tau_x

    Args:
        S   : depth
        tau : list of per-step time costs

    Returns:
        Scalar total time
    """
    return sum(tau[:S])


def c_bot(S: int, rho: List[float], a_bot: float, b_bot: float) -> float:
    """
    Training/development cost, convex in p_succ.

    c_bot(S) = a_bot * (100 * p_succ(S))^b_bot

    Args:
        S     : depth
        rho   : step-level success probs
        a_bot : scale parameter
        b_bot : convexity parameter

    Returns:
        Scalar cost
    """
    ps = p_succ(S, rho)
    return a_bot * (100.0 * ps) ** b_bot


def V_bot(S: int, params: dict) -> float:
    """
    Expected value of chatbot policy at depth S.

    V_bot(S) = p_succ(S) * r
             + (1 - p_succ(S)) * (V_agent - c_c)
             - R_star * T(S)
             - c_bot(S)

    Args:
        S      : depth
        params : model parameters dict

    Returns:
        Scalar expected value
    """
    rho    = params["rho"]
    tau    = params["tau"]
    r      = params["r"]
    c_c    = params["c_c"]
    V_ag   = params["V_agent"]
    R_star = params["R_star"]
    a_bot  = params["a_bot"]
    b_bot  = params["b_bot"]

    ps   = p_succ(S, rho)
    T    = total_time(S, tau)
    cost = c_bot(S, rho, a_bot, b_bot)

    return (
        ps * r
        + (1.0 - ps) * (V_ag - c_c)
        - R_star * T
        - cost
    )


# ─────────────────────────────────────────────
# 3. Optimization
# ─────────────────────────────────────────────

def optimize(params: dict) -> Tuple[int, float, List[float]]:
    """
    Find optimal depth S* = argmax_S V_bot(S).

    Evaluates all feasible depths S = 1, ..., max_S.

    Args:
        params : model parameters dict

    Returns:
        (S_star, V_star, all_values)
        S_star    : optimal depth
        V_star    : optimal expected value
        all_values: list of V_bot(S) for S = 0, 1, ..., max_S
    """
    max_S = len(params["rho"])
    values = [V_bot(S, params) for S in range(max_S + 1)]

    # S=0 means no chatbot at all (baseline)
    S_star = int(np.argmax(values[1:]) + 1)  # search over S >= 1
    V_star = values[S_star]

    return S_star, V_star, values


# ─────────────────────────────────────────────
# 4. Results Summary
# ─────────────────────────────────────────────

def print_summary(params: dict) -> None:
    """Prints a full summary of model results."""
    rho    = params["rho"]
    tau    = params["tau"]
    max_S  = len(rho)

    S_star, V_star, values = optimize(params)

    print("=" * 60)
    print("CHATBOT OPTIMIZATION MODEL — RESULTS")
    print("=" * 60)

    print("\n--- Parameters ---")
    print(f"  Reward if resolved   (r)      : {params['r']}")
    print(f"  Cold transfer cost   (c_c)    : {params['c_c']}")
    print(f"  Human agent value    (V_agent): {params['V_agent']}")
    print(f"  Time cost rate       (R*)     : {params['R_star']}")
    print(f"  Training cost (a, b)          : {params['a_bot']}, {params['b_bot']}")

    print("\n--- Step-level Profile ---")
    print(f"  {'S':>4}  {'rho_x':>8}  {'tau_x':>8}  {'p_succ':>8}  {'T(S)':>8}  {'V_bot':>10}")
    print("  " + "-" * 52)
    for S in range(1, max_S + 1):
        ps  = p_succ(S, rho)
        T   = total_time(S, tau)
        V   = values[S]
        marker = " <-- OPTIMAL" if S == S_star else ""
        print(f"  {S:>4}  {rho[S-1]:>8.3f}  {tau[S-1]:>8.2f}  {ps:>8.4f}  {T:>8.2f}  {V:>10.4f}{marker}")

    print("\n--- Optimal Policy ---")
    print(f"  S*        = {S_star}")
    print(f"  p_succ*   = {p_succ(S_star, rho):.4f}  ({100*p_succ(S_star, rho):.2f}%)")
    print(f"  T*        = {total_time(S_star, tau):.2f}")
    print(f"  V_bot*    = {V_star:.4f}")
    print("=" * 60)


# ─────────────────────────────────────────────
# 5. Plotting
# ─────────────────────────────────────────────

def plot_results(params: dict, save_path: str = None) -> None:
    """
    Plots V_bot(S) across all depths, highlighting S*.
    """
    rho   = params["rho"]
    max_S = len(rho)

    S_star, V_star, values = optimize(params)

    S_range = list(range(1, max_S + 1))
    V_range = values[1:]
    ps_range = [p_succ(S, rho) for S in S_range]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Chatbot Optimization Model", fontsize=14, fontweight="bold")

    # --- Left: V_bot(S) ---
    ax = axes[0]
    ax.plot(S_range, V_range, "o-", color="#3266ad", linewidth=2, markersize=5, label="V_bot(S)")
    ax.axvline(S_star, color="#e04f3a", linestyle="--", linewidth=1.5, label=f"S* = {S_star}")
    ax.scatter([S_star], [V_star], color="#e04f3a", s=100, zorder=5)
    ax.set_xlabel("Depth S (steps allowed)", fontsize=12)
    ax.set_ylabel("Expected Value V_bot(S)", fontsize=12)
    ax.set_title("Expected Value vs. Depth")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Right: p_succ(S) ---
    ax2 = axes[1]
    ax2.plot(S_range, [p * 100 for p in ps_range], "s-", color="#2a9d6e",
             linewidth=2, markersize=5, label="p_succ(S)")
    ax2.axvline(S_star, color="#e04f3a", linestyle="--", linewidth=1.5, label=f"S* = {S_star}")
    ax2.set_xlabel("Depth S (steps allowed)", fontsize=12)
    ax2.set_ylabel("Success Probability (%)", fontsize=12)
    ax2.set_title("Cumulative Success Probability vs. Depth")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved to: {save_path}")
    else:
        plt.show()


# ─────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    params = default_params()

    print_summary(params)

    # Uncomment to save plot:
    # plot_results(params, save_path="chatbot_optimization.png")
    plot_results(params)
