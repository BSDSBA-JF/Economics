"""
R* Solver — Theorem 2
======================
Solves for R* (the optimal long-run reward rate) using the dynamic programming
formulation from the proof. All G(·) constants are linear in R*, so the final
equation (step 2.2) reduces to a single linear equation which is solved exactly.

Parameters
----------
S       : int   — maximum number of attempts
q       : float — probability queue is congested (Q=1)
a       : float — probability agent is active (A=1)
r       : float — reward for resolving the issue
cw      : float — cost of a warm transfer
cc      : float — cost of a cold transfer
tau_w   : int   — waiting time for a warm transfer
rho     : list  — resolution probabilities rho[x] for x = 1..S  (length S)
tau     : list  — attempt delay tau[x] for x = 1..S             (length S)

Returns
-------
Rstar   : float — optimal reward rate
G       : dict  — all G(·) constants evaluated at R*
policy  : dict  — optimal action at each (X, Q, A) for X = 1..S-1
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

# A linear expression in R*: value = coef * R* + const
LinearExpr = Tuple[float, float]   # (coef, const)

def eval_expr(expr: LinearExpr, Rstar: float) -> float:
    return expr[0] * Rstar + expr[1]


@dataclass
class Params:
    S:     int
    q:     float
    a:     float
    r:     float
    cw:    float
    cc:    float
    tau_w: int
    rho:   list   # rho[0] unused; rho[1..S] used
    tau:   list   # tau[0] unused; tau[1..S] used


# ---------------------------------------------------------------------------
# Step 1 — Base constants (all linear in R*)
# ---------------------------------------------------------------------------

def base_constants(p: Params) -> Dict[str, LinearExpr]:
    G = {}

    # Step 0: normalize G(0i, 0, -) = 0
    G["0i_0"] = (0.0, 0.0)

    # Step 2.1: G(0i, 1, -) = R*/q
    G["0i_1"] = (1.0 / p.q, 0.0)

    # Step 2.4: G(0w, -, 1) = -R*(tau_w - 1)
    G["0w_1"] = (-(p.tau_w - 1), 0.0)

    # Step 2.3: G(0w, -, 0) = -R*(a*(tau_w-1)+1)/a
    G["0w_0"] = (-(p.a * (p.tau_w - 1) + 1) / p.a, 0.0)

    # Step 2.6: G(S, Q, A)
    # G(S, 0, -) = r
    G["S_00"] = (0.0, p.r)
    G["S_01"] = (0.0, p.r)
    # G(S, 1, -) = r + R*/q
    G["S_10"] = (1.0 / p.q, p.r)
    G["S_11"] = (1.0 / p.q, p.r)

    return G


# ---------------------------------------------------------------------------
# Step 2 — Compute N_{X+1} (value-to-go for continuing), linear in R*
# ---------------------------------------------------------------------------

def compute_N(x_next: int, p: Params, G: Dict[str, LinearExpr]) -> LinearExpr:
    """
    N_{x_next} = -R* * tau_{x_next}
                 + (1-q)(1-a) G(x_next,0,0)
                 + q(1-a)     G(x_next,1,0)
                 + (1-q)a     G(x_next,0,1)
                 + q*a        G(x_next,1,1)
    """
    tau_xp = p.tau[x_next]
    wt  = (1 - p.q) * (1 - p.a)
    wq  = p.q        * (1 - p.a)
    wa  = (1 - p.q)  * p.a
    wqa = p.q        * p.a

    def g(Q, A):
        key = f"S_{Q}{A}" if x_next == p.S else f"{x_next}_{Q}{A}"
        return G[key]

    coef  = -tau_xp + wt*g(0,0)[0] + wq*g(1,0)[0] + wa*g(0,1)[0] + wqa*g(1,1)[0]
    const =           wt*g(0,0)[1] + wq*g(1,0)[1] + wa*g(0,1)[1] + wqa*g(1,1)[1]
    return (coef, const)


# ---------------------------------------------------------------------------
# Step 3 — Recursion from X = S-1 down to X = 1
# ---------------------------------------------------------------------------

def compute_G_recursion(
    p: Params,
    G: Dict[str, LinearExpr],
    Rstar_trial: float = 1.0,
) -> Tuple[Dict[str, LinearExpr], Dict[Tuple[int,int,int], str]]:
    """
    For each X from S-1 down to 1, and each (Q, A) in {0,1}^2:
      - Evaluate the three candidate continuation values at Rstar_trial
        to determine the optimal action.
      - Store G(X,Q,A) as a LinearExpr and record the policy.

    Returns updated G dict and policy dict.
    """
    policy: Dict[Tuple[int,int,int], str] = {}

    for x in range(p.S - 1, 0, -1):
        N = compute_N(x + 1, p, G)   # linear in R*

        for Q in (0, 1):
            for A in (0, 1):
                G0i = G[f"0i_{Q}"]
                G0w = G[f"0w_{A}"]
                rho_x = p.rho[x]

                # Three second-term candidates (EC.6):
                candidates = {
                    "continue": N,
                    "warm":     (G0w[0],            G0w[1] - (p.cw - p.r)),
                    "cold":     (G0i[0],             G0i[1] - (p.cc - p.r)),
                }

                # Pick optimal action at trial R*
                scores = {action: eval_expr(expr, Rstar_trial)
                          for action, expr in candidates.items()}
                best_action = max(scores, key=scores.__getitem__)
                policy[(x, Q, A)] = best_action

                chosen = candidates[best_action]

                # G(X,Q,A) = rho_x*(r + G(0i,Q,-)) + (1-rho_x)*chosen
                g_coef  = rho_x * G0i[0]             + (1 - rho_x) * chosen[0]
                g_const = rho_x * (p.r + G0i[1])     + (1 - rho_x) * chosen[1]
                G[f"{x}_{Q}{A}"] = (g_coef, g_const)

    return G, policy


# ---------------------------------------------------------------------------
# Step 4 — Solve for R* via equation (2.2)
# ---------------------------------------------------------------------------

def solve_Rstar(p: Params, G: Dict[str, LinearExpr]) -> float:
    """
    Equation (2.2):
      R*/q = -R*tau_1 + sum_{Q,A} w(Q,A) * G(1, Q, A)

    Rearranged:
      R* * (1/q - rhs_coef) = rhs_const
      R* = rhs_const / (1/q - rhs_coef)
    """
    wt  = (1 - p.q) * (1 - p.a)
    wq  = p.q        * (1 - p.a)
    wa  = (1 - p.q)  * p.a
    wqa = p.q        * p.a

    def g1(Q, A):
        key = f"S_{Q}{A}" if p.S == 1 else f"1_{Q}{A}"
        return G[key]

    rhs_coef  = -p.tau[1] + wt*g1(0,0)[0] + wq*g1(1,0)[0] + wa*g1(0,1)[0] + wqa*g1(1,1)[0]
    rhs_const =              wt*g1(0,0)[1] + wq*g1(1,0)[1] + wa*g1(0,1)[1] + wqa*g1(1,1)[1]

    denom = (1.0 / p.q) - rhs_coef
    if abs(denom) < 1e-12:
        raise ValueError("Degenerate equation: denominator is zero.")
    return rhs_const / denom


# ---------------------------------------------------------------------------
# Main solver — iterates policy evaluation until stable
# ---------------------------------------------------------------------------

def solve(
    p: Params,
    max_iter: int = 100,
    tol: float = 1e-9,
    verbose: bool = True,
) -> Tuple[float, Dict[str, float], Dict[Tuple[int,int,int], str]]:
    """
    Full iterative solver:
      1. Start with Rstar_trial = 0.
      2. Run recursion to get policy and G constants (as linear exprs).
      3. Solve linear equation for Rstar.
      4. Re-run recursion with new Rstar_trial.
      5. Repeat until Rstar converges.

    Returns
    -------
    Rstar   : float
    G_vals  : dict mapping state key -> evaluated G value
    policy  : dict mapping (X, Q, A) -> action string
    """
    Rstar_trial = 0.0

    for iteration in range(max_iter):
        G = base_constants(p)
        G, policy = compute_G_recursion(p, G, Rstar_trial)
        Rstar_new = solve_Rstar(p, G)

        if verbose:
            print(f"  Iter {iteration+1:3d}: R* = {Rstar_new:.8f}")

        if abs(Rstar_new - Rstar_trial) < tol:
            Rstar = Rstar_new
            break

        Rstar_trial = Rstar_new
    else:
        Rstar = Rstar_new
        if verbose:
            print("  Warning: did not fully converge within max_iter.")

    # Evaluate all G constants at final R*
    G_vals = {k: eval_expr(v, Rstar) for k, v in G.items()}

    return Rstar, G_vals, policy


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_results(
    p: Params,
    Rstar: float,
    G_vals: Dict[str, float],
    policy: Dict[Tuple[int,int,int], str],
) -> None:
    print("\n" + "="*55)
    print(f"  R* (optimal reward rate) = {Rstar:.6f}")
    print("="*55)

    print("\nG constants:")
    base_keys = ["0i_0", "0i_1", "0w_0", "0w_1",
                 "S_00", "S_01", "S_10", "S_11"]
    for k in base_keys:
        if k in G_vals:
            print(f"  G({k.replace('_', ', ')}) = {G_vals[k]:+.6f}")

    for x in range(1, p.S):
        for qa in ["00", "10", "01", "11"]:
            k = f"{x}_{qa}"
            if k in G_vals:
                print(f"  G(X={x}, Q={qa[0]}, A={qa[1]}) = {G_vals[k]:+.6f}")

    print("\nOptimal policy:")
    print(f"  {'X':>4}  {'Q':>3}  {'A':>3}  {'Action':<12}")
    print("  " + "-"*28)
    for (x, Q, A), action in sorted(policy.items()):
        print(f"  {x:>4}  {Q:>3}  {A:>3}  {action:<12}")


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Example parameters
    p = Params(
        S     = 3,
        q     = 0.5,
        a     = 0.4,
        r     = 5.0,
        cw    = 2.0,
        cc    = 3.0,
        tau_w = 2,
        rho   = [None, 0.30, 0.40, 0.50],   # index 0 unused; rho[1..S]
        tau   = [None,    1,    1,    1],    # index 0 unused; tau[1..S]
    )

    print(f"Parameters: S={p.S}, q={p.q}, a={p.a}, r={p.r}, "
          f"cw={p.cw}, cc={p.cc}, tau_w={p.tau_w}")
    print(f"rho = {p.rho[1:]},  tau = {p.tau[1:]}\n")
    print("Iterating to find R*:")

    Rstar, G_vals, policy = solve(p, verbose=True)
    print_results(p, Rstar, G_vals, policy)

    # -------------------------------------------------------------------
    # Sensitivity: vary q from 0.1 to 0.9
    # -------------------------------------------------------------------
    print("\n\nSensitivity of R* to q:")
    print(f"  {'q':>6}  {'R*':>12}")
    print("  " + "-"*22)
    for q_val in np.linspace(0.1, 0.9, 9):
        p_sens = Params(
            S=p.S, q=round(q_val,2), a=p.a, r=p.r,
            cw=p.cw, cc=p.cc, tau_w=p.tau_w,
            rho=p.rho, tau=p.tau,
        )
        Rs, _, _ = solve(p_sens, verbose=False)
        print(f"  {q_val:>6.2f}  {Rs:>12.6f}")
