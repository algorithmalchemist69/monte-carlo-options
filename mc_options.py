"""
Monte Carlo Option Pricing with GBM
- Prices European Call/Put options via simulation
- Benchmarks against Black-Scholes closed-form
- Estimates Delta and Vega via pathwise/bump methods
- Computes 95% Value-at-Risk for a simple options portfolio
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Black-Scholes closed-form ────────────────────────────────────────────────

def bs_price(S, K, T, r, sigma, option="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bs_delta(S, K, T, r, sigma, option="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if option == "call" else norm.cdf(d1) - 1

# ── Monte Carlo engine ───────────────────────────────────────────────────────

def simulate_paths(S0, r, sigma, T, n_steps, n_sims, seed=42):
    """Simulate GBM paths using vectorised Euler-Maruyama."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    Z = rng.standard_normal((n_sims, n_steps))
    # log-price increments  ←  exact GBM solution
    increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.log(S0) + np.cumsum(increments, axis=1)
    paths = np.concatenate(
        [np.full((n_sims, 1), S0), np.exp(log_paths)], axis=1
    )
    return paths  # shape: (n_sims, n_steps+1)

def mc_price(S0, K, T, r, sigma, n_sims=100_000, option="call", seed=42):
    """Price European option; return (price, std_error)."""
    paths = simulate_paths(S0, r, sigma, T, n_steps=1, n_sims=n_sims, seed=seed)
    S_T = paths[:, -1]
    if option == "call":
        payoffs = np.maximum(S_T - K, 0)
    else:
        payoffs = np.maximum(K - S_T, 0)
    discounted = np.exp(-r * T) * payoffs
    return discounted.mean(), discounted.std() / np.sqrt(n_sims)

# ── Greeks via bump-and-reprice ──────────────────────────────────────────────

def mc_delta(S0, K, T, r, sigma, h=0.01, n_sims=100_000, option="call"):
    """Central-difference Delta: (V(S+h) - V(S-h)) / 2h."""
    p_up, _ = mc_price(S0 + h, K, T, r, sigma, n_sims, option, seed=0)
    p_dn, _ = mc_price(S0 - h, K, T, r, sigma, n_sims, option, seed=0)
    return (p_up - p_dn) / (2 * h)

def mc_vega(S0, K, T, r, sigma, h=0.001, n_sims=100_000, option="call"):
    """Central-difference Vega: (V(σ+h) - V(σ-h)) / 2h."""
    p_up, _ = mc_price(S0, K, T, r, sigma + h, n_sims, option, seed=0)
    p_dn, _ = mc_price(S0, K, T, r, sigma - h, n_sims, option, seed=0)
    return (p_up - p_dn) / (2 * h)

# ── Portfolio VaR ────────────────────────────────────────────────────────────

def portfolio_var(S0, K, T, r, sigma, n_contracts=100, horizon=10/252,
                  confidence=0.95, n_sims=50_000, seed=7):
    """
    1-day (horizon) VaR for a long call portfolio.
    Simulate S after `horizon` days; reprice option; compute P&L.
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_sims)
    S_h = S0 * np.exp((r - 0.5 * sigma**2) * horizon + sigma * np.sqrt(horizon) * Z)

    T_rem = T - horizon
    V0, _ = mc_price(S0, K, T, r, sigma, n_sims=200_000, option="call", seed=99)

    # reprice at each scenario
    V_h = np.array([bs_price(s, K, T_rem, r, sigma) for s in S_h])
    pnl = n_contracts * (V_h - V0)

    var = -np.percentile(pnl, (1 - confidence) * 100)
    return var, pnl

# ── Convergence study ────────────────────────────────────────────────────────

def convergence_study(S0, K, T, r, sigma, option="call"):
    sizes = [500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000]
    prices, errors = [], []
    true_price = bs_price(S0, K, T, r, sigma, option)
    for n in sizes:
        p, se = mc_price(S0, K, T, r, sigma, n, option)
        prices.append(p)
        errors.append(se)
    return sizes, prices, errors, true_price

# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_results(S0, K, T, r, sigma, n_sims=50_000):
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # 1) Sample GBM paths
    ax1 = fig.add_subplot(gs[0, 0])
    paths = simulate_paths(S0, r, sigma, T, n_steps=252, n_sims=200, seed=0)
    t = np.linspace(0, T, 253)
    for path in paths:
        ax1.plot(t, path, lw=0.4, alpha=0.3, color="steelblue")
    ax1.axhline(K, color="red", lw=1.2, linestyle="--", label=f"Strike K={K}")
    ax1.set_title("GBM Simulated Paths")
    ax1.set_xlabel("Time (years)")
    ax1.set_ylabel("Stock Price ($)")
    ax1.legend(fontsize=8)

    # 2) Convergence of MC price to BS
    ax2 = fig.add_subplot(gs[0, 1])
    sizes, prices, errors, true_price = convergence_study(S0, K, T, r, sigma)
    errors = np.array(errors)
    prices = np.array(prices)
    ax2.semilogx(sizes, prices, "o-", color="steelblue", label="MC Price")
    ax2.fill_between(sizes, prices - 1.96 * errors, prices + 1.96 * errors,
                     alpha=0.2, color="steelblue", label="95% CI")
    ax2.axhline(true_price, color="red", lw=1.2, linestyle="--", label=f"BS Price={true_price:.4f}")
    ax2.set_title("MC Convergence to Black-Scholes")
    ax2.set_xlabel("Number of Simulations")
    ax2.set_ylabel("Option Price ($)")
    ax2.legend(fontsize=8)

    # 3) Option price vs Spot (smile-like payoff diagram)
    ax3 = fig.add_subplot(gs[1, 0])
    spots = np.linspace(70, 130, 60)
    bs_calls = [bs_price(s, K, T, r, sigma, "call") for s in spots]
    bs_puts  = [bs_price(s, K, T, r, sigma, "put")  for s in spots]
    mc_calls = [mc_price(s, K, T, r, sigma, 30_000, "call", seed=1)[0] for s in spots]
    ax3.plot(spots, bs_calls, "b-",  lw=2,   label="BS Call")
    ax3.plot(spots, bs_puts,  "r-",  lw=2,   label="BS Put")
    ax3.plot(spots, mc_calls, "b--", lw=1.2, label="MC Call")
    ax3.axvline(K, color="gray", lw=0.8, linestyle=":")
    ax3.set_title("Option Price vs Spot Price")
    ax3.set_xlabel("Spot Price ($)")
    ax3.set_ylabel("Option Price ($)")
    ax3.legend(fontsize=8)

    # 4) Portfolio P&L distribution + VaR
    ax4 = fig.add_subplot(gs[1, 1])
    var, pnl = portfolio_var(S0, K, T, r, sigma)
    ax4.hist(pnl, bins=80, color="steelblue", alpha=0.7, edgecolor="none")
    ax4.axvline(-var, color="red", lw=2, label=f"95% VaR = ${var:,.0f}")
    ax4.set_title("Portfolio P&L Distribution (10-day)")
    ax4.set_xlabel("P&L ($)")
    ax4.set_ylabel("Frequency")
    ax4.legend(fontsize=8)

    plt.suptitle("Monte Carlo Option Pricing — European Call/Put", fontsize=13, fontweight="bold")
    plt.savefig("mc_options_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: mc_options_results.png")

# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Parameters
    S0    = 100.0   # current stock price
    K     = 100.0   # strike (at-the-money)
    T     = 1.0     # time to expiry (1 year)
    r     = 0.05    # risk-free rate
    sigma = 0.20    # volatility
    N     = 100_000 # simulations

    print("=" * 55)
    print("  Monte Carlo Option Pricing — European Options")
    print("=" * 55)
    print(f"  S0={S0}, K={K}, T={T}yr, r={r:.0%}, σ={sigma:.0%}")
    print()

    for opt in ("call", "put"):
        mc, se    = mc_price(S0, K, T, r, sigma, N, opt)
        bs        = bs_price(S0, K, T, r, sigma, opt)
        delta_mc  = mc_delta(S0, K, T, r, sigma, option=opt)
        delta_bs  = bs_delta(S0, K, T, r, sigma, opt)
        vega_mc   = mc_vega(S0, K, T, r, sigma, option=opt)
        print(f"  [{opt.upper()}]")
        print(f"    MC Price : {mc:.4f}  ±{1.96*se:.4f}  (95% CI)")
        print(f"    BS Price : {bs:.4f}")
        print(f"    Error    : {abs(mc-bs):.4f}  ({abs(mc-bs)/bs*100:.2f}%)")
        print(f"    Delta    : MC={delta_mc:.4f}  BS={delta_bs:.4f}")
        print(f"    Vega     : MC={vega_mc:.4f}")
        print()

    var, _ = portfolio_var(S0, K, T, r, sigma)
    print(f"  Portfolio VaR (100 calls, 10-day, 95%): ${var:,.2f}")
    print()
    print("  Generating plots…")
    plot_results(S0, K, T, r, sigma)
