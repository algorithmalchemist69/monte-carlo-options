# Monte Carlo Option Pricing — Interview Guide

This guide explains every part of the project at the depth a quant interviewer will probe.

---

## 1. The Big Picture (say this first)

> "I simulate thousands of stock price paths under the risk-neutral measure, compute the option payoff at expiry for each path, discount back to today, and average. By the law of large numbers this converges to the true risk-neutral expectation — which is the no-arbitrage option price."

---

## 2. Geometric Brownian Motion (GBM)

### The SDE
```
dS = r·S·dt + σ·S·dW
```
- **S**: stock price
- **r**: risk-free rate (drift under risk-neutral measure Q)
- **σ**: volatility
- **dW**: Wiener process increment ~ N(0, dt)

### Why log-normal?
Taking logs and applying Itô's lemma gives:
```
d(ln S) = (r - σ²/2)·dt + σ·dW
```
So log-returns are normally distributed → prices are **log-normal** (can't go negative).

### Exact simulation (what the code does)
```python
S_T = S0 * exp((r - 0.5*σ²)*T + σ*√T * Z),   Z ~ N(0,1)
```
This is the **exact** solution — not an approximation. The `(r - σ²/2)` term is the **Itô correction**: when you exponentiate a normal random variable, the mean shifts, so you subtract half the variance to keep the expectation at `S0 * exp(r*T)`.

### Interview question: "Why subtract σ²/2?"
> "It's the Itô correction. The expected value of exp(X) where X~N(μ,σ²) is exp(μ + σ²/2). To make E[S_T] = S0·exp(rT), the drift in log-space must be (r - σ²/2), not r."

---

## 3. Risk-Neutral Pricing

### Why use r as drift, not the real-world drift μ?

Under the **risk-neutral measure Q** (guaranteed by no-arbitrage), every asset grows at the risk-free rate. This lets us price by:

```
V₀ = e^(-rT) · E^Q[payoff(S_T)]
```

We don't need to know μ (the real-world drift). This is the **fundamental theorem of asset pricing**.

### Interview question: "What is the risk-neutral measure?"
> "It's a probability measure under which all assets have the same expected return — the risk-free rate. It exists if and only if there's no arbitrage. Under this measure, option pricing becomes computing a discounted expectation, which Monte Carlo can estimate directly."

---

## 4. Black-Scholes as a Benchmark

### Closed-form for European Call
```
C = S·N(d1) - K·e^(-rT)·N(d2)
d1 = [ln(S/K) + (r + σ²/2)·T] / (σ√T)
d2 = d1 - σ√T
```
N(·) is the standard normal CDF.

### Why does MC converge to BS?
Both use the same model (GBM, risk-neutral) — BS is the analytical solution to the same expectation that MC estimates numerically. As N → ∞, MC → BS.

### Interview question: "When would you use MC over BS?"
> "Black-Scholes only works for European options with simple payoffs. Monte Carlo handles: path-dependent options (Asian, barrier, lookback), early exercise (American options via Longstaff-Schwartz), stochastic volatility models (Heston), jump diffusion — anything where there's no closed form."

---

## 5. Convergence and Error

### Standard error of MC estimate
```
SE = σ_payoff / √N
```
So to halve the error, you need **4× more simulations** (√N scaling).

### 95% Confidence Interval
```
Price ± 1.96 · SE
```

### Interview question: "How do you reduce MC error without more simulations?"
> "Variance reduction techniques:
> 1. **Antithetic variates** — use Z and -Z together; payoffs are negatively correlated, averaging them reduces variance
> 2. **Control variates** — use BS price as a control; the MC error on a known quantity tells you how to correct the unknown
> 3. **Quasi-Monte Carlo** — use low-discrepancy sequences (Sobol, Halton) instead of pseudo-random numbers for faster convergence O(1/N) vs O(1/√N)"

---

## 6. Greeks

Greeks measure option price sensitivity to parameters.

### Delta (Δ)
```
Δ = ∂V/∂S ≈ [V(S+h) - V(S-h)] / 2h
```
- Call Delta ∈ (0,1), Put Delta ∈ (-1,0)
- Interpretation: if S increases by $1, call price increases by ~Δ dollars
- Used for **delta hedging**: hold -Δ shares of stock per long call to be instantaneously market-neutral

### Vega (ν)
```
ν = ∂V/∂σ ≈ [V(σ+h) - V(σ-h)] / 2h
```
- Always positive for long options (both calls and puts)
- Interpretation: if σ increases by 1%, option price increases by ν/100 dollars
- At-the-money options have highest vega

### Interview question: "What is delta hedging?"
> "If I'm long a call with Δ = 0.63, I'm long 0.63 shares of delta exposure. To be delta-neutral, I short 0.63 shares. As S changes, Δ changes (that's Gamma), so I need to rebalance — this is dynamic delta hedging. The cost of continuous rehedging is what Black-Scholes prices into the option premium."

### Interview question: "Why is vega always positive for long options?"
> "Higher volatility means larger potential swings. Since options give you the right but not the obligation, you benefit from big moves in either direction but your downside is capped at the premium. More volatility = more chance of a large favourable move = higher option value."

---

## 7. Value at Risk (VaR)

### Definition
```
VaR(α, T) = -Percentile(P&L distribution, 1-α)
```
95% VaR of $380 means: with 95% confidence, your portfolio won't lose more than $380 over 10 days.

### How the code computes it
1. Simulate S after 10 trading days (horizon = 10/252 years)
2. Reprice the call at each scenario using BS (T remaining)
3. P&L = n_contracts × (V_new - V_old)
4. VaR = -5th percentile of the P&L distribution

### Interview question: "What are the limitations of VaR?"
> "Three main ones: (1) it says nothing about the magnitude of losses beyond the threshold — Expected Shortfall (CVaR) fixes this by averaging the tail losses; (2) it assumes the historical/simulated distribution is representative of future risk; (3) VaR is not sub-additive in general — two portfolios can each have low VaR but their combination can have high VaR, violating the intuition that diversification reduces risk."

---

## 8. Questions You Should Be Ready For

### Theory
- What assumptions does GBM make? *(constant σ, continuous trading, no jumps, log-normal returns)*
- What is put-call parity? *(C - P = S - K·e^(-rT); must hold by no-arbitrage)*
- What is implied volatility? *(the σ that makes BS price equal the market price; markets trade implied vol)*
- What is the volatility smile/skew? *(implied vol varies by strike; BS assumes constant σ so it's wrong; skew = OTM puts have higher IV than calls, reflecting crash risk)*

### Coding
- How would you vectorise this further? *(all paths in one matrix multiply; already done in the code)*
- How would you price an Asian option? *(payoff = max(mean(S_t) - K, 0); compute path mean, not just final S_T)*
- How would you price an American option? *(Longstaff-Schwartz: use regression to estimate continuation value at each step, exercise if intrinsic > continuation)*

### Risk/Finance
- What is gamma risk? *(second derivative of V w.r.t. S; delta changes as S moves; you accumulate gamma P&L when dynamically hedging)*
- What is theta? *(time decay; long options lose value as T decreases, all else equal)*
- What happens to option price as σ→0? *(call → max(S - K·e^(-rT), 0); the option converges to its intrinsic value)*

---

## 9. One-line Answers for Fast-Fire Questions

| Question | Answer |
|---|---|
| What is an option? | Right but not obligation to buy (call) or sell (put) at strike K |
| What is the premium? | The no-arbitrage price of that right |
| What is in-the-money? | Call: S > K. Put: S < K |
| What is intrinsic value? | max(S-K, 0) for call; extrinsic = premium - intrinsic |
| Why does time value exist? | More time = more chance stock moves favorably |
| What is a straddle? | Long call + long put at same K; profits from large moves either way |
| What is risk-neutral? | Pricing as if all investors are indifferent to risk |
| What is Ito's lemma? | Chain rule for stochastic processes; used to derive GBM solution and BS PDE |

---

## 10. Elevator Pitch (30 seconds)

> "I built a Monte Carlo pricer for European options. The core idea is: under the risk-neutral measure, every asset grows at the risk-free rate, so I simulate 100,000 stock price paths using Geometric Brownian Motion, compute the call or put payoff at expiry for each path, discount at the risk-free rate, and average. This converges to the Black-Scholes price — I verified within 0.3% error. I also estimated Delta and Vega using finite differences, and computed 10-day 95% VaR for a portfolio of 100 contracts by simulating the P&L distribution."
