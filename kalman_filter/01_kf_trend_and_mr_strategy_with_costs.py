# =========================================================
# Kalman Filter for Trend Extraction + Trading Strategies
#
# Features:
#   - Download data automatically from Yahoo Finance (set ticker)
#   - OR load a local CSV file with 'Date' and 'Close' columns
#   - Includes transaction costs (default: 0.1% per trade)
#   - Strategies:
#       1) Trend Following
#       2) Mean Reversion
#   - Outputs:
#       - Trading signals plotted on price
#       - Cumulative returns vs Buy & Hold
#       - Summary table with performance metrics
# =========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# 1. Choose data source
# -------------------------
# Option A: Yahoo Finance (requires internet + yfinance installed)
# pip install yfinance
use_yahoo = True
ticker = "AAPL"  # change this to any valid Yahoo Finance ticker, e.g. "BTC-USD", "MSFT", "TSLA"

# Option B: Local CSV file (must contain "Date" and "Close")
local_csv = "AAPL.csv"

if use_yahoo:
    import yfinance as yf
    df = yf.download(ticker, period="5y")  # last 5 years daily
    df = df.reset_index()[["Date", "Close"]]
    df = df.set_index("Date")
else:
    df = pd.read_csv(local_csv, parse_dates=['Date'])
    df = df.set_index('Date')

prices = df['Close'].astype(float)

# -------------------------
# 2. Set Kalman Filter parameters
# -------------------------
Q = 1e-3   # Process variance (smoothness of trend)
R = 4.0    # Observation variance (noise level)

x_est = prices.iloc[0]
P = 1.0
kf_trend = []

# -------------------------
# 3. Run Kalman Filter loop
# -------------------------
for z in prices:
    x_pred = x_est
    P_pred = P + Q
    K = P_pred / (P_pred + R)
    x_est = x_pred + K * (z - x_pred)
    P = (1 - K) * P_pred
    kf_trend.append(x_est)

df['KF_trend'] = kf_trend
df['KF_slope'] = df['KF_trend'].diff()
df['Residual'] = df['Close'] - df['KF_trend']

# -------------------------
# 4. Create trading signals
# -------------------------

# --- Trend Following ---
df['Signal_TF'] = 0
df.loc[(df['Close'] > df['KF_trend']) & (df['KF_slope'] > 0), 'Signal_TF'] = 1
df.loc[(df['Close'] < df['KF_trend']) & (df['KF_slope'] < 0), 'Signal_TF'] = -1
df['Position_TF'] = df['Signal_TF'].shift(1).fillna(0)

# --- Mean Reversion ---
threshold = df['Residual'].std()
df['Signal_MR'] = 0
df.loc[df['Residual'] > threshold, 'Signal_MR'] = -1
df.loc[df['Residual'] < -threshold, 'Signal_MR'] = 1
df['Position_MR'] = df['Signal_MR'].shift(1).fillna(0)

# -------------------------
# 5. Backtest strategies with transaction costs
# -------------------------
df['Return'] = np.log(df['Close'] / df['Close'].shift(1))
transaction_cost = 0.001  # 0.1% cost per trade

def compute_strategy_returns(position, returns, cost=0.001):
    strat_ret = position * returns
    trades = position.diff().abs()
    strat_ret = strat_ret - trades * cost
    return strat_ret

df['TF_Return'] = compute_strategy_returns(df['Position_TF'], df['Return'], transaction_cost)
df['MR_Return'] = compute_strategy_returns(df['Position_MR'], df['Return'], transaction_cost)

df['BuyHold_CumRet'] = df['Return'].cumsum().apply(np.exp)
df['TF_CumRet'] = df['TF_Return'].cumsum().apply(np.exp)
df['MR_CumRet'] = df['MR_Return'].cumsum().apply(np.exp)

# -------------------------
# 6. Visualization
# -------------------------
plt.figure(figsize=(12,6))
plt.plot(df.index, df['Close'], label="Observed Price", alpha=0.6)
plt.plot(df.index, df['KF_trend'], label="Kalman Trend", linewidth=2)
plt.scatter(df.index[df['Signal_TF'] == 1], df['Close'][df['Signal_TF'] == 1],
            marker='o', facecolors='none', edgecolors='green', label="TF Buy")
plt.scatter(df.index[df['Signal_TF'] == -1], df['Close'][df['Signal_TF'] == -1],
            marker='o', facecolors='none', edgecolors='red', label="TF Sell")
plt.scatter(df.index[df['Signal_MR'] == 1], df['Close'][df['Signal_MR'] == 1],
            marker='^', color='green', label="MR Buy", alpha=0.7)
plt.scatter(df.index[df['Signal_MR'] == -1], df['Close'][df['Signal_MR'] == -1],
            marker='v', color='red', label="MR Sell", alpha=0.7)
plt.title(f"{ticker} - Kalman Filter Strategies with Trading Signals")
plt.legend()
plt.show()

plt.figure(figsize=(12,6))
plt.plot(df.index, df['BuyHold_CumRet'], label="Buy & Hold", alpha=0.7)
plt.plot(df.index, df['TF_CumRet'], label="Trend Following (with costs)", linewidth=2)
plt.plot(df.index, df['MR_CumRet'], label="Mean Reversion (with costs)", linewidth=2)
plt.title(f"Backtest: {ticker} KF Strategies vs Buy & Hold (with Transaction Costs)")
plt.ylabel("Cumulative Return (log scale)")
plt.yscale("log")
plt.legend()
plt.show()

# -------------------------
# 7. Performance metrics
# -------------------------
def performance_metrics(returns, name="Strategy"):
    ann_factor = 252
    mean_ret = returns.mean() * ann_factor
    vol = returns.std() * np.sqrt(ann_factor)
    sharpe = mean_ret / vol if vol != 0 else np.nan
    cum_ret = np.exp(returns.cumsum().iloc[-1])
    cum_curve = np.exp(returns.cumsum())
    max_dd = (cum_curve / cum_curve.cummax() - 1).min()
    return {
        "Strategy": name,
        "Annualized Return": round(mean_ret, 4),
        "Annualized Vol": round(vol, 4),
        "Sharpe": round(sharpe, 2),
        "Cumulative Return": round(cum_ret, 2),
        "Max Drawdown": round(max_dd, 2)
    }

summary = []
summary.append(performance_metrics(df['Return'].dropna(), "Buy & Hold"))
summary.append(performance_metrics(df['TF_Return'].dropna(), "Trend Following (with costs)"))
summary.append(performance_metrics(df['MR_Return'].dropna(), "Mean Reversion (with costs)"))

summary_table = pd.DataFrame(summary)
print(summary_table)

# -------------------------
# 8. Save outputs
# -------------------------
df.to_csv(f"{ticker}_with_KF_strategies_with_costs.csv")
summary_table.to_csv(f"{ticker}_strategy_summary_with_costs.csv", index=False)