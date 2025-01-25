import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# 计算 RSI 的函数
def calculate_rsi(data, period=14):
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    data['RSI'] = rsi

    return data

# RSI 回测函数
def backtest_rsi(data, rsi_period=14, overbought=70, oversold=30, stop_loss_pct=0.05, initial_balance=10000):
    data = calculate_rsi(data, period=rsi_period)
    balance = initial_balance
    position = 0
    entry_price = 0

    for i in range(1, len(data)):
        row = data.iloc[i]
        prev_row = data.iloc[i - 1]

        # Buy signal: RSI crosses above the oversold level
        if position == 0 and prev_row['RSI'] < oversold and row['RSI'] >= oversold:
            position = balance / row['Close']
            entry_price = row['Close']
            balance = 0

        # Sell signal: RSI crosses below the overbought level or stop-loss triggered
        elif position > 0:
            stop_loss_price = entry_price * (1 - stop_loss_pct)
            if (row['RSI'] <= overbought and prev_row['RSI'] > overbought) or row['Close'] <= stop_loss_price:
                balance = position * row['Close']
                position = 0

    # Final balance
    if position > 0:
        balance += position * data.iloc[-1]['Close']

    return balance

# 优化参数函数
def optimize_rsi(data, rsi_periods, overbought_levels, oversold_levels, stop_loss_pcts):
    results = []
    for rsi_period, overbought, oversold, stop_loss_pct in product(rsi_periods, overbought_levels, oversold_levels, stop_loss_pcts):
        if overbought <= oversold:
            continue  # Skip invalid combinations

        final_balance = backtest_rsi(data, rsi_period, overbought, oversold, stop_loss_pct)
        results.append({
            'RSI Period': rsi_period,
            'Overbought Level': overbought,
            'Oversold Level': oversold,
            'Stop Loss %': stop_loss_pct,
            'Final Balance': final_balance
        })

    results_df = pd.DataFrame(results)
    return results_df

# 可视化 2D 热图
def plot_optimization_results(results_df):
    pivot = results_df.pivot_table(
        index='RSI Period',
        columns='Overbought Level',
        values='Final Balance',
        aggfunc='max'
    )

    plt.figure(figsize=(10, 8))
    plt.title("Optimization Results: Final Balance")
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis")
    plt.xlabel("Overbought Level")
    plt.ylabel("RSI Period")
    plt.show()

# 可视化 3D 图
def plot_3d_optimization_results(results_df):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    rsi_period = results_df['RSI Period']
    overbought = results_df['Overbought Level']
    oversold = results_df['Oversold Level']
    final_balance = results_df['Final Balance']

    scatter = ax.scatter(rsi_period, overbought, oversold, c=final_balance, cmap='viridis', alpha=0.8)

    ax.set_title('3D Optimization Results: Final Balance', pad=20)
    ax.set_xlabel('RSI Period', labelpad=10)
    ax.set_ylabel('Overbought Level', labelpad=10)
    ax.set_zlabel('Oversold Level', labelpad=10)

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, aspect=10)
    cbar.set_label('Final Balance')

    plt.show()

# 数据获取与处理
ticker = "002175.sz"  # Replace with your preferred stock ticker
data = yf.download(ticker, start="2014-01-01", end="2024-01-01")
data_clean = data.copy()
data_clean = data_clean.ffill()
data = data_clean.copy()

# 参数优化
rsi_periods = range(5, 21, 1)  # RSI periods from 5 to 20
overbought_levels = range(65, 86, 5)  # Overbought levels from 65 to 85
oversold_levels = range(15, 36, 5)  # Oversold levels from 15 to 35
stop_loss_pcts = [0.02, 0.05, 0.1]  # Stop-loss percentages

results_df = optimize_rsi(data, rsi_periods, overbought_levels, oversold_levels, stop_loss_pcts)

# 输出最佳参数
best_params = results_df.loc[results_df['Final Balance'].idxmax()]
print("Best Parameters:")
print(best_params)

# 绘制优化结果
plot_optimization_results(results_df)
plot_3d_optimization_results(results_df)
