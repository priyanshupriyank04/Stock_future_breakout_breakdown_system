## Strategy Overview

This system analyzes **NSE stock futures** using daily OHLC data and technical indicators to detect potential breakout and breakdown signals.  
The strategy calculates:
- **ADX and DI+ / DI−** for trend strength.
- **CBOE-based Odd Bull / Odd Bear sentiment metrics**.
- **Stochastic K/D crossovers** for momentum confirmation.

Two types of alerts are generated:
1. **Relative Strength Crossovers** – Triggers when Odd Bull/Bear and DI lines cross each other, with confirmation from Stochastic K/D.
2. **Absolute Threshold Crossovers** – Triggers when Odd Bull/Bear and DI cross below key levels (15 for sentiment, 9 for DI) with Stochastic confirmation.

All triggered futures are saved into CSV files for easy monitoring:
- `bull_relative_stoch.csv`, `bear_relative_stoch.csv`  
- `green_absolute.csv`, `red_absolute.csv`
