# CryptoBot: Advanced Crypto Signal Telegram Bot

## Overview

CryptoBot is an advanced Telegram bot for cryptocurrency traders, providing automated technical analysis, signal generation, and charting for any crypto pair. It leverages a suite of custom indicators and multi-timeframe analysis to deliver actionable trading signals, visualizations, and user-customizable preferences—all via Telegram.

---

## Features

- **Automated Signal Generation**: Multi-timeframe, probability-based bullish/bearish/neutral signals for any crypto pair.
- **Custom Technical Indicators**: Includes Order Blocks, Fair Value Gaps (FVGs), Liquidity Levels, and Breaker Blocks.
- **Interactive Telegram Bot**: Manage signals, select indicators, and receive charts directly in Telegram.
- **User Preferences**: Per-user indicator selection, chart options (legend, volume), and signal frequency.
- **Backtesting & Strategy Optimization**: Built-in backtesting and training modules for strategy evaluation.
- **Database-Backed**: User preferences and signal jobs are persisted in a local SQLite database.
- **Beautiful Charting**: Candlestick charts with overlays for all supported indicators.

---

## Supported Indicators

- **Order Blocks**: Detects key supply/demand zones based on price impulses and reversals.
- **Fair Value Gaps (FVGs)**: Identifies price gaps (inefficiencies) that may act as support/resistance.
- **Liquidity Levels**: Finds significant support/resistance using ATR-based fractal pivots and clustering.
- **Breaker Blocks**: Detects liquidity sweeps and reversals, highlighting potential breakout/reversal zones.

All indicators can be enabled/disabled per user.

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd cryptoBot
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   - Create a `.env` file in the root directory.
   - Add your Telegram bot token:
     ```env
     API_TELEGRAM_KEY=your_telegram_bot_token_here
     ```

4. **(Optional) Run tests:**
   ```bash
   pytest
   ```

---

## Running the Bot

1. **Start the bot:**
   ```bash
   python bot.py
   ```
   The bot will initialize the database (`preferences.db`) automatically if it does not exist.

2. **Add your bot to Telegram and start a chat.**

---

## Customer/User Instructions

### Main Commands

- `/chart <symbol> <hours> <interval> <tolerance>`
  - Get a candlestick chart with all enabled indicators.
  - Example: `/chart BTCUSDT 48 1h 0.05`

- `/text_result <symbol> <hours> <interval> <tolerance>`
  - Get a text summary of all detected indicators.
  - Example: `/text_result ETHUSDT 24 15m 0.03`

- `/preferences`
  - Interactive menu to select which indicators to use and chart options (legend, volume).

- `/create_signal <symbol> <minutes> [<is_with_chart>]`
  - Start receiving auto-signals for a pair at a given frequency (in minutes).
  - Example: `/create_signal BTCUSDT 60 true`

- `/delete_signal <symbol>`
  - Stop auto-signals for a pair.
  - Example: `/delete_signal BTCUSDT`

- `/manage_signals`
  - Interactive menu to view, add, or delete your signal jobs.

### Signal Jobs
- Signals are sent automatically at your chosen frequency, with or without charts, based on your preferences.
- You can have up to 10 active signal jobs per user.

### Indicator Selection
- Use `/preferences` to enable/disable Order Blocks, FVGs, Liquidity Levels, Breaker Blocks, and chart options.
- Preferences are saved per user.

---

## Charting & Visualization
- Charts are generated using `mplfinance` and `matplotlib`.
- Overlays include:
  - Order Blocks (bullish/bearish)
  - FVGs (highlighted gaps)
  - Liquidity Levels (horizontal lines)
  - Breaker Blocks (colored zones)
  - Optional legend and volume

---

## Database
- Uses SQLite (`preferences.db`) for user preferences and signal job persistence.
- No manual setup required; tables are created automatically on first run.

---

## Backtesting & Strategy Training
- The `back_tester/` directory contains scripts for backtesting and optimizing signal logic.
- See `back_tester/strategy.py` and `back_tester/trainer.py` for details.

---

## Contributing & Testing
- Contributions are welcome! Please add tests for new features.
- Run all tests with:
  ```bash
  pytest
  ```

---

## Requirements

All dependencies are listed in `requirements.txt`. Key packages:
- python-telegram-bot
- pandas, numpy, scikit-learn, scipy
- matplotlib, mplfinance
- python-dotenv
- requests

---

## License

TBD

---

## Contact

For questions or support, open an issue or contact the maintainer (@yariks5s).

## Contibutions

You also have a chance to contribute to this project. Please create a pull request containing your ideas or solutions. All contributions are welcome.

### Todo list (can be extended by anyone):
- [x] Create the Breaker Block indicator
[Maybe other indicators will be useful]
- [x] Create every indicator output as classes
- [x] Create a possibility to return non-visual (numeric) data
- [x] Add tests for this
- [x] Create a possibility to enable/disable indicators
- [x] Connect the bot to database (probably local), to store the user_ids and their preferences
- [x] Possibility to calculate indicators only on demand
- [x] Add tests for this
- [x] Create a logic to find predictable movements, based on the created indicators
- [ ] Tune this logic properly
- [ ] Add tests for this
- [x] Don't count very small FVGs
- [x] Impove handling the cases where some indicators are not shown (Covered FVGs in text representation)
- [x] Tune order block detection - **Important**
- [x] Keep previous 1000 candles in history and send signal based on the extended analysis (probably multi-frame) - done in another way (merging requests responses)
- [x] Create a logic regarding pinging users about signals
- [ ] Add tests for this
- [ ] Check if a requested cryptocurrency pair exists on a Bybit (suggest the correct name using Levenstein distance) - not important
- [x] Refactor obsolete function for 1k candles, put the logic into the basic function
- [x] Create a signal finding pipeline
- [ ] Tune the signal finding logic
- [x] Create an adequate UI for managing signals
- [x] Prevent from creating a multiple signal queries for the same currency
- [x] Possibility to choose whether we need a chart along with the signal using query setting
- [x] Clarify the exception message for this use-case: '/create_signal ARBUSDT 1 2'
- [x] Possibility to apply only chosen indicators to signals
- [x] Limit frequency of signals and amount of signals to not violate the API limits
- [x] Testing system (backtesting)
- [x] Abilty to toggle legend
- [x] Ability to toggle volumes
- [x] Optimize the process of normalizing the liquidity levels if no settings are specified
- [x] Add a test for the task above (automatically create the right amount of liquidity levels if setting is not specified)
- [x] Maybe do not add covered FVGs at all? UPD: Yes
- [ ] Logging system
- [ ] Disable info logging for training
- [ ] Liquidity pools implementation
- [ ] Tests for liquidity pools
- [ ] Ability to show historical (not real-time) data
- [ ] Disclaimer in /help temporary debugging commands
- [ ] Implement testing for proper backtesting
- [ ] Multi language support
- [ ] Implement advanced risk management tools (e.g., stop-loss/take-profit suggestions)
- [ ] Allow users to set custom indicator parameters (e.g., ATR period, FVG min size)
- [ ] Add portfolio tracking: Let users track their holdings and PnL
- [ ] Add a web dashboard for visualizing signals and statistics outside Telegram
- [ ] Add a “strategy marketplace” where users can share and use custom signal strategies
- [ ] Implement a notification system for major market events (e.g., high volatility, news)
- [ ] Add onboarding/tutorial messages for new users
- [ ] Provide inline help for each command (e.g., /help chart)
- [ ] Allow users to export their signal history (CSV, Excel, etc.)
- [ ] Add dark/light mode for charts
- [ ] Implement rate limiting and abuse prevention
- [ ] Make the system that will set the right logic coefficients based on backtesting
- [ ] If possible, make the bot to continue sending signals after it is restarted
- [ ] Create a neural network and make it learn on own data
- [ ] Integrate a neural network to the predictions system
- [ ] Prettify the bot and make it easy to use
