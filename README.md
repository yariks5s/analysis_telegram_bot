<img width="872" alt="image" src="https://github.com/user-attachments/assets/98c74f20-6201-48bc-8134-38e004c9f0d0" />

Todo:
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
- [ ] Tune order block detection - **Important**
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
- [ ] Make the system that will set the right logic coefficients based on backtesting
- [ ] If possible, make the bot to continue sending signals after it is restarted
- [ ] Create a neural network and make it learn on own data
- [ ] Integrate a neural network to the predictions system
- [ ] Prettify the bot and make it easy to use
