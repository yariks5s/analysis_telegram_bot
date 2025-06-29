from typing import Dict, Any, Optional
import pandas as pd

class TransactionCostsModel:
    """
    Model for simulating realistic trading costs including:
    - Exchange fees (maker/taker)
    - Slippage based on order size and volatility
    - Market impact estimation
    """
    
    def __init__(
        self,
        maker_fee_pct: float = 0.1,
        taker_fee_pct: float = 0.1,
        slippage_model: str = "fixed",
        fixed_slippage_pct: float = 0.05,
        slippage_vol_factor: float = 0.5,
        market_impact_factor: float = 0.0,
    ):
        """
        Initialize transaction costs model with specified parameters.
        
        Args:
            maker_fee_pct: Maker fee percentage (e.g., 0.1 for 0.1%)
            taker_fee_pct: Taker fee percentage (e.g., 0.1 for 0.1%)
            slippage_model: Type of slippage model to use ('fixed', 'volatility', or 'advanced')
            fixed_slippage_pct: Fixed slippage percentage when using 'fixed' model
            slippage_vol_factor: Multiplier for ATR-based slippage when using 'volatility' model
            market_impact_factor: Factor for estimating market impact based on position size
        """
        self.maker_fee_pct = maker_fee_pct
        self.taker_fee_pct = taker_fee_pct
        self.slippage_model = slippage_model
        self.fixed_slippage_pct = fixed_slippage_pct
        self.slippage_vol_factor = slippage_vol_factor
        self.market_impact_factor = market_impact_factor
    
    def calculate_entry_costs(
        self,
        entry_price: float,
        amount_to_invest: float,
        signal_type: str,
        atr: Optional[float] = None,
        volume_24h: Optional[float] = None,
        is_market_order: bool = True,
    ) -> Dict[str, float]:
        """
        Calculate entry costs including fees and slippage.
        
        Args:
            entry_price: The theoretical entry price
            amount_to_invest: Amount to be invested in the trade
            signal_type: 'buy' or 'sell'
            atr: Average True Range for volatility-based slippage
            volume_24h: 24-hour trading volume for market impact estimation
            is_market_order: Whether it's a market order (True) or limit order (False)
            
        Returns:
            Dictionary with cost breakdown and adjusted entry price
        """
        # Determine fee percentage based on order type
        fee_pct = self.taker_fee_pct if is_market_order else self.maker_fee_pct
        
        # Calculate base fee
        fee_amount = amount_to_invest * (fee_pct / 100)
        
        # Calculate slippage based on model
        slippage_pct = self._calculate_slippage(
            entry_price, 
            amount_to_invest, 
            signal_type, 
            atr, 
            volume_24h
        )
        
        # Determine slippage direction based on signal
        slippage_direction = 1 if signal_type.lower() == "buy" else -1
        slippage_amount = entry_price * (slippage_pct / 100) * slippage_direction
        
        # Calculate actual entry price after slippage
        actual_entry_price = entry_price + slippage_amount
        
        # Adjust amount after fees
        actual_amount = amount_to_invest - fee_amount
        actual_position = actual_amount / actual_entry_price
        
        return {
            "theoretical_entry_price": entry_price,
            "actual_entry_price": actual_entry_price,
            "fee_percentage": fee_pct,
            "fee_amount": fee_amount,
            "slippage_percentage": slippage_pct,
            "slippage_amount": slippage_amount,
            "original_amount": amount_to_invest,
            "actual_amount": actual_amount,
            "actual_position": actual_position
        }
    
    def calculate_exit_costs(
        self,
        exit_price: float,
        position_size: float,
        signal_type: str,
        atr: Optional[float] = None,
        volume_24h: Optional[float] = None,
        is_market_order: bool = True,
    ) -> Dict[str, float]:
        """
        Calculate exit costs including fees and slippage.
        
        Args:
            exit_price: The theoretical exit price
            position_size: Size of the position being exited
            signal_type: 'buy' or 'sell' (original entry signal, used to determine exit direction)
            atr: Average True Range for volatility-based slippage
            volume_24h: 24-hour trading volume for market impact estimation
            is_market_order: Whether it's a market order (True) or limit order (False)
            
        Returns:
            Dictionary with cost breakdown and adjusted exit price
        """
        # Determine fee percentage based on order type
        fee_pct = self.taker_fee_pct if is_market_order else self.maker_fee_pct
        
        # Calculate position value before fees
        position_value = position_size * exit_price
        
        # Calculate base fee
        fee_amount = position_value * (fee_pct / 100)
        
        # Calculate slippage based on model
        slippage_pct = self._calculate_slippage(
            exit_price, 
            position_value, 
            signal_type, 
            atr, 
            volume_24h
        )
        
        # For exit, slippage direction is opposite of entry
        slippage_direction = -1 if signal_type.lower() == "buy" else 1
        slippage_amount = exit_price * (slippage_pct / 100) * slippage_direction
        
        # Calculate actual exit price after slippage
        actual_exit_price = exit_price + slippage_amount
        
        # Calculate actual amount after fees
        actual_amount = position_size * actual_exit_price - fee_amount
        
        return {
            "theoretical_exit_price": exit_price,
            "actual_exit_price": actual_exit_price,
            "fee_percentage": fee_pct,
            "fee_amount": fee_amount,
            "slippage_percentage": slippage_pct,
            "slippage_amount": slippage_amount,
            "position_value": position_value,
            "actual_amount_received": actual_amount
        }
    
    def _calculate_slippage(
        self,
        price: float,
        amount: float,
        signal_type: str,
        atr: Optional[float] = None,
        volume_24h: Optional[float] = None
    ) -> float:
        """
        Calculate slippage percentage based on the chosen model.
        
        Args:
            price: Current price
            amount: Trade amount
            signal_type: 'buy' or 'sell'
            atr: Average True Range
            volume_24h: 24-hour trading volume
            
        Returns:
            Slippage as a percentage of price
        """
        if self.slippage_model == "fixed":
            return self.fixed_slippage_pct
        
        elif self.slippage_model == "volatility" and atr is not None:
            # Calculate relative ATR as percentage of price
            relative_atr_pct = (atr / price) * 100
            # Scale slippage based on volatility
            return relative_atr_pct * self.slippage_vol_factor
        
        elif self.slippage_model == "advanced" and volume_24h is not None and volume_24h > 0:
            # Advanced model considering market impact
            market_impact = (amount / volume_24h) * 100 * self.market_impact_factor
            
            # Base slippage using volatility component if available
            base_slippage = 0
            if atr is not None:
                relative_atr_pct = (atr / price) * 100
                base_slippage = relative_atr_pct * self.slippage_vol_factor
            else:
                base_slippage = self.fixed_slippage_pct
                
            # Combine base slippage and market impact
            return base_slippage + market_impact
        
        # Default to fixed slippage if other conditions not met
        return self.fixed_slippage_pct

def apply_transaction_costs_to_backtest(
    trades: pd.DataFrame,
    costs_model: TransactionCostsModel,
    atr_series: Optional[pd.Series] = None,
    volume_24h_series: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Apply transaction costs to a dataframe of backtest trades.
    
    Args:
        trades: DataFrame containing trade information
        costs_model: Transaction costs model instance
        atr_series: Optional time series of ATR values
        volume_24h_series: Optional time series of 24h volume
        
    Returns:
        DataFrame with adjusted trade values including costs
    """
    # Create a copy to avoid modifying the original
    adjusted_trades = trades.copy()
    
    # Skip processing if empty
    if adjusted_trades.empty:
        return adjusted_trades
        
    # Check if we have the necessary fields in the trades DataFrame
    # Handle different field name conventions
    field_mappings = {
        'entry_price': ['entry_price', 'entry', 'price'],
        'exit_price': ['exit_price', 'exit', 'close_price'],
        'amount_traded': ['amount_traded', 'amount', 'position_size', 'size'],
        'entry_signal': ['entry_signal', 'signal', 'side', 'trade_type'],
        'profit_loss': ['profit_loss', 'profit', 'pnl', 'realized_pnl']
    }
    
    # Map fields to their actual names in the DataFrame
    actual_fields = {}
    for field, alternatives in field_mappings.items():
        for alt in alternatives:
            if alt in adjusted_trades.columns:
                actual_fields[field] = alt
                break
    
    # Check if we have enough fields to proceed
    required_fields = ['entry_price', 'amount_traded']
    missing_fields = [field for field in required_fields if field not in actual_fields]
    
    if missing_fields:
        print(f"Warning: Cannot apply transaction costs - missing required fields: {missing_fields}")
        print(f"Available columns: {list(adjusted_trades.columns)}")
        return adjusted_trades
    
    # Set default values for optional fields
    if 'entry_signal' not in actual_fields:
        actual_fields['entry_signal'] = None
        adjusted_trades['temp_signal'] = 'buy'  # Default to 'buy' if no signal is provided
    
    # Process each trade
    for i, trade in adjusted_trades.iterrows():
        # Get ATR and volume if available
        atr = None
        volume = None
        
        timestamp_field = 'entry_timestamp' if 'entry_timestamp' in trade else 'timestamp' if 'timestamp' in trade else None
        
        if timestamp_field and atr_series is not None and trade.get(timestamp_field) in atr_series.index:
            atr = atr_series[trade[timestamp_field]]
            
        if timestamp_field and volume_24h_series is not None and trade.get(timestamp_field) in volume_24h_series.index:
            volume = volume_24h_series[trade[timestamp_field]]
        
        # Get values using mapped field names
        entry_price = trade[actual_fields['entry_price']]
        amount_to_invest = trade[actual_fields['amount_traded']]
        signal_type = trade[actual_fields['entry_signal']] if actual_fields.get('entry_signal') else 'buy'
        
        # Calculate entry costs
        try:
            entry_costs = costs_model.calculate_entry_costs(
                entry_price=entry_price,
                amount_to_invest=amount_to_invest,
                signal_type=signal_type,
                atr=atr,
                volume_24h=volume
            )
            
            # Update position size based on fees
            adjusted_position = entry_costs['actual_position']
            
            # Calculate exit costs if trade has been exited
            if 'exit_price' in actual_fields and trade[actual_fields['exit_price']] > 0:
                exit_price = trade[actual_fields['exit_price']]
                
                exit_costs = costs_model.calculate_exit_costs(
                    exit_price=exit_price,
                    position_size=adjusted_position,
                    signal_type='sell' if signal_type.lower() == 'buy' else 'buy',  # Opposite of entry signal
                    atr=atr,
                    volume_24h=volume
                )
                
                # Recalculate profit/loss with transaction costs
                original_profit = trade[actual_fields['profit_loss']] if 'profit_loss' in actual_fields else 0
                adjusted_profit = exit_costs['actual_amount_received'] - entry_costs['original_amount']
                
                # Update trade data
                adjusted_trades.loc[i, actual_fields['entry_price']] = entry_costs['actual_entry_price']
                adjusted_trades.loc[i, actual_fields['exit_price']] = exit_costs['actual_exit_price']
                
                # Add new fields for transaction cost details
                adjusted_trades.loc[i, 'adjusted_position'] = adjusted_position
                adjusted_trades.loc[i, 'entry_fee'] = entry_costs['fee_amount']
                adjusted_trades.loc[i, 'exit_fee'] = exit_costs['fee_amount']
                adjusted_trades.loc[i, 'entry_slippage'] = entry_costs['slippage_amount']
                adjusted_trades.loc[i, 'exit_slippage'] = exit_costs['slippage_amount']
                
                # Store both original and adjusted profit
                if 'profit_loss' in actual_fields:
                    adjusted_trades.loc[i, 'original_' + actual_fields['profit_loss']] = original_profit
                    adjusted_trades.loc[i, actual_fields['profit_loss']] = adjusted_profit
                else:
                    adjusted_trades.loc[i, 'profit_loss'] = adjusted_profit
                
                # Calculate total transaction costs
                adjusted_trades.loc[i, 'transaction_costs'] = (
                    entry_costs['fee_amount'] + 
                    exit_costs['fee_amount'] + 
                    abs(entry_costs['slippage_amount'] * adjusted_position) + 
                    abs(exit_costs['slippage_amount'] * adjusted_position)
                )
        except Exception as e:
            print(f"Warning: Error applying transaction costs to trade {i}: {str(e)}")
            continue
    
    # Clean up temporary columns if created
    if 'temp_signal' in adjusted_trades.columns:
        adjusted_trades = adjusted_trades.drop('temp_signal', axis=1)
    
    return adjusted_trades
