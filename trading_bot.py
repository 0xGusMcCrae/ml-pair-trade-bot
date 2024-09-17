import os
import time
import ccxt
import pandas as pd
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler
import joblib

from dotenv import load_dotenv

from logger_config import LoggerConfig

logger_config = LoggerConfig()
log = logger_config.get_logger()

load_dotenv()


class TradingBot():
    """CLASS DOCSTRING"""

    def __init__(self):
        self.model = joblib.load('joblib/crypto_trading_model.pkl')
        self.scaler = joblib.load('joblib/scaler.pkl')
        self.class_to_label = joblib.load('joblib/class_to_label.pkl')
        print(f"class_to_label: {self.class_to_label}")
        self.test_run = os.getenv("TEST_RUN") == "True"
        self.exchange = ccxt.hyperliquid({
            'walletAddress': os.getenv("ACCOUNT_ADDRESS"),
            'privateKey': os.getenv("AGENT_PRIVATE_KEY") if not self.test_run else os.getenv("TESTNET_PRIVATE_KEY"),
            'enableRateLimit': True,
        })
        if self.test_run:
            self.exchange.set_sandbox_mode(True)
        self.symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'DOGE/USDT:USDT',
                        'AAVE/USDT:USDT']
        self.leverage_multiplier = int(os.getenv("LEVERAGE_MULTIPLIER"))
        self.usdt_to_usdc_mapping = { # Model is trained using binance's USDT pairs, so we need to convert from USDT to USDC to interact with hyperliquid api
            'BTC/USDT:USDT': 'BTC/USDC:USDC',
            'ETH/USDT:USDT': 'ETH/USDC:USDC',
            'SOL/USDT:USDT': 'SOL/USDC:USDC',
            'DOGE/USDT:USDT': 'DOGE/USDC:USDC',
            'INJ/USDT:USDT': 'INJ/USDC:USDC',
            'FTM/USDT:USDT': 'FTM/USDC:USDC',
            'SAGA/USDT:USDT': 'SAGA/USDC:USDC',
            'TAO/USDT:USDT': 'TAO/USDC:USDC',
            'SEI/USDT:USDT': 'SEI/USDC:USDC',
            'SUI/USDT:USDT': 'SUI/USDC:USDC',
            'AAVE/USDT:USDT': 'AAVE/USDC:USDC',
            '1000PEPE/USDT:USDT': 'kPEPE/USDC:USDC'
        }
        self.usdc_to_usdt_mapping = {v: k for k, v in self.usdt_to_usdc_mapping.items()}

    def convert_symbol_to_usdc(self, symbol):
        return self.usdt_to_usdc_mapping[symbol]
    
    def convert_symbol_to_usdt(self, symbol):
        return self.usdc_to_usdt_mapping[symbol]
    
    def fetch_latest_data(self, symbols):
        dfs = []
        for symbol in symbols:
            try: 
                df = self.exchange.fetch_ohlcv(symbol, timeframe='1d', limit=100)
            except Exception as e:
                log.error(f"Error fetching latest data for {symbol}: {e}")
                continue #probably need to just shut down the whole bot if we get this error - missing data would mess up the model
            df = pd.DataFrame(df, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = self.convert_symbol_to_usdt(symbol) #convert HL's usdc symbol back to model's binance usdt symbol
            dfs.append(df)
        combined_df = pd.concat(dfs)
        combined_df.set_index('Date', inplace=True)
        return combined_df
    
    def preprocess_data(self, df):
        # Assign symbol_code
        df['symbol_code'] = df['symbol'].apply(lambda x: self.symbols.index(x))

        # Initialize an empty list to store processed groups
        dfs = []

        # Group the data by symbol and process each group
        for symbol, group in df.groupby('symbol'):
            group = group.copy()
            # Calculate indicators
            group['returns'] = group['close'].pct_change()
            group['rsi'] = ta.rsi(group['close'], length=14)
            stoch = ta.stoch(group['high'], group['low'], group['close'])
            group['stoch_d'] = stoch['STOCHd_14_3_3']
            # Append the processed group to the list
            dfs.append(group)

        # Concatenate all the processed groups
        df = pd.concat(dfs)

        # Drop rows with NaNs
        df.dropna(inplace=True)
        return df

    def prepare_features(self, df):
        features = ['returns', 'rsi', 'symbol_code']  
        X = df[features]
        X_scaled = self.scaler.transform(X)
        return X_scaled

    def predict_signals(self, X_scaled):
        y_pred = self.model.predict(X_scaled)
        y_pred_labels = pd.Series(y_pred).map(self.class_to_label)
        return y_pred_labels
    
    def calculate_order_size(self, symbol, current_price):
        """Half of the account size, inclusive of leverage"""
        account_balance = float(self.exchange.fetch_balance()['info']['marginSummary']['accountValue'])
        usd_value = account_balance * self.leverage_multiplier / 2
        # convert to # of coins
    
        return usd_value / current_price / 10
        # return 0
    
    def get_current_price(self, symbol):
        ticker = self.exchange.fetch_ticker(symbol)
        current_price = ticker['last']
        return current_price
    
    def close_position(self, position, current_price):
        symbol = position['symbol']
        contracts = position['contracts']  # Number of contracts in the open position
        side = position['side']
        
        # Determine the opposite order type
        close_side = 'sell' if side == 'long' else 'buy'
        
        # Close the position by placing an order in the opposite direction
        self.exchange.create_order(
            symbol=symbol, 
            type='market', 
            side=close_side, 
            amount=contracts, 
            price=current_price,
            params={
                'slippage': 0.005
            }
        )
        log.info(f"Closed {contracts} contracts of {symbol} in a {side} position.")
        
    def execute_trade(self, symbol, signal):
        # convert binance usdt symbols to hyperliquid usdc symbols
        symbol = self.convert_symbol_to_usdc(symbol)

        current_price = self.get_current_price(symbol)

        # if the buy/sell is the same as the currently open long/short, nothing happens. 
        # likewise, if it's NOT the same, we need to close the open position
        positions = self.exchange.fetch_positions()
        for position in positions:
            if position['symbol'] == symbol:
                if (signal == 1 and position['side'] == 'long') or (signal == -1 and position['side'] == 'short'):
                    # the desired position is already open, do nothing
                    log.info(f"Keeping {position['side']} position open for {symbol}")
                    return
                else:
                    # either closing (signal == 0) or flipping between long and short (i.e. signal was 1, now it's -1) for this position
                    # the opposite order will be opened below where necessary
                    self.close_position(position, current_price)                

        print(f"Symbol: {symbol}, Signal: {signal}")
        
        
        amount = self.calculate_order_size(symbol, current_price)

        if signal == 1:
            # Place a market buy order
            self.exchange.create_order(
                symbol=symbol, 
                type='market', 
                side='buy', 
                amount=amount,
                price=current_price,
                params={
                    'slippage': 0.005
                }
            )
            log.info(f"Bought {amount} of {symbol}")
        elif signal == -1:
            # Place a market sell order
            self.exchange.create_order(
                symbol=symbol, 
                type='market', 
                side='sell', 
                amount=amount,
                price=current_price,
                params={
                    'slippage': 0.005
                }
            )
            log.info(f"Sold {amount} of {symbol}")

    def run(self):
        while True:
            # Fetch data
            df = self.fetch_latest_data(
                [self.convert_symbol_to_usdc(symbol) for symbol in self.symbols] #convert model's binance usdt symbol to HL usdc symbol to interact with api
            )
            
            if df.empty:
                log.error("df is empty. Latest data not fetched")

            # Preprocess data
            df = self.preprocess_data(df)

            # Reset index to ensure proper alignment
            df.reset_index(drop=True, inplace=True)

            # Prepare features
            X_scaled = self.prepare_features(df)

            # Predict signals
            df['predicted_label'] = self.predict_signals(X_scaled)
            # print(f"predicted_label: {df['predicted_label']}")
            
            for symbol in self.symbols:
                symbol_df = df[df['symbol'] == symbol]
                if symbol_df.empty:
                    continue
                latest_entry = symbol_df.iloc[-1]
                signal = latest_entry['predicted_label']

                # Execute trade
                try:
                    self.execute_trade(symbol, signal)
                except Exception as e:
                    log.error(f"Error executing trade for {symbol}: {e}")
                    continue
        
            log.info("Waiting for next daily candle...")
            time.sleep(24*60*60) #wait for next daily candle


if __name__ == "__main__":
    bot = TradingBot()
    while True:
        try:
            bot.run()
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            log.error(f"Connection lost error: {e}")
        except KeyboardInterrupt:
            log.warning("Keyboard interrupt")
            break
