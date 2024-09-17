import ccxt
import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
import joblib

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

##### DATA COLLECTION #####

# List of cryptocurrencies to include in the basket
# symbols = ['BTC/USDC:USDC', 'ETH/USDC:USDC', 'SOL/USDC:USDC', 'DOGE/USDC:USDC', 'INJ/USDC:USDC', 'FTM/USDC:USDC',
#            'SAGA/USDC:USDC', 'TAO/USDC:USDC', 'SEI/USDC:USDC', 'SUI/USDC:USDC', 'AAVE/USDC:USDC', 'kPEPE/USDC:USDC']

symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'DOGE/USDT:USDT', 
           'AAVE/USDT:USDT']

# Initialize the exchange
exchange = ccxt.binance({'enableRateLimit': True}) # Use binance for more robust historical data

# Define the timeframe and since date
timeframe = '1d'
since = exchange.parse8601('2021-01-01T00:00:00Z')  # Start date for data

# Dictionary to store data
data = {}

# Fetch historical data for each symbol
for symbol in symbols:
    print(f"Fetching data for {symbol}...")
    ohlcv = []
    limit = 1000  # Maximum number of candles per request
    all_since = since
    while True:
        candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=all_since, limit=limit)
        if not candles:
            break
        ohlcv.extend(candles)
        all_since = candles[-1][0] + 1  # Move to the next timestamp
        if len(candles) < limit:
            break
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('Date', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']]
    data[symbol] = df


##### FEATURE ENGINEERING #####

# Function to calculate technical indicators
def add_indicators(df):
    df['returns'] = df['close'].pct_change()
    df['rsi'] = ta.rsi(df['close'], length=14)
    macd = ta.macd(df['close'])
    df['macd'] = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']
    stoch = ta.stoch(df['high'], df['low'], df['close'])
    df['stoch_k'] = stoch['STOCHk_14_3_3']
    df['stoch_d'] = stoch['STOCHd_14_3_3']
    df['willr'] = ta.willr(df['high'], df['low'], df['close'], length=14)
    df['cci'] = ta.cci(df['high'], df['low'], df['close'], length=14)
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    adx = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['adx'] = adx['ADX_14']
    df['mfi'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14)
    df['sma21'] = ta.sma(df['close'], length=21)
    df['ema21'] = ta.ema(df['close'], length=21)
    df['sma100'] = ta.sma(df['close'], length=100)
    df['ema100'] = ta.ema(df['close'], length=100)
    df['next_return'] = df['returns'].shift(-1)
    df.dropna(inplace=True)
    return df

# Apply the function to each DataFrame
for symbol in symbols:
    data[symbol] = add_indicators(data[symbol])


##### DATA PREPARATION #####

# Combine all data into a single DataFrame
dfs = []
for symbol in symbols:
    df = data[symbol].copy()
    df['symbol'] = symbol
    dfs.append(df)

combined_df = pd.concat(dfs)
combined_df.reset_index(inplace=True)

# Remove any duplicate dates per symbol
combined_df.drop_duplicates(subset=['Date', 'symbol'], inplace=True)

# Sort by date and symbol
combined_df.sort_values(by=['Date', 'symbol'], inplace=True)

# Create the target variable
# Rank cryptocurrencies by next day's return for each date
combined_df['rank'] = combined_df.groupby('Date')['next_return'].rank(method='first', ascending=False)

# Number of symbols
num_symbols = len(symbols)

# Assign labels: 1 for top performer, -1 for worst performer, 0 for others
def assign_label(row):
    if row['rank'] == 1:
        return 1
    elif row['rank'] == num_symbols:
        return -1
    else:
        return 0

combined_df['Label'] = combined_df.apply(assign_label, axis=1)

# Drop rows with missing values in features or target
combined_df.dropna(subset=['Label', 'returns', 'volume', 'rsi', 'macd', 'macd_signal', 'stoch_k', 'stoch_d',
                           'willr', 'cci', 'atr', 'adx', 'mfi', 'sma21', 'ema21', 'sma100', 'ema100'], inplace=True)

# Encode the symbol as a categorical variable
combined_df['symbol_code'] = combined_df['symbol'].apply(lambda x: symbols.index(x))

# Features and target
features = ['returns', 'rsi', 'symbol_code']#, 'rsi', 'macd', 'macd_signal', 'stoch_k', 'stoch_d',
         #   'willr', 'cci', 'atr', 'adx', 'mfi', 'symbol_code']
X = combined_df[features]
y = combined_df['Label']


##### Model Training #####

# Check class distribution
counter = Counter(y)
print('Label distribution:', counter)

# Split data into training and testing sets based on date
dates = combined_df['Date'].sort_values().unique()
train_dates, test_dates = train_test_split(dates, test_size=0.2, shuffle=False)

train_df = combined_df[combined_df['Date'].isin(train_dates)]
test_df = combined_df[combined_df['Date'].isin(test_dates)]

X_train = train_df[features]
y_train = train_df['Label']
X_test = test_df[features]
y_test = test_df['Label']

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the fitted scaler to a file
joblib.dump(scaler, 'joblib/scaler.pkl')


# Map labels to class indices
label_to_class = {label: idx for idx, label in enumerate(sorted(y_train.unique()))}
print(f"label_to_class mapping: {label_to_class}")
y_train_mapped = y_train.map(label_to_class)
y_test_mapped = y_test.map(label_to_class)

# Calculate class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_mapped), y=y_train_mapped)
class_weights_dict = dict(zip(np.unique(y_train_mapped), class_weights))
print('Class weights:', class_weights_dict)

# Map class weights to sample weights
sample_weights = y_train_mapped.map(class_weights_dict)

# Initialize XGBoost classifier
model = XGBClassifier(
    objective='multi:softmax',
    num_class=3,
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42
)

# Fit the model with sample weights
model.fit(X_train_scaled, y_train_mapped, sample_weight=sample_weights)


##### Strategy Implementation #####

# Predict on test data
y_pred = model.predict(X_test_scaled)

# Map class indices back to labels
class_to_label = {v: k for k, v in label_to_class.items()}
print(f"Class to Label Mapping: {class_to_label}")
joblib.dump(class_to_label, 'joblib/class_to_label.pkl')
y_pred_labels = pd.Series(y_pred).map(class_to_label)

# Add predictions to test_df
test_df = test_df.reset_index(drop=True)
test_df['predicted_label'] = y_pred_labels.values

# Initialize portfolio
initial_capital = 100000  # $100,000
portfolio = initial_capital
portfolio_values = []
dates = sorted(test_df['Date'].unique())

for date in dates[:-1]:  # Exclude the last date
    day_df = test_df[test_df['Date'] == date]
    next_day_df = test_df[test_df['Date'] == date + timedelta(days=1)]
    
    # Identify positions
    long_positions = day_df[day_df['predicted_label'] == 1]
    short_positions = day_df[day_df['predicted_label'] == -1]
    
    num_positions = len(long_positions) + len(short_positions)
    if num_positions == 0:
        portfolio_values.append({'Date': date, 'Portfolio': portfolio})
        continue
    
    # Equal allocation to each position
    position_size = portfolio / num_positions
    
    daily_pnl = 0  # Profit and loss for the day
    
    # Long positions
    for idx, row in long_positions.iterrows():
        symbol = row['symbol']
        entry_price = row['close']
        exit_row = next_day_df[next_day_df['symbol'] == symbol]
        if exit_row.empty:
            continue
        exit_price = exit_row['close'].values[0]
        returns = (exit_price - entry_price) / entry_price
        pnl = position_size * returns
        daily_pnl += pnl
        # Subtract transaction cost
        daily_pnl -= position_size * 0.001  # Assuming 0.1% per trade
    
    # Short positions
    for idx, row in short_positions.iterrows():
        symbol = row['symbol']
        entry_price = row['close']
        exit_row = next_day_df[next_day_df['symbol'] == symbol]
        if exit_row.empty:
            continue
        exit_price = exit_row['close'].values[0]
        returns = (entry_price - exit_price) / entry_price
        pnl = position_size * returns
        daily_pnl += pnl
        # Subtract transaction cost
        daily_pnl -= position_size * 0.0005  # Assuming 0.1% per trade
    
    portfolio += daily_pnl
    portfolio_values.append({'Date': date, 'Portfolio': portfolio})


##### Backtesting and Evaluation ##### 

portfolio_df = pd.DataFrame(portfolio_values)
portfolio_df.set_index('Date', inplace=True)

plt.figure(figsize=(12, 6))
plt.plot(portfolio_df.index, portfolio_df['Portfolio'])
plt.title('Portfolio Value Over Time')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.grid(True)
plt.show()

# Calculate daily returns
portfolio_df['Daily_Return'] = portfolio_df['Portfolio'].pct_change().fillna(0)

# Cumulative Return
cumulative_return = (portfolio_df['Portfolio'].iloc[-1] / initial_capital) - 1
print(f"Cumulative Return: {cumulative_return * 100:.2f}%")

# Annualized Return
num_days = len(portfolio_df)
annualized_return = ((1 + cumulative_return) ** (365 / num_days)) - 1
print(f"Annualized Return: {annualized_return * 100:.2f}%")

# Sharpe Ratio
sharpe_ratio = (portfolio_df['Daily_Return'].mean() / portfolio_df['Daily_Return'].std()) * np.sqrt(252)
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

# Maximum Drawdown
running_max = portfolio_df['Portfolio'].cummax()
drawdown = (portfolio_df['Portfolio'] - running_max) / running_max
max_drawdown = drawdown.min()
print(f"Maximum Drawdown: {max_drawdown * 100:.2f}%")

print("Classification Report:")
print(classification_report(y_test, y_pred_labels))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_labels)
print("Confusion Matrix:")
print(cm)

# Save the model
joblib.dump(model, 'joblib/crypto_trading_model.pkl')