import pandas as pd
import os
import matplotlib.pyplot as plt

def get_max_close(symbol):
    '''Return the maximum closing value for stock indicated by symbol.'''
    df = pd.read_csv("data/{}.csv".format(symbol)) # read in data
    return df['Close'].max() # compute and return max

def get_mean_volume(symbol):
    '''Return the mean volume for stock indicated by symbol.'''
    df = pd.read_csv("data/{}.csv".format(symbol))
    return df['Volume'].mean()

def get_data(symbols, dates):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols.insert(0, 'SPY')

    for symbol in symbols:
        # TODO: Read and join data for each symbol
        df_temp = pd.read_csv("data/{}.csv".format(symbol), 
                              index_col="Date",
                              parse_dates=True,
                              usecols=['Date', 'Adj Close'],
                              na_values='nan')
        df_temp = df_temp.rename(columns={'Adj Close':symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':
            df = df.dropna()

    return df

def plot_selected(df, columns, start_index, end_index):
    plot_data(df.ix[start_index:end_index, columns], title="Selected data")

def normalize_data(df):
    '''Normalize stock prices using the first row of the df.'''
    #df.ix[0, :] will give us the first row
    return df / df.ix[0, :]

def plot_data(df, title="Stock prices"):
    '''Plot stock prices'''
    ax = df.plot(title=title, fontsize=8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.show()

def test_run_mc1p1():
    df = pd.read_csv("data/AAPL.csv")
    # Print the last 5 rows of the data frame
    # print df.tail(5)
    # print df[10:21]
    # for symbol in ['AAPL', 'IBM']:
    #     # print "Max close"
    #     # print symbol, get_max_close(symbol)
    #     print "Mean volume"
    #     print symbol, get_mean_volume(symbol)
    # df = pd.read_csv("data/IBM.csv")
    # print df['Adj Close']
    # print df['High']
    # df['Adj Close'].plot()
    # df['High'].plot()
    df[['Close', 'Adj Close']].plot()
    plt.show()

def test_run_mc1p2():
    # Define a date range
    dates = pd.date_range('2010-01-01', '2010-12-31')

    # Choose stock symbols to read
    symbols = ['GOOG', 'IBM', 'GLD']
    
    # Get stock data
    df = get_data(symbols, dates)
    
    # Slice by row range (dates) using DataFrame.ix[] selector
    # print df.ix['2010-01-01':'2010-01-31']
    
    # Slice by column (symbols)
    # print df['GOOG']
    # print df[['IBM', 'GLD']]

    # Slice by row and column
    print df.ix['2010-03-10':'2010-03-15', ['IBM', 'GLD']]
    #plot_data(df)
    #plot_selected(df, ['SPY', 'IBM'], '2010-03-01', '2010-04-01')
    plot_data(normalize_data(df))


if __name__ == '__main__':
    test_run_mc1p1()
    # test_run_mc1p2()