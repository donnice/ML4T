import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import scipy.optimize as spo

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

def plot_data(df, title="Stock prices", ylabel="Price", xlabel="Date"):
    '''Plot stock prices'''
    ax = df.plot(title=title, fontsize=8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.show()


def get_max_index(a):
    """Return the index of the maximum value in given 1D array."""
    return a.argmax()

def how_long(func, *args):
    t0 = time.time()
    result = func(*args)
    t1 = time.time()
    return result, t1 - t0

def manual_mean(arr):
    """Compute mean (average) of all elements in the given 2D array"""
    sum = 0
    for i in xrange(0, arr.shape[0]):
        for j in xrange(0, arr.shape[1]):
            sum = sum + arr[i, j]
    return sum / arr.size

def numpy_mean(arr):
    return arr.mean()

def get_rolling_mean(values, window):
    """Return rolling mean of given values, using specified window size."""
    return pd.rolling_mean(values, window=window)

def get_rolling_std(values, window):
    """Return rolling mean of given values, using specified window size."""
    return pd.rolling_std(values, window=window)

def get_bollinger_bands(rm, rstd):
    """Return upper and lower Bollinger Bands."""
    upper_band = rm + rstd * 2
    lower_band = rm - rstd * 2
    return upper_band, lower_band

def compute_daily_returns(df):
    """ Compute and return daily return values."""
    daily_returns = df.copy()
    # computer daily returns for row 1 onwards
    daily_returns[1:] = (df[1:]/df[:-1].values)-1
    daily_returns.ix[0, :] = 0
    return daily_returns

def fill_missing_values(df_data):
    """Fill missing values in data frame, in place."""
    df_data.fillna(method="ffill", inplace=True) # forward fill
    df_data.fillna(method="bfill", inplace=True) # backward fill
    
    return df_data

def f(X):
    """Given a scalar X, return some value"""
    Y = (X - 1.5)**2 + 0.5
    print "X = {}, Y = {}".format(X, Y)
    return Y

def error(line, data):
    """ error between given line model and observed data. 

    Parameters
    ------------
    line: tuple/list/array ((C0, C1)) where C0 is slope and C1 is Y-intercept
    data: 2D array where each row is a point (x, y)

    Returns error as a single real value
    """
    # Metric: Sum of squared Y-axis differences
    err = np.sum((data[:, 1] - (line[0] * data[:, 0] + line[1])) ** 2)
    return err

def error_poly(C, data):
    """ error between given polynomial and observed data. 

    Parameters
    ------------
    line: numpy.poly1d object or equivalent array representing polynomial coefficients
    data: 2D array where each row is a point (x, y)

    Returns error as a single real value
    """
    # Metric: Sum of squared Y-axis differences
    err = np.sum((data[:, 1] - np.polyval(C, data[:, 0])) ** 2)
    return err

def fit_line(data, error_func):
    """ Fit a line to given data, using a supplied error function

    Parameters
    -----------
    data: 2D array where each row is a point (X0, Y)
    error_func: function that computes the error function.
    """
    # Generate initial guess for line model
    l = np.float32([0, np.mean(data[:, 1])]) # slope = 0, intercept = mean y

    # Call optimizer to minimize err function
    result = spo.minimize(error_func, l, args=(data,), method='SLSQP', options={'display': True})
    return result.x

def fit_poly(data, error_func, degree=3):
    """ Fit a polynomial to given data, using a supplied error function

    Parameters
    -----------
    data: 2D array where each row is a point (X0, Y)
    error_func: function that computes the error function.
    """
    # Generate initial guess for line model
    Cguess = np.poly1d(np.ones(degree + 1, dtype=np.float32))

    # Call optimizer to minimize err function
    result = spo.minimize(error_func, Cguess, args=(data,), method='SLSQP', options={'display': True})
    return np.poly1d(result.x)


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

def test_run_mc1p3():
    # print np.array([(2, 3, 4), (5, 6, 7)])
    # print np.empty(5)
    # print np.empty((5, 4))
    # print np.ones((5, 4), dtype=np.int_)

    # filled with specified value
    # print np.full((3, 5), 7, dtype=np.int_)

    # Generate an array full of random numbers, uniformly sampled from [0, 1)
    # print np.random.random((5, 4))
    # print np.random.rand(5, 4) #function arguments (not a tuple), not recommended

    # Sample numbers from a Gaussian (normal distribution)
    # print np.random.normal(size=(2, 3)) # Standard normal (mean = 0, s.d. = 1)
    # print np.random.normal(50, 10, size=(2, 3)) # mean = 50, s.d. = 10

    # Random integers
    # print np.random.randint(10) # a single integer from [0, 10)
    # print np.random.randint(5, 20) # a single integer from [5, 20) 
    # print np.random.randint(0, 10, size=5) # 5 random numbers
    # print np.random.randint(0, 10, size=(2, 3)) 2x3 random numbers

    # Shape
    # a = np.random.random((5, 4, 3, 2)) # 5x4 number of array
    # print a.shape[0]
    # print a.shape[1]
    # print len(a.shape)
    # print a.size # 5x4x3x2
    # print a.dtype

    # Mathematics
    np.random.seed(693) # seed the random number generator to make the random output same
    # a = np.random.randint(0, 10, size=(5, 4))
    # a = np.random.rand(5, 4)
    # print "Array:\n", a
    # # Sum of all elements
    # print "Sum of all the elements:", a.sum()
    # # Iterate over rows, to compute sum of each column: 0 = rows
    # print "Sum of each column:\n", a.sum(axis=0)
    # # Iterate over columns, to compute sum of each row: 1 = columns
    # print "Sum of each row:\n", a.sum(axis=1)
    # #Statistics: min, max, mean (across rows, columns and overall)
    # print "Minimum of each column:\n", a.min(axis=0)
    # print "Maximum of each row:\n", a.max(axis=1)
    # print "Mean of all elements:\n", a.mean()

    # Maximum index
    # a = np.array([9, 6, 2, 3, 12, 14, 7, 10], dtype=np.int32)  # 32-bit integer array
    # print "Array:", a
    # # Find the maximum and its index in array
    # print "Maximum value:", a.max()
    # print "Index of max:", get_max_index(a)

    # Test function time
    # t1 = time.time()
    # print "ML4T"
    # t2 = time.time()
    # print "The time taken by the print statement is", t2 - t1, "seconds"

    # nd1 = np.random.random((1000, 10000)) # sufficient large array

    # # Time the two functions, retriving results and execute times
    # res_manual, t_manual = how_long(manual_mean, nd1)
    # res_numpy, t_numpy = how_long(numpy_mean, nd1)
    # print "Manual: {:.6f} ({:.3f} secs.) vs Numpy: {:.6f} ({:.3f} secs.)".format(res_manual, t_manual, res_numpy, t_numpy)
    # # Make sure both gives us the same answer
    # assert abs(res_manual - res_numpy) <= 10e-06, "results are not equal!!"
    # # Compute speedup
    # speedup = t_manual / t_numpy
    # print "Numpy mean is", speedup, "times faster than manual loops."

    # Accessing elements
    # element = a[3, 2]
    # print a[0, 1:3]

    # Slicing n:m:t specifies a range that starts at n, ends before m in t for every row
    # print a[:, 0:3:2]

    # Modify the array
    # a[0, 0] = 1
    # a[0, :] = 2
    # a[:, 3] = [1, 2, 3, 4, 5]
    # print "\nModified array:", a

    # a = np.random.rand(5)
    # # accessing using list of indices
    # indices = np.array([1, 1, 2, 3])
    # print a[indices]

    # a = np.array([(20, 25, 10, 23, 26, 32, 10, 5, 0), (0, 2, 50, 20, 0, 1, 28, 5, 0)])
    # print a
    # # calculate mean
    # mean = a.mean()
    # print mean
    # # masking
    # # print a[a<mean]
    # a[a<mean] = mean
    # print a

    # Arithmetic operations
    a = np.array([(1, 2, 3, 4, 5), (10, 20, 30, 40, 50)])
    print "Original array a:\n", a
    # Multiply a by 2
    print "\nMultiply a by 2:\n", a/2.0
    b = np.array([(100, 200, 300, 400, 500), (10, 20, 30, 40, 50)])
    print "Original array a:\n", b
    # Add the two arrays
    print "\nAdd a + b:\n", a + b
    # Multiply the two arrays
    print "\nAdd a * b:\n", a * b

def test_run_mc1p4():
    # # Read data
    # dates = pd.date_range('2010-01-01', '2010-12-31')
    # symbols = ['GOOG', 'IBM', 'GLD']
    # df = get_data(symbols, dates)
    # plot_data(df)

    # # Compute global statistics for each stock
    # print df.mean()
    # print df.std()

    # Read data
    dates = pd.date_range('2012-07-01', '2012-07-31')
    symbols = ['SPY', 'XOM']
    df = get_data(symbols, dates)

    # Plot SPY data, retain matplotlib axis object
    # ax = df['SPY'].plot(title="SPY rolling mean", label="SPY")
    # Compute the rolling mean using a 20-day window
    # rm_SPY = pd.rolling_mean(df['SPY'], window=20)
    # # Add rolling mean to same plot
    # rm_SPY.plot(label='Rolling mean', ax=ax) # add to the existing plot
    
    # # Add axis labels and legend
    # ax.set_xlabel("Date")
    # ax.set_ylabel("Price")
    # ax.legend(loc='upper left')
    # plt.show()
    # 1. Compute rolling mean
    # rm_SPY = get_rolling_mean(df['SPY'], window=20)

    # # 2. Compute rolling standard deviation
    # rstd_SPY = get_rolling_std(df['SPY'], window=20)

    # # 3. Compute upper and lower bands
    # upper_band, lower_band = get_bollinger_bands(rm_SPY, rstd_SPY)

    # # Plot raw SPY values, rolling mean and Bollinger Bands
    # ax = df['SPY'].plot(title="Bollinger Bands", label="SPY")
    # # Add rolling mean to same plot
    # rm_SPY.plot(label='Rolling mean', ax=ax) # add to the existing plot
    # upper_band.plot(label='Upper band', ax=ax)
    # lower_band.plot(label='lower band', ax=ax)
    # plt.show()
    daily_returns = compute_daily_returns(df)
    plot_data(daily_returns)

def test_run_mc1p5():
    symbollist=["FAKE2"]
    # date range
    start_date='2005-12-31'
    end_date='2014-12-07'
    # create date range
    idx = pd.date_range(start_date, end_date)
    df_data=get_data(symbollist, idx)
    #get adjusted close of each symbol
    fill_missing_values(df_data)
    
    plot_data(df_data)

def test_run_mc1p6():
    # Read data
    dates = pd.date_range('2009-01-01', '2012-12-31')
    symbols = ['SPY', 'XOM', 'GLD']
    df = get_data(symbols, dates)
    # plot_data(df)

    # Compute daily return
    daily_returns = compute_daily_returns(df)
    # plot_data(daily_returns, title="Daily returns", ylabel="Daily returns", xlabel="Days")

    # Plot a histogram
    # daily_returns.hist(bins=20)
    # daily_returns['SPY'].hist(bins=20, label='SPY')
    # daily_returns['XOM'].hist(bins=20, label='XOM')

    # Get mean and standard deviation
    # mean = daily_returns['SPY'].mean()
    # print "mean=", mean
    # std = daily_returns['SPY'].std()
    # print "std=", std

    # verticle line
    # plt.axvline(mean, color='w', linestyle='dashed', linewidth=2) # mean line
    # plt.axvline(std, color='r', linestyle='dashed', linewidth=2) # standard deviation line
    # plt.axvline(-std, color='r', linestyle='dashed', linewidth=2) # standard deviation line

    # Scatter plot SPY vs XOM
    daily_returns.plot(kind='scatter', x='SPY', y='XOM')
    # alpha means how well it performs with x value
    beta_XOM, alpha_XOM = np.polyfit(daily_returns['SPY'], daily_returns['XOM'], 1) # make x and y fit a line
    plt.plot(daily_returns['SPY'], beta_XOM*daily_returns['SPY']+alpha_XOM, '-', color='r')
    
    # daily_returns.plot(kind='scatter', x='SPY', y='GLD')
    plt.show()

    # Calculate correlation coefficient
    print daily_returns.corr(method='pearson')

def test_run_mc1p8():
    # Xguess = 2.0
    # min_result = spo.minimize(f, Xguess, method='SLSQP', options={'disp': True})
    # print "Minima found at:"
    # print "X = {}, Y = {}".format(min_result.x, min_result.fun)

    # # Plot function values, mark minima
    # Xplot = np.linspace(0.5, 2.5, 21)
    # Yplot = f(Xplot)
    # plt.plot(Xplot, Yplot)
    # plt.plot(min_result.x, min_result.fun, 'ro')
    # plt.title("Minima of an objective function")
    # plt.show()

    # Define original line
    l_orig = np.float32([4, 2])
    print "Original line: C0 = {}, C1 = {}".format(l_orig[0], l_orig[1])
    Xorig = np.linspace(0, 10, 21)
    Yorig = l_orig[0] * Xorig + l_orig[1]
    plt.plot(Xorig, Yorig, 'b--', linewidth=2.0, label="Original line")

    # Generate nosidy data point
    noise_sigma = 3.0
    noise = np.random.normal(0, noise_sigma, Yorig.shape)
    data = np.asarray([Xorig, Yorig + noise]).T
    plt.plot(data[:, 0], data[:, 1], 'go', label="Data points")

    # Try to fit a line to this data
    # l_fit = fit_line(data, error)
    l_fit = fit_poly(data, error)
    print "Fitted line: C0={}, C1={}".format(l_fit[0], l_fit[1])
    plt.plot(data[:, 0], l_fit[0] * data[:, 0]+l_fit[1], 'r--', linewidth=2.0, label="Fit line")

    plt.show()


if __name__ == '__main__':
    # test_run_mc1p1()
    # test_run_mc1p2()
    # test_run_mc1p3()
    # test_run_mc1p4()
    # test_run_mc1p5()
    # test_run_mc1p6()
    test_run_mc1p8()
