import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import time

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


if __name__ == '__main__':
    # test_run_mc1p1()
    # test_run_mc1p2()
    test_run_mc1p3()
