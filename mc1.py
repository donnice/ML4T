import pandas as pd

def test_run():
    start_date = '2012-01-22'
    end_date = '2012-01-26'
    dates = pd.date_range(start_date, end_date)

    #Create an empty dataframe
    df1 = pd.DataFrame(index=dates)

    #Read SPY data into temporary dataframe
    dfSPY = pd.read_csv("data/SPY.csv", index_col="Date", parse_dates=True)

    #Join the two dataframes using DataFrame.join()
    df1 = df1.join(dfSPY)
    print df1
    

if __name__ == '__main__':
    test_run()