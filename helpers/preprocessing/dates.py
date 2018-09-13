import pandas as pd


def parse_date(series):
    """
    Separates date object into separate days and months columns
    and returns dataframe
    :param df: pd.DataFrame
    :param date_column_name: str
    :return: pd.DataFrame
    """
    if type(series) != pd.Series:
        raise TypeError(series + ' is not of type pd.Series')
    date = pd.to_datetime(series)
    days = {0: 'Mon', 1: 'Tues', 2: 'Weds', 3: 'Thurs', 4: 'Fri', 5: 'Sat',
            6: 'Sun'}
    df = pd.DataFrame()
    df['day_of_week'] = date.apply(lambda x: days[x.dayofweek])
    df['month'] = date.dt.month
    return df





