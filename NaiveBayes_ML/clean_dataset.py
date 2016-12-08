import math
import pandas as pd
from datetime import datetime


def get_year(x):
    if x is not None and type(x) is not float:
        try:
            return datetime.strptime(x, '%Y-%m-%d').year
        except ValueError:
            return datetime.strptime(x, '%Y-%m-%d %H:%M:%S').year
    else:
        return 2013     # default to 2013
    pass


def get_month(x):
    if x is not None and type(x) is not float:
        try:
            return datetime.strptime(x, '%Y-%m-%d').month
        except:
            return datetime.strptime(x, '%Y-%m-%d %H:%M:%S').month
    else:
        return 1     # default to 1 (Jan)
    pass



# perform left-join
def left_merge_dataset(left_dframe, right_dframe, merge_column):
    return pd.merge(left_dframe, right_dframe, on=merge_column, how='left')


if __name__ == '__main__':
    # get expedia dataset
    print('Reading data files')
    train_data = pd.read_csv('../../../Datasets/expedia_train.csv')
    dest_data = pd.read_csv('../../../Datasets/destinations.csv')

    # Perform operations on train_data

    """ work on date_time column """
    # add new year,month cols copied from date_time
    print('Creating 2 new columns from date_time, 1 each for year, month')
    train_data['date_time_year'] = pd.Series(train_data.date_time, index=train_data.index)
    train_data['date_time_month'] = pd.Series(train_data.date_time, index=train_data.index)

    # convert year & months to int
    print('Converting the formats for year, month')
    train_data.date_time_year = train_data.date_time_year.apply(lambda x: get_year(x))
    train_data.date_time_month = train_data.date_time_month.apply(lambda x: get_month(x))

    # now remove the date_time column
    print('Deleting date_time column')
    del train_data['date_time']

    # print(train_data.info())

    """ work on srch_ci column """
    # add new year,month cols copied from date_time
    print('Creating 2 new columns from srch_ci, 1 each for year, month')
    train_data['srch_ci_year'] = pd.Series(train_data.srch_ci, index=train_data.index)
    train_data['srch_ci_month'] = pd.Series(train_data.srch_ci, index=train_data.index)

    # convert year & months to int
    print('Converting the formats for year, month')
    train_data.srch_ci_year = train_data.srch_ci_year.apply(lambda x: get_year(x))
    train_data.srch_ci_month = train_data.srch_ci_month.apply(lambda x: get_month(x))

    # now remove the date_time column
    print('Deleting srch_ci column')
    del train_data['srch_ci']

    """ work on srch_co column """
    # add new year,month cols copied from date_time
    print('Creating 2 new columns from srch_co, 1 each for year, month')
    train_data['srch_co_year'] = pd.Series(train_data.srch_co, index=train_data.index)
    train_data['srch_co_month'] = pd.Series(train_data.srch_co, index=train_data.index)

    # convert year & months to int
    print('Converting the formats for year, month')
    train_data.srch_co_year = train_data.srch_co_year.apply(lambda x: get_year(x))
    train_data.srch_co_month = train_data.srch_co_month.apply(lambda x: get_month(x))

    # now remove the date_time column
    print('Deleting srch_co column')
    del train_data['srch_co']

    """ now merge the data-set with destinations """
    print('Merging dataset with destinations data')
    merged_data_set = left_merge_dataset(train_data, dest_data, 'srch_destination_id')

    # now rearranging columns to move hotel_cluster to the end
    print('Moving hotel_cluster column to the end')
    cols = list(merged_data_set.columns.values)
    cols.pop(cols.index('hotel_cluster'))
    merged_data_set = merged_data_set[cols+['hotel_cluster']]

    # write to file
    print('Writing to new file')
    merged_data_set.to_csv('../../../Datasets/expedia_train_cleaned.csv', index=False)

