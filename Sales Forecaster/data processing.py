import pandas as pd
import numpy as np


# reading in data, cleaning column names, initial data cleansing/filtering

df = pd.read_csv('/Users/gabrielreyes/Documents/VScode Projects/sales_forecaster/Sales Forecaster/historical_sales.csv')

rename_cols = {'Item Description: Product Name':'Product', 'Item Description: Country':'Country',
'Item Description: Product Family':'Family', 'Item Description: Size':'Size', 'Shipping State/Province':'Shipping State',
'Item Description: MSRP / Bottle':'MSRP', 'Item Description: Percent Alcohol':'% Alc',
'Item Description: Total Cases OH':'Total Cases OH'}

df = df.rename(columns=rename_cols)

df = df[df['Total Cases OH'] != '#Error!']
df['Total Cases OH'] = pd.to_numeric(df['Total Cases OH'])

df['Delivery Date'] = pd.to_datetime(df['Delivery Date']).dt.to_period('M')
df['Year'] = df['Delivery Date'].dt.year
df['Month'] = df['Delivery Date'].dt.month

df = df[(df['Sample'] == 'N') & (df['Warehouse'] != 'DSW')]


# df of active products to predict - to use in later prediction df

active = df[df['Total Cases OH'] != 0][['Family', 'Size']].drop_duplicates()


# formula to get first month of activity for each product, used to differentiate NaNs when data is pivoted

def get_first_sale(df):
    first_sale = pd.pivot_table(df, index=['Family', 'Size'], values='Delivery Date', aggfunc=np.min).reset_index()
    first_sale = first_sale.rename(columns={'Delivery Date':'First Sale'})

    merge = pd.merge(df, first_sale, how='left', on=['Family', 'Size'])

    return merge

df = get_first_sale(df)


# selecting main columns to allow for quick iterations

cols = ['Brand', 'Family', 'Size', 'First Sale']


# tranforming data into monthly sales, grouping by above columns

df = (pd.pivot_table(df,
            index=cols,
            columns=['Delivery Date'],
            values=['Cases Sold'],
            aggfunc=np.sum)
            .sort_values(['Family', 'Size']))


# cleaning multi-level columns for cleaner output

df = df.droplevel(level=0, axis=1).sort_index(axis=1, ascending=False).reset_index()


# now that no monthly gaps, melting back to format that allows groupby shift for previous months

df = pd.melt(df, id_vars=['Brand', 'Family', 'Size', 'First Sale'], value_name='Cases Sold')


# eliminating erroneous data (this can be combined w/ initial df processing)

df = df[df['Delivery Date'] <= pd.Timestamp.today().to_period('M')]


# two step process to segment NaNs into true zero sale months vs months before product arrival

df = df.fillna(0)

df['Cases Sold'] = (np.where(df['Delivery Date'] < df['First Sale'], np.nan, df['Cases Sold']))



# adding on all previous month data

df['Last Month'] = df.groupby(['Brand', 'Family', 'Size'])['Cases Sold'].shift(-1)
df['2 Months Ago'] = df.groupby(['Brand', 'Family', 'Size'])['Cases Sold'].shift(-2)
df['3 Months Ago'] = df.groupby(['Brand', 'Family', 'Size'])['Cases Sold'].shift(-3)
df['4 Months Ago'] = df.groupby(['Brand', 'Family', 'Size'])['Cases Sold'].shift(-4)
df['5 Months Ago'] = df.groupby(['Brand', 'Family', 'Size'])['Cases Sold'].shift(-5)
df['6 Months Ago'] = df.groupby(['Brand', 'Family', 'Size'])['Cases Sold'].shift(-6)
df['7 Months Ago'] = df.groupby(['Brand', 'Family', 'Size'])['Cases Sold'].shift(-7)
df['8 Months Ago'] = df.groupby(['Brand', 'Family', 'Size'])['Cases Sold'].shift(-8)
df['9 Months Ago'] = df.groupby(['Brand', 'Family', 'Size'])['Cases Sold'].shift(-9)
df['10 Months Ago'] = df.groupby(['Brand', 'Family', 'Size'])['Cases Sold'].shift(-10)
df['11 Months Ago'] = df.groupby(['Brand', 'Family', 'Size'])['Cases Sold'].shift(-11)
df['Last Year'] = df.groupby(['Brand', 'Family', 'Size'])['Cases Sold'].shift(-12)