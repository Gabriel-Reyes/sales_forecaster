import pandas as pd
import numpy as np


# reading in data, cleaning column names, initial data cleansing/filtering

df = pd.read_csv('Sales Forecaster/historical_sales.csv')

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

monthly = (pd.pivot_table(df,
            index=cols,
            columns=['Delivery Date'],
            values=['Cases Sold'],
            aggfunc=np.sum)
            .sort_values(['Family', 'Size']))


# cleaning multi-level columns for cleaner output

monthly = monthly.droplevel(level=0, axis=1).sort_index(axis=1, ascending=False).reset_index()


# now that no monthly gaps, melting back to format that allows groupby shift for previous months

monthly = pd.melt(monthly, id_vars=['Brand', 'Family', 'Size', 'First Sale'], value_name='Cases Sold')


# eliminating erroneous data (this can be combined w/ initial df processing)

monthly = monthly[monthly['Delivery Date'] <= pd.Timestamp.today().to_period('M')]


# two step process to segment NaNs into true zero sale months vs months before product was available

monthly = monthly.fillna(0)

monthly['Cases Sold'] = (np.where(monthly['Delivery Date'] < monthly['First Sale'], np.nan, monthly['Cases Sold']))



# adding on all previous month data

monthly['Last Month'] = monthly.groupby(['Brand', 'Family', 'Size'])['Cases Sold'].shift(-1)
monthly['Last Month Delta YoY'] = monthly['Last Month'] - monthly.groupby(['Family', 'Brand', 'Size'])['Cases Sold'].shift(13)
monthly['2 Months Ago'] = monthly.groupby(['Brand', 'Family', 'Size'])['Cases Sold'].shift(-2)
monthly['3 Months Ago'] = monthly.groupby(['Brand', 'Family', 'Size'])['Cases Sold'].shift(-3)
monthly['4 Months Ago'] = monthly.groupby(['Brand', 'Family', 'Size'])['Cases Sold'].shift(-4)
monthly['5 Months Ago'] = monthly.groupby(['Brand', 'Family', 'Size'])['Cases Sold'].shift(-5)
monthly['6 Months Ago'] = monthly.groupby(['Brand', 'Family', 'Size'])['Cases Sold'].shift(-6)
monthly['7 Months Ago'] = monthly.groupby(['Brand', 'Family', 'Size'])['Cases Sold'].shift(-7)
monthly['8 Months Ago'] = monthly.groupby(['Brand', 'Family', 'Size'])['Cases Sold'].shift(-8)
monthly['9 Months Ago'] = monthly.groupby(['Brand', 'Family', 'Size'])['Cases Sold'].shift(-9)
monthly['10 Months Ago'] = monthly.groupby(['Brand', 'Family', 'Size'])['Cases Sold'].shift(-10)
monthly['11 Months Ago'] = monthly.groupby(['Brand', 'Family', 'Size'])['Cases Sold'].shift(-11)
monthly['Last Year'] = monthly.groupby(['Brand', 'Family', 'Size'])['Cases Sold'].shift(-12)


# creating dataframes for predicting X months out

next_month = monthly.drop(columns=['Last Month', 'Last Month Delta YoY', 'Last Year'])
next_month = next_month.rename(columns={'11 Months Ago':'Last Year'})

two_months_out = next_month.drop(columns=['2 Months Ago', 'Last Year'])
two_months_out = two_months_out.rename(columns={'10 Months Ago':'Last Year'})

three_months_out = two_months_out.drop(columns=['3 Months Ago', 'Last Year'])
three_months_out = three_months_out.rename(columns={'9 Months Ago':'Last Year'})