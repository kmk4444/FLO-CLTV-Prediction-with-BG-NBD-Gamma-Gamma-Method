# Bussines Problem
#Segmenting the customers of FLO, an online shoe store, wants to make sense according to these segments.
#It will be designed accordingly and will be created according to this particular clustering.
#FLO, Wants to determine marketing strategies according to these segments.

# Variables
# master_id : Unique Customer Number
# order_channel : Which channel of the shopping platform is used (Android, IOS, Desktop, Mobile)
# last_order_channel : The channel where the most recent purchase was made
# first_order_date : Date of the customer's first purchase
# last_order_channel : Customer's previous shopping history
# last_order_date_offline : The date of the last purchase made by the customer on the offline platform
# order_num_total_ever_online : Total number of purchases made by the customer on the online platform
# order_num_total_ever_offline : Total number of purchases made by the customer on the offline platform
# customer_value_total_ever_offline : Total fees paid for the customer's offline purchases
# customer_value_total_ever_online : Total fees paid for the customer's online purchases
# interested_in_categories_12 : List of categories the customer has shopped in the last 12 months

import os
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
import seaborn as sns
sns.set_style('whitegrid')
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
from sklearn.preprocessing import MinMaxScaler

df_ = pd.read_csv("WEEK_3/Ödevler/FLO_RFM/flo_data_20k.csv")
df = df_.copy()
df.head()

###################################### TASK 1 ################################
# Prepare the data

def outlier_threshold(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_threshold(dataframe,variable)
    dataframe.loc[dataframe[variable] < low_limit, :] = low_limit
    dataframe.loc[dataframe[variable] > up_limit, :] = up_limit


replace_with_thresholds(df, "order_num_total_ever_online")
replace_with_thresholds(df, "order_num_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_online")

df["order_num_total"] = (df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]).astype(int)
df["customer_value_total"] = (df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]).astype(int)

df.loc[:,df.columns.str.contains("date")] = df.loc[:,df.columns.str.contains("date")].astype("datetime64[ns]")

###################################### TASK 2 ################################
###################################################################
# 1.Preparation of Lifetime Data Structure
###################################################################

df["last_order_date"].max()
today_date = dt.datetime(2021, 6, 1)

df["date_subtract"] = df.apply(lambda x: int((x["last_order_date"]-x["first_order_date"]).days), axis=1)

cltv = df.groupby("master_id").agg(
    {"date_subtract": lambda x: x.sum(),
     "first_order_date": lambda x: (today_date- x.min()).days,
     "order_num_total": lambda x: x.sum(),
     "customer_value_total": lambda x : x.sum()})

cltv.head()
cltv.columns = ["recency_cltv_weekly", "T_weekly", "frequency", "monetary_cltv_avg"]

cltv["monetary_cltv_avg"] = (cltv["monetary_cltv_avg"] / cltv["frequency"]).astype(int)
cltv = cltv[cltv["frequency"] > 1].astype(int)
cltv["recency_cltv_weekly"] = (cltv["recency_cltv_weekly"] / 7).astype(int)
cltv["T_weekly"] = (cltv["T_weekly"] /7 ).astype(int)

cltv = cltv.drop(index = [16.0 , 2928.0540000000146, 7418.990000000026], axis=0)

###################################################################
# Establishment of BG-NBD Model
###################################################################

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv["frequency"],
        cltv["recency_cltv_weekly"],
        cltv["T_weekly"])

cltv["expected_purch_3_months"] = bgf.predict(4*3,
            cltv["frequency"],
            cltv["recency_cltv_weekly"],
            cltv["T_weekly"])

cltv["expected_purch_6_months"] = bgf.predict(4*6,
            cltv["frequency"],
            cltv["recency_cltv_weekly"],
            cltv["T_weekly"])

cltv.head()

# our model has superb prediction in some points;nevertheless, the other points is bad.
plot_period_transactions(bgf)
plt.show()

###################################################################
# Establishing the Gamma - Gamma Model
###################################################################

ggf = GammaGammaFitter(penalizer_coef = 0.01)
ggf.fit(cltv["frequency"], cltv["monetary_cltv_avg"])

cltv["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv["frequency"],
                                        cltv["monetary_cltv_avg"])


####################################################################
#Calculation of CLTV with BG-NBD and GG Model
###################################################################

cltv_last =ggf.customer_lifetime_value(bgf,
                                       cltv['frequency'],
                                       cltv['recency_cltv_weekly'],
                                       cltv['T_weekly'],
                                       cltv['monetary_cltv_avg'],
                                       time=6,  # 6 month
                                       freq="W",  # information of frequency of T and recency
                                       discount_rate=0.01)

cltv_final = cltv.merge(cltv_last, on="master_id", how="left")

print(cltv_final.sort_values(by = "clv", ascending=False).head(20))

###################################################################
# Creating the Customer Segment
###################################################################

cltv_final["segments"] = pd.qcut(cltv_final["clv"],4, labels = ["D", "C", "B", "A"])

print(cltv_final.groupby("segments").agg({"count", "mean", "sum"}))

#graph 1

cltv_final.groupby('segments').agg('expected_average_profit').mean().plot(kind='bar', colormap='copper_r');

plt.ylabel("profit");

#graph 2

cltv_final.groupby('segments').agg('expected_purch_6_months').mean().plot(kind='bar', colormap='copper_r');

plt.ylabel("expected purchase");

#graph 3

cltv_final.groupby('segments').agg('clv').mean().plot(kind='bar', colormap='copper_r');

plt.ylabel("clv");

