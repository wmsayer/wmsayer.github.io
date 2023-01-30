#!/usr/bin/env python
# coding: utf-8

# # Pandas Cheat Sheet

# Run Jupter:
# 
# `jupyter notebook`
# 
# Build book:
# 
# `jupyter-book build pandas_cheatsheet.ipynb`

# ## Setup Environment

# In[1]:


# ensures you are running the pip version associated with the current Python kernel
import sys
get_ipython().system('{sys.executable} -m pip install requests')
get_ipython().system('{sys.executable} -m pip install scipy')
get_ipython().system('{sys.executable} -m pip install matplotlib')
get_ipython().system('{sys.executable} -m pip install scikit-learn')
get_ipython().system('{sys.executable} -m pip install plotly')


# In[2]:


import pandas as pd
import numpy as np
import json
import requests

import datetime as dt
from dateutil.relativedelta import relativedelta

import scipy
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import plotly.offline as pyo
# Set notebook mode to work in offline
pyo.init_notebook_mode()


# ## Data & File I/O
# 
# ### JSON Handling

# #### Read JSON File

# In[3]:


# load local JSON file and return as Python dict 
def json_load(f_path):
    f = open(f_path, )
    json_dict = json.load(f)
    f.close()
    return json_dict


# #### Flattening (JSON Normalize)

# In[4]:


json_norm_dict = [
     {
         "id": 1,
         "name": "Cole Volk",
         "fitness": {"height": 130, "weight": 60},
     },
     {"name": "Mark Reg", "fitness": {"height": 130, "weight": 60}},
     {
         "id": 2,
         "name": "Faye Raker",
         "fitness": {"height": 130, "weight": 60},
     },
]

pd.json_normalize(json_norm_dict, max_level=0)


# In[5]:


pd.json_normalize(json_norm_dict, max_level=1)


# In[6]:


json_norm_dict = [
     {
         "state": "Florida",
         "shortname": "FL",
         "info": {"governor": "Rick Scott"},
         "counties": [
             {"name": "Dade", "population": 12345},
             {"name": "Broward", "population": 40000},
             {"name": "Palm Beach", "population": 60000},
         ],
     },
     {
         "state": "Ohio",
         "shortname": "OH",
         "info": {"governor": "John Kasich"},
         "counties": [
             {"name": "Summit", "population": 1234},
             {"name": "Cuyahoga", "population": 1337},
         ],
     },
 ]

pd.json_normalize(json_norm_dict, 
                  record_path="counties",
                  meta=["state", "shortname", ["info", "governor"]])


# #### Load JSON via REST API

# In[7]:


# make GET request to REST API and return as Python dict
def run_json_get(url, params={}, headers={}, print_summ=True, print_resp=False):

    if print_summ:
        print("/"*69)
        print("GET Address: %s\nHeaders %s:\nParameters %s:" % (url, repr(headers), repr(params)))

    response = requests.get(url, params=params, headers=headers)
    status_code = response.status_code

    if status_code == 200:
        json_dict = response.json()
    else:
        json_dict = {}

    if print_summ:
        print("Status Code: %d" % response.status_code)
        # print("Message: %d" % response.messa)
        if type(json_dict) == dict:
            print("Response Keys: %s\n" % json_dict.keys())
    
    if print_resp:
        print("Response: %s\n" % json_dict)

    return json_dict, status_code


# ### Read SQL Query w/ Params

# In[8]:


def read_sql(f_path, params={}):
    f = open(f_path, "r")
    query = f.read()
    f.close()
    
    if params:
        query = query.format(**params)
    
    return query


# #### I/O tests

# In[9]:


# read JSON
test_json_path = "C:/Users/wsaye/PycharmProjects/CashAppInterview/templates/data.json"
print(json_load(test_json_path))


# In[10]:


# read SQL (no params)
test_sql_path = "C:/Users/wsaye/PycharmProjects/CashAppInterview/templates/query.sql"
print(read_sql(test_sql_path))


# In[11]:


# read SQL (w/ params)
test_sql_params_path = "C:/Users/wsaye/PycharmProjects/CashAppInterview/templates/query_w_params.sql"
test_params = {"my_id": 102393, "max_date": "2000/01/01"}
print(read_sql(test_sql_params_path, params=test_params))


# ### Sample REST Dataset (CoinMetrics)

# In[12]:


def get_asset_metrics(assets, metrics, freq, alt_params={}, page_size=10000, print_summ=True):
    # freq options 1b, 1s, 1m, 1h, 1d
    # for 'start_time' and 'end_time', formats "2006-01-20T00:00:00Z" and "2006-01-20" are supported among others
    # https://docs.coinmetrics.io/api/v4#operation/getTimeseriesAssetMetrics
    # https://docs.coinmetrics.io/info/metrics

    assets_str = ", ".join(assets)
    metrics_str = ", ".join(metrics)
    
    api_root = 'https://community-api.coinmetrics.io/v4'
    data_key = "data"

    url = "/".join([api_root, "timeseries/asset-metrics"])
    params = {'assets': assets_str, 'metrics': metrics_str, 'frequency': freq,
              'page_size': page_size}
    params.update(alt_params)
    
    result_dict, status_code = run_json_get(url, params=params, headers={}, print_summ=print_summ)

    result_df = pd.DataFrame(result_dict[data_key])
    result_df.sort_values(by=["asset", "time"], inplace=True)
    result_df.reset_index(inplace=True, drop=True)

    for m in metrics:
        result_df[m] = result_df[m].astype(float)

    return result_df


# In[13]:


def load_asset_metric_data(pull_new):
    if pull_new:
        df = get_asset_metrics(test_assets, test_metrics, test_freq, print_summ=False)
        df.to_csv(test_df_cache, index=False)
    else:
        df = pd.read_csv(test_df_cache)
    return df


# #### Get data w/ cache

# In[14]:


test_assets = ['btc', 'eth']
test_metrics = ['AdrActCnt', 'PriceUSD']
test_freq = '1d'
test_df_cache = "C:/Users/wsaye/PycharmProjects/CashAppInterview/data/cm_test_data.csv"


# In[15]:


test_df = load_asset_metric_data(True)
test_df = test_df.dropna(subset=test_metrics).reset_index(drop=True)

test_df["datetime"] = pd.to_datetime(test_df["time"], utc=True) # str timestamp to datetime
test_df["dayname"] = test_df["datetime"].dt.day_name()
test_df["date"] = pd.to_datetime(test_df["time"], utc=True).dt.date # datetime to date
test_df


# ## Pandas Options & Settings
# 
# Docs and available options [here](https://pandas.pydata.org/docs/user_guide/options.html#available-options)

# In[16]:


pd.get_option("display.max_rows")
pd.set_option("display.max_rows", 10)
pd.set_option("display.max_columns", 10)


# ## Summarize Dataframe

# ### General Info

# In[17]:


test_df.describe()  # summary stats of columns


# In[18]:


test_df.info()  # dataframe schema info, column types


# In[19]:


test_df.dtypes


# In[20]:


test_df.head()


# In[21]:


print(type(test_df.loc[0, "time"]))  # type of particular entry


# In[22]:


test_df.nlargest(5, "PriceUSD")
# test_df.nsmallest(5, "PriceUSD")


# In[23]:


test_df["asset"].unique()
# test_df["asset"].nunique()


# ### Crosstab

# In[24]:


cross_df = test_df.loc[test_df["asset"]== "btc", ["datetime", "dayname", "PriceUSD"]].copy().dropna()
cross_df = cross_df.sort_values(by="datetime")
cross_df["7d_SMA"] = cross_df["PriceUSD"].rolling(7).mean()
cross_df["beating_SMA"] = cross_df["PriceUSD"] > cross_df["7d_SMA"]
cross_df["return"] = cross_df["PriceUSD"].pct_change()
cross_df.dropna(inplace=True)
cross_df


# In[25]:


pd.crosstab(cross_df['beating_SMA'], cross_df['dayname'])


# In[26]:


pd.crosstab(cross_df['beating_SMA'], cross_df['dayname'], normalize=True)


# In[27]:


pd.crosstab(cross_df['beating_SMA'], cross_df['dayname'], values=cross_df['return'], aggfunc=np.mean)


# ## Sort/ Rank

# In[28]:


sort_df = test_df[["date", "asset", "PriceUSD"]].copy()
sort_df['price_rank'] = sort_df["PriceUSD"].rank(ascending=True, pct=False)
sort_df['price_pct'] = sort_df["PriceUSD"].rank(ascending=True, pct=True)
sort_df


# In[29]:


sort_df.sort_values(by="price_rank", ascending=False)


# ## Cleaning
# 
# Deleting Rows/Columns [here](https://www.shanelynn.ie/pandas-drop-delete-dataframe-rows-columns/)

# ### Replace

# In[30]:


replace_df = test_df[["date", "asset", "dayname"]].copy()
# replace_df.replace("Sunday", "Sun")
replace_df.replace({"Sunday": "S",  "Monday": "M", "Tuesday": "T"})


# ### Drop/Fill NA()

# In[31]:


cleaning_df = test_df[["date", "asset", "PriceUSD"]].pivot(index="date", columns="asset", values="PriceUSD")
cleaning_df


# In[32]:


# cleaning_df.dropna()  # drops N/A looking in all columns
cleaning_df.dropna(subset=["eth"])  # drops N/A in subset only


# In[33]:


cleaning_df.fillna(-1)


# In[34]:


cleaning_df.fillna(method="ffill")


# In[35]:


cleaning_df.fillna(method="bfill")


# In[36]:


# setup df for interpolation
interp_df = cleaning_df.iloc[cleaning_df.shape[0] - 5:, :].copy()
interp_df["btc_og"] = interp_df["btc"]
interp_df["eth_og"] = interp_df["eth"]
interp_df.iloc[1, 0:2] = [np.nan ,np.nan]

interp_df.interpolate(method="linear")


# ## Selecting/Sampling

# In[37]:


test_df.select_dtypes(include='float64')


# In[38]:


# test_df.sample(n = 200)
test_df.sample(frac = 0.25, random_state=42)


# ### Boolean Selection

# In[39]:


bool_df = test_df[["date", "asset", "PriceUSD"]].pivot(index="date", columns="asset", values="PriceUSD")
bool_df


# In[40]:


# returns Series of same shape w/ np.NaN at failing rows (default)
# bool_df['PriceUSD'].where(bool_df['PriceUSD'] > 10**4)  # returns np.Nan in failing rows
bool_df['eth'].where(bool_df['eth'] > 10**3, 0)  # returns 0 in failing rows


# In[41]:


test_df["asset"].isin(["btc"])


# In[42]:


na_check_df = bool_df.isna()
na_check_series = na_check_df.any(axis=1)  # aggregate booleans
bool_df.loc[na_check_series, :]


# In[43]:


bool_df.loc[(bool_df['eth'] > 10**3), :]


# ### Boolean Operators

# In[44]:


bool_df.loc[~(bool_df['eth'] > 10**3), :]  # NOT


# In[45]:


bool_df.loc[(bool_df['eth'] > 10**3) & (bool_df['eth'] > 10), :]  # AND


# In[46]:


bool_df.loc[(bool_df['eth'] > 10**3) | (bool_df['eth'] > 10), :]  # OR


# ## Datetime
# 
# Python datetime <-> string formatting [here](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior)
# 
# ### Datetime to string

# Python:

# In[47]:


test_date_str = "2022-01-01"
print(dt.datetime.strptime(test_date_str, "%Y-%m-%d"))
print(dt.datetime.strptime(test_date_str, "%Y-%m-%d").date())


# Pandas:

# In[48]:


test_df["datetime"] = pd.to_datetime(test_df["time"], utc=True) # str timestamp to datetime
test_df['datetime_alt'] = pd.to_datetime(test_df["time"], format='%Y-%m-%dT%H:%M:%S.%fZ', utc=True)
test_df["date"] = pd.to_datetime(test_df["time"], utc=True).dt.date # datetime to date
test_df['datetime_str'] = test_df["datetime"].dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')  # datetime to str
test_df['date_str'] = pd.to_datetime(test_df["date"]).dt.strftime('%Y-%m-%d')  # date to str

test_df[["datetime", "datetime_alt", "date", "datetime_str", "date_str"]]


# ### Datetime Altering

# Python:

# In[49]:


test_date_str = "2022-01-01"
test_dt = dt.datetime.strptime(test_date_str, "%Y-%m-%d")
print(test_dt + dt.timedelta(days=2))
print(test_dt + relativedelta(years=5, months=4))


# Pandas:

# In[50]:


# (also pd.Timedelta)
test_df["date_offset"] = test_df["date"] + pd.DateOffset(years=2, months=2, days=1)

test_df[["date", "date_offset"]]


# ### Date Parts (Week & Weekday)

# Python:

# In[51]:


test_date_str = "2022-01-01"
test_dt = dt.datetime.strptime(test_date_str, "%Y-%m-%d")
print(test_dt.isocalendar().week)
print(test_dt.weekday())
print(test_dt.strftime('%A'))  # get day name


# Pandas:

# In[52]:


test_df["week"] = test_df["datetime"].dt.isocalendar().week
test_df["weekday"] = test_df["datetime"].dt.weekday
test_df["dayname"] = test_df["datetime"].dt.day_name()
test_df["monthname"] = test_df["datetime"].dt.month_name()
test_df["year"] = test_df["datetime"].dt.year

test_df[["date", "week", "weekday", "dayname", "monthname", "year"]]


# ## Numerical

# In[53]:


num_df = test_df.loc[3000:, ["date", "asset", "PriceUSD"]].copy()
num_df["PriceUSD_rnd"] = num_df["PriceUSD"].round()
num_df["PriceUSD_rnd1"] = num_df["PriceUSD"].round(1)
num_df["PriceUSD_floor"] = np.floor(num_df["PriceUSD"])
num_df["PriceUSD_ceil"] = np.ceil(num_df["PriceUSD"])
num_df


# ## Transforms

# ### Indexes
# 
# * df.set_index(keys, drop=True, verify_integrity=False)
# * df.reset_index(drop=False)
# * df.reindex()

# ### Pivot & Melt
# 
# * pd.unstack() - pivot multilevel index
# 
# #### Pivot to MultiIndex

# In[54]:


pivot_df = test_df.pivot(index="date", columns="asset", values=['AdrActCnt', 'PriceUSD'])
pivot_df.reset_index(drop=False, inplace=True)
pivot_df


# #### Melt from MultiIndex

# In[55]:


# pivot_df = pd.melt(pivot_df, col_level=0, id_vars=["date"])
pivot_df = pd.melt(pivot_df, id_vars=[("date", "")])
pivot_df.columns = ["date", "metric", "asset", "value"]
pivot_df


# #### Pivot back to OG (single Index)

# In[56]:


pivot_df = pivot_df.pivot(index=["date", "asset"], columns="metric", values="value")
pivot_df.columns = pivot_df.columns.rename("")
pivot_df.reset_index(drop=False, inplace=True)
pivot_df


# #### Pivot to Date Index (Fill missing dates)

# In[57]:


date_rng = pd.date_range(test_df["date"].min(), test_df["date"].max())
pivot_fill_df = test_df.pivot(index="date", columns="asset", values='PriceUSD')
# pivot_fill_df = pivot_fill_df.drop(labels=[1, 3, 4524], axis=0)  # drop by index value
pivot_fill_df = pivot_fill_df.drop(pivot_fill_df.index[[1, 3, 4524]])  # drop by row num
pivot_fill_df = pivot_fill_df.reindex(date_rng, fill_value=np.nan)
pivot_fill_df


# ### Join & Merge
# 
# * article on when to use [join vs merge](https://towardsdatascience.com/pandas-join-vs-merge-c365fd4fbf49)
# 
# Merge:
# * can join on indices or columns
# * validate - check 1:1, 1:many, etc. (also available for join)
# * indicator - produces additional column to indicate "left_only", "right_only", or "both"

# In[58]:


join_merge_df = test_df.loc[3000:, ["date", "datetime", "asset", 'AdrActCnt', 'PriceUSD']].copy()
join_merge_df = join_merge_df.sort_values(by="date").reset_index(drop=True)
join_merge_df


# In[59]:


join_cols = ["date", "asset"]
join_merge_df1 = join_merge_df.loc[:3000, join_cols + ["AdrActCnt"]]
join_merge_df2 = join_merge_df.loc[:, join_cols + ["PriceUSD"]]
print(join_merge_df1)
print(join_merge_df2)


# In[60]:


joined_df = join_merge_df1.join(join_merge_df2.set_index(keys=join_cols), how="outer", on=join_cols)
joined_df


# In[61]:


merged_df = join_merge_df1.merge(join_merge_df2, how="outer", on=join_cols, indicator=True)
merged_df


# ### Explode

# In[62]:


explode_df = pd.DataFrame({"city": ['A', 'B', 'C'],
                   "day1": [22, 25, 21],
                   'day2':[31, 12, 67],
                   'day3': [27, 20, 15],
                   'day4': [34, 37, [41, 45, 67, 90, 21]],
                   'day5': [23, 54, 36]})
explode_df


# In[63]:


explode_df.explode("day4", ignore_index=False)


# ## Aggregation
# 
# ### Aggregation functions
# 
# * mean(): Compute mean of groups
# * sum(): Compute sum of group values
# * size(): Compute group sizes
# * count(): Compute count of group
# * std(): Standard deviation of groups
# * var(): Compute variance of groups
# * sem(): Standard error of the mean of groups
# * first(): Compute first of group values
# * last(): Compute last of group values
# * nth() : Take nth value, or a subset if n is a list
# * min(): Compute min of group values
# * max(): Compute max of group values

# In[64]:


agg_df = test_df[test_metrics]
agg_df.count()


# In[65]:


agg_df.nunique()


# In[66]:


agg_df.median()


# In[67]:


agg_df.quantile(q=0.10)


# ### Groupby
# 
# DataFrame.groupby(by=None, axis=0, level=None, as_index=True, sort=True, group_keys=_NoDefault.no_default, observed=False, dropna=True)

# In[68]:


group_df = test_df[["date", "year", "asset"] + test_metrics]
group_df.groupby(by="asset").nunique()


# In[69]:


group_df.groupby(by=["year", "asset"]).nunique()


# #### Groupby Map

# In[70]:


agg_map = {'AdrActCnt':['count', 'nunique'], 
           'PriceUSD':['max', 'min', lambda x: x.max() - x.min()]
          }
group_df.groupby(by=["year", "asset"]).agg(agg_map)


# #### Groupby Datetime
# 
# DataFrame.groupby(pd.Grouper(key="dfgfgdf", axis=0, freq='M'))

# In[71]:


group_dt_df = test_df.pivot(index="datetime", columns="asset", values=test_metrics)
group_dt_df


# In[72]:


group_dt_df.groupby(pd.Grouper(axis=0, freq='M')).last()


# In[73]:


group_dt_df.groupby(pd.Grouper(axis=0, freq='Y')).first()


# ### Rolling/Window
# 
# DataFrame.rolling(window, min_periods=None, center=False, win_type=None, on=None, axis=0, closed=None, step=None, method='single')
# 
# Shorthand methods:
# * cumsum()
# * pct_change()

# In[74]:


rolling_df = test_df.pivot(index="datetime", columns="asset", values=test_metrics)
rolling_df


# In[75]:


rolling_df.rolling(5, min_periods=3, center=False).mean()


# Gaussian Window:

# In[76]:


num = 18
std_dev = 4.5
window = scipy.signal.windows.gaussian(num, std=std_dev)
z_score = np.linspace(-num/2/std_dev, num/2/std_dev, num)
plt.plot(z_score, window)
plt.title(r"Gaussian window ($\sigma$=7)")
plt.ylabel("Amplitude")
plt.xlabel("Z-Score")
plt.figure()


# ### Binning

# #### cut
# 
# *Bins are of equal size* (width), but number of entries per bin may not similar.

# In[77]:


# returns pd.Series
pd.cut(test_df['PriceUSD'], bins=5).value_counts()


# #### qcut
# 
# *Bins are **not** of equal size* (width), but number of entries per bin are similar.

# In[78]:


# returns pd.Series
pd.qcut(test_df['PriceUSD'], q=5).value_counts()


# ## Strings
# 
# Regex cheat sheet [here](https://www.dataquest.io/blog/regex-cheatsheet/)
# 
# Python string formatting [cookbook](https://mkaz.blog/code/python-string-format-cookbook/)

# In[79]:


nba_df = pd.read_csv("C:/Users/wsaye/PycharmProjects/CashAppInterview/data/nba.csv")
nba_df.dropna(inplace=True)
nba_df


# ### Search/Replace

# In[80]:


# does not assume regex
# (nba_df["Name"].str.find("a") >= 1).sum()
nba_df["Name"].str.find("J") >= 1


# In[81]:


# assumes regex
# nba_df["Name"].str.contains("^[jJ]")
# nba_df["Name"].str.contains("^(Jo)")
nba_df["Name"].str.extract("^(Jo)")


# In[82]:


nba_df["Name"].str.replace("^(Jo)", "PORKY", regex=True)


# ### Split, Concat

# In[83]:


# assumes regex if len > 1, else literal
nba_df["Name"].str.split(" ")


# In[84]:


# assumes regex if len > 1, else literal
nba_df["Name"].str.split(" ", n=1, expand=True)


# In[85]:


nba_df["Name"].str.split(" ").str.len().describe()


# In[86]:


nba_df["Name"].str.cat(others=[nba_df["Team"], nba_df["College"]], sep="-")


# ### Formatting

# In[87]:


# nba_df["Name"].str.upper()
nba_df["Name"].str.lower()


# In[88]:


nba_df['Salary_str'] = (nba_df['Salary']/10**6).map('${:,.2f}M'.format)  # for single Series
nba_df['Salary_str']
# cohort_crosstab = cohort_crosstab.applymap('{:,.0f}%'.format)  # for entire df


# ## Modeling
# 
# ### Linear Regression

# In[89]:


lin_reg_df = pd.read_csv('C:/Users/wsaye/PycharmProjects/CashAppInterview/data/lin_reg_test_data.csv')  # load data set
X = lin_reg_df.iloc[:, 0].values.reshape(-1, 1)  # values converts it into a numpy array
Y = lin_reg_df.iloc[:, 1].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
COLOR = lin_reg_df.iloc[:, 2].values.reshape(-1, 1)

ohe = OneHotEncoder(sparse_output=False)
ohe_vals = ohe.fit_transform(COLOR)

X_mat = np.concatenate([X, ohe_vals], axis=1)


# In[90]:


def regression_results(y_true, y_pred, lin_reg):

    # Regression metrics
    explained_variance = metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred) 
    mse = metrics.mean_squared_error(y_true, y_pred) 
    mean_squared_log_error = metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error = metrics.median_absolute_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)

    print('explained_variance: ', round(explained_variance,4))    
    print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    print('r^2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))
    print(f"Coefficients: {lin_reg.coef_}")
    print(f"Intercept: {lin_reg.intercept_}")


# #### Test/Train Split

# In[91]:


X_train, X_test, y_train, y_test = train_test_split(X_mat, Y, test_size = 0.25)


# #### Fit, Predict, Summarize

# In[92]:


regr = LinearRegression()

regr.fit(X_train, y_train)
y_test_pred = regr.predict(X_test)
lin_reg_df["y_pred"] = regr.predict(X_mat)

regression_results(y_test, y_test_pred, regr)


# ### K-Means

# In[93]:


from sklearn.cluster import KMeans

X_ktrain, X_ktest, y_ktrain, y_ktest = train_test_split(X, Y, test_size = 0.25)

train_set = np.concatenate([X_ktrain, y_ktrain], axis=1)
test_set = np.concatenate([X_ktest, y_ktest], axis=1)

kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(train_set)
train_labels = kmeans.labels_
test_labels = kmeans.predict(test_set)


# Plot Model:

# In[94]:


fig = make_subplots(rows=1, cols=2)
fig.add_trace(go.Scatter(x=X_ktrain[:, 0], y=y_ktrain[:, 0], mode='markers', name='train',
                         marker=dict(color=train_labels, line_color="black", line_width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=X_ktest[:, 0], y=y_ktest[:, 0], mode='markers', name='test',
                         marker=dict(color=test_labels, line_color="black", line_width=1, symbol="x")), row=1, col=2)
fig.show()


# ## Plotting
# 
# ### Matplotlib

# In[95]:


# plt.scatter(X, Y)
# plt.plot(X, Y_pred, color="red", linestyle='None', marker="x")
plt.scatter(lin_reg_df["x"], lin_reg_df["y"])
plt.plot(lin_reg_df["x"], lin_reg_df["y_pred"], color="green", linestyle='None', marker="x")

plt.show()


# ### Plotly (Standard)

# In[96]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=X[:, 0], y=Y[:, 0], mode='markers', name='raw data', marker=dict(color="grey")))
# fig.add_trace(go.Scatter(x=X[:, 0], y=Y_pred[:, 0], mode='markers', name='prediction', marker=dict(color=COLOR[:, 0])))

for c in list(np.unique(COLOR[:, 0])):
    temp_x = lin_reg_df.loc[lin_reg_df["color"]==c, "x"]
    temp_y = lin_reg_df.loc[lin_reg_df["color"]==c, "y_pred"]
    fig.add_trace(go.Scatter(x=temp_x, y=temp_y, mode='lines', name='pred-' + c, line_color=c))

fig.show()


# ### Plotly Express

# In[97]:


fig = px.scatter(lin_reg_df, x="x", y="y", color="color")
fig.show()


# In[98]:


fig = px.scatter(lin_reg_df, x="x", y="y", color="color", 
                 facet_col="color", 
#                  facet_row="time", 
#                  trendline="ols"
                )
fig.show()


# ## Sample Analyses

# ### Cohort

# In[99]:


rand_gen = np.random.RandomState(2021)  # set seed

start_date = dt.datetime.strptime("2022-01-01", "%Y-%m-%d")
end_date = dt.datetime.strptime("2022-01-10", "%Y-%m-%d")
date_rng = pd.date_range(start_date, end_date).values

total_days = len(date_rng)
num_users = 1000
user_df_list = []

for u in range(0, num_users):
    num_active_days = rand_gen.randint(low=2, high=total_days)
    active_days_index = rand_gen.randint(low=0, high=total_days, size=(1, num_active_days))
    active_dates = pd.Series(date_rng[active_days_index[0, :]])
    user_id = pd.Series([u]*num_active_days)
    user_df = pd.concat([active_dates, user_id], axis=1)
    user_df.columns = ["date", "user_id"]
    user_df_list.append(user_df)

cohort_df = pd.concat(user_df_list)
cohort_df


# In[100]:


first_date = cohort_df.groupby(by=["user_id"]).min().rename(columns={"date": "start_date"})
cohort_df = cohort_df.join(first_date, on="user_id", how="left")
cohort_df


# In[101]:


cohort_crosstab = pd.crosstab(cohort_df['start_date'], cohort_df['date'])
cohort_totals = np.diag(cohort_crosstab).reshape(-1, 1)

cohort_crosstab[cohort_crosstab.columns] = 100 * cohort_crosstab.values / cohort_totals
cohort_crosstab = cohort_crosstab.applymap('{:,.0f}%'.format)
cohort_crosstab


# ### Funnel

# In[102]:


stages = ["Website visit", "Downloads", "Potential customers", "Requested price", "Invoice sent"]
df_mtl = pd.DataFrame(dict(number=[39, 27.4, 20.6, 11, 3], stage=stages))
df_mtl['office'] = 'Montreal'
df_toronto = pd.DataFrame(dict(number=[52, 36, 18, 14, 5], stage=stages))
df_toronto['office'] = 'Toronto'
df = pd.concat([df_mtl, df_toronto], axis=0)

fig = px.funnel(df, x='number', y='stage', color='office')
display(fig)


# In[103]:


trace0 = go.Funnel(
        y = stages,
        x = [49, 29, 26, 11, 2], 
        textinfo = "value+percent initial")

# Fill out data with our traces
traces = [trace0]
# Plot it and save as basic-line.html
pyo.iplot(traces, filename = 'funnel_2')


# In[ ]:




