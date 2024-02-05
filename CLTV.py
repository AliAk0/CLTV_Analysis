import pandas as pd
import datetime as dt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option('display.width',1000)

df_=pd.read_csv("C:/Users/Ali/Desktop/PROJECTS/FLOMusteriSegmentasyonu/flo_data_20k.csv")
df=df_.copy()

#Defining functions for eliminating the outliers
def outlier_thresholds(dataframe, variable):
    quartile1=dataframe[variable].quantile(0.01)
    quartile3=dataframe[variable].quantile(0.99)
    interquantile_range=quartile3 - quartile1
    up_limit= quartile3 + 1.5 * interquantile_range
    low_limit=quartile1 + 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit,0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit,0)


#Eliminating the outliers at the numerical columns
columns = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline","customer_value_total_ever_online"]
for col in columns:
    replace_with_thresholds(df, col)


#Total order and value
df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
print(df.head())


date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)


print(df["last_order_date"].max())#2021-05-30
analysis_date=dt.datetime(2021,6,1)



cltv_df=pd.DataFrame()
cltv_df["customer_id"]=df["master_id"]
cltv_df["recency_weekly"]=((df["last_order_date"] - df["first_order_date"]).astype("timedelta64[s]")) /7
cltv_df["T_weekly"]= ((analysis_date - df["last_order_date"]).astype("timedelta64[s]")) /7
cltv_df["frequency"]=df["order_num_total"]
cltv_df["monetary_avg"]=df["customer_value_total"]/ df["order_num_total"]

print(cltv_df.head())



#For determining the CLTV first we build a model to find number of transaction for each customer.
bgf=BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df["recency_weekly"],
        cltv_df["frequency"],
        cltv_df["T_weekly"])


cltv_df["exp_sales_3months"]=bgf.predict(4*3,
                                     cltv_df["recency_weekly"],
                                     cltv_df["frequency"],
                                     cltv_df["T_weekly"])


cltv_df["exp_sales_6months"]=bgf.predict(4*6,
                                     cltv_df["recency_weekly"],
                                     cltv_df["frequency"],
                                     cltv_df["T_weekly"])


cltv_df.sort_values("exp_sales_3_month",ascending=False)[:10]

cltv_df.sort_values("exp_sales_6_month",ascending=False)[:10]



#Building gamma model for the expected average profit
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary_avg'])
cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                cltv_df['monetary_avg'])
print(cltv_df.head())



cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_cltv_weekly'],
                                   cltv_df['T_weekly'],
                                   cltv_df['monetary_avg'],
                                   time=6,
                                   freq="W",
                                   discount_rate=0.01)
cltv_df["cltv"] = cltv

print(cltv_df.sort_values("cltv",ascending=False)[:20])



#Grouping the customers
cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 5, labels=["E","D", "C", "B", "A"])
cltv_df.head()

#For the "A" segment customers we can arrange an event with the special invitation to introduce our new product or
#we can send them some of the samples of our products.
#For not losing the customer "D" AND "E" it is better to call them directly or sending e-mail to them
#to show how our product is beneficial for them.
