from flask import Flask, request,jsonify,render_templete
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
import seaborn as sns
import matplotlib.pyplot as plt
import json

app = Flask(__name__)
model = pickle.load(open('keras_model.pkl','rb'))

def load_and_clean_data(file_path):
    #Load_data
    retail = pd.read_csv('OnlineRetail.csv', sep=",",encoding="ISO-8859-1",header=0)

    # changing the datatype od customer ID
    retail['CustomerID'] = retail['CustomerID'].astype(str)
    retail['Amount'] = retail['Quantity']*retail['UnitPrice']

    rfm_m = retail.groupby('CustomerID')['Amount'].sum()
    rfm_f = retail.groupby('CustomerID')['InvoiceNo'].count()
    rfm_f.column = ['CustomerID', 'Frrequency']
    retail['InvoiceDate'] = pd.to_datetime(retail['InvoiceDate'],infer_datetime_format=True)
    max_date = max(retail['InvoiceDate'])
    retail['Diff'] = max_date - retail['InvoiceDate']
    rfm_p = retail.groupby('CustomerID')['Diff'].min()
    rfm_p['Diff'] = rfm_p['Diff'].dt.days
    # Re-create rfm by merging monetary and frequency data
    rfm = pd.merge(rfm_m, rfm_f, on='CustomerID', how='inner')

# Merge with recency data
    rfm = pd.merge(rfm, rfm_p, on='CustomerID', how='inner')

# Rename the columns to final RFM attributes
    rfm.columns = ['CustomerID', 'Amount', 'Frequency', 'Recency']

#remove outliers
    # removing outliers for Amount
    Q1 = rfm.quantile(0.05)
    Q3 = rfm.quantile(0.95)
    IQR = Q3 - Q1
    rfm = rfm[(rfm.Amount >= Q1 - 1.5*IQR) & (rfm.Amount <= Q3 + 1.5*IQR)]
    rfm = rfm[(rfm.Recency >= Q1 - 1.5*IQR) & (rfm.Recency <= Q3 + 1.5*IQR)] 
    rfm = rfm[(rfm.Frequency >= Q1 - 1.5*IQR) & (rfm.Frequency <= Q3 + 1.5*IQR)]

    return rfm

def preprocessing_data(file_path):
    