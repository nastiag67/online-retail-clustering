from datetime import datetime
import pandas as pd
import numpy as np


def test():
    current_time = datetime.now().strftime("%H:%M:%S")
    print("Test time :", current_time)


class Dataset:
    """ """
    def __init__(self, features, features_ohe=None):
        self.features = features
        self.features_ohe = features_ohe
        self.df = self.get_original()

    # show this when do `print(instance)`
    def __str__(self):
        return f"Data transformation class. \
        \n---------------------------\
        \nInputted features: {self.features}. \
        \n---------------------------\
        \nTransformation steps: \
        \n1. Correct data types \
        \n2. Feature engineering: Revenue \
        \n3. One Hot Encoding of {self.features_ohe} \
       "

    @staticmethod
    def get_original():
        return pd.read_csv("../data/OnlineRetail.csv", encoding='unicode_escape')

    def get_transformed(self):
        # 1. Correct data types
        self.df = self._correct_types()
        # 2. Feature Engineering
        self.df = self._add_features()
        return self.df

    def _correct_types(self):
        self.df['CustomerID'] = self.df['CustomerID'].astype('Int64')
        return self.df

    def _get_features(self):
        assert set(self.features_ohe).issubset(set(self.features)), \
            "`features_ohe` to be OHE must be a part of `features`."
        # subset of df with only necessary features
        self.df = self.df[self.features]
        self.df = self._ohe(self.features_ohe)
        return self.df

    def _ohe(self, cols):
        """Performs One Hot Encoding for features in `cols`.

        Parameters:
        ----------
        cols : List
            List of column names which need to be One-Hot-Encoded.

        Returns
        -------
        pd.DataFrame - pd.dataframe with one-hot-encoded features.

        """
        if cols:
            for colname in cols:
                # get df of dummies from a column, where "prefix_" is name of the column the dummy comes from
                dummies = pd.get_dummies(self.df[colname]).add_prefix(f"{colname}_")
                # append dummies to df
                self.df = pd.concat([self.df, dummies], axis=1)
                # drop colname which was transformed into dummies
                self.df.drop(columns=colname, inplace=True)
        return self.df

    def _add_features(self):
        self.df['InvoiceYear'] = pd.to_datetime(self.df['InvoiceDate'], errors='coerce').dt.year
        self.df['InvoiceMonth'] = pd.to_datetime(self.df['InvoiceDate'], errors='coerce').dt.month
        self.df['InvoiceDay'] = pd.to_datetime(self.df['InvoiceDate'], errors='coerce').dt.day
        self.df.drop('InvoiceDate', inplace=True, axis=1)
        self.features.remove('InvoiceDate')
        self.features.extend(['InvoiceYear', 'InvoiceMonth', 'InvoiceDay'])

        self.df['Revenue'] = self.df['UnitPrice'] * self.df['Quantity']
        return self.df

    def get_clustering_df(self):
        self.df = self.get_profiling_df()

        # 3. One Hot Encoding
        if self.features_ohe is not None:
            self.df = self._get_features()

        return self.df

    def get_profiling_df(self):
        # number of products per customer
        df_stockCode = pd.DataFrame(self.df.groupby('CustomerID')['StockCode'].count()).rename(
            columns={'StockCode': '#_stockCode'})
        # number of invoices per customer
        df_InvoiceNo = pd.DataFrame(self.df.groupby('CustomerID')['InvoiceNo'].nunique()).rename(
            columns={'InvoiceNo': '#_InvoiceNo'})
        # avg quantity of all products per customer
        df_Q = pd.DataFrame(self.df.groupby('CustomerID')['Quantity'].mean()).rename(columns={'Quantity': 'avg_Q'})
        # avg P of all products per customer
        df_P = pd.DataFrame(self.df.groupby('CustomerID')['UnitPrice'].mean()).rename(columns={'UnitPrice': 'avg_P'})
        # avg Revenue of all products per customer
        df_Revenue = pd.DataFrame(self.df.groupby('CustomerID')['Revenue'].mean()).rename(
            columns={'Revenue': 'avg_Revenue'})
        # month with highest Revenue per customer (month when customer spent most money)
        df_RevenuePerMonth = pd.DataFrame(
            self.df.groupby(['CustomerID', 'InvoiceMonth'])['Revenue'].sum()).reset_index().sort_values(['Revenue'],
                                                                                                    ascending=False)
        df_HighRevenueMonth = pd.DataFrame(
            df_RevenuePerMonth[['CustomerID', 'InvoiceMonth']].drop_duplicates(subset='CustomerID',
                                                                               keep='first')).rename(
            columns={'InvoiceMonth': 'HighRevenueMonth'})

        self.df = pd.merge(self.df, df_stockCode, how='left', left_on='CustomerID', right_index=True)
        self.df = pd.merge(self.df, df_InvoiceNo, how='left', left_on='CustomerID', right_index=True)
        self.df = pd.merge(self.df, df_Q, how='left', left_on='CustomerID', right_index=True)
        self.df = pd.merge(self.df, df_P, how='left', left_on='CustomerID', right_index=True)
        self.df = pd.merge(self.df, df_Revenue, how='left', left_on='CustomerID', right_index=True)
        self.df = pd.merge(self.df, df_HighRevenueMonth, how='left', left_on='CustomerID', right_on='CustomerID')
        return self.df
