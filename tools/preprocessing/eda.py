from datetime import datetime
import pandas as pd
from typing import Dict, Union, List, Tuple, Optional, NoReturn
import logging
import sys


def test():
    current_time = datetime.now().strftime("%H:%M:%S")
    print("Test time :", current_time)


class Dataset:
    """ """
    def __init__(self, features: List[str], features_ohe: Optional[List[str]] = None):
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

    @staticmethod
    def _ohe(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
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
                dummies = pd.get_dummies(df[colname]).add_prefix(f"{colname}_")
                # append dummies to df
                df = pd.concat([df, dummies], axis=1)
                # drop colname which was transformed into dummies
                df.drop(columns=colname, inplace=True)
        return df

    def get_transformed(self) -> pd.DataFrame:
        # 1. Correct data types
        self.df = self._correct_types()
        # 2. Add engineered features
        self.df = self._add_features()
        return self.df

    def _correct_types(self) -> pd.DataFrame:
        self.df['CustomerID'] = self.df['CustomerID'].astype('Int64')
        return self.df

    def _add_features(self) -> pd.DataFrame:
        self.df['InvoiceYear'] = pd.to_datetime(self.df['InvoiceDate'], errors='coerce').dt.year
        self.df['InvoiceMonth'] = pd.to_datetime(self.df['InvoiceDate'], errors='coerce').dt.month
        self.df['InvoiceDay'] = pd.to_datetime(self.df['InvoiceDate'], errors='coerce').dt.day
        self.df.drop('InvoiceDate', inplace=True, axis=1)
        self.features.remove('InvoiceDate')
        self.features.extend(['InvoiceYear', 'InvoiceMonth', 'InvoiceDay'])

        self.df['Revenue'] = self.df['UnitPrice'] * self.df['Quantity']
        return self.df

    def _get_features(self) -> pd.DataFrame:
        assert set(self.features_ohe).issubset(set(self.features)), \
            "`features_ohe` to be OHE must be a part of `features`."
        # subset of df with only necessary features
        df = self.df[self.features]
        df = self._ohe(df, self.features_ohe)
        return df

    def get_clustering_df(self) -> pd.DataFrame:

        # One Hot Encoding
        if self.features_ohe is not None:
            df = self._get_features()

        df.set_index('CustomerID', inplace=True)
        return df

    def get_profiling_df(self) -> pd.DataFrame:
        # number of products per customer
        df_stockCode = pd.DataFrame(self.df.groupby('CustomerID')['StockCode'].count()) \
            .rename(columns={'StockCode': '#_stockCode'})
        # number of invoices per customer
        df_InvoiceNo = pd.DataFrame(self.df.groupby('CustomerID')['InvoiceNo'].nunique()) \
            .rename(columns={'InvoiceNo': '#_InvoiceNo'})
        # avg quantity of all products per customer
        df_Q = pd.DataFrame(self.df.groupby('CustomerID')['Quantity'].mean()) \
            .rename(columns={'Quantity': 'avg_Q'})
        # avg P of all products per customer
        df_P = pd.DataFrame(self.df.groupby('CustomerID')['UnitPrice'].mean()) \
            .rename(columns={'UnitPrice': 'avg_P'})
        # avg Revenue of all products per customer
        df_Revenue = pd.DataFrame(self.df.groupby('CustomerID')['Revenue'].mean()) \
            .rename(columns={'Revenue': 'avg_Revenue'})
        # month with highest Revenue per customer (month when customer spent most money)
        df_RevenuePerMonth = pd.DataFrame(self.df.groupby(['CustomerID', 'InvoiceMonth'])['Revenue'].sum()) \
            .reset_index() \
            .sort_values(['Revenue'], ascending=False)
        df_HighRevenueMonth = pd.DataFrame(
            df_RevenuePerMonth[['CustomerID', 'InvoiceMonth']].drop_duplicates(subset='CustomerID', keep='first')
            ).rename(columns={'InvoiceMonth': 'HighRevenueMonth'})
        df_HighRevenueMonth.index = df_HighRevenueMonth.CustomerID
        df_HighRevenueMonth.drop(columns='CustomerID', inplace=True)

        lst_df_profiles = [df_stockCode, df_InvoiceNo, df_Q, df_P, df_Revenue, df_HighRevenueMonth]
        df = self.df.copy()
        for d in lst_df_profiles:
            df = pd.merge(df, d, how='left', left_on='CustomerID', right_index=True)

        df.index = df.CustomerID
        df.drop(columns=self.df.columns, inplace=True)
        df.drop_duplicates(inplace=True)
        return df
