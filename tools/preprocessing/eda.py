from datetime import datetime
import pandas as pd
import numpy as np


def test():
    current_time = datetime.now().strftime("%H:%M:%S")
    print("Test time :", current_time)


class Dataset:
    """ """
    def __init__(self, features, features_ohe):
        self.features = features
        self.features_ohe = features_ohe
        self.df = self.get_original()
        # self.df_transformed = self.get_transformed()

    # show this when do `print(instance)`
    def __str__(self):
        return f"Data transformation class. \
        \n---------------------------\
        \nInputted features: {self.features}. \
        \n---------------------------\
        \nTransformation steps: \
        \n1. Correct data types \
        \n2. One Hot Encoding of {self.features_ohe} \
        "

    @staticmethod
    def get_original():
        return pd.read_csv("../data/OnlineRetail.csv", encoding='unicode_escape')

    def get_transformed(self):
        # 1. Correct data types
        self.df = self._correct_types()
        # 2. One Hot Encoding
        self.df = self._get_features()
        return self.df

    def _correct_types(self):
        # df_new = self.df.copy()

        self.df['CustomerID'] = self.df['CustomerID'].astype('Int64')

        self.df['InvoiceYear'] = pd.to_datetime(self.df['InvoiceDate'], errors='coerce').dt.year
        self.df['InvoiceMonth'] = pd.to_datetime(self.df['InvoiceDate'], errors='coerce').dt.month
        self.df['InvoiceDay'] = pd.to_datetime(self.df['InvoiceDate'], errors='coerce').dt.day
        self.features.remove('InvoiceDate')
        self.features.extend(['InvoiceYear', 'InvoiceMonth', 'InvoiceDay'])
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
