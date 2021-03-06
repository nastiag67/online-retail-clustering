import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Set, Dict, Tuple, NoReturn, Optional

from sklearn.pipeline import Pipeline
from sklearn.cluster import SpectralClustering, OPTICS, MeanShift, KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class Clustering:
    def __init__(self, X: pd.Series) -> NoReturn:
        self.X = np.array(X)

    def show_clusters(self, ypred: pd.Series) -> NoReturn:
        clusters = np.unique(ypred)
        for cluster in clusters:
            # get row indexes for samples with this cluster
            row_ix = np.where(ypred == cluster)
            # create scatter of these samples
            plt.scatter(self.X[row_ix, 0], self.X[row_ix, 1])
        # show the plot
        plt.show()

    def check_model(self,
                    name: str,
                    model: Pipeline,
                    steps: Optional[List[str]] = [],
                    plot: Optional[bool] = False):
        steps_model = steps[:]

        steps_model.append((name, model))
        pipeline = Pipeline(steps_model)

        pipeline.fit(self.X)

        if isinstance(model, SpectralClustering) or isinstance(model, OPTICS) or isinstance(model, MeanShift):
            ypred = pipeline.fit_predict(self.X)
        else:
            ypred = pipeline.predict(self.X)

        print(model)

        if plot:
            self.show_clusters(ypred)

        return pipeline, ypred

    def build_model(self,
                    models: Pipeline,
                    steps: Optional[List[str]],
                    plot: Optional[bool] = True,
                    add_to_df: Optional[pd.DataFrame] = None):
        if add_to_df is not None:
            df_clustered = pd.DataFrame(add_to_df)
        else:
            df_clustered = pd.DataFrame(self.X)
        for name, model in models.items():
            res_model, res_ypred = self.check_model(name, model, steps, plot)
            df_clustered[f"ypred_{name}"] = res_ypred
        return df_clustered

