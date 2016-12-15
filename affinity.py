import numpy as np
from itertools import count
import pandas as pd


class AffinityMatrix(object):


    def __init__(self):
        self.count_df = None
        self.order = None
        self.trials = 0


    def initialize_counts(self, clust_series):
        order = clust_series.index
        N = len(clust_series)
        return pd.DataFrame(np.zeros(shape=(N,N), dtype=float), index=order, columns=order)


    def add_stuff(self, cseries):

        assert isinstance(cseries, pd.Series)
        if self.count_df is None:
            count_df = self.initialize_counts(clust_series=cseries)
        else:
            count_df = self.count_df

        for clust_id in cseries.unique():
            clust_items = cseries[cseries==clust_id].index
            count_df.loc[clust_items, clust_items] += 1

        self.count_df = count_df
        return count_df


    def affinity(self):
        return self.count_df / self.trials
