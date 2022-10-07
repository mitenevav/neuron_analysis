import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from analysis.minian import MinianAnalysis
from analysis.functions import active_df_to_dict, corr_df_to_distribution


def iqr(x):
    return x.quantile(0.75) - x.quantile(0.25)


class Data:
    def __init__(self,
                 path_to_data,
                 dates,
                 fps,
                 labels):

        self.labels = labels
        self.dates = dates
        self.data = None

        self.models = {}
        for date in dates:
            ma = MinianAnalysis(f'{path_to_data}/{date}/minian/', fps)
            ma.active_state_df = pd.read_csv(f'{path_to_data}/{date}/results/active_states_spike.csv',
                                             index_col=0).astype(bool)
            ma.active_state = active_df_to_dict(ma.active_state_df)

            ma.smooth_signals = ma.signals.rolling(window=10, center=True, min_periods=0).mean()
            ma.smooth_diff = ma.smooth_signals.diff()[1:].reset_index(drop=True)

            self.models[date] = ma

    def _get_burst_rate_data(self):
        df_br = pd.DataFrame()
        for date in self.dates:
            df_ptr = pd.DataFrame()
            df_ptr['br'] = self.models[date].burst_rate()['activations per min']
            df_ptr['model'] = date
            df_br = df_br.append(df_ptr)

        df_br = df_br.reset_index(drop=True)

        return df_br

    def _get_nsp_data(self):
        df_nsp = pd.DataFrame()
        for date in self.dates:
            df_ptr = pd.DataFrame()
            df_ptr['nsp'] = self.models[date].network_spike_peak(1).T['peak']
            df_ptr['model'] = date
            df_nsp = df_nsp.append(df_ptr)

        df_nsp = df_nsp.reset_index(drop=True)

        return df_nsp

    def _get_nsr_data(self):
        df_nsr = pd.DataFrame()
        for date in self.dates:
            df_ptr = pd.DataFrame()
            df_ptr['nsr'] = self.models[date].network_spike_rate(1).T['spike rate']
            df_ptr['model'] = date
            df_nsr = df_nsr.append(df_ptr)

        df_nsr = df_nsr.reset_index(drop=True)

        return df_nsr

    def _get_corr_data(self, method='signal'):
        df_corr = pd.DataFrame()
        for date in self.dates:
            df_ptr = pd.DataFrame()
            df_ptr[f'corr_{method}'] = corr_df_to_distribution(self.models[date].get_correlation(method))
            df_ptr['model'] = date
            df_corr = df_corr.append(df_ptr)

        df_corr = df_corr.reset_index(drop=True)
        df_corr = df_corr.fillna(0)

        return df_corr

    def _get_nd_data(self,
                     df_corr,
                     method='signal',
                     thrs=None,
                     ):

        if thrs is None:
            thrs = [.1, .2, .3, .4, .5]

        df_network_degree = pd.DataFrame()
        for date in self.dates:
            df_ptr = pd.DataFrame()
            corr = np.array(df_corr[df_corr['model'] == date][f'corr_{method}'])
            for thr in thrs:
                df_ptr[f'nd_{method}_{thr}'] = [(corr > thr).sum() / len(corr)]
            df_ptr['model'] = date
            df_network_degree = df_network_degree.append(df_ptr)

        df_network_degree = df_network_degree.reset_index(drop=True)
        df_network_degree = df_network_degree.fillna(0)
        df_network_degree = df_network_degree.set_index('model')

        return df_network_degree

    def _get_conn_data(self,
                       df_corr,
                       method='signal',
                       q=0.9):
        df_conn = pd.DataFrame()

        total_distr = df_corr[f'corr_{method}'].dropna().tolist()
        thr = np.quantile(total_distr, q=q)

        for date in self.dates:
            df_ptr = pd.DataFrame()
            corr_df = self.models[date].get_correlation(method)
            df_ptr[f'connectivity_{method}'] = ((corr_df > thr).sum() - 1) / len(corr_df)
            df_ptr['model'] = date
            df_conn = df_conn.append(df_ptr)

        df_conn = df_conn.reset_index(drop=True)

        return df_conn

    def get_data(self):
        df_nsr = self._get_nsr_data()
        df_nsp = self._get_nsp_data()
        df_br = self._get_burst_rate_data()

        agg_functions = ['mean', 'std', 'max', 'min', np.ptp, iqr]

        nsr = df_nsr.groupby('model').agg(agg_functions)
        nsp = df_nsp.groupby('model').agg(agg_functions)
        br = df_br.groupby('model').agg(agg_functions)

        df_corr_sign = self._get_corr_data('signal')
        df_corr_diff = self._get_corr_data('diff')
        df_corr_active = self._get_corr_data('active')
        df_corr_active_acc = self._get_corr_data('active_acc')

        corr_sign = df_corr_sign.groupby('model').agg(agg_functions)
        corr_diff = df_corr_diff.groupby('model').agg(agg_functions)
        corr_active = df_corr_active.groupby('model').agg(agg_functions)
        corr_active_acc = df_corr_active_acc.groupby('model').agg(agg_functions)

        df_network_degree_sign = self._get_nd_data(df_corr_sign, method='signal')
        df_network_degree_diff = self._get_nd_data(df_corr_diff, method='diff')
        df_network_degree_active = self._get_nd_data(df_corr_active, method='active')
        df_network_degree_active_acc = self._get_nd_data(df_corr_active_acc, method='active_acc')

        df_conn_sign = self._get_conn_data(df_corr_sign, 'signal')
        df_conn_diff = self._get_conn_data(df_corr_diff, 'diff')
        df_conn_active = self._get_conn_data(df_corr_active, 'active')
        df_conn_active_acc = self._get_conn_data(df_corr_active_acc, 'active_acc')

        conn = df_conn_sign.groupby('model').agg(agg_functions)
        conn_diff = df_conn_diff.groupby('model').agg(agg_functions)
        conn_act = df_conn_active.groupby('model').agg(agg_functions)
        conn_act_acc = df_conn_active_acc.groupby('model').agg(agg_functions)

        data = nsp.join(nsr)
        data = data.join(br)

        data = data.join(conn)
        data = data.join(conn_diff)
        data = data.join(conn_act)
        data = data.join(conn_act_acc)

        data = data.join(corr_sign)
        data = data.join(corr_diff)
        data = data.join(corr_active)
        data = data.join(corr_active_acc)

        data = data.T.set_index(data.columns.map('_'.join)).T

        data = data.reset_index()

        data = data.merge(df_network_degree_sign.reset_index(), on='model')
        data = data.merge(df_network_degree_diff.reset_index(), on='model')
        data = data.merge(df_network_degree_active.reset_index(), on='model')
        data = data.merge(df_network_degree_active_acc.reset_index(), on='model')

        data = data.set_index('model')

        data = data[data.columns[data.apply(lambda x: len(x.unique()) > 1)]]

        self.data = data

        return data

    @staticmethod
    def drop_strong_corr(data, thr=0.9):

        while (data.corr() > thr).sum().sum() + (data.corr() < -thr).sum().sum() > len(data):
            strong_corr = (data.corr() > thr).sum() + (data.corr() < -thr).sum()
            val = strong_corr.max()
            if val <= 1:
                break
            col = strong_corr.idxmax()
            data = data.drop(columns=[col])

        return data

    @staticmethod
    def data_reduction(data, model=PCA(n_components=2, random_state=42)):
        scaler = StandardScaler()
        data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

        data = model.fit_transform(data)

        return data, model

    @staticmethod
    def show_result(data, labels):
        plt.figure(figsize=(8, 6))
        plt.title('PCA', fontsize=18)
        sns.scatterplot(data[:, 0], data[:, 1], hue=labels, s=120)
        plt.legend(fontsize=16)
        plt.tick_params(axis='both', labelsize=14)
        plt.show()