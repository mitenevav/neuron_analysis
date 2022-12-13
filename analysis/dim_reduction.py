import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm.notebook import tqdm

from analysis.minian import MinianAnalysis
from analysis.functions import active_df_to_dict, corr_df_to_distribution


def iqr(x):
    return x.quantile(0.75) - x.quantile(0.25)


def q95(x):
    return x.quantile(0.95)


def q5(x):
    return x.quantile(0.05)


class Data:
    def __init__(self,
                 path_to_data,
                 sessions,
                 verbose=False):

        self.sessions = sessions.keys()
        self.data = None
        self.disable_verbose = not verbose
        self.params = sessions
        self.data_reduced = None

        self.models = {}
        for date in tqdm(self.sessions, disable=self.disable_verbose):
            session_path = self.params[date]["path"]
            ma = MinianAnalysis(f'{path_to_data}/{session_path}/minian/', self.params[date]['fps'])
            ma.active_state_df = pd.read_csv(f'{path_to_data}/{session_path}/results/active_states_spike.csv',
                                             index_col=0).astype(bool)
            ma.active_state = active_df_to_dict(ma.active_state_df)

            ma.smooth_signals = ma.signals.rolling(window=10, center=True, min_periods=0).mean()
            ma.smooth_diff = ma.smooth_signals.diff()[1:].reset_index(drop=True)

            self.models[date] = ma

    def _get_burst_rate_data(self):
        df_br = pd.DataFrame()
        for date in tqdm(self.sessions,
                         disable=self.disable_verbose,
                         desc='Step 1/6: Burst rate computing...'):
            df_ptr = pd.DataFrame()
            df_ptr['br'] = self.models[date].burst_rate()['activations per min']
            df_ptr['model'] = date
            df_br = df_br.append(df_ptr)

        df_br = df_br.reset_index(drop=True)

        return df_br

    def _get_nsp_data(self):
        df_nsp = pd.DataFrame()
        for date in tqdm(self.sessions,
                         disable=self.disable_verbose,
                         desc='Step 2/6: Network spike peak computing...'):
            df_ptr = pd.DataFrame()
            df_ptr['nsp'] = self.models[date].network_spike_peak(1).T['peak']
            df_ptr['model'] = date
            df_nsp = df_nsp.append(df_ptr)

        df_nsp = df_nsp.reset_index(drop=True)

        return df_nsp

    def _get_nsr_data(self):
        df_nsr = pd.DataFrame()
        for date in tqdm(self.sessions,
                         disable=self.disable_verbose,
                         desc='Step 3/6: Network spike rate computing...'):
            df_ptr = pd.DataFrame()
            df_ptr['nsr'] = self.models[date].network_spike_rate(1).T['spike rate']
            df_ptr['model'] = date
            df_nsr = df_nsr.append(df_ptr)

        df_nsr = df_nsr.reset_index(drop=True)

        return df_nsr

    def _get_corr_data(self, method='signal'):
        df_corr = pd.DataFrame()
        for date in self.sessions:
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
        for date in self.sessions:
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

        for date in self.sessions:
            df_ptr = pd.DataFrame()
            corr_df = self.models[date].get_correlation(method)
            df_ptr[f'connectivity_{method}'] = ((corr_df > thr).sum() - 1) / len(corr_df)
            df_ptr['model'] = date
            df_conn = df_conn.append(df_ptr)

        df_conn = df_conn.reset_index(drop=True)

        return df_conn

    def get_data(self):

        df_br = self._get_burst_rate_data()
        df_nsp = self._get_nsp_data()
        df_nsr = self._get_nsr_data()

        agg_functions = ['mean',
                         'std',
                         q95,
                         q5,
                         iqr]

        nsr = df_nsr.groupby('model').agg(agg_functions)
        nsp = df_nsp.groupby('model').agg(agg_functions)
        br = df_br.groupby('model').agg(agg_functions)

        corr_types = ['signal', 'diff', 'active']#, 'active_acc']

        df_corr = {}
        corrs = {}
        for corr in tqdm(corr_types,
                         disable=self.disable_verbose,
                         desc='Step 4/6: Correlation computing...'):
            df_corr[corr] = self._get_corr_data(corr)
            corrs[corr] = df_corr[corr].groupby('model').agg(agg_functions)

        df_network_degree = {}
        for corr in tqdm(corr_types,
                         disable=self.disable_verbose,
                         desc='Step 5/6: Network degree computing...'):
            df_network_degree[corr] = self._get_nd_data(df_corr[corr], method=corr)

        df_conn = {}
        for corr in tqdm(corr_types,
                         disable=self.disable_verbose,
                         desc='Step 6/6: Connectivity computing...'):
            df_conn[corr] = self._get_conn_data(df_corr[corr], corr)
            df_conn[corr] = df_conn[corr].groupby('model').agg(agg_functions)

        data = nsp.join(nsr)
        data = data.join(br)

        for corr in corr_types:
            data = data.join(df_conn[corr])

        for corr in corr_types:
            data = data.join(corrs[corr])

        data = data.T.set_index(data.columns.map('_'.join)).T

        data = data.reset_index()

        for corr in corr_types:
            data = data.merge(df_network_degree[corr].reset_index(), on='model')

        data = data.set_index('model')

        data = data[data.columns[data.apply(lambda x: len(x.unique()) > 1)]]

        self.data = data

        # return data

    def drop_strong_corr(self, thr=0.9):

        while (self.data.corr() > thr).sum().sum() + (self.data.corr() < -thr).sum().sum() > len(self.data):
            strong_corr = (self.data.corr() > thr).sum() + (self.data.corr() < -thr).sum()
            val = strong_corr.max()
            if val <= 1:
                break
            col = strong_corr.idxmax()
            self.data = self.data.drop(columns=[col])

            if len(self.data.columns) <= 2:
                continue

        # return self.data

    def data_reduction(self, model=PCA(n_components=2, random_state=42), scaler=StandardScaler()):
        sessions = self.data.index

        data = self.data.copy()

        if scaler:
            data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

        data_reduced = model.fit_transform(data)

        data_reduced = pd.DataFrame(data_reduced, columns=['x', 'y'])

        data_reduced['session'] = sessions
        data_reduced['mouse'] = [self.params[session]['mouse'] for session in sessions]
        data_reduced['condition'] = [self.params[session]['condition'] for session in sessions]

        self.data_reduced = data_reduced
        return data_reduced, model

    def show_result(self, mouse, condition_order=None):
        if self.data_reduced is None:
            data = self.data_reduction()[0]
        else:
            data = self.data_reduced.copy()

        if condition_order is None:
            condition_order = data['condition'].unique()

        cmap = mcolors.LinearSegmentedColormap.from_list("", ["red", "yellow", "green"], len(condition_order))
        palette = [cmap(i) for i in range(len(condition_order))]
        palette = [palette[-1]] + palette[:-1]

        palette_dict = {}
        for cond, color in zip(condition_order, palette):
            palette_dict[cond] = color

        plt.figure(figsize=(9, 8))

        data_mouse = data[data['mouse'] == mouse]
        plt.title(f'Mouse {mouse}', fontsize=18)

        sns.scatterplot(data=data_mouse,
                        x='x', y='y',
                        hue='condition',
                        hue_order=condition_order,
                        palette=palette_dict,
                        s=120,
                        zorder=2
                        )

        conditions = data_mouse['condition'].value_counts()
        centers = data_mouse.groupby('condition').mean()

        for cond in conditions[conditions > 1].index:
            sns.scatterplot(x=[centers.loc[cond]['x']],
                            y=[centers.loc[cond]['y']],
                            label=f'{cond} mean',
                            color=palette_dict[cond],
                            marker='s',
                            s=120,
                            zorder=2
                            )

        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim()

        d = max(xmax - xmin, ymax - ymin)
        for start_cond, end_cond in zip(condition_order[:-1], condition_order[1:]):
            plt.arrow(centers.loc[start_cond]['x'],
                      centers.loc[start_cond]['y'],
                      centers.loc[end_cond]['x'] - centers.loc[start_cond]['x'],
                      centers.loc[end_cond]['y'] - centers.loc[start_cond]['y'],
                      width=d * 0.0045,
                      length_includes_head=True,
                      color=palette_dict[end_cond],
                      zorder=1
                      )

        plt.tick_params(axis='both', labelsize=14)

        plt.show()

    def show_stats_deviation(self, condition='all', topn=8):
        if condition == 'all':
            df = self.data.copy()
        else:
            df = self.data[[self.params[x]['condition'] == condition for x in self.data.index]]

        df = pd.DataFrame(MinMaxScaler().fit_transform(df), columns=df.columns, index=df.index)

        df['mouse'] = [self.params[session]['mouse'] for session in df.index]

        if condition == 'all':
            df['condition'] = [self.params[session]['condition'] for session in df.index]

            df = df.groupby(['mouse', 'condition']).mean()
            feat_std = df.reset_index().groupby('mouse').mad().mean().sort_values()
        else:
            df = df.groupby('mouse').mean()
            feat_std = df.mad().sort_values()

        plt.figure(figsize=(7, 6))
        plt.barh(feat_std[:topn].index, feat_std[:topn])
        plt.barh(feat_std[-topn:].index, feat_std[-topn:])
        plt.show()

    def show_stat(self, stat, condition='all'):
        if condition == 'all':
            df = self.data[[stat]]
        else:
            df = self.data[[self.params[x]['condition'] == condition for x in self.data.index]][[stat]]

        df = pd.DataFrame(MinMaxScaler().fit_transform(df), columns=df.columns, index=df.index)

        df['mouse'] = [self.params[session]['mouse'] for session in df.index]

        if condition == 'all':
            df['condition'] = [self.params[session]['condition'] for session in df.index]

            df = df.groupby(['mouse', 'condition']).mean().reset_index()

            fig, ax = plt.subplots(1, 2, figsize=(18, 8))

            ax[0].set_title(stat)

            sns.barplot(data=df,
                        y=stat,
                        x='mouse',
                        ax=ax[0]
                        )

            sns.barplot(data=df,
                        hue='condition',
                        y=stat,
                        x='mouse',
                        ax=ax[1]
                        )

            plt.legend(loc='upper right')
        else:
            df = df.groupby('mouse').mean().reset_index()
            sns.barplot(data=df,
                        y=stat,
                        x='mouse',
                        )

        plt.show()

    def get_stat_list(self):
        return self.data.columns.tolist()

