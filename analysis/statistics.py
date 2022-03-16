import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import kstest, mannwhitneyu, ks_2samp
from tqdm.notebook import tqdm

from analysis.functions import corr_df_to_distribution, active_df_to_dict
from analysis.minian import MinianAnalysis

sns.set(color_codes=True)


class StatTests:
    def __init__(self, path_to_data, dates, fps):
        self.corr_types = [
            'signal',
            'diff',
            'active',
            'active_acc',
        ]
        self.fps = fps

        self.corr_dfs = {}
        self.corr_distr = {}
        self.signals = {}
        self.dates = dates
        self.path_to_data = path_to_data

        for date in self.dates:
            distr_dict = {}
            df_dict = {}
            for t in self.corr_types:
                corr_df = pd.read_csv(f'{self.path_to_data}/{date}/results/correlation_spike_{t}.csv', index_col=0)
                df_dict[t] = corr_df.copy()

                distr_dict[t] = corr_df_to_distribution(corr_df)

            self.corr_dfs[date] = df_dict

            self.corr_distr[date] = distr_dict

            ma = MinianAnalysis(f'{self.path_to_data}/{date}/minian/', self.fps)
            ma.active_state_df = pd.read_csv(f'{self.path_to_data}/{date}/results/active_states_spike.csv',
                                             index_col=0).astype(bool)
            ma.active_state = active_df_to_dict(ma.active_state_df)

            self.signals[date] = ma

        br = pd.DataFrame()
        nsr = pd.DataFrame()
        nsp = pd.DataFrame()
        nsd = pd.DataFrame()

        for date in tqdm(self.signals):
            br = br.append(self.signals[date].burst_rate()['activations per min'].rename(date))
            nsr = nsr.append(self.signals[date].network_spike_rate(1).T['spike rate'].rename(date))
            nsp = nsp.append(self.signals[date].network_spike_peak(1).T['peak'].rename(date))
            nsd = nsd.append(
                self.signals[date].network_spike_duration(np.arange(3, 63, 3))['Network spike duration'].rename(date))

        self.br = br.T
        self.nsr = nsr.T
        self.nsp = nsp.T
        self.nsd = nsd.T

        self.all_corr = pd.DataFrame()

        for date in dates:
            for t in self.corr_types:
                self.all_corr = self.all_corr.append(
                    pd.DataFrame({'values': self.corr_distr[date][t], 'date': date, 'type': t}))

    def show_correlation_distribution(self, method='kde'):
        if method == 'box':
            plt.figure(figsize=(12, 10))
            sns.boxplot(data=self.all_corr, y='values', hue='date', x='type')
            plt.title('Correlation distribution', fontsize=17)
            plt.xlabel('type', fontsize=16)
            plt.ylabel('Correlation', fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.legend(fontsize=16)
            plt.show()

        elif method == 'hist':
            fig, axs = plt.subplots(2, 2, figsize=(18, 15))

            for t, ax in zip(self.corr_types, axs.flatten()):
                df = pd.DataFrame()

                for date in self.dates:
                    label = f'{len(self.corr_dfs[date][t])} neurons'
                    df = df.append(pd.DataFrame(self.corr_distr[date][t]).T.rename(index=lambda x: label))

                sns.histplot(df.T, stat='percent', bins=20, ax=ax)

                ax.set_title(f'Correlation distribution {t}', fontsize=17)
                ax.set_xlabel('Correlation', fontsize=16)
                ax.set_ylabel('Density', fontsize=16)
                ax.tick_params(axis='both', labelsize=14)
            plt.show()

        else:
            fig, axs = plt.subplots(2, 2, figsize=(18, 15))

            for t, ax in zip(self.corr_types, axs.flatten()):

                for date in self.dates:
                    sns.kdeplot(self.corr_distr[date][t],
                                label=f'{len(self.corr_dfs[date][t])} neurons',
                                hue_norm=True,
                                linewidth=3,
                                ax=ax)

                ax.set_title(f'Correlation distribution {t}', fontsize=17)
                ax.set_xlabel('Correlation', fontsize=16)
                ax.set_ylabel('Density', fontsize=16)
                ax.tick_params(axis='both', labelsize=14)
                ax.legend(fontsize=16)
            plt.show()

    def __distr_test(self, data):
        dist_tets = pd.DataFrame(columns=[True, False])

        n = (len(self.dates) ** 2 - len(self.dates)) / 2

        alpha = 0.05 / n

        for test, title in zip([mannwhitneyu, ks_2samp],
                               ['Mann–Whitney U test', 'Kolmogorov–Smirnov test']
                               ):

            test_res = {}
            for t in data:
                df = data[t]
                lst = []
                for i in range(len(self.dates) - 1):
                    for j in range(i + 1, len(self.dates)):
                        p = test(df[self.dates[i]].dropna(), df[self.dates[j]].dropna())[1]
                        lst.append(p)
                test_res[t] = (pd.Series(lst) > alpha).value_counts()

            test_res = pd.DataFrame(test_res).T.reset_index().rename(columns={'index': 'type'})
            test_res['test'] = title
            dist_tets = dist_tets.append(test_res)

        dist_tets = dist_tets.groupby(['test', 'type']).agg({True: 'first', False: 'first'}).fillna(0).astype(int)

        return dist_tets

    def __norm_test(self, data):
        test_df = pd.DataFrame(columns=[True, False])

        alpha = 0.05 / len(self.dates)

        test_res = {}
        for t in data:
            df = data[t]
            lst = []
            for i in range(len(self.dates)):
                lst.append(kstest(df[self.dates[i]], 'norm')[1])
                # lst.append(normaltest(df[dates[i]])[1])

            test_res[t] = (pd.Series(lst) > alpha).value_counts()

        test_df = test_df.append(pd.DataFrame(test_res).T).fillna(0)

        return test_df

    def __get_test_data(self, data_type):
        if data_type == 'corr':
            data = {}
            for t in self.corr_types:
                type_df = pd.DataFrame()
                for date in self.dates:
                    type_df = type_df.append(pd.Series(self.corr_distr[date][t], name=date))

                data[t] = type_df.T
            return data
        else:
            data = {}
            for df, name in zip([self.br, self.nsr, self.nsp, self.nsd],
                                ['Burst rate', 'Network spike rate', 'Network spike peak', 'Network spike duration']):
                data[name] = df
            return data

    def get_test(self, test_type='distr', data_type='stat'):
        data = self.__get_test_data(data_type)
        if test_type == 'norm':
            return self.__norm_test(data)
        else:
            return self.__distr_test(data)

    def show_stats_distribution(self):

        fig, axs = plt.subplots(2, 2, figsize=(18, 15))
        for df, name, ax in zip([self.br, self.nsr, self.nsp, self.nsd],
                                ['Burst rate', 'Network spike rate', 'Network spike peak', 'Network spike duration'],
                                axs.flatten()
                                ):

            for day in df:
                sns.kdeplot(df[day], label=f"{len(self.signals[day].active_state)} neurons", hue_norm=True, linewidth=3,
                            ax=ax)

            ax.set_title(f'{name} distribution', fontsize=17)
            ax.set_xlabel(name, fontsize=16)
            ax.set_ylabel('Density', fontsize=16)
            ax.tick_params(axis='both', labelsize=14)
            ax.legend(fontsize=16)

        plt.show()
