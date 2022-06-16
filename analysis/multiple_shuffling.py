from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from analysis.shuffling import ShuffleAnalysis
from analysis.functions import corr_df_to_distribution

sns.set(color_codes=True)


class MultipleShuffler:
    def __init__(self,
                 path_to_data,
                 dates,
                 fps,
                 num_of_shuffles=10,
                 shuffle_fractions=None,
                 correlation_type='active',
                 verbose=True):
        """

        :param path_to_data:
        :param dates:
        :param fps:
        :param num_of_shuffles:
        :param shuffle_fractions:
        :param correlation_type:
        :param verbose:
        """

        if shuffle_fractions is None:
            shuffle_fractions = [0.25, 0.5, 0.75, 1.]

        self.path_to_data = path_to_data
        self.dates = dates
        self.fps = fps
        self.num_of_shuffles = num_of_shuffles
        self.shuffle_fractions = shuffle_fractions
        self.correlation_type = correlation_type
        self.verbose = verbose

        self.models = self._generate_models()

        self.stat_df = self._create_stats()

        self.corr_df = self._create_corrs()

    def _generate_models(self):
        models = {0: {0: ShuffleAnalysis(self.path_to_data,
                                         self.dates, self.fps,
                                         shuffle_fraction=0,
                                         verbose=False)
                      }}

        for shuffle_fraction in tqdm(self.shuffle_fractions,
                                     disable=(not self.verbose),
                                     desc='Generating models...'):
            ptr = {}
            for i in range(self.num_of_shuffles):
                ptr[i] = ShuffleAnalysis(self.path_to_data,
                                         self.dates,
                                         self.fps,
                                         shuffle_fraction=shuffle_fraction,
                                         verbose=False)
            models[shuffle_fraction] = ptr
        return models

    def _create_stats(self):
        stat_df = pd.DataFrame()
        for shuffle_fraction in tqdm(self.models,
                                     disable=(not self.verbose),
                                     desc='Computing statistics...'
                                     ):
            for i in self.models[shuffle_fraction]:
                model = self.models[shuffle_fraction][i]
                for date in model.shuffled_data:
                    ptr_df = pd.DataFrame()

                    ptr_df['network spike rate'] = model.shuffled_data[date].network_spike_rate(1).T[
                        'spike rate'].tolist()
                    ptr_df['network spike peak'] = model.shuffled_data[date].network_spike_peak(1).T[
                        'peak'].tolist()
                    ptr_df['date'] = date
                    ptr_df['shuffle_fraction'] = shuffle_fraction
                    ptr_df['attempt'] = i

                    stat_df = stat_df.append(ptr_df)

        return stat_df.reset_index(drop=True)

    def _create_corrs(self):
        corr_df = pd.DataFrame()

        for shuffle_fraction in tqdm(self.models,
                                     disable=(not self.verbose),
                                     desc='Computing correlations...'
                                     ):
            for i in self.models[shuffle_fraction]:
                model = self.models[shuffle_fraction][i]

                for date in model.shuffled_data:
                    for position in [False, True]:
                        ptr_df = pd.DataFrame()

                        ptr_df['corr'] = corr_df_to_distribution(
                            model.shuffled_data[date].get_correlation(self.correlation_type, position=position))
                        ptr_df['date'] = date
                        ptr_df['position'] = position
                        ptr_df['attempt'] = i
                        ptr_df['shuffle_fraction'] = shuffle_fraction

                        corr_df = corr_df.append(ptr_df)

        corr_df['corr'] = corr_df['corr'].fillna(0)

        return corr_df.reset_index(drop=True)

    def show_day_mean_correlation_range(self,
                                        position=False):
        ptr = (
            self.corr_df[self.corr_df['position'] == position]
                .groupby(['shuffle_fraction', 'date', 'attempt']).agg({'corr': np.ptp})
                .groupby(['date', 'shuffle_fraction']).agg({'corr': np.mean})
                .reset_index()
        )

        plt.figure(figsize=(15, 8))
        sns.barplot(data=ptr, x='date', y='corr', hue='shuffle_fraction')
        plt.xlabel('Session', fontsize=16)
        plt.ylabel('Range of values', fontsize=16)
        plt.title(f'Mean correlation range ({self.correlation_type})', fontsize=18)
        plt.tick_params(axis='both', labelsize=14)
        plt.legend(title='Shuffle ratio', fontsize=14, bbox_to_anchor=(1, 1))  # , loc= 'lower right', shadow = True)
        plt.show()

    def show_position_mean_correlation_range(self):
        ptr = (
            self.corr_df
                .groupby(['position', 'shuffle_fraction', 'date', 'attempt']).agg({'corr': np.ptp})
                .groupby(['position', 'shuffle_fraction']).agg({'corr': np.mean})
                .reset_index()
        )

        plt.figure(figsize=(15, 8))
        sns.barplot(data=ptr, x='position', y='corr', hue='shuffle_fraction')
        plt.xlabel('Session', fontsize=16)
        plt.ylabel('Range of values', fontsize=16)
        plt.title(f'Mean correlation range ({self.correlation_type})', fontsize=18)
        plt.tick_params(axis='both', labelsize=14)
        plt.legend(title='Shuffle ratio', fontsize=14, bbox_to_anchor=(1, 1))  # , loc= 'lower right', shadow = True)
        plt.show()

    def show_mean_statistic_peak(self,
                                 daily=False,
                                 statistic_type='network spike peak'):
        plt.figure(figsize=(15, 8))

        if daily:
            ptr = (
                self.stat_df
                    .groupby(['shuffle_fraction', 'date', 'attempt']).agg({f'{statistic_type}': np.max})
                    .groupby(['date', 'shuffle_fraction']).agg({f'{statistic_type}': np.mean}).reset_index()
            )
            sns.barplot(data=ptr, x='date', y=f'{statistic_type}', hue='shuffle_fraction')
            plt.xlabel('Session', fontsize=16)
            plt.legend(title='Shuffle ratio',
                       fontsize=14,
                       bbox_to_anchor=(1, 1))  # , loc= 'lower right', shadow = True)
        else:
            ptr = (
                self.stat_df
                    .groupby(['shuffle_fraction', 'date', 'attempt']).agg({f'{statistic_type}': np.max})
                    .groupby(['shuffle_fraction']).agg({f'{statistic_type}': np.mean}).reset_index()
            )
            sns.barplot(data=ptr, x='shuffle_fraction', y=f'{statistic_type}')
            plt.xlabel('Shuffle fraction', fontsize=16)

        plt.ylabel('Peak values', fontsize=16)
        plt.title(f'Mean {statistic_type} peaks', fontsize=18)
        plt.tick_params(axis='both', labelsize=14)
        plt.show()
