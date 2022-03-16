import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm

from analysis.functions import corr_df_to_distribution, active_df_to_dict
from analysis.minian import MinianAnalysis

sns.set(color_codes=True)


class ShuffleAnalysis:
    def __init__(self, path_to_data, dates, fps):
        self.dates = dates
        self.path_to_data = path_to_data
        self.fps = fps

        self.original_data = {}
        self.shuffled_data = {}

        for date in tqdm(self.dates):
            ma_o = MinianAnalysis(f'{self.path_to_data}/{date}/minian/', self.fps)
            ma_o.active_state_df = pd.read_csv(f'{self.path_to_data}/{date}/results/active_states_spike.csv',
                                               index_col=0).astype(bool)
            ma_o.active_state = active_df_to_dict(ma_o.active_state_df)

            ma_s = MinianAnalysis(f'{self.path_to_data}/{date}/minian/', self.fps)
            ma_s.active_state_df = ma_o.active_state_df.apply(self.shuffle_signal)
            ma_s.active_state = active_df_to_dict(ma_s.active_state_df)

            self.original_data[date] = ma_o
            self.shuffled_data[date] = ma_s

    @staticmethod
    def shuffle_signal(signal):
        res = []
        sleep = signal[signal == 0].reset_index()
        sleep_min = sleep['index'].min()

        if len(sleep) == 0:
            res = [np.arange(0, len(signal), dtype='int').tolist()]
        else:
            if sleep_min > 0:
                res.append(np.arange(0, sleep_min, dtype='int').tolist())

            sleep['index_diff'] = sleep['index'].diff()

            changes = sleep[sleep['index_diff'] > 1].copy()

            if len(changes) > 0:
                changes['start'] = changes['index'] - changes['index_diff'] + 1
                changes['end'] = changes['index']

                res += changes.apply(lambda x: np.arange(x['start'], x['end'], dtype='int').tolist(), axis=1).tolist()

                sleep_max = sleep['index'].max() + 1
                if sleep_max < len(signal):
                    res.append(np.arange(sleep_max, len(signal), dtype='int').tolist())

        starts = []
        lens = []
        for x in res:
            starts.append(x[0])
            lens.append(len(x))

        df = pd.DataFrame({'start': starts, 'len': lens})

        intervals = np.random.randint(100, size=len(df) + 1)
        intervals = intervals / intervals.sum() * len(signal[signal == 0])
        intervals = intervals.astype(int)
        intervals[-1] += len(signal[signal == 0]) - intervals.sum()

        order = np.arange(len(df))
        np.random.shuffle(order)

        shuff = []

        for i in range(len(order)):
            shuff += [0] * intervals[i]
            shuff += [1] * df.iloc[order[i]]['len']

        shuff += [0] * intervals[-1]

        return shuff

    def correlation_dist(self, corr_type='active'):
        values = []
        models = []
        dates_df = []

        for date in tqdm(self.dates):
            corr = corr_df_to_distribution(self.original_data[date].get_correlation(corr_type))
            values += corr
            models += ['original'] * len(corr)
            dates_df += [date] * len(corr)

            corr = corr_df_to_distribution(self.shuffled_data[date].get_correlation(corr_type))
            values += corr
            models += ['shuffle'] * len(corr)
            dates_df += [date] * len(corr)

        df = pd.DataFrame({'values': values, 'model': models, 'date': dates_df})
        ptp = df.groupby(['model', 'date']).agg({'values': np.ptp}).reset_index()
        diff = ptp.groupby(['date']).agg({'model': list, 'values': np.diff}).reset_index()
        diff['values'] = -diff['values']

        fig, ax = plt.subplots(1, 2, figsize=(15, 5))

        ax[0].set_title('Размах', fontsize=16)
        sns.barplot(data=ptp, hue='model', x='date', y='values', ax=ax[0])
        ax[0].tick_params(axis='x', rotation=45)
        ax[0].tick_params(axis='both', labelsize=13)
        ax[0].set_ylabel('Размах', fontsize=14)
        ax[0].set_xlabel('Дата', fontsize=14)

        ax[1].set_title('Разность размаха', fontsize=16)
        sns.barplot(data=diff, x='date', y='values', ax=ax[1])
        ax[1].tick_params(axis='x', rotation=45)
        ax[1].tick_params(axis='both', labelsize=13)
        ax[1].set_ylabel('Разность', fontsize=14)
        ax[1].set_xlabel('Дата', fontsize=14)

        plt.show()

    def statistic_dist(self, stat_type='network_spike_rate'):
        values = []
        models = []
        dates_df = []

        for data, name in zip([self.original_data, self.shuffled_data],
                              ['original', 'shuffle']):

            for date in self.dates:

                if stat_type == 'network_spike_peak':
                    val = data[date].network_spike_peak(1).T['peak'].tolist()
                else:
                    val = data[date].network_spike_rate(1).T['spike rate'].tolist()

                values += val
                models += [name] * len(val)
                dates_df += [date] * len(val)

        df = pd.DataFrame({'values': values, 'model': models, 'date': dates_df})
        maximum = df.groupby(['model', 'date']).agg({'values': np.max}).reset_index()
        mean = df.groupby(['model', 'date']).agg({'values': np.mean}).reset_index()

        fig, ax = plt.subplots(1, 2, figsize=(15, 5))

        ax[0].set_title('Максимум', fontsize=16)
        sns.barplot(data=maximum, hue='model', x='date', y='values', ax=ax[0])
        ax[0].tick_params(axis='x', rotation=45)
        ax[0].tick_params(axis='both', labelsize=13)
        ax[0].set_ylabel('Максимум', fontsize=14)
        ax[0].set_xlabel('Дата', fontsize=14)

        ax[1].set_title('Среднее', fontsize=16)
        sns.barplot(data=mean, hue='model', x='date', y='values', ax=ax[1])
        ax[1].tick_params(axis='x', rotation=45)
        ax[1].tick_params(axis='both', labelsize=13)
        ax[1].set_ylabel('Среднее', fontsize=14)
        ax[1].set_xlabel('Дата', fontsize=14)

        plt.show()

    def show_shuffling(self, date):
        fig, ax = plt.subplots(2, 1, figsize=(15, 10))

        ax[0].set_title('Исходные данные', fontsize=20)
        sns.heatmap(self.original_data[date].active_state_df, cbar=False, ax=ax[0])
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[0].set_ylabel('Нейроны', fontsize=18)

        ax[1].set_title('Перемешанные данные', fontsize=20)
        sns.heatmap(self.shuffled_data[date].active_state_df, cbar=False, ax=ax[1])
        ax[1].set_yticks([])
        ax[1].set_xticks([])
        ax[1].set_xlabel('Время \u2192', fontsize=18)
        ax[1].set_ylabel('Нейроны', fontsize=18)

        plt.show()
