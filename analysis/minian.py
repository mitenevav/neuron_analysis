import numpy as np
import zarr
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.notebook import tqdm
from scipy import ndimage

sns.set(color_codes=True)
from os import path, mkdir


class MinianAnalysis:
    def __init__(self, path_to_data, fps, path_to_results=None):
        """
        Initialization function
        :param path_to_data: path to minian output directory
        :param fps: frames per second
        :path_to_results: path to folder for results
        """
        if path_to_results is None:
            path_to_results = path_to_data + '../results'
        signals = zarr.open_group(path_to_data + 'C.zarr')
        self.signals = pd.DataFrame(signals.C).set_index(pd.DataFrame(signals.unit_id)[0].rename('unit_id')).T

        positions = zarr.open_group(path_to_data + 'A.zarr')
        positions_centers = np.array([ndimage.measurements.center_of_mass(x) for x in np.array(positions.A)])
        self.positions = pd.DataFrame({'unit_id': positions.unit_id,
                                       'x': positions_centers[:, 0],
                                       'y': positions_centers[:, 1],
                                       }).set_index('unit_id')

        self.fps = fps

        self.smooth_signals = None
        self.diff = None
        self.smooth_diff = None

        self.active_state = {}
        self.active_state_df = pd.DataFrame()

        self.type_of_activity = None
        self.results_folder = path_to_results

    @staticmethod
    def __get_active_states(signal, threshold):
        """
        Function for determining the active states of the input signal
        :param signal:
        :param threshold:
        :return: list of lists with active states indexes
        """
        res = []
        sleep = signal[signal <= threshold].reset_index()
        sleep_min = sleep['index'].min()

        if len(sleep) == 0:
            return [np.arange(0, len(signal), dtype='int').tolist()]
        elif sleep_min > 0:
            # res.append(np.arange(0, sleep_min + 1, dtype='int').tolist())
            res.append(np.arange(0, sleep_min, dtype='int').tolist())

        sleep['index_diff'] = sleep['index'].diff()

        changes = sleep[sleep['index_diff'] > 1].copy()

        if len(changes) == 0:
            return res

        changes['start'] = changes['index'] - changes['index_diff'] + 1
        changes['end'] = changes['index']  # + 1

        res += changes.apply(lambda x: np.arange(x['start'], x['end'], dtype='int').tolist(), axis=1).tolist()

        sleep_max = sleep['index'].max() + 1
        if sleep_max < len(signal):
            res.append(np.arange(sleep_max, len(signal), dtype='int').tolist())
        return res

    @staticmethod
    def __get_peaks(spike, decay=None, cold=0, warm=0):
        """
        Function for post-processing of found activity states.
        :param spike: list of growing parts
        :param decay: list of decaying parts
        :param cold: minimum duration of active state
        :param warm: minimum duration of inactive state
        :return: list of lists with active states indexes
        """
        peaks_idx = spike + decay if decay else spike
        peaks_idx.sort(key=lambda x: x[0])
        new_peaks = [peaks_idx[0]]

        for i in range(1, len(peaks_idx)):
            gap = peaks_idx[i][0] - new_peaks[-1][-1]
            if gap <= warm:
                new_peaks[-1] = (new_peaks[-1] +
                                 [i for i in range(new_peaks[-1][-1] + 1, peaks_idx[i][0])] +
                                 peaks_idx[i])
            else:
                new_peaks.append(peaks_idx[i])

        peaks = []
        for i in new_peaks:
            if len(i) > cold:
                peaks.append(i)

        return peaks

    def find_active_state(self, window, cold, warm, method='spike', verbose=True):
        """
        Function for preprocessing signals and determining the active states
        :param window: size of the moving window for smoothing
        :param cold: minimum duration of active state
        :param warm: minimum duration of inactive state
        :param method: ['spike', 'full'] type of active state
            spike - only the stage of intensity growth
            full - the stage of growth and weakening of intensity
        :param verbose: verbose
        """
        self.type_of_activity = method

        # rolling mean
        self.smooth_signals = self.signals.rolling(window=window, center=True, min_periods=0).mean()

        # derivative
        self.diff = self.signals.diff()[1:].reset_index(drop=True)
        self.smooth_diff = self.smooth_signals.diff()[1:].reset_index(drop=True)

        for num in tqdm(self.smooth_signals.columns):

            y = self.smooth_diff[num]

            y_pos = y[y >= 0]
            mad_pos = np.mean(np.abs(np.median(y_pos) - y_pos))
            threshold_pos = np.median(y_pos) + mad_pos
            peaks_pos_idx = self.__get_active_states(y, threshold_pos)

            if method == 'full':
                y_neg = -y[y <= 0]
                mad_neg = np.mean(np.abs(np.median(y_neg) - y_neg))
                threshold_neg = np.median(y_neg) + mad_neg
                peaks_neg_idx = self.__get_active_states(-y, threshold_neg)
            else:
                peaks_neg_idx = []

            peaks_idx = self.__get_peaks(peaks_pos_idx, peaks_neg_idx, cold=cold, warm=warm)

            self.active_state[num] = peaks_idx

            if verbose:

                signal = self.signals[num]

                plt.figure(figsize=(15, 10))
                plt.title(f'Neuron {num}', fontsize=18)

                plt.plot(signal, label='sleep')
                for peak in peaks_idx:
                    plt.plot(signal.iloc[peak], c='r')

                if len(peaks_idx) > 0:
                    plt.plot(signal.iloc[peaks_idx[0]], c='r', label='active')

                plt.plot(range(len(signal)), [0] * len(signal), c='b', lw=3)
                for peak in peaks_idx:
                    plt.plot(peak, [0] * len(peak), c='r', lw=3)

                plt.legend(fontsize=18)
                plt.show()

        for neuron in self.active_state:
            self.active_state_df[neuron] = [False] * len(self.signals)
            for peak in self.active_state[neuron]:
                self.active_state_df[neuron].iloc[peak] = True

    def get_active_state(self, neuron, window, cold, warm, method='spike'):
        """
        Function for preprocessing neuron signal and determining the active states
        :param neuron: neuron number
        :param window: size of the moving window for smoothing
        :param cold: minimum duration of active state
        :param warm: minimum duration of inactive state
        :param method: ['spike', 'full'] type of active state
            spike - only the stage of intensity growth
            full - the stage of growth and weakening of intensity
        """
        signal = self.signals[neuron]
        # rolling mean
        smooth_signal = signal.rolling(window=window, center=True, min_periods=0).mean()

        # derivative
        diff = signal.diff()[1:].reset_index(drop=True)
        smooth_diff = smooth_signal.diff()[1:].reset_index(drop=True)

        y_pos = smooth_diff[smooth_diff >= 0]
        mad_pos = np.mean(np.abs(np.median(y_pos) - y_pos))
        threshold_pos = np.median(y_pos) + mad_pos
        peaks_pos_idx = self.__get_active_states(smooth_diff, threshold_pos)

        if method == 'full':
            y_neg = -smooth_diff[smooth_diff <= 0]
            mad_neg = np.mean(np.abs(np.median(y_neg) - y_neg))
            threshold_neg = np.median(y_neg) + mad_neg
            peaks_neg_idx = self.__get_active_states(-smooth_diff, threshold_neg)
        else:
            peaks_neg_idx = []

        peaks_idx = self.__get_peaks(peaks_pos_idx, peaks_neg_idx, cold=cold, warm=warm)

        plt.figure(figsize=(15, 10))
        plt.title(f'Neuron {neuron}', fontsize=22)

        plt.plot(signal, label='sleep', c='b', lw=4)
        for peak in peaks_idx:
            plt.plot(signal.iloc[peak], c='r', lw=4)

        if len(peaks_idx) > 0:
            plt.plot(signal.iloc[peaks_idx[0]], c='r', label='active', lw=4)

        plt.plot(range(len(signal)), [0] * len(signal), c='b', lw=5)
        for peak in peaks_idx:
            plt.plot(peak, [0] * len(peak), c='r', lw=5)

        plt.legend(fontsize=20)
        plt.show()

    def burst_rate(self):
        """
        Function for computing burst rate
        Burst rate - number of cell activations per minute
        """
        num_of_activations = []
        for neuron in self.active_state:
            num_of_activations.append(len(self.active_state[neuron]))

        burst_rate = pd.DataFrame({'num_of_activations': num_of_activations})

        burst_rate['activations per min'] = burst_rate['num_of_activations'] / len(self.active_state_df) * self.fps * 60

        burst_rate['activations per min'] = burst_rate['activations per min'].round(2)

        return burst_rate

    def network_spike_rate(self, period):
        """
        Function for computing network spike rate
        Network spike rate - percentage of active neurons per period
        :param period: period in seconds
        """
        step = period * self.fps

        periods = pd.DataFrame()
        for i in range(0, len(self.active_state_df), step):
            ptr = []
            for neuron in self.active_state_df:
                ptr.append(True in self.active_state_df[neuron][i:i + step].tolist())

            periods[i] = ptr

        nsr = {}
        for x in periods:
            nsr[f'{x}-{x + step}'] = len(periods[x][periods[x] == True])

        nsr = pd.DataFrame(nsr, index=['spike rate'])
        nsr = nsr / len(self.active_state_df.columns) * 100

        return nsr

    def network_spike_duration(self, thresholds, verbose=True):
        """
        Function for computing network spike duration
        Network spike duration - duration when the percentage of active cells is above the set thresholds
        :param thresholds: threshold values in percentages
        :param verbose: progressbar
        """
        spike_durations = []

        for thr in tqdm(thresholds, disable=(not verbose)):
            percent_thr = len(self.active_state_df.columns) * thr / 100
            duration = 0
            for _, row in self.active_state_df.iterrows():
                if len(row[row == True]) > percent_thr:
                    duration += 1
            spike_durations.append(duration)

        nsd_df = pd.DataFrame({'percentage': thresholds,
                               'Network spike duration': np.array(spike_durations) / self.fps
                               })
        return nsd_df

    def network_spike_peak(self, period):
        """
        Function for computing network spike peak
        Network spike peak - maximum percentage of active cells per second
        :param period: period in seconds
        """
        step = period * self.fps

        spike_peaks = {}
        for i in range(0, len(self.active_state_df), step):
            peak = 0
            for _, row in self.active_state_df[i:i + step].iterrows():
                current_peak = len(row[row == True])
                if current_peak > peak:
                    peak = current_peak

            spike_peaks[f'{i}-{i + step}'] = peak

        nsp_df = pd.DataFrame(spike_peaks, index=['peak'])
        nsp_df = nsp_df / len(self.active_state_df.columns) * 100

        return nsp_df

    def show_burst_rate(self, max_bins=15):
        """
        Function for plotting burst rate
        Burst rate - number of cell activations per minute
        :param max_bins: maximum number of columns
        """
        burst_rate = self.burst_rate()

        plt.figure(figsize=(8, 6))
        plt.title('Burst rate', fontsize=17)

        if burst_rate['activations per min'].nunique() > max_bins:
            sns.histplot(data=burst_rate, x='activations per min', bins=max_bins, stat='percent')
        else:
            burst_rate = (
                burst_rate['activations per min']
                    .value_counts(normalize=True)
                    .mul(100)
                    .rename('percent')
                    .reset_index()
                    .rename(columns={'index': 'activations per min'})
            )
            sns.barplot(data=burst_rate, x='activations per min', y='percent')

        plt.xlabel('activations per min', fontsize=16)
        plt.ylabel('percent', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()

    def show_network_spike_rate(self, period):
        """
        Function for plotting network spike rate
        Network spike rate - percentage of active neurons per period
        :param period: period in seconds
        """
        nsr = self.network_spike_rate(period)
        plt.figure(figsize=(8, 6))
        plt.title('Network spike rate', fontsize=17)
        sns.histplot(data=nsr.T, x='spike rate', stat='percent')
        plt.xlabel(f'percentage of active neurons per {period} second', fontsize=16)
        plt.ylabel('percent', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()

    def show_network_spike_duration(self, thresholds):
        """
        Function for plotting network spike duration
        Network spike duration - duration when the percentage of active cells is above the set thresholds
        :param thresholds: threshold values in percentages
        """
        nsd_df = self.network_spike_duration(thresholds)
        plt.figure(figsize=(8, 6))
        plt.title('Network spike duration', fontsize=17)
        sns.barplot(data=nsd_df, x='percentage', y='Network spike duration')
        plt.xlabel('percentage of active neurons', fontsize=16)
        plt.ylabel('seconds', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()

    def show_network_spike_peak(self, period):
        """
        Function for plotting network spike peak
        Network spike peak - maximum percentage of active cells per second
        :param period: period in seconds
        """
        nsp_df = self.network_spike_peak(period)
        plt.figure(figsize=(8, 6))
        plt.title('Network spike peak', fontsize=17)
        sns.histplot(data=nsp_df.T, x='peak', bins=8, stat='percent')
        plt.xlabel(f'max percentage of active neurons per {period} second', fontsize=16)
        plt.ylabel('percent', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()

    def save_burst_rate(self):
        """
        Function for saving burst rate
        Burst rate - number of cell activations per minute
        """
        burst_rate = self.burst_rate()
        burst_rate.to_csv(self.results_folder + '/burst_rate.csv')

    def save_network_spike_rate(self, period):
        """
        Function for saving network spike rate
        Network spike rate - percentage of active neurons per period
        :param period: period in seconds
        """
        nsr = self.network_spike_rate(period)
        nsr.to_csv(self.results_folder + '/network_spike_rate.csv')

    def save_network_spike_duration(self, thresholds):
        """
        Function for saving network spike duration
        Network spike duration - duration when the percentage of active cells is above the set thresholds
        :param thresholds: threshold values in percentages
        """
        nsd_df = self.network_spike_duration(thresholds)
        nsd_df.to_csv(self.results_folder + '/network_spike_duration.csv')

    def save_network_spike_peak(self, period):
        """
        Function for saving network spike peak
        Network spike peak - maximum percentage of active cells per second
        :param period: period in seconds
        """
        nsp_df = self.network_spike_peak(period)
        nsp_df.to_csv(self.results_folder + '/network_spike_peak.csv')

    def compute_nzsfi(self):
        """
        Function for computing NonZeroSpikeFramesIntersection
        :return: FataFrame with NonZeroSpikeFramesIntersection values
        """
        nzsfi = pd.DataFrame(columns=self.active_state_df.columns)
        for i in self.active_state_df:
            nzsfi[i] = [self.active_state_df[i].sum() / (self.active_state_df[i] & self.active_state_df[j]).sum() for j
                        in self.active_state_df]

        return nzsfi.T

    def compute_spike_accuracy(self):
        """
        Function for computing spike accuracy (intersection / union)
        :return: FataFrame with spike accuracy
        """
        spike_acc = pd.DataFrame()
        for i in self.active_state_df:
            row = []
            for j in self.active_state_df:
                union = (self.active_state_df[i] & self.active_state_df[j]).sum()
                intersec = (self.active_state_df[i] | self.active_state_df[j]).sum()
                row.append((union / intersec))
            spike_acc[i] = row

        return spike_acc.T.set_axis(self.active_state_df.columns, axis=1)

    def get_correlation(self, method='signal', position=False):
        """
        Function for computing correlation
        :param method: ['signal', 'diff', 'active', 'active_acc'] type of correlated sequences
            signal - Pearson correlation for signal intensity
            diff   - Pearson correlation for derivative of signal intensity
            active - Pearson correlation for binary segmentation of active states (depends on the chosen method for find_active_state)
            active_acc - ratio of intersection to union of active states (depends on the chosen method for find_active_state)
        :param position: consideration of spatial position
        """
        if method == 'signal':
            corr_df = self.signals.corr()
        elif method == 'diff':
            corr_df = self.smooth_diff.corr()
        elif method == 'active':
            corr_df = self.active_state_df.corr()
        elif method == 'active_acc':
            corr_df = self.compute_spike_accuracy()
        else:
            print(f'Method {method} is not supported!')
            return

        if position:
            distances = self.positions.apply(
                lambda x: ((self.positions['x'] - x['x']) ** 2 + (self.positions['y'] - x['y']) ** 2) ** (1 / 2),
                axis=1
            )

            corr_df = (1 - 100 / (distances + 100)) * corr_df.values  # 100 is 25% of distance

        return corr_df

    def save_active_states(self):
        """
        Function for saving active states matrix to results folder (depends on the chosen method for find_active_state)
        """
        if not path.exists(self.results_folder):
            mkdir(self.results_folder)
        self.active_state_df.astype(int).to_csv(
            path.join(self.results_folder, f'active_states_{self.type_of_activity}.csv'))

    def save_correlation_matrix(self, method='signal', position=False):
        """
        Function for saving correlation matrix to results folder
        :param method: ['signal', 'diff', 'active', 'active_acc'] type of correlated sequences
            signal - Pearson correlation for signal intensity
            diff   - Pearson correlation for derivative of signal intensity
            active - Pearson correlation for binary segmentation of active states (depends on the chosen method for find_active_state)
            active_acc - ratio of intersection to union of active states (depends on the chosen method for find_active_state)
        :param position: consideration of spatial position
        """
        corr_df = self.get_correlation(method, position)

        if not path.exists(self.results_folder):
            mkdir(self.results_folder)

        corr_df.to_csv(path.join(self.results_folder,
                                 f"correlation_{self.type_of_activity}_{method}{'_position' if position else ''}.csv"))

    def show_corr(self, threshold, method='signal', position=False):
        """
        Function for plotting correlation distribution and map
        :param threshold: threshold for displayed correlation
        :param method: ['signal', 'diff', 'active', 'active_acc'] type of correlated sequences
            signal - Pearson correlation for signal intensity
            diff   - Pearson correlation for derivative of signal intensity
            active - Pearson correlation for binary segmentation of active states (depends on the chosen method for find_active_state)
            active_acc - ratio of intersection to union of active states (depends on the chosen method for find_active_state)
        :param position: consideration of spatial position
        """
        corr_df = self.get_correlation(method, position)

        fig, ax = plt.subplots(1, 2, figsize=(20, 10))

        c = 1
        corr = []
        for i, row in corr_df.iterrows():
            for j in corr_df.columns.tolist()[c:]:
                corr.append(row[j])

            c += 1

        sns.histplot(corr, stat='percent', ax=ax[0])
        ax[0].set_ylabel('percent', fontsize=20)
        ax[0].set_title(f'Correlation distribution for {method} method', fontsize=24)

        corr_df = corr_df[(corr_df > threshold) & (corr_df.abs() < 1)]
        corr_df.dropna(axis=0, how='all', inplace=True)
        corr_df.dropna(axis=1, how='all', inplace=True)

        ax[1].set_title(f'Correlation map for {method} method', fontsize=24)

        c = 0
        for i, row in corr_df.iterrows():
            for j in corr_df.columns.tolist()[c:]:
                if not np.isnan(row[j]):
                    ax[1].plot(
                        self.positions.loc[[i, j]]['y'], self.positions.loc[[i, j]]['x'],
                        color='r',
                        lw=0.5 + (row[j] - threshold) / (1 - threshold) * 4,
                    )

            ax[1].scatter(x=self.positions.loc[i]['y'], y=self.positions.loc[i]['x'], color='w', zorder=5)
            c += 1

        ax[1].scatter(x=self.positions['y'], y=self.positions['x'], s=100, zorder=4)
        plt.show()
