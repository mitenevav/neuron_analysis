import pandas as pd
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import numpy as np
from analysis.minian import MinianAnalysis
from analysis.functions import active_df_to_dict, corr_df_to_distribution

sns.set(color_codes=True)


class DistanceAnalysis:
    """
    Class for analysing distance between neurons
    """

    def __init__(self, path_to_data, dates, fps):
        """
                Initialization function
        :param path_to_data: path to directory with sessions folders
        :param dates: folders session names
        :param fps: frames per second
        """
        self.base_correlations = [
            "signal",
            "active",
            "active_acc",
        ]
        self.dates = dates

        self.fps = fps
        self.models = {}

        for date in self.dates:
            ma = MinianAnalysis(f"{path_to_data}/{date}/minian/", fps)
            ma.active_state_df = pd.read_excel(
                f"{path_to_data}/{date}/results/active_states_spike.xlsx", index_col=0
            ).astype(bool)
            ma.active_state = active_df_to_dict(ma.active_state_df)
            self.models[date] = ma

        self.correlations = {}

        for date in tqdm(self.models):
            distr_dict = {}
            df_dict = {}
            for t in self.base_correlations:
                corr_df = self.models[date].get_correlation(t, position=False)

                df_dict[t] = corr_df.copy()
                distr_dict[t] = corr_df_to_distribution(corr_df)

            self.correlations[date] = {}
            self.correlations[date]["df"] = df_dict
            self.correlations[date]["distr"] = distr_dict

        for date in self.models:
            model = self.models[date]

            x = model.positions["x"] - model.positions["x"].mean()
            y = model.positions["y"] - model.positions["y"].mean()

            model.positions["r"] = (x**2 + y**2) ** (1 / 2)
            model.positions["phi"] = np.arctan2(y, x)

        self.distance_df = self.create_distance_df()

    def plot_radius_dependency(self):
        """
        Function for plotting distribution of rho and the dependence of the average signal intensity and activity on it
        """
        df = pd.DataFrame()
        for date in self.models:
            model = self.models[date]
            poss = model.positions.copy()
            poss["mean_sig"] = model.signals.mean()
            # poss['br'] = model.burst_rate()['activations per min'].rename(date).dropna().tolist()
            poss["active_ratio"] = (
                model.active_state_df.sum() / len(model.active_state_df)
            ).tolist()
            poss["date"] = date
            df = df.append(poss)

        df = df.reset_index(drop=True)

        fig = plt.figure(figsize=(15, 12))
        gs = GridSpec(3, 4, figure=fig)

        ax1 = fig.add_subplot(gs[0, :])

        sns.boxplot(data=df, x="r", y="date", ax=ax1)
        ax1.set_ylabel("Session", fontsize=16)
        ax1.set_xlabel("Rho distance, pixels", fontsize=16)
        ax1.set_title(f"Rho distribution", fontsize=18)
        ax1.tick_params(axis="both", labelsize=14)

        ax2 = fig.add_subplot(gs[1:, :2])
        sns.scatterplot(data=df, x="r", y="mean_sig", ax=ax2)
        ax2.set_xlabel("Rho distance, pixels", fontsize=16)
        ax2.set_ylabel(f"Signal mean", fontsize=16)
        ax2.set_title(f"Scatterplot", fontsize=18)
        ax2.tick_params(axis="both", labelsize=14)

        ax3 = fig.add_subplot(gs[1:, 2:])
        sns.scatterplot(data=df, x="r", y="active_ratio", ax=ax3)
        ax3.set_xlabel("Rho distance, pixels", fontsize=16)
        ax3.set_ylabel(f"Active ratio", fontsize=16)
        ax3.set_title(f"Scatterplot", fontsize=18)
        ax3.tick_params(axis="both", labelsize=14)

        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        plt.show()

    def create_distance_df(self):
        """
        Function for creating DataFrame with distance information
        """
        df = pd.DataFrame()
        for date in self.dates:
            day_df = pd.DataFrame()

            poss = self.models[date].positions
            day_df["euclidean"] = corr_df_to_distribution(
                poss.apply(
                    lambda x: ((poss["x"] - x["x"]) ** 2 + (poss["y"] - x["y"]) ** 2)
                    ** (1 / 2),
                    axis=1,
                )
            )

            day_df["radial"] = corr_df_to_distribution(
                poss.apply(lambda x: np.abs(x["r"] - poss["r"]), axis=1)
            )

            for t in self.correlations[date]["distr"]:
                day_df[t] = self.correlations[date]["distr"][t]

            day_df["date"] = date

            df = df.append(day_df)

        df = df.reset_index(drop=True)

        return df

    def plot_distance_distribution(self, dist_type="euclidean", thr=-1):
        """
        Function for plotting distance distribution
        :param dist_type: type of distance
                * 'euclidean'
                * 'radial'
        :param thr: correlation threshold for visualization
        """
        df = self.distance_df[self.distance_df['active'] >= thr]

        if len(df) > 0:
            plt.figure(figsize=(8, 6))

            sns.boxplot(data=df, x=dist_type, y="date")

            plt.xlabel("Distance, pixels", fontsize=16)
            plt.ylabel("Session", fontsize=16)
            plt.title(f"{dist_type} distance", fontsize=18)
            plt.tick_params(axis="both", labelsize=14)

            plt.show()
        else:
            print('There is no data for the selected parameters')


    def plot_dependency(self, x, y):
        """
        Function for plotting dependence of x on y
        :param x, y: column
                * 'euclidean' (distance)
                * 'radial' (distance)
                * 'signal' (correlation)
                * 'active' (correlation)
                * 'active_acc' (correlation)
        """
        plt.figure(figsize=(8, 6))

        sns.scatterplot(data=self.distance_df, x=x, y=y, hue="date")

        plt.xlabel(x, fontsize=16)
        plt.ylabel(y, fontsize=16)
        plt.title(f"Dependency between {x} and {y}", fontsize=18)
        plt.tick_params(axis="both", labelsize=14)
        plt.show()
