from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import ticker

import numpy as np
import pandas as pd

from scipy.interpolate import interp1d
from scipy.signal import find_peaks, peak_widths
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from sklearn.metrics import auc


def get_auc(data_table, sample_name: str, prominence=0.9, plot=False, **kwargs) -> dict:
    """Detects peaks and calculates area under the peak for time vs cps HPLC ICPMS data"""
    wlen = kwargs.get('wlen', 10)  # minimum width for a peak
    threshold = kwargs.get('threshold', 100)  # how many cps units higher than it's neighbors
    width = kwargs.get('width', (1,))
    distance = kwargs.get('distance', 50)
    height = kwargs.get('height', 1)
    return_fig: bool = kwargs.get('return_fig', False)
    kwargs["prom"] = prominence

    peaks, peak_info = find_peaks(data_table["cps"],
                                  height=height,
                                  threshold=threshold,
                                  distance=distance,
                                  prominence=prominence,
                                  width=width,
                                  wlen=wlen)

    # further refine the peaks
    results_full = peak_widths(x=data_table["cps"],
                               peaks=peaks,
                               rel_height=prominence)
    widths, h_eval, left_ips, right_ips = results_full

    auc_info = defaultdict(float)
    auc_info["sample_name"] = sample_name
    auc_info["arsenite"] = 0
    auc_info["arsenate"] = 0

    saved_peaks = dict(arsenite=[15, 35],
                       arsenate=[90, 160])

    sample_labels = []

    peak_coordinates = defaultdict(dict)

    # find the area under the curve for each peak
    for i, [left, right] in enumerate(zip(left_ips, right_ips)):
        y = data_table["cps"][int(left):int(right)]

        # interploate a line across the bottom of each peak
        y_base = get_peak_baseline(data_table["time"], data_table["cps"], [int(left), int(right)])

        x = data_table["time"][int(left):int(right)]

        peak_auc = auc(x, y) - auc(x, y_base)

        # limit the peak areas to the ones I'm interested in
        for key, [l_limit, r_limit] in saved_peaks.items():
            if l_limit < peaks[i] < r_limit:
                auc_info[key] = peak_auc
                peak_coordinates[key] = {"peak_id": peaks[i],
                                         "peak_area": peak_auc,
                                         "x": x,
                                         "y_base": y_base,
                                         "y": y}
                sample_labels.append(key)

    if plot:
        plot_auc(data_table, peak_coordinates, title=sample_name, **kwargs)

    if return_fig:
        auc_info["fig"], auc_info["ax"] = plot_auc(data_table, peak_coordinates, title=sample_name, **kwargs)
    return dict(auc_info)


def get_peak_baseline(time, cps, peaks: list):
    left, right = peaks
    slope, intercept, r_value, p_value, std_err = linregress(x=[time[left], time[right]],
                                                             y=[cps[left], cps[right]])
    x = time[left:right]
    y = (x*slope+intercept)
    func = interp1d(x, y)

    # Get baseline values
    base_time = time[left:right]
    base_cps = func(base_time)

    return base_cps


def plot_auc(table_data, peaks_coords: dict, **kwargs):
    figsize = kwargs.get('figsize', (4, 3))
    yscale = kwargs.get('yscale', 'log')
    save_fig = kwargs.get('save_fig', False)
    return_fig = kwargs.get('return_fig', False)

    fig, ax = plt.subplots(figsize=figsize)
    table_data.plot("time", "cps", ax=ax, style={"cps": "k-"}, linewidth= 0.5, legend=False)
    peak_colors = ["blue", "red", "green"]

    for i, (key, ars_dict) in enumerate(peaks_coords.items()):
        ax.plot(ars_dict["x"], ars_dict["y_base"], linewidth=0.75)
        ax.fill_between(ars_dict["x"], ars_dict["y"], ars_dict["y_base"], alpha=0.3, color=peak_colors[i])
        pk_time = table_data["time"][ars_dict["peak_id"]]
        pk_cps = table_data["cps"][ars_dict["peak_id"]]
        pk_area = ars_dict["peak_area"]
        ax.text(pk_time + 12, pk_cps * .9,
                f"{int(pk_time)}: {pk_area:.2E}",
                ha="left")

    for kwarg, val in kwargs.items():
        if "title" in kwarg:
            ax.set_title(val)

    ax.set_ylabel("cps")

    ax.set_yscale(yscale)

    formatter = ticker.LogFormatterSciNotation(labelOnlyBase=True)
    formatter.labelOnlyBase = {'style': 'sci', 'useMathText': True}
    ax.yaxis.set_major_formatter(formatter)

    minor_x_locator = ticker.FixedLocator(range(25, 350, 25))
    ax.xaxis.set_minor_locator(minor_x_locator)



    for k,v in kwargs.items():
        print(f"{k}: {v}")

    if return_fig:
        return fig, ax
    if save_fig:
        plt.savefig("auc_out.png", bbox_inches="tight", dpi=300)
        plt.show()
    else:
        plt.show()


def calc_concentrations(concentrations: pd.Series, std_areas: pd.Series, unknown_areas: pd.Series):
    X = concentrations.values
    Y = std_areas.values.reshape(-1, 1)
    model = LinearRegression()
    model.fit(Y, X)
    predicted = model.predict(unknown_areas.values.reshape(-1, 1))
    return np.round(predicted, 1)


def clean_icpms_data(input_file: str|Path, rows=5) -> pd.DataFrame:
    # create a dataframe from the tsv file, must change based on file
    df1 = pd.read_csv(input_file, sep="\t", header=None, skiprows=list(range(0, rows)))
    df1 = df1.iloc[:, 0:2].dropna(axis=1)

    df1.columns = ["time", "cps"]
    return df1


def nearest(values, query):
    nearest_idx = np.argmin(np.abs(values - query))
    return nearest_idx


if __name__ == "__main__":
    infile = Path("100_uM_As3As5_rep2.TXT")
    sample_df = pd.read_csv(infile, sep="\t", skiprows=5, header=None).dropna(axis=1)
    sample_df.columns = ["time", "cps"]
    ##%
    wlen = 40
    sample_auc = get_auc(sample_df, infile.name, prominence=0.9995, plot=True, wlen=wlen, return_fig=True)

    print(sample_auc)

    ##%
    fig, ax = sample_auc.get('fig'), sample_auc.get('ax')
    plt.show()

    #%%
    # Make a standard curve and calcualte concentrations
    arsenic_std_data = pd.DataFrame({'concentration': [10, 100, 1000],
                                     "area": [7080000, 68400000, 713000000]})
    unknowns = pd.Series([7080000, 68400000, 713000000], name="unknowns")

    print(calc_concentrations(arsenic_std_data['concentration'], arsenic_std_data['area'], unknowns))
