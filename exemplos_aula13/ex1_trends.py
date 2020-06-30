""" Trend and Seasonality """
# pip install pytest
# pip install statsmodels

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import autocorrelation_plot


def main():
    """ Let's create a time series that just trends upward: """
    time = np.arange(4 * 365 + 1)
    baseline = 10
    series = trend(time, 0.1)
    plt.figure(figsize=(10, 6))
    plot_series(time, series)
    plt.show()

    """Now let's generate a time series with a seasonal pattern:"""
    baseline = 10
    amplitude = 40
    series = seasonality(time, period=365, amplitude=amplitude)
    plt.figure(figsize=(10, 6))
    plot_series(time, series)
    plt.show()

    """Now let's create a time series with both trend and seasonality:"""
    slope = 0.05
    series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
    plt.figure(figsize=(10, 6))
    plot_series(time, series)
    plt.show()

    """# Noise
    In practice few real-life time series have such a smooth signal. They usually have some noise, and the signal-to-noise ratio can sometimes be very low. Let's generate some white noise:
    """
    noise_level = 5
    noise = white_noise(time, noise_level, seed=42)
    plt.figure(figsize=(10, 6))
    plot_series(time, noise)
    plt.show()

    """Now let's add this white noise to the time series:"""
    series += noise
    plt.figure(figsize=(10, 6))
    plot_series(time, series)
    plt.show()

    """All right, this looks realistic enough for now. Let's try to forecast it. We will split it into two periods: 
    the training period and the validation period (in many cases, you would also want to have a test period). The 
    split will be at time step 1000. """
    split_time = 1000
    time_train = time[:split_time]
    x_train = series[:split_time]
    time_valid = time[split_time:]
    x_valid = series[split_time:]
    series = autocorrelation2(time, 10, seed=42)
    plot_series(time[:200], series[:200])
    plt.show()
    series = autocorrelation2(time, 10, seed=42) + trend(time, 2)
    plot_series(time[:200], series[:200])
    plt.show()
    series = autocorrelation1(time, 10, seed=42) + seasonality(time, period=50, amplitude=150) + trend(time, 2)
    plot_series(time[:200], series[:200])
    plt.show()
    series = autocorrelation1(time, 10, seed=42) + seasonality(time, period=50, amplitude=150) + trend(time, 2)
    series2 = autocorrelation1(time, 5, seed=42) + seasonality(time, period=50, amplitude=2) + trend(time, -1) + 550
    series[200:] = series2[200:]
    # series += noise(time, 30)
    plot_series(time[:300], series[:300])
    plt.show()
    series = impulses(time, 10, seed=42)
    plot_series(time, series)
    plt.show()
    signal = impulses(time, 10, seed=42)
    series = autocorrelation3(signal, {1: 0.99})
    plot_series(time, series)
    plt.plot(time, signal, "k-")
    plt.show()
    signal = impulses(time, 10, seed=42)
    series = autocorrelation3(signal, {1: 0.70, 50: 0.2})
    plot_series(time, series)
    plt.plot(time, signal, "k-")
    plt.show()
    series_diff1 = series[1:] - series[:-1]
    plot_series(time[1:], series_diff1)
    autocorrelation_plot(series)
    model = ARIMA(series, order=(5, 1, 0))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())


def autocorrelation1(time, amplitude, seed=None):
    rnd = np.random.RandomState(seed)
    φ1 = 0.5
    φ2 = -0.1
    ar = rnd.randn(len(time) + 50)
    ar[:50] = 100
    for step in range(50, len(time) + 50):
        ar[step] += φ1 * ar[step - 50]
        ar[step] += φ2 * ar[step - 33]
    return ar[50:] * amplitude


def autocorrelation2(time, amplitude, seed=None):
    rnd = np.random.RandomState(seed)
    φ = 0.8
    ar = rnd.randn(len(time) + 1)
    for step in range(1, len(time) + 1):
        ar[step] += φ * ar[step - 1]
    return ar[1:] * amplitude


def autocorrelation3(source, φs):
    ar = source.copy()
    max_lag = len(φs)
    for step, value in enumerate(source):
        for lag, φ in φs.items():
            if step - lag > 0:
                ar[step] += φ * ar[step - lag]
    return ar


def plot_series(time, series, format="-", start=0, end=None, label=None):
    plt.plot(time[start:end], series[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    if label:
        plt.legend(fontsize=14)
    plt.grid(True)


def impulses(time, num_impulses, amplitude=1, seed=None):
    rnd = np.random.RandomState(seed)
    impulse_indices = rnd.randint(len(time), size=10)
    series = np.zeros(len(time))
    for index in impulse_indices:
        series[index] += rnd.rand() * amplitude
    return series


def trend(time, slope=0):
    return slope * time


def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))


def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)


def white_noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


if __name__ == '__main__':
    main()
