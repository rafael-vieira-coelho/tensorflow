import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv


def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

def main():
    time_step = []
    sunspots = []

    with open('./sunspots.csv') as csvfile:
      reader = csv.reader(csvfile, delimiter=',')
      next(reader)
      for row in reader:
        sunspots.append(float(row[2]))
        time_step.append(int(row[0]))

    series = np.array(sunspots)
    time = np.array(time_step)
    plt.figure(figsize=(10, 6))
    plot_series(time, series)

    split_time = 3000
    time_train = time[:split_time]
    x_train = series[:split_time]
    time_valid = time[split_time:]
    x_valid = series[split_time:]

    window_size = 60
    batch_size = 32
    shuffle_buffer_size = 1000

    def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
      dataset = tf.data.Dataset.from_tensor_slices(series)
      dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
      dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
      dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
      dataset = dataset.batch(batch_size).prefetch(1)
      return dataset

    dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(20, input_shape=[window_size], activation="relu"),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(1)
    ])

    model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=1e-7, momentum=0.9))
    model.fit(dataset,epochs=100,verbose=0)

    forecast=[]
    for time in range(len(series) - window_size):
      forecast.append(model.predict(series[time:time + window_size][np.newaxis]))

    forecast = forecast[split_time-window_size:]
    results = np.array(forecast)[:, 0, 0]

    plt.figure(figsize=(10, 6))

    plot_series(time_valid, x_valid)
    plot_series(time_valid, results)

    print(tf.keras.metrics.mean_absolute_error(x_valid, results).numpy())


if __name__ == '__main__':
    main()
