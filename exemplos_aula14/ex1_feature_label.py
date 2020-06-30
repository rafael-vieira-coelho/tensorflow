import tensorflow as tf


def main():
    data = tf.data.Dataset.range(10)
    for val in data:
        print(val.numpy())

    data = tf.data.Dataset.range(10)
    data = data.window(5, shift=1)
    for window_dataset in data:
        for val in window_dataset:
            print(val.numpy(), end=" ")
        print()

    data = tf.data.Dataset.range(10)
    data = data.window(5, shift=1, drop_remainder=True)
    for window_dataset in data:
        for val in window_dataset:
            print(val.numpy(), end=" ")
        print()

    data = tf.data.Dataset.range(10)
    data = data.window(5, shift=1, drop_remainder=True)
    data = data.flat_map(lambda window: window.batch(5))
    for window in data:
        print(window.numpy())

    data = tf.data.Dataset.range(10)
    data = data.window(5, shift=1, drop_remainder=True)
    data = data.flat_map(lambda window: window.batch(5))
    data = data.map(lambda window: (window[:-1], window[-1:]))
    for x, y in data:
        print(x.numpy(), y.numpy())

    data = tf.data.Dataset.range(10)
    data = data.window(5, shift=1, drop_remainder=True)
    data = data.flat_map(lambda window: window.batch(5))
    data = data.map(lambda window: (window[:-1], window[-1:]))
    data = data.shuffle(buffer_size=10)
    for x, y in data:
        print(x.numpy(), y.numpy())

    data = tf.data.Dataset.range(10)
    data = data.window(5, shift=1, drop_remainder=True)
    data = data.flat_map(lambda window: window.batch(5))
    data = data.map(lambda window: (window[:-1], window[-1:]))
    data = data.shuffle(buffer_size=10)
    data = data.batch(2).prefetch(1)
    for x, y in data:
        print("x = ", x.numpy())
        print("y = ", y.numpy())


if __name__ == '__main__':
    main()
