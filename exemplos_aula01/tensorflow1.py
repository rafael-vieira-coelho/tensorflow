import tensorflow as tf

# Testando se a instalação do Tensorflow foi bem sucedida.

def main():
	msg = tf.constant('Hello, TensorFlow!')
	tf.print(msg)

if __name__ == '__main__':
    main()
