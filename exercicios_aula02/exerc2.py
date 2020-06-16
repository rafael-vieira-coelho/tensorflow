import tensorflow as tf
import keras.datasets.mnist as mnist

'''
Exercício 2:

Vamos agora dar uma olhada nas camadas do seu modelo. 
Execute o modelo com 512 neurônios na camada oculta. 

a) Como se comportaram os resultados para erro, tempo de treinamento, etc.? 

b) Por que você acha que os valores de (a) se comportaram assim?

c) Aumente para 1024 neurônios. O que ocorre?

opção 1) O treinamento demora mais e aumenta a acurácia.
opção 2) O treinamento demora mais, mas não melhora a acurácia.
opção 3) O treinamento leva o mesmo tempo, mas aumenta a acurácia.

RESPOSTA:
A opção correta é a 1. Adicionando mais neurônios, precisamos fazer mais cálculos, o que torna
o processo mais lento. Mas neste caso tem um bom impacto pois se torna mais preciso. Mas nem
sempre se tem este comportamento.
'''

def main():
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
    training_images = training_images / 255.0
    test_images = test_images / 255.0
    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                        tf.keras.layers.Dense(512, activation=tf.nn.relu),
                                        tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy')
    model.fit(training_images, training_labels, epochs=5)
    model.evaluate(test_images, test_labels)
    classifications = model.predict(test_images)
    print(classifications[0])
    print(test_labels[0])


if __name__ == "__main__":
    main()
