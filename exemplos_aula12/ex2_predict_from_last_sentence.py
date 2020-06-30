from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from ex1_predict_from_last_word import create_model


def main():
    # create neural network model
    tokenizer, max_sequence_len, model = create_model()

    # predict next 100 words from sentece "Laurence went to Dublin"
    seed_text = "Laurence went to Dublin"
    next_words = 100
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1)
        predicted = np.argmax(model.predict(token_list), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    print(seed_text)


if __name__ == '__main__':
    main()
