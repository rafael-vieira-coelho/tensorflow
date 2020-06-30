from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def main():
    sentences = [
        'i love my dog',
        'I, love my cat',
        'You love my dog!',
        "Do you think my dog is amazing?"
    ]

    tokenizer = Tokenizer(num_words=100, oov_token="OOV")
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index
    print(word_index)

    sequences = tokenizer.texts_to_sequences(sentences)
    print(sequences)

    padded = pad_sequences(sequences, padding='post', truncating='post', maxlen=5)
    print(padded)


if __name__ == '__main__':
    main()
