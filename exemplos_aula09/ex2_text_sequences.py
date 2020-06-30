from tensorflow.keras.preprocessing.text import Tokenizer


def main():
    sentences = [
        'i love my dog',
        'I, love my cat',
        'You love my dog!',
        "Do you think my dog is amazing?"
    ]

    tokenizer = Tokenizer(num_words=100)
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index
    print(word_index)

    sequences = tokenizer.texts_to_sequences((sentences))
    print(sequences)


if __name__ == '__main__':
    main()
