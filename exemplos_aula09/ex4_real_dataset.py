import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def main():
    filename = 'sarcasm.json'

    with open(filename, 'r') as f:
        data = json.load(f)

    sentences = []
    labels = []
    urls = []
    for item in data:
        sentences.append(item['headline'])
        labels.append(item['is_sarcastic'])
        urls.append(item['article_link'])

    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(sentences)

    word_index = tokenizer.word_index
    for word, index in word_index.items():
        print(str(index) + ': ' + word)
    print('\n We have ' + str(len(word_index)) + " tokens.")
    sequences = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sequences, padding='post')
    print(sequences[0])
    print(sentences[0])
    print(padded[0])
    print(padded.shape)


if __name__ == '__main__':
    main()
