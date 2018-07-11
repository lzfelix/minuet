from minuet import Minuet

MODEL_PATH = '../models/small_ner/'

if __name__ == '__main__':
    model = Minuet.load(MODEL_PATH)
    print('NER model loaded. Type exit to leave.')
    while True:
        sentence = input('> ')
        if sentence == 'exit':
            break

        sentence = sentence.split()
        model_in = [sentence]

        labels = model.decode_predictions(model.predict(model_in))[0]
        for word, label in zip(sentence, labels):
            print(f'{word}/{label} ', end='')
        print()
