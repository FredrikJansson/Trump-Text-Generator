import numpy as np
import tensorflow as tf
import os
import sys

def split_input(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


def process_text(filepath):
    text = open(filepath, 'rb').read().decode(encoding='utf-8')
    vocabulary = sorted(set(text))
    chridx = {k: i for i, k in enumerate(vocabulary)}
    idxchr = np.array(vocabulary)
    text_int = np.array([chridx[c] for c in text])
    return text_int, vocabulary, chridx, idxchr


def create_dataset(text_int, seq_l = 100, batch_s = 64, buffer_s = 10000):
    char_dataset = tf.data.Dataset.from_tensor_slices(text_int)
    dataset = char_dataset.batch(seq_l + 1, drop_remainder=True).map(split_input)
    dataset = dataset.shuffle(buffer_s).batch(batch_s, drop_remainder=True)
    return dataset


def build_model(vocabulary_s, embedding_d = 256, rnn_u = 1024, batch_s = 64):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocabulary_s, embedding_d, batch_input_shape=[batch_s, None]),
        tf.keras.layers.LSTM(rnn_u, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LSTM(rnn_u, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(vocabulary_s)
    ])
    return model


def generate_text(model, chridx, idxchr, text_start, generate_number_char = 100, temperature = 1.0):
    input_eval = [chridx[s] for s in text_start]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []
    model.reset_states()
    for i in range(generate_number_char):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions /= temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        input_eval = tf.expand_dims([predicted_id], axis=0)
        text_generated.append(idxchr[predicted_id])
    return text_start + ''.join(text_generated)

if len(sys.argv) > 1:
    if sys.argv[1] in ['-t', '--train']:
        # Training
        filename = 'tweets.txt'

        if not os.path.exists(filename):
            print(f'{filename} does not exist. Exits')
            exit(1)

        text_int, vocabulary, chridx, idxchr = process_text(filename)
        voc_s = len(vocabulary)
        print('Vocabulary size:', voc_s)
        dataset = create_dataset(text_int)
        model = build_model(vocabulary_s=voc_s)
        model.compile(optimizer='adam', loss=loss)
        model.summary()
        history = model.fit(dataset, epochs=50)
        model.save_weights('gen_text_weights.h5', save_format='h5')
        model = build_model(vocabulary_s=voc_s, batch_s=1)
        model.load_weights('gen_text_weights.h5')
        model.summary()

        model_name = 'models/model_' + str(voc_s)
        model.save(model_name)
    elif sys.argv[1] in ['-a', '--ask']:
        filename = 'tweets.txt'
        text_int, vocabulary, chridx, idxchr = process_text(filename)
        query_string = str(sys.argv[1])
        model_name = 'models/model_' + str(len(vocabulary))
        model = tf.keras.models.load_model(model_name)
        
        text_in = 'temp'
        while text_in:
            text_in = str(input('- '))
            if not text_in:
                break
            generated_text = generate_text(model, chridx, idxchr, text_start=text_in, generate_number_char=1000, temperature=int(sys.argv[2]))
            print(generated_text)
            print('\n\n')
    else:
        # Querying
        filename = 'tweets.txt'
        text_int, vocabulary, chridx, idxchr = process_text(filename)
        query_string = str(sys.argv[1])
        model_name = 'models/model_' + str(len(vocabulary))
        model = tf.keras.models.load_model(model_name)
        generated_text = generate_text(model, chridx, idxchr, text_start=query_string, generate_number_char=1000, temperature=int(sys.argv[2]))
        print(generated_text)
else:
    print('You need to provide a parameter.')
    print('\t-t/--train for training the model')
    print('\tAny singular text string for querying the model.')