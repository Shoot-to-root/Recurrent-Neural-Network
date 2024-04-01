import numpy as np
import io
import tensorflow as tf
import sys
import os
import matplotlib.pyplot as plt

train_data_URL = 'shakespeare_train.txt'
valid_data_URL = 'shakespeare_valid.txt'

# read data
with io.open(train_data_URL, 'r', encoding='utf8') as f:
    train_text = f.read()

with io.open(valid_data_URL, 'r', encoding='utf8') as f:
    test_text = f.read()
    
# Characters' collection
vocab = sorted(list(set(train_text)))

# Construct character dictionary
vocab_to_int = {c:i for i, c in enumerate (vocab)}
int_to_vocab = dict(enumerate(vocab))

# Encode data, shape = [number of characters]
train = np.array([vocab_to_int [c] for c in train_text], dtype=np.int32)
test = np.array([vocab_to_int [c] for c in test_text], dtype=np.int32)
#print(train[:100])

maxlen = 40
step = 3
x = []
y = []

def build_dataset(text):
    for i in range(0, len(text) - maxlen, step):
        x.append(text[i: i + maxlen])
        y.append(text[i + maxlen])
        
    return x, y

train_x, train_y = build_dataset(train_text)
test_x, test_y = build_dataset(test_text)

# vectorize input
def vectorization(x_, y_):    
    x_ = np.zeros((len(x), maxlen, len(vocab)), dtype=bool)
    y_ = np.zeros((len(x), len(vocab)), dtype=bool)

    for i, sentence in enumerate(x):
        for t, char in enumerate(sentence):
            x_[i, t, vocab_to_int[char]] = 1
            y_[i, vocab_to_int[y[i]]] = 1
    
    return x_, y_

train_x, train_y = vectorization(train_x, train_y)
test_x, test_y = vectorization(test_x, test_y)
   
# build model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.SimpleRNN(256, input_shape=(maxlen, len(vocab))))
model.add(tf.keras.layers.Dense(len(vocab), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.01))
#print(model.summary())

def predict(preds, diversity=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / diversity
    exp_preds = np.exp(preds)
    preds = exp_preds/np.sum(exp_preds)
    probs = np.random.multinomial(1, preds, 1)
    return np.argmax(probs)

# create checkpoint
checkpoint_path = "checkpoints/cp_{epoch}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True)

batch_size = 512
epochs = 200

for e in range(epochs):
    if(e%50 == 0):
        print('Epoch', e+1)
        
        hist = model.fit(train_x, train_y, batch_size=batch_size, epochs=1, validation_data=(test_x, test_y), callbacks=[cp_callback])
        
        start_index = np.random.randint(0, len(train_text)-maxlen-1)
        
        for diversity in [0.2]:
            generated = ''
            sentence = train_text[start_index: start_index + maxlen]
            generated += sentence
            print('------Seed: "' + sentence + '"')
            sys.stdout.write(generated)
            
            for i in range(500):
                x = np.zeros((1, maxlen, len(vocab)))
                
                for t, char in enumerate(sentence):
                    x[0, t, vocab_to_int[char]] = 1.
                    
                preds = model.predict(x, verbose=0)[0]
                next_index = predict(preds, diversity)
                pred_char = int_to_vocab[next_index]
                generated += pred_char
                sentence = sentence[1:] + pred_char
                sys.stdout.write(pred_char)
                sys.stdout.flush()
            print()
'''
def generate(model, start_string, num_generate = 1000, diversity=1.0):
    vectorized_start_string = [vocab_to_int[s] for s in start_string]
    vectorized_start_string = tf.expand_dims(vectorized_start_string, 0)

    text_generated = []

    # clean slates
    model.reset_states()
    for char_index in range(num_generate):
        preds = model(vectorized_start_string)
        preds = tf.squeeze(preds, 0) # remove batch dimension

        # predict
        preds = preds / diversity
        predicted = tf.random.categorical(preds, num_samples=1)[-1,0].numpy()

        # pass the predicted character as the next input to the model along with the previous hidden state
        vectorized_start_string = tf.expand_dims([predicted], 0)

        text_generated.append(int_to_vocab[predicted])

    return (start_string + ''.join(text_generated))

print(generate(model, start_string=u"JULIET: "))

#model.save('rnn.h5', save_format='h5')

# plot
loss = hist.history['loss']
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.plot(loss, label='train')
plt.legend()
plt.show()

train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.title('Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(train_loss, label='train_loss')
#plt.plot(val_loss, label='valid_loss')
plt.legend()
plt.grid(linestyle='--', linewidth=1, alpha=0.5)
plt.show()

plt.title('Error_rate')
plt.plot(train_acc, label='train_error')
plt.plot(val_acc, label = 'val_error')
plt.xlabel('epoch')
plt.ylabel('Error_rate')
plt.legend()
plt.grid(linestyle='--', linewidth=1, alpha=0.5)
plt.show()
'''

