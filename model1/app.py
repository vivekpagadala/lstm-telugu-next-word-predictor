from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
import numpy as np
import random
import sys
import os

# Define Save Directory
SAVE_DIR = "saved_weights"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# LOAD TEXT
# Save Notepad file as UTF-8
filename = "files/agnigundam.txt"
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()
print(raw_text[:1000])

# CLEAN TEXT - Remove numbers
raw_text = ''.join(c for c in raw_text if not c.isdigit())

# Create character mapping dictionaries
chars = sorted(list(set(raw_text)))  # Unique characters
char_to_int = {c: i for i, c in enumerate(chars)}
int_to_char = {i: c for i, c in enumerate(chars)}

# Summary of data
n_chars = len(raw_text)
n_vocab = len(chars)
print(f"Total Characters in the text: {n_chars}")
print(f"Total Unique Characters (Vocabulary Size): {n_vocab}")

# CREATE INPUT/OUTPUT SEQUENCES FOR TRAINING
seq_length = 60  # Length of each input sequence
step = 10  # Instead of moving 1 letter at a time, try skipping a few
sentences = []  # X values (Sequences)
next_chars = []  # Y values (Next character to predict)

for i in range(0, n_chars - seq_length, step):
    sentences.append(raw_text[i: i + seq_length])
    next_chars.append(raw_text[i + seq_length])

n_patterns = len(sentences)
print(f"Number of Sequences: {n_patterns}")

# VECTORIZE INPUT AND OUTPUT
x = np.zeros((len(sentences), seq_length, n_vocab), dtype=np.bool_)
y = np.zeros((len(sentences), n_vocab), dtype=np.bool_)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_to_int[char]] = 1
    y[i, char_to_int[next_chars[i]]] = 1

print("Input Shape (x):", x.shape)
print("Output Shape (y):", y.shape)

# BUILD THE LSTM MODEL
model = Sequential()
model.add(LSTM(128, input_shape=(seq_length, n_vocab), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(n_vocab, activation='softmax'))

# COMPILE THE MODEL
optimizer = RMSprop(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
model.summary()

# DEFINE MODEL CHECKPOINT
filepath = os.path.join(SAVE_DIR, "saved_weights-{epoch:02d}-{loss:.4f}.keras")
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# TRAIN THE MODEL
history = model.fit(x, y, batch_size=128, epochs=50, callbacks=callbacks_list)

# SAVE FINAL MODEL WEIGHTS
model.save_weights(os.path.join(SAVE_DIR, "weights.weights.h5"))  # Fixed filename issue

# SAVE COMPLETE MODEL
model.save(os.path.join(SAVE_DIR, "my_saved_weights_telugu_50epochs.h5"))

# PLOT TRAINING LOSS
import matplotlib.pyplot as plt

loss = history.history['loss']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'y', label='Training loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# TEXT GENERATION FUNCTION
def sample(preds):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# LOAD TRAINED WEIGHTS
model.load_weights(os.path.join(SAVE_DIR, "my_saved_weights_telugu_50epochs.h5"))

# PICK RANDOM SEED SENTENCE FROM TEXT
start_index = random.randint(0, n_chars - seq_length - 1)
generated = ''
sentence = raw_text[start_index: start_index + seq_length]
generated += sentence

print(f"----- Seed for Text Generation: \"{sentence}\"")
sys.stdout.write(generated)

# GENERATE CHARACTERS
for i in range(400):  # Generate 400 characters
    x_pred = np.zeros((1, seq_length, n_vocab))
    
    for t, char in enumerate(sentence):
        x_pred[0, t, char_to_int[char]] = 1.

    preds = model.predict(x_pred, verbose=0)[0]
    next_index = sample(preds)
    next_char = int_to_char[next_index]

    generated += next_char
    sentence = sentence[1:] + next_char

    sys.stdout.write(next_char)
    sys.stdout.flush()

print()