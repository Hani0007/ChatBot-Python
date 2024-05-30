import random
import pickle
import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

# Download the 'punkt' and 'wordnet' tokenizer models
nltk.download('punkt')
nltk.download('wordnet')

# Load intents from JSON file
with open('intents.json') as file:
    intents = json.load(file)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Initialize lists for words, classes, and documents
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.']

# Process intents data
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word in the pattern
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # Add the pattern along with its tag to documents list
        documents.append((word_list, intent['tags']))
        # Add the tag to classes if it's not already there
        if intent['tags'] not in classes:
            classes.append(intent['tags'])

# Lemmatize and normalize words
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

# Save words and classes to pickle files
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Initialize training data
training = []
output_empty = [0] * len(classes)

# Create bag of words for each document
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1 if word in word_patterns else 0)
    # Create one-hot encoded output row
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# Shuffle training data
random.shuffle(training)

# Separate bag of words vectors and one-hot encoded output rows
train_x = np.array([i[0] for i in training])
train_y = np.array([i[1] for i in training])

# Build and compile the model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation='softmax'))

# Compile the model with the learning rate specified
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Save the trained model
model.save('chatbot_model.h5')

print("Training is Done ")
