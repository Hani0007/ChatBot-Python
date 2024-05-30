# import random
# import json
# import pickle
# import numpy as np
#
# import nltk
# from nltk.stem import WordNetLemmatizer
# from tensorflow.keras.models import load_model
#
# # Initialize lemmatizer
# lemmatizer = WordNetLemmatizer()
#
# # Load data
# with open('intents.json') as file:
#     intents = json.load(file)
#
# words = pickle.load(open('words.pkl', 'rb'))
# classes = pickle.load(open('classes.pkl', 'rb'))
# model = load_model('chatbot_model.h5')
#
# def clean_up_sentence(sentence):
#     sentence_words = nltk.word_tokenize(sentence)
#     sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
#     return sentence_words
#
# def bag_of_words(sentence):
#     sentence_words = clean_up_sentence(sentence)
#     bag = [0] * len(words)
#     for w in sentence_words:
#         for i, word in enumerate(words):
#             if word == w:
#                 bag[i] = 1
#     return np.array(bag)
#
# def predict(sentence):
#     bow = bag_of_words(sentence)
#     result = model.predict(np.array([bow]))[0]
#     ERROR_THRESHOLD = 0.25
#     results = [[i, r] for i, r in enumerate(result) if r > ERROR_THRESHOLD]
#     results.sort(key=lambda x: x[1], reverse=True)
#     return_list = []
#     for r in results:
#         return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
#     return return_list
#
# def get_response(intents_list, intents_json):
#     tag = intents_list[0]['intent']
#     list_of_intents = intents_json['intents']
#     for i in list_of_intents:
#         if i['tags'] == tag:
#             result = random.choice(i['responses'])
#             break
#     return result
#
# print("Go! The Bot is Running")
#
# while True:
#     message = input("")
#     ints = predict(message)
#     result = get_response(ints, intents)
#     print(result)
import sys
import os
import random
import json
import pickle
import numpy as np

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QLabel, QScrollArea, QFrame, QSizePolicy
from PyQt5.QtCore import Qt

lemmatizer = WordNetLemmatizer()

# Load data and model
with open('intents.json') as file:
    intents = json.load(file)
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

# Functions to process the input and predict the response
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    result = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(result) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tags'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# PyQt5 GUI
class ChatBotApp(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('ChatBot')
        self.setGeometry(100, 100, 400, 400)

        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(10, 10, 10, 10)

        self.chat_area = QVBoxLayout()
        self.chat_widget = QWidget()
        self.chat_widget.setLayout(self.chat_area)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.chat_widget)
        self.layout.addWidget(self.scroll_area)

        self.input_area = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Type your message here...")
        self.chat_input.setStyleSheet("""
            QLineEdit {
                border: 2px solid #ccc;
                border-radius: 15px;
                padding: 10px;
                font-size: 16px;
            }
        """)

        self.send_button = QPushButton("Send")
        self.send_button.setStyleSheet("background-color: lightgreen; padding: 10px; border-radius: 15px; font-size: 16px;")
        self.send_button.clicked.connect(self.handle_send)
        self.input_area.addWidget(self.chat_input)
        self.input_area.addWidget(self.send_button)

        self.layout.addLayout(self.input_area)
        self.setLayout(self.layout)

    def handle_send(self):
        user_message = self.chat_input.text()
        if user_message:
            self.add_message("User", user_message)
            self.chat_input.clear()

            ints = predict_class(user_message)
            response = get_response(ints, intents)
            self.add_message("Bot", response)

    def add_message(self, sender, message):
        message_label = QLabel(message)
        message_label.setWordWrap(True)
        message_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        message_frame = QFrame()
        message_layout = QHBoxLayout()

        if sender == "User":
            message_label.setStyleSheet("background-color: lightblue; padding: 10px; border-radius: 10px;")
            message_layout.addStretch(1)
            message_layout.addWidget(message_label, alignment=Qt.AlignRight)
        else:
            message_label.setStyleSheet("background-color: lightgreen; padding: 10px; border-radius: 10px;")
            message_layout.addWidget(message_label, alignment=Qt.AlignLeft)
            message_layout.addStretch(1)

        message_frame.setLayout(message_layout)
        self.chat_area.addWidget(message_frame)
        self.chat_area.addStretch(1)

        self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    chatbot_app = ChatBotApp()
    chatbot_app.show()
    sys.exit(app.exec_())



