import os
import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Fungsi untuk membuat tokenizer
def create_tokenizer(sequences):
    tokenizer = Tokenizer(char_level=True)  # Karakter DNA
    tokenizer.fit_on_texts(sequences)
    return tokenizer

# Fungsi untuk memuat data DNA
def load_dna_sequences(file_path):
    try:
        with open(file_path, "r") as file:
            sequences = [line.strip() for line in file if line.strip()]
        return sequences
    except FileNotFoundError:
        print(f"File tidak ditemukan: {file_path}")
        return []

# Fungsi preprocessing data
def preprocess_data(sequences, tokenizer, max_length):
    encoded = tokenizer.texts_to_sequences(sequences)
    padded = pad_sequences(encoded, maxlen=max_length, padding="post", truncating="post")
    return padded

# Fungsi untuk membuat model LSTM
def build_lstm_model(input_length, vocab_size):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=input_length))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(32))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # Load dataset
    healthy_sequences = load_dna_sequences("d:/CODE/Projects/Salmon_Sehat.txt")
    sick_sequences = load_dna_sequences("d:/CODE/Projects/Salmon_Sakit.txt")

    if not healthy_sequences or not sick_sequences:
        print("Dataset kosong atau tidak valid.")
        exit()

    # Gabungkan semua data
    all_sequences = healthy_sequences + sick_sequences
    labels = [1] * len(healthy_sequences) + [0] * len(sick_sequences)

    # Buat tokenizer
    tokenizer = create_tokenizer(all_sequences)
    vocab_size = len(tokenizer.word_index) + 1
    max_length = 100

    # Simpan tokenizer
    with open("tokenizer.pkl", "wb") as handle:
        pickle.dump(tokenizer, handle)
    print("Tokenizer berhasil disimpan.")

    # Preprocessing data
    X = preprocess_data(all_sequences, tokenizer, max_length)
    y = np.array(labels)

    # Split dataset menjadi train dan test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Buat model LSTM
    lstm_model = build_lstm_model(input_length=max_length, vocab_size=vocab_size)
    lstm_model.summary()

    # Latih model
    lstm_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

    # Simpan model LSTM
    lstm_model.save("model_lstm.h5")
    print("Model LSTM berhasil disimpan.")
