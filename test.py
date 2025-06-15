import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

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

if __name__ == "__main__":
    # Load tokenizer
    try:
        with open("d:/CODE/Projects/tokenizer.pkl", "rb") as handle:
            tokenizer = pickle.load(handle)
        print("Tokenizer berhasil dimuat.")
    except FileNotFoundError:
        print("Tokenizer tidak ditemukan. Pastikan file tokenizer.pkl tersedia.")
        exit()

    # Load model LSTM
    try:
        lstm_model = load_model('D:\CODE\Projects\model_lstm.h5')
        print("Model LSTM berhasil dimuat.")
    except Exception as e:
        print(f"Error memuat model LSTM: {e}")
        exit()

    # Load data uji
    test_sequences = load_dna_sequences("D:\CODE\Projects\Salmon_Uji.txt")
    if not test_sequences:
        print("Error: File uji kosong atau tidak memiliki urutan DNA yang valid.")
        exit()

    # Preprocessing data uji
    max_length = 100
    X_test = preprocess_data(test_sequences, tokenizer, max_length)

    # Prediksi data uji
    predictions = lstm_model.predict(X_test)
    for i, pred in enumerate(predictions):
        print(f"Sequence {i+1}: Prediksi = {'Sehat' if pred > 0.5 else 'Sakit'}")
