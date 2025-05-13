#!/usr/bin/env python
# -*- coding: utf-8 -*-

# train.py

import os
import pickle
import numpy as np
from config import CONFIG
from utils.data_loader import load_conversations, save_processed_data
from utils.preprocessor import TextPreprocessor
from utils.tokenizer import SimpleTokenizer
from models.neural_network import SequenceToSequenceModel


def preprocess_data():
    """
    Konuşma verilerini yükler, işler ve eğitim için hazırlar
    """
    print("Veri ön işleme başlatılıyor...")
    
    # Konuşma verilerini yükle
    conversations = load_conversations(CONFIG['data']['conversations_path'])
    
    if not conversations:
        raise ValueError("Konuşma verileri yüklenemedi veya boş.")
    
    # Giriş ve çıkış metinlerini ayır
    inputs = [conv["input"] for conv in conversations]
    responses = [conv["response"] for conv in conversations]
    
    print(f"Toplam {len(inputs)} konuşma çifti yüklendi.")
    
    # Tokenizer oluştur ve metinlere uygula
    tokenizer = SimpleTokenizer()
    tokenizer.fit_on_texts(inputs + responses)
    
    print(f"Kelime dağarcığı boyutu: {len(tokenizer.word_index)}")
    
    # Metinleri dizilere dönüştür
    input_sequences = tokenizer.texts_to_sequences(inputs)
    response_sequences = tokenizer.texts_to_sequences(responses)
    
    # Dizileri doldur
    max_length = CONFIG['data']['max_sequence_length']
    padded_inputs = tokenizer.pad_sequences(input_sequences, maxlen=max_length)
    padded_responses = tokenizer.pad_sequences(response_sequences, maxlen=max_length)
    
    # One-hot kodlama için kelime dağarcığı boyutu
    vocab_size = len(tokenizer.word_index) + 1  # +1 çünkü indeks 0'dan başlıyor
    
    # Dizileri one-hot kodlanmış matrise dönüştür
    def to_one_hot(sequences, vocab_size):
        results = np.zeros((len(sequences), sequences.shape[1], vocab_size), dtype=np.float32)
        for i, sequence in enumerate(sequences):
            for j, idx in enumerate(sequence):
                if idx > 0:  # 0 padding değeri
                    results[i, j, idx] = 1.0
        return results
    
    # Eğitim ve test setlerine ayır
    test_size = int(len(padded_inputs) * CONFIG['data']['test_split'])
    train_size = len(padded_inputs) - test_size
    
    # Numpy formatına dönüştür
    x_train = padded_inputs[:train_size].reshape(train_size, -1)
    y_train = to_one_hot(padded_responses[:train_size], vocab_size).reshape(train_size, -1)
    
    x_test = padded_inputs[train_size:].reshape(test_size, -1)
    y_test = to_one_hot(padded_responses[train_size:], vocab_size).reshape(test_size, -1)
    
    print(f"Eğitim seti boyutu: {len(x_train)}, Test seti boyutu: {len(x_test)}")
    
    # İşlenmiş veriyi kaydet
    processed_data = {
        'x_train': x_train,
        'y_train': y_train,
        'x_test': x_test,
        'y_test': y_test,
        'tokenizer': tokenizer,
        'vocab_size': vocab_size,
        'max_length': max_length
    }
    
    # Veri klasörünü oluştur (eğer yoksa)
    os.makedirs(os.path.dirname(CONFIG['data']['processed_data_path']), exist_ok=True)
    
    save_processed_data(processed_data, CONFIG['data']['processed_data_path'])
    print(f"İşlenmiş veri kaydedildi: {CONFIG['data']['processed_data_path']}")
    
    return processed_data


def train_model(processed_data):
    """
    Seq2Seq modelini eğitir
    """
    print("Model eğitimi başlatılıyor...")
    
    x_train = processed_data['x_train']
    y_train = processed_data['y_train']
    vocab_size = processed_data['vocab_size']
    
    # Model boyutlarını ayarla
    input_dim = x_train.shape[1]
    hidden_dim = CONFIG['model']['hidden_dim']
    output_dim = y_train.shape[1]
    
    print(f"Model boyutları: input_dim={input_dim}, hidden_dim={hidden_dim}, output_dim={output_dim}")
    
    # Seq2Seq modelini oluştur
    model = SequenceToSequenceModel(input_dim, hidden_dim, output_dim)
    
    # Modeli eğit
    history = model.fit(
        x_train,
        y_train,
        epochs=CONFIG['training']['epochs'],
        learning_rate=CONFIG['training']['learning_rate'],
        batch_size=CONFIG['training']['batch_size'],
        verbose=True
    )
    
    # Model dosya yolunu oluştur
    model_dir = os.path.dirname(CONFIG['model']['saved_model_path'])
    os.makedirs(model_dir, exist_ok=True)
    
    # Modeli kaydet
    model.save(CONFIG['model']['saved_model_path'])
    print(f"Model kaydedildi: {CONFIG['model']['saved_model_path']}")
    
    return model, history


def evaluate_model(model, processed_data):
    """
    Modeli test seti üzerinde değerlendirir
    """
    print("Model değerlendiriliyor...")
    
    x_test = processed_data['x_test']
    y_test = processed_data['y_test']
    tokenizer = processed_data['tokenizer']
    vocab_size = processed_data['vocab_size']
    
    # Test örnekleri üzerinde tahmin yap
    predictions = model.predict(x_test)
    
    # Tahminleri one-hot'tan indekslere dönüştür
    pred_indices = np.argmax(predictions.reshape(-1, vocab_size), axis=1)
    
    # Test örneklerini göster
    print("\nÖrnek Tahminler:")
    for i in range(min(5, len(x_test))):
        # Girişi diziye çevir
        input_indices = x_test[i]
        input_indices = input_indices[input_indices != 0]  # Padding'i kaldır
        input_text = tokenizer.sequences_to_texts([input_indices.astype(np.int32)])[0]
        
        # Gerçek çıkışı diziye çevir
        true_indices = np.argmax(y_test[i].reshape(-1, vocab_size), axis=1)
        true_indices = true_indices[true_indices != 0]  # Padding'i kaldır
        true_text = tokenizer.sequences_to_texts([true_indices.astype(np.int32)])[0]
        
        # Tahmini çıkışı diziye çevir
        pred_idx = pred_indices[i * vocab_size:(i + 1) * vocab_size]
        pred_idx = pred_idx[pred_idx != 0]  # Padding'i kaldır
        pred_text = tokenizer.sequences_to_texts([pred_idx.astype(np.int32)])[0]
        
        print(f"Giriş: {input_text}")
        print(f"Gerçek: {true_text}")
        print(f"Tahmin: {pred_text}")
        print("-" * 50)


def main():
    """
    Ana eğitim işlemi
    """
    print("Derin Öğrenme Chatbot Eğitimi Başlatılıyor")
    print("=" * 50)
    
    # Veriyi ön işle
    processed_data = preprocess_data()
    
    # Modeli eğit
    model, history = train_model(processed_data)
    
    # Modeli değerlendir
    evaluate_model(model, processed_data)
    
    print("Eğitim tamamlandı!")


if __name__ == "__main__":
    main()