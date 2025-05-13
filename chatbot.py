#!/usr/bin/env python
# -*- coding: utf-8 -*-

# chatbot.py

import os
import numpy as np
from config import CONFIG
from utils.data_loader import load_processed_data
from models.neural_network import SequenceToSequenceModel


def load_model_and_data():
    """
    Eğitilmiş modeli ve işlenmiş verileri yükler
    """
    # İşlenmiş verileri yükle
    processed_data = load_processed_data(CONFIG['data']['processed_data_path'])
    if not processed_data:
        raise ValueError(f"İşlenmiş veri bulunamadı: {CONFIG['data']['processed_data_path']}")
    
    # Modeli yükle
    model = SequenceToSequenceModel.load(CONFIG['model']['saved_model_path'])
    print("Model ve işlenmiş veriler yüklendi.")
    
    return model, processed_data


def predict_response(model, processed_data, input_text):
    """
    Girilen metne chatbot yanıtını tahmin eder
    """
    tokenizer = processed_data['tokenizer']
    vocab_size = processed_data['vocab_size']
    max_length = processed_data['max_length']
    
    # Giriş metnini tokenize et ve doldur
    input_sequence = tokenizer.texts_to_sequences([input_text])
    padded_input = tokenizer.pad_sequences(input_sequence, maxlen=max_length)
    
    # Tek bir örnek olduğunu belirt
    x_input = padded_input.reshape(1, -1)
    
    # Modelle tahmin yap
    prediction = model.predict(x_input)
    
    # Tahmini tokenlarına dönüştür
    pred_indices = np.argmax(prediction.reshape(-1, vocab_size), axis=1)
    pred_indices = pred_indices[pred_indices != 0]  # Padding'i kaldır
    
    # Tokenleri metne dönüştür
    predicted_text = tokenizer.sequences_to_texts([pred_indices.astype(np.int32)])[0]
    
    return predicted_text


def chat():
    """
    Kullanıcı ile chatbot arasında interaktif sohbet başlatır
    """
    print("Derin Öğrenme Chatbot'a Hoş Geldiniz!")
    print("Çıkmak için 'q' veya 'exit' yazabilirsiniz.")
    print("=" * 50)
    
    try:
        # Model ve verileri yükle
        model, processed_data = load_model_and_data()
        
        while True:
            # Kullanıcı girdisini al
            user_input = input("Siz: ")
            
            # Çıkış kontrolü
            if user_input.lower() in ['q', 'exit', 'quit']:
                print("Görüşmek üzere!")
                break
            
            # Chatbot yanıtını tahmin et
            try:
                response = predict_response(model, processed_data, user_input)
                print(f"Chatbot: {response}")
            except Exception as e:
                print(f"Yanıt üretilirken bir hata oluştu: {e}")
                print("Lütfen tekrar deneyin.")
                
    except Exception as e:
        print(f"Bir hata oluştu: {e}")
        print("Lütfen modelin eğitildiğinden ve veri dosyalarının mevcut olduğundan emin olun.")


def sample_conversations(count=5):
    """
    Örnek konuşmalar gösterir
    """
    print(f"{count} Örnek Konuşma Gösteriliyor:")
    print("=" * 50)
    
    try:
        # Model ve verileri yükle
        model, processed_data = load_model_and_data()
        
        # Örnek girdiler
        sample_inputs = [
            "Merhaba, nasılsın?",
            "Derin öğrenme nedir?",
            "Yapay zekada son gelişmeler nelerdir?",
            "Python programlama dili hakkında bilgi verir misin?",
            "Kendini tanıtır mısın?"
        ]
        
        # Her örnek girdi için yanıt üret
        for i, input_text in enumerate(sample_inputs[:count], 1):
            response = predict_response(model, processed_data, input_text)
            print(f"Örnek {i}:")
            print(f"Girdi: {input_text}")
            print(f"Yanıt: {response}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Bir hata oluştu: {e}")
        print("Lütfen modelin eğitildiğinden ve veri dosyalarının mevcut olduğundan emin olun.")


def main():
    """
    Ana program
    """
    print("Derin Öğrenme Chatbot Başlatılıyor")
    print("=" * 50)
    
    # Komut satırı argümanlarını işle
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "--samples":
            count = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            sample_conversations(count)
        else:
            print("Geçersiz argüman. Kullanım: python chatbot.py [--samples [count]]")
    else:
        # İnteraktif sohbet başlat
        chat()


if __name__ == "__main__":
    main()