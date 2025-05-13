# data_loader.py

import json
import pickle
import numpy as np


def load_conversations(file_path):
    """
    JSON dosyasından konuşma verilerini yükler
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            conversations = json.load(file)
        return conversations
    except FileNotFoundError:
        print(f"Hata: {file_path} dosyası bulunamadı.")
        return []
    except json.JSONDecodeError:
        print(f"Hata: {file_path} dosyası geçerli bir JSON formatında değil.")
        return []


def save_processed_data(data, file_path):
    """
    İşlenmiş verileri pickle formatında kaydeder
    """
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)


def load_processed_data(file_path):
    """
    İşlenmiş verileri pickle dosyasından yükler
    """
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data
    except FileNotFoundError:
        print(f"Hata: {file_path} dosyası bulunamadı.")
        return None
    except Exception as e:
        print(f"Veri yükleme hatası: {e}")
        return None