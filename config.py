# config.py

# Proje yapılandırma ayarları
CONFIG = {
    # Veri ayarları
    'data': {
        'conversations_path': './data/conversations.json',
        'processed_data_path': './data/preprocessed_data.pkl',
        'max_sequence_length': 20,
        'min_word_count': 2,
        'test_split': 0.2,
    },
    
    # Model ayarları
    'model': {
        'hidden_dim': 128,
        'embedding_dim': 64,
        'dropout_rate': 0.2,
        'saved_model_path': './models/saved_model',
    },
    
    # Eğitim ayarları
    'training': {
        'batch_size': 32,
        'epochs': 100,
        'learning_rate': 0.001,
        'early_stopping_patience': 5,
    },
    
    # Chatbot ayarları
    'chatbot': {
        'max_response_length': 50,
        'temperature': 0.7,
    }
}