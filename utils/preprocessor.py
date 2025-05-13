# preprocessor.py

import re
import string
import numpy as np


class TextPreprocessor:
    def __init__(self, min_word_count=1):
        self.min_word_count = min_word_count
        self.word_counts = {}
        self.vocabulary = {}
        self.inverse_vocabulary = {}
        self.vocab_size = 0

    def build_vocabulary(self, texts):
        """
        Metinlerden kelime dağarcığı oluşturur
        """
        # Kelime sayılarını topla
        for text in texts:
            for word in self.tokenize(text.lower()):
                if word in self.word_counts:
                    self.word_counts[word] += 1
                else:
                    self.word_counts[word] = 1
        
        # Belirli bir eşik değerinden fazla geçen kelimeleri kelime dağarcığına ekle
        vocab_index = 0
        for word, count in self.word_counts.items():
            if count >= self.min_word_count:
                self.vocabulary[word] = vocab_index
                self.inverse_vocabulary[vocab_index] = word
                vocab_index += 1
        
        # Özel tokenler ekle
        self.vocabulary['<PAD>'] = vocab_index  # Padding token
        self.inverse_vocabulary[vocab_index] = '<PAD>'
        vocab_index += 1
        
        self.vocabulary['<UNK>'] = vocab_index  # Bilinmeyen kelime token'ı
        self.inverse_vocabulary[vocab_index] = '<UNK>'
        vocab_index += 1
        
        self.vocab_size = len(self.vocabulary)
        
        return self.vocabulary

    def tokenize(self, text):
        """
        Metni basitçe kelimelere ayırır
        """
        # Noktalama işaretlerini kaldır
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Fazla boşlukları temizle
        text = re.sub(r'\s+', ' ', text).strip()
        # Kelimelere ayır
        return text.split()

    def texts_to_sequences(self, texts):
        """
        Metinleri token dizilerine dönüştürür
        """
        sequences = []
        for text in texts:
            sequence = []
            for word in self.tokenize(text.lower()):
                if word in self.vocabulary:
                    sequence.append(self.vocabulary[word])
                else:
                    sequence.append(self.vocabulary['<UNK>'])
            sequences.append(sequence)
        return sequences

    def sequences_to_texts(self, sequences):
        """
        Token dizilerini metinlere dönüştürür
        """
        texts = []
        for sequence in sequences:
            text = []
            for token_id in sequence:
                if token_id in self.inverse_vocabulary:
                    text.append(self.inverse_vocabulary[token_id])
                else:
                    text.append(self.inverse_vocabulary[self.vocabulary['<UNK>']])
            texts.append(' '.join(text))
        return texts

    def pad_sequences(self, sequences, max_length=None):
        """
        Dizileri belirli bir uzunluğa doldurur
        """
        if max_length is None:
            max_length = max(len(seq) for seq in sequences)
        
        padded_sequences = []
        for seq in sequences:
            if len(seq) > max_length:
                padded = seq[:max_length]
            else:
                padded = seq + [self.vocabulary['<PAD>']] * (max_length - len(seq))
            padded_sequences.append(padded)
        
        return np.array(padded_sequences)

    def one_hot_encode(self, sequences):
        """
        Dizileri one-hot kodlamasına dönüştürür
        """
        encoded = np.zeros((len(sequences), len(sequences[0]), self.vocab_size), dtype=np.float32)
        for i, sequence in enumerate(sequences):
            for j, idx in enumerate(sequence):
                encoded[i, j, idx] = 1.0
        return encoded