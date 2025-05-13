# tokenizer.py

import numpy as np


class SimpleTokenizer:
    def __init__(self):
        self.word_index = {}  # kelime -> index eşleştirmesi
        self.index_word = {}  # index -> kelime eşleştirmesi
        self.word_count = {}  # kelime frekansları
        self.document_count = 0
        self.num_words = None
        self.char_level = False
        
    def fit_on_texts(self, texts):
        """
        Verilen metinlerden kelime dağarcığı oluşturur
        """
        for text in texts:
            self.document_count += 1
            seq = self._text_to_word_sequence(text)
            
            for w in seq:
                if w in self.word_count:
                    self.word_count[w] += 1
                else:
                    self.word_count[w] = 1
        
        # Kelimeleri frekanslarına göre sırala
        sorted_word_counts = sorted(self.word_count.items(), key=lambda x: x[1], reverse=True)
        
        # Kelime indexi oluştur
        for idx, (word, _) in enumerate(sorted_word_counts):
            self.word_index[word] = idx + 1  # 0 indeksi padding için ayrılmıştır
            self.index_word[idx + 1] = word
        
        # Özel tokenler ekle
        self.word_index['<PAD>'] = 0
        self.index_word[0] = '<PAD>'
        
    def _text_to_word_sequence(self, text):
        """
        Metni kelimelere ayırır
        """
        if self.char_level:
            return list(text)
        else:
            return text.lower().split()
        
    def texts_to_sequences(self, texts):
        """
        Metinleri indeks dizilerine dönüştürür
        """
        sequences = []
        for text in texts:
            seq = self._text_to_word_sequence(text)
            vect = []
            for w in seq:
                if w in self.word_index:
                    i = self.word_index[w]
                    if self.num_words is None or i < self.num_words:
                        vect.append(i)
                # Kelime bulunamazsa atla (UNK token kullanmıyoruz)
            sequences.append(vect)
        return sequences
    
    def sequences_to_texts(self, sequences):
        """
        İndeks dizilerini metinlere dönüştürür
        """
        texts = []
        for seq in sequences:
            words = []
            for idx in seq:
                if idx in self.index_word:
                    words.append(self.index_word[idx])
            texts.append(" ".join(words))
        return texts
    
    def pad_sequences(self, sequences, maxlen=None, padding='post'):
        """
        Dizileri belirli bir uzunluğa doldurur
        """
        if maxlen is None:
            maxlen = max(len(s) for s in sequences)
            
        output = np.zeros((len(sequences), maxlen), dtype=np.int32)
        
        for i, seq in enumerate(sequences):
            if len(seq) > maxlen:
                output[i, :] = seq[:maxlen]
            else:
                if padding == 'post':
                    output[i, :len(seq)] = seq
                else:  # padding == 'pre'
                    output[i, -len(seq):] = seq
                    
        return output