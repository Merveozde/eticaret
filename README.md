# import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:

pip install keras
pip install nltk
pip install scikit-learn
pip install collections
pip install tensorflow
pip install seaborn

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import GridSearchCV

nltk.download('stopwords')

stop_words = set(stopwords.words('turkish'))

try:
    data = pd.read_csv('/kaggle/input/eticaret-urun-yorumlari/e-ticaret_urun_yorumlari.csv',sep=';', encoding='utf-8')
except Exception as e:
    print(e)


    data.info()

    data.tail()

    data['Durum'] = data['Durum'].map({0: 'olumsuz', 1: 'olumlu', 2: 'nötr'})

    yorumlar = data['Metin']
tokenized_yorumlar =[word_tokenize(comment.lower()) for comment in yorumlar]
all_words = [word for yorumlar in tokenized_yorumlar for word in yorumlar if word not in stop_words]

word_freq = Counter(all_words)
most_common_words = word_freq.most_common(20)

# Sık kullanılan kelimeler
words, counts = zip(*most_common_words)

# Görselleştirme
plt.figure(figsize=(10, 6))
sns.barplot(x=list(counts), y=list(words))  
plt.title("En Sık Kullanılan Kelimeler")
plt.xlabel("Frekans")
plt.ylabel("Kelimeler")
plt.show()

# Duygu durumu dağılımı
plt.figure(figsize=(8, 6))
sns.countplot(x=data['Durum'])
plt.title("Duygu Durumu Dağılımı")
plt.show()

# Yorum uzunluklarını hesapla (kelime sayısı)
comment_lengths = [len(yorum) for yorum in tokenized_yorumlar]

# Görselleştirme
plt.figure(figsize=(10, 6))
sns.histplot(comment_lengths, kde=True, bins=30)
plt.title("Yorum Uzunlukları Dağılımı")
plt.xlabel("Yorum Uzunluğu (Kelime Sayısı)")
plt.ylabel("Frekans")
plt.show()

tfidf = TfidfVectorizer(stop_words=None, max_features=20)
X_tfidf = tfidf.fit_transform(yorumlar)

# Stop word'leri manuel olarak çıkarma
tokens = [word for word in tfidf.get_feature_names_out() if word not in stop_words]
print(tokens)

max_vocab_size = len(tokenized_yorumlar)
max_sequence_length = max(data['Metin'].str.len())

wordTokenizer = Tokenizer(num_words=max_vocab_size)
label_encoder = LabelEncoder()

duygular = data['Durum']

wordTokenizer.fit_on_texts(data['Metin']) #Yorumların fit edilmesi
data['Metin'] = wordTokenizer.texts_to_sequences(data['Metin'])

padded_sequences = pad_sequences(data['Metin'], maxlen=max_sequence_length, padding='post')

data['Metin']

duygular_encoded = label_encoder.fit_transform(duygular)
data['Durum'] = duygular_encoded

 Her duygu için en sık kelimeleri bul
for duygus in label_encoder.classes_:
    print(f"\nDuygu: {duygus}")
    label_index = label_encoder.transform([duygus])[0]
    sentiment_comments = [tokenized_yorumlar[i] for i in range(len(data)) if duygular_encoded[i] == label_index]
    sentiment_words = [word for yorum in sentiment_comments for word in yorum if word not in stop_words]
    sentiment_word_freq = Counter(sentiment_words)
    most_common_sentiment_words = sentiment_word_freq.most_common(10)
    print(most_common_sentiment_words)

    # Veriyi eğitim ve test setlerine bölün
X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, duygular_encoded, test_size=0.2, random_state=42
)
vectorizer = TfidfVectorizer(max_features=10000)


X_train_list = X_train.tolist()
X_test_list = X_test.tolist()


X_train_str = [' '.join(str(word) for word in yorum) for yorum in X_train_list]
X_test_str = [' '.join(str(word) for word in yorum) for yorum in X_test_list]


X_train_tfidf = vectorizer.fit_transform(X_train_str)
X_test_tfidf = vectorizer.transform(X_test_str)
rf = RandomForestClassifier(random_state=42)
Modeli Eğitme 
param_grid = {
    'n_estimators': [100, 200, 300],         
    'max_depth': [None, 10, 20, +30],          
    'min_samples_split': [2, 5, 10],           
    'min_samples_leaf': [1, 2, 4],             
    'max_features': ['auto', 'sqrt', 'log2']  
}
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                           cv=5, n_jobs=-1, verbose=2, scoring='accuracy')

# Fit
grid_search.fit(X_train_tfidf, y_train)
print(f"En iyi hiperparametreler: {grid_search.best_params_}")

print(f"En iyi çapraz doğrulama doğruluk oranı: {grid_search.best_score_:.2f}")
En iyi hiperparametreler: {'max_depth': None, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}
En iyi çapraz doğrulama doğruluk oranı: 0.87
best_rf_model = grid_search.best_estimator_
test_accuracy = best_rf_model.score(X_test_tfidf, y_test)

print(f"Test verisi doğruluk oranı: {test_accuracy:.2f}")
