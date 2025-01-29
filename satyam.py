!pip install pandas numpy scikit-learn gensim tensorflow seaborn

# Import necessary libraries
import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# import os
# for dirname, _, filenames in os.walk('Fake.csv'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

file_path = "fake.csv"

# Read the CSV file
df = pd.read_csv(file_path)

# Display the first 

# Check if 'test.csv' exists in the current directory
if "fake.csv" in os.listdir('.'):
    print("fake.csv is present in the folder.")
else:
    print("fake.csv is not found.")

file_path = "fake.csv"  # Use just the filename since it's in the same directory
try:
    with open(file_path, 'r') as f:
        print("File opened successfully! ✅")
except FileNotFoundError:
    print("File not found. ❌")
except Exception as e:
    print(f"An error occurred: {e}")

print(os.getcwd())  # Prints the current working directory

print(os.listdir('.'))  # Lists all files in the current directory

# Step 1: Load the datasets
# Read the CSV files into pandas DataFrames
fake_df = pd.read_csv("fake.csv")
true_df = pd.read_csv("true.csv")

# Add a label column to distinguish between fake and real news
fake_df['label'] = 0  # 0 for fake news
true_df['label'] = 1  # 1 for real news

# Combine the two DataFrames into one
data = pd.concat([fake_df, true_df], ignore_index=True)

# Assuming the dataset has columns: 'text' and 'label'
texts = data['text']
labels = data['label']

# Step 2: Preprocessing
# Download stopwords if not already downloaded

nltk.download('wordnet')
nltk.download('stopwords')

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords and stem words
    text = ' '.join([stemmer.stem(word) for word in text.split() if word not in stop_words])
    return text

# Apply preprocessing to all texts
texts = texts.apply(preprocess_text)

# Step 3: Train Word2Vec from scratch
# Tokenize texts for Word2Vec
tokenized_texts = [text.split() for text in texts]

# Train Word2Vec model
word2vec_model = Word2Vec(sentences=tokenized_texts, vector_size=300, window=10, min_count=5, workers=4, sg=1)  # sg=1 for skip-gram

# Convert texts to Word2Vec vectors
def text_to_vector(text):
    words = text.split()
    word_vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
    if len(word_vectors) == 0:
        return np.zeros(300)  # vector_size=300
    return np.mean(word_vectors, axis=0)

X_word2vec = np.array([text_to_vector(text) for text in texts])

# Step 4: Prepare data for LSTM
# Tokenize texts for LSTM
tokenizer = Tokenizer(num_words=10000)  # Increase vocabulary size
tokenizer.fit_on_texts(texts)
X_sequences = tokenizer.texts_to_sequences(texts)
X_padded = pad_sequences(X_sequences, maxlen=200)  # Pad sequences to a fixed length

# Split data into training and testing sets
X_train_lstm, X_test_lstm, y_train, y_test = train_test_split(X_padded, labels, test_size=0.25, random_state=42)

# Pad sequences to a fixed length (e.g., 200 tokens)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=200)

lstm_model.add(Embedding(input_dim=10000, output_dim=300, weights=[embedding_matrix], trainable=False, input_length=None))

# Tokenize text data
sequences = tokenizer.texts_to_sequences(texts)

# Pad sequences to the same length
padded_sequences = pad_sequences(sequences, maxlen=200)  # Adjust maxlen if needed

# Now feed padded_sequences into your LSTM model
lstm_model.add(Embedding(input_dim=10000, output_dim=300, weights=[embedding_matrix], trainable=False))

# Step 5: Build LSTM model with improvements
lstm_model = Sequential()

# Embedding layer (using Word2Vec embeddings)
embedding_matrix = np.zeros((10000, 300))  # vocab_size=10000, vector_size=300
for word, i in tokenizer.word_index.items():
    if i < 10000 and word in word2vec_model.wv:
        embedding_matrix[i] = word2vec_model.wv[word]

lstm_model.add(Embedding(input_dim=10000, output_dim=300, weights=[embedding_matrix], input_length=200, trainable=False))  # Use pre-trained Word2Vec

# LSTM layer with more neurons and dropout
lstm_model.add(LSTM(128, dropout=0.3, recurrent_dropout=0.3, return_sequences=True))  # First LSTM layer
lstm_model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))  # Second LSTM layer

# Fully connected layer
lstm_model.add(Dense(64, activation='relu'))
lstm_model.add(Dropout(0.3))

# Output layer
lstm_model.add(Dense(1, activation='sigmoid'))

# Compile the model
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early Stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Step 6: Train the LSTM model
history = lstm_model.fit(
    X_train_lstm, y_train,
    epochs=5,  # Increase number of epochs
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stopping]
)

# Step 7: Evaluate the LSTM model
y_pred_lstm = (lstm_model.predict(X_test_lstm) > 0.5).astype(int)

# Calculate evaluation metrics
acc = accuracy_score(y_test, y_pred_lstm)
prec = precision_score(y_test, y_pred_lstm)
rec = recall_score(y_test, y_pred_lstm)
f1 = f1_score(y_test, y_pred_lstm)

print(f"LSTM with Word2Vec - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1-Score: {f1:.4f}")

# Step 8: Classification Report and Confusion Matrix
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lstm, target_names=['Fake', 'Real']))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_lstm)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Step 9: Plot Training and Validation Metrics
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Accuracy Plot
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_acc, label='Train Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Model Summary
lstm_model.summary()