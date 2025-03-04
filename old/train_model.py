import librosa
import soundfile as sf

print("Librosa version:", librosa.__version__)
print("Soundfile version:", sf.__version__)

# import os
# os.environ['PATH'] = r'C:/Users/chris/radioconda/envs/ChordIdentifier_tf/Library/bin;' + os.environ['PATH']

# import numpy as np
# import librosa
# import tensorflow as tf
# from sklearn.preprocessing import LabelEncoder
# from keras.utils import to_categorical
# from sklearn.model_selection import train_test_split

# NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# def extract_features(file_path):
#     y, sr = librosa.load(file_path, sr=None)
#     mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
#     chroma = librosa.feature.chroma_stft(y=y, sr=sr)
#     mel = librosa.feature.melspectrogram(y=y, sr=sr)
#     contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
#     tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    
#     features = np.hstack([np.mean(mfccs, axis=1), np.mean(chroma, axis=1), np.mean(mel, axis=1), np.mean(contrast, axis=1), np.mean(tonnetz, axis=1)])
#     return features

# def load_data(data_dir):
#     features = []
#     labels = []
#     for root, dirs, files in os.walk(data_dir):
#         for file in files:
#             if file.endswith('.wav'):
#                 file_path = os.path.join(root, file)
#                 feature = extract_features(file_path)
#                 label = file.split('_')[0]  # Assuming the chord name is the first part of the file name
#                 features.append(feature)
#                 labels.append(label)
#     return np.array(features), np.array(labels)

# def create_model(input_shape, num_classes):
#     model = tf.keras.Sequential()
#     model.add(tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))
#     model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
#     model.add(tf.keras.layers.Dropout(0.25))
#     model.add(tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu'))
#     model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
#     model.add(tf.keras.layers.Dropout(0.25))
#     model.add(tf.keras.layers.Flatten())
#     model.add(tf.keras.layers.Dense(128, activation='relu'))
#     model.add(tf.keras.layers.Dropout(0.5))
#     model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model

# # Load data
# data_dir = 'C:/School/2025_Winter/Python for Engineers/Project/ChordIdentifier/ChordIdentifier/GeneratedChords'
# features, labels = load_data(data_dir)

# # Encode labels
# label_encoder = LabelEncoder()
# labels_encoded = label_encoder.fit_transform(labels)
# labels_categorical = to_categorical(labels_encoded)

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(features, labels_categorical, test_size=0.2, random_state=42)

# # Create model
# input_shape = (X_train.shape[1], 1)
# num_classes = len(label_encoder.classes_)
# model = create_model(input_shape, num_classes)

# # Train model
# model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# # Save the model
# model.save('chord_identifier_model.h5')

# # Save the label encoder classes
# np.save('classes.npy', label_encoder.classes_)