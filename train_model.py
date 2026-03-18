import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, Flatten,
                                     Dense, Dropout, BatchNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        ModelCheckpoint)

print("=" * 70)
print("STEP 1: CHECK DATA RANGE AND NORMALIZE")
print("=" * 70)

train = pd.read_csv("data/mitbih_train.csv", header=None)
test  = pd.read_csv("data/mitbih_test.csv",  header=None)

print("\n✓ Raw data range BEFORE normalization:")
print(f"  Min: {train.iloc[:, :-1].values.min():.2f}")
print(f"  Max: {train.iloc[:, :-1].values.max():.2f}")
print(f"  Mean: {train.iloc[:, :-1].values.mean():.2f}")
print(f"  Std: {train.iloc[:, :-1].values.std():.2f}")

# Sample 20% for speed
train = train.sample(frac=0.2, random_state=42)
print(f"\n✓ Using 20% of training data: {len(train)} samples")

X_train_raw = train.iloc[:, :-1].values.astype(np.float32)
y_train = train.iloc[:, -1].values.astype(int)
X_test_raw  = test.iloc[:,  :-1].values.astype(np.float32)
y_test  = test.iloc[:,  -1].values.astype(int)

# Per-sample normalization (normalize each ECG signal individually to [0, 1])
def normalize_ecg(X):
    """Normalize each ECG sample to [0, 1] range"""
    X_norm = np.zeros_like(X)
    for i in range(len(X)):
        sig = X[i]
        sig_min = sig.min()
        sig_max = sig.max()
        if sig_max - sig_min > 1e-8:
            X_norm[i] = (sig - sig_min) / (sig_max - sig_min)
        else:
            X_norm[i] = sig - sig_min
    return X_norm

print("\n✓ Applying per-sample normalization...")
X_train = normalize_ecg(X_train_raw).reshape(-1, 187, 1)
X_test  = normalize_ecg(X_test_raw).reshape(-1, 187, 1)

print(f"\n✓ After normalization:")
print(f"  Train range: [{X_train.min():.4f}, {X_train.max():.4f}]")
print(f"  Test range:  [{X_test.min():.4f}, {X_test.max():.4f}]")
print(f"  Train shape: {X_train.shape}")
print(f"  Test shape:  {X_test.shape}")
print(f"  Class distribution: {np.bincount(y_train)}")

y_train_cat = to_categorical(y_train, 5)
y_test_cat  = to_categorical(y_test,  5)

cw = compute_class_weight('balanced',
                           classes=np.unique(y_train),
                           y=y_train)
class_weights = dict(enumerate(cw))
print(f"✓ Class weights: {class_weights}")


# STEP 2 - Build model
print("\n" + "=" * 70)
print("STEP 2: BUILD CNN MODEL")
print("=" * 70)

model = Sequential([
    Conv1D(32, 5, activation='relu', padding='same', input_shape=(187, 1)),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.2),

    Conv1D(64, 5, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.2),

    Conv1D(128, 3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.2),

    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(5, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.0003),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n✓ Model architecture:")
model.summary()

# STEP 3 - Train
print("\n" + "=" * 70)
print("STEP 3: TRAINING ON NORMALIZED DATA")
print("=" * 70)

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=5,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                     patience=3, min_lr=1e-6, verbose=1),
    ModelCheckpoint("models/ecg_cnn_bilstm.keras",
                    monitor='val_accuracy',
                    save_best_only=True, verbose=1)
]

history = model.fit(
    X_train, y_train_cat,
    epochs=20,
    batch_size=32,
    validation_split=0.15,
    class_weight=class_weights,
    callbacks=callbacks,
    shuffle=True,
    verbose=1
)

# STEP 4 - Evaluate
print("\n" + "=" * 70)
print("STEP 4: EVALUATION")
print("=" * 70)

test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\n✓ Test Accuracy: {test_acc * 100:.2f}%")
print(f"✓ Test Loss: {test_loss:.4f}")

y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
class_names = ["Normal", "Supraventricular", "PVC", "Fusion", "MI/Unknown"]
print("\n✓ Per-class accuracy:")
for i, name in enumerate(class_names):
    mask = y_test == i
    if mask.sum() > 0:
        acc = (y_pred[mask] == i).mean()
        print(f"  {name:20}: {acc*100:6.1f}%  (n={mask.sum():5d})")

# Save model
model.save("models/ecg_cnn_bilstm.keras")
print("\n" + "=" * 70)
print("✓ TRAINING COMPLETE!")
print("✓ Model saved to models/ecg_cnn_bilstm.keras")
print("=" * 70)
