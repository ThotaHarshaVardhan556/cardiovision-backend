import numpy as np
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("=" * 80)
print("STEP 1: LOAD ALL ECG IMAGES FROM DATASET")
print("=" * 80)

DATASET_PATH = r"C:\Users\HARSHA VARDHAN\Desktop\ECG\dataset"
SAVE_MODEL_PATH = r"C:\Users\HARSHA VARDHAN\Desktop\Project 143\backend\models\ecg_image_cnn.keras"

def get_label(folder_name):
    name = folder_name.lower()
    if "abnormal" in name:
        return 0, "Abnormal"
    elif "history" in name or "pmi" in name:
        return 1, "HistoryMI"
    elif "mi" in name or "infarction" in name:
        return 2, "MI"
    elif "normal" in name:
        return 3, "Normal"
    else:
        return 0, "Abnormal"

X, y = [], []
for folder_name in sorted(os.listdir(DATASET_PATH)):
    folder_path = os.path.join(DATASET_PATH, folder_name)
    if not os.path.isdir(folder_path):
        continue
    label, class_name = get_label(folder_name)
    count = 0
    for fname in os.listdir(folder_path):
        if not fname.lower().endswith(('.jpg','.png','.jpeg','.bmp')):
            continue
        img = cv2.imread(os.path.join(folder_path, fname), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (128, 128))
        X.append(img / 255.0)
        y.append(label)
        count += 1
    print(f"{folder_name} → {class_name}: {count} images")

X = np.array(X, dtype=np.float32).reshape(-1, 128, 128, 1)
y = np.array(y, dtype=np.int32)
print(f"Total: {X.shape}, Classes: {np.bincount(y)}")

# STEP 2 - Fast offline augmentation using numpy/cv2 (no generator loop)
print("\n" + "=" * 80)
print("STEP 2: FAST OFFLINE AUGMENTATION")
print("=" * 80)

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical

def fast_augment(X_class, target):
    """Fast numpy+cv2 augmentation without generator loop"""
    result = list(X_class)
    needed = target - len(X_class)
    if needed <= 0:
        return np.array(result)
    
    rng = np.random.RandomState(42)
    indices = rng.choice(len(X_class), needed, replace=True)
    
    for idx in indices:
        img = X_class[idx].copy().reshape(128, 128)
        
        # Random small shift
        shift_x = rng.randint(-4, 4)
        shift_y = rng.randint(-4, 4)
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        img = cv2.warpAffine(img, M, (128, 128),
                             borderMode=cv2.BORDER_REPLICATE)
        
        # Random tiny zoom
        if rng.random() > 0.5:
            scale = rng.uniform(0.97, 1.03)
            M2 = cv2.getRotationMatrix2D((64, 64), 0, scale)
            img = cv2.warpAffine(img, M2, (128, 128),
                                borderMode=cv2.BORDER_REPLICATE)
        
        result.append(img.reshape(128, 128, 1))
    
    return np.array(result, dtype=np.float32)

TARGET = 300
X_aug_list, y_aug_list = [], []
for class_idx in range(4):
    class_samples = X[y == class_idx]
    augmented = fast_augment(class_samples, TARGET)
    X_aug_list.append(augmented)
    y_aug_list.append(np.full(len(augmented), class_idx, dtype=np.int32))
    print(f"Class {class_idx}: {len(class_samples)} → {len(augmented)} samples")

X_aug = np.concatenate(X_aug_list, axis=0)
y_aug = np.concatenate(y_aug_list, axis=0)
print(f"Augmented: {X_aug.shape}")

# Shuffle
perm = np.random.permutation(len(X_aug))
X_aug, y_aug = X_aug[perm], y_aug[perm]

# Split
X_train, X_temp, y_train, y_temp = train_test_split(
    X_aug, y_aug, test_size=0.30, random_state=42, stratify=y_aug)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

y_train_cat = to_categorical(y_train, 4)
y_val_cat   = to_categorical(y_val, 4)
y_test_cat  = to_categorical(y_test, 4)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(cw))
print("Class weights:", class_weights)

# STEP 3 - Build and train CNN
print("\n" + "=" * 80)
print("STEP 3: BUILD AND TRAIN CNN")
print("=" * 80)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense,
                                     Dropout, BatchNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        ModelCheckpoint)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', padding='same',
           input_shape=(128, 128, 1)),
    BatchNormalization(),
    Conv2D(32, (3,3), activation='relu', padding='same'),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.30),

    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(4, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel architecture:")
model.summary()

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=12,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                     patience=5, min_lr=1e-7, verbose=1),
    ModelCheckpoint(SAVE_MODEL_PATH,
                    monitor='val_accuracy',
                    save_best_only=True, verbose=1)
]

print("\nTraining CNN (60 epochs)...")
print("=" * 80)

history = model.fit(
    X_train, y_train_cat,
    epochs=60,
    batch_size=16,
    validation_data=(X_val, y_val_cat),
    class_weight=class_weights,
    callbacks=callbacks,
    shuffle=True,
    verbose=1
)

# STEP 4 - Evaluate
print("\n" + "=" * 80)
print("STEP 4: EVALUATION")
print("=" * 80)

from sklearn.metrics import classification_report

test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
y_true = np.argmax(y_test_cat, axis=1)

class_names = ["Abnormal", "HistoryMI", "MI", "Normal"]
print("\nPer-class accuracy:")
for i, name in enumerate(class_names):
    mask = y_true == i
    if mask.sum() > 0:
        acc = (y_pred[mask] == i).mean()
        print(f"  {name:12}: {acc*100:.1f}%  (n={mask.sum()})")

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

model.save(SAVE_MODEL_PATH)

print("\n" + "=" * 80)
print("✓ IMAGE CNN TRAINING COMPLETE!")
print(f"✓ Model saved to: {SAVE_MODEL_PATH}")
print("=" * 80)
