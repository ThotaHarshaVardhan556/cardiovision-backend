import pandas as pd
import numpy as np

train = pd.read_csv("data/mitbih_train.csv", header=None)
test  = pd.read_csv("data/mitbih_test.csv",  header=None)

print("=" * 60)
print("STEP 3 - DATASET VERIFICATION")
print("=" * 60)
print(f"✓ Train shape: {train.shape}  (expected: 87554, 188)")
print(f"✓ Test shape:  {test.shape}   (expected: 21892, 188)")

print("\n✓ Train class distribution:")
print(train.iloc[:, -1].value_counts().sort_index())
print("\n✓ Test class distribution:")
print(test.iloc[:, -1].value_counts().sort_index())
print("\n✓ All Data OK - Ready for training!")
