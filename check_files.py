import os
from pathlib import Path

base = Path(r"C:\Users\HARSHA VARDHAN\Desktop\Project 143\backend")

required_files = {
    "main.py": base / "main.py",
    "MIT-BIH signal model": base / "models" / "ecg_cnn_bilstm.keras",
    "MobileNetV2 image model": base / "models" / "ecg_mobilenetv2_final.keras",
    "MIT-BIH train CSV": base / "data" / "mitbih_train.csv",
    "MIT-BIH test CSV": base / "data" / "mitbih_test.csv",
}

print("=== File Check ===")
all_ok = True
for name, path in required_files.items():
    exists = path.exists()
    if exists:
        size = f"{path.stat().st_size/1024/1024:.1f} MB"
    else:
        size = "MISSING"
    status = "✅" if exists else "❌"
    print(f"{status} {name}: {size}")
    if not exists:
        all_ok = False

print(f"\n{'✅ All files ready!' if all_ok else '❌ Some files missing'}")
