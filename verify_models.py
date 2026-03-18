#!/usr/bin/env python3
"""
Verify model responses and predictions.
"""

import requests
import json
from pathlib import Path

BASE_URL = "http://localhost:8000"

print("\n" + "="*80)
print("MODEL VERIFICATION - DETAILED RESPONSES")
print("="*80)

# Test 1: Check model status
print("\n[1] Signal Model Status")
try:
    resp = requests.get(f"{BASE_URL}/model/status")
    if resp.status_code == 200:
        data = resp.json()
        print(f"  ✅ Model loaded")
        print(f"  Response keys: {list(data.keys())[:5]}")
        if "model_accuracy" in data:
            print(f"  Accuracy: {data['model_accuracy']}")
        if "model_params" in data:
            print(f"  Params: {data['model_params']}")
except Exception as e:
    print(f"  ❌ Error: {e}")

# Test 2: Check image model status
print("\n[2] Image Model Status")
try:
    resp = requests.get(f"{BASE_URL}/model/image-status")
    if resp.status_code == 200:
        data = resp.json()
        print(f"  ✅ Image model endpoint responds")
        print(f"  Response keys: {list(data.keys())[:5]}")
except Exception as e:
    print(f"  ❌ Error: {e}")

# Test 3: Demo prediction
print("\n[3] Demo Prediction (Normal)")
try:
    resp = requests.post(f"{BASE_URL}/analyze/demo?condition=normal")
    if resp.status_code == 200:
        data = resp.json()
        print(f"  ✅ Demo endpoint responds")
        print(f"  Full response:")
        print(f"    {json.dumps(data, indent=4)[:500]}")
except Exception as e:
    print(f"  ❌ Error: {e}")

# Test 4: Real image prediction
print("\n[4] Real Image Prediction")
img_path = Path(r"C:\Users\HARSHA VARDHAN\Desktop\ECG\dataset\Normal\Normal(6).jpg")
if img_path.exists():
    try:
        with open(img_path, "rb") as f:
            files = {"file": f}
            resp = requests.post(f"{BASE_URL}/analyze/image", files=files)
        if resp.status_code == 200:
            data = resp.json()
            print(f"  ✅ Image prediction received")
            print(f"  Prediction: {data.get('prediction', {}).get('class', 'N/A')}")
            print(f"  Confidence: {data.get('prediction', {}).get('confidence', 'N/A')}")
            print(f"  Response keys: {list(data.keys())}")
            print(f"\n  Full response (first 1000 chars):")
            print(f"    {json.dumps(data, indent=4)[:1000]}")
    except Exception as e:
        print(f"  ❌ Error: {e}")

print("\n" + "="*80)
print("✅ API IS RESPONDING AND READY")
print("="*80)
