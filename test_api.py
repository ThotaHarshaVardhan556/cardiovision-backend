#!/usr/bin/env python3
"""
Comprehensive test of CardioVision FastAPI endpoints.
"""

import requests
import json
import time
from pathlib import Path

BASE_URL = "http://localhost:8000"
DATASET_ROOT = Path(r"C:\Users\HARSHA VARDHAN\Desktop\ECG\dataset")

print("\n" + "="*80)
print("CARDIOVISION FASTAPI ENDPOINT TESTS")
print("="*80)

results = []

def test_endpoint(name, method, endpoint, **kwargs):
    """Test a single endpoint."""
    url = f"{BASE_URL}{endpoint}"
    print(f"\n[TEST] {name}")
    print(f"  URL: {method} {url}")
    try:
        if method == "GET":
            resp = requests.get(url, timeout=10, **kwargs)
        elif method == "POST":
            resp = requests.post(url, timeout=10, **kwargs)
        
        status = "✅" if resp.status_code == 200 else "❌"
        print(f"  {status} Status: {resp.status_code}")
        
        if resp.status_code == 200:
            try:
                data = resp.json()
                if "prediction" in data:
                    pred = data["prediction"]
                    print(f"    Prediction: {pred.get('class', 'N/A')}")
                    print(f"    Confidence: {pred.get('confidence', 'N/A'):.1%}")
                if "status" in data:
                    print(f"    Status: {data['status']}")
            except:
                print(f"    Response: {resp.text[:200]}")
            results.append({"test": name, "status": "PASS", "code": resp.status_code})
        else:
            print(f"    Error: {resp.text[:200]}")
            results.append({"test": name, "status": "FAIL", "code": resp.status_code})
    except Exception as e:
        print(f"  ❌ Error: {str(e)[:100]}")
        results.append({"test": name, "status": "ERROR", "error": str(e)[:50]})

# Test 1: Health Check
test_endpoint("Health Check", "GET", "/health")

# Test 2: Root Info
test_endpoint("Root Info", "GET", "/")

# Test 3: Model Status (Signal)
test_endpoint("Model Status (Signal)", "GET", "/model/status")

# Test 4: Model Status (Image)
test_endpoint("Model Status (Image)", "GET", "/model/image-status")

# Test 5-8: Demo Endpoints
for condition in ["normal", "mi", "afib", "pvc"]:
    test_endpoint(
        f"Demo - {condition.upper()}", 
        "POST", 
        f"/analyze/demo?condition={condition}"
    )

# Test 9-12: Image Predictions
image_tests = [
    ("Normal(6).jpg", "Normal Sinus Rhythm"),
    ("MI(9).jpg", "Myocardial Infarction"),
    ("HB(1).jpg", "Abnormal Heartbeat"),
    ("PMI(1).jpg", "History of MI"),
]

image_paths = [
    (DATASET_ROOT / "Normal" / "Normal(6).jpg", "Normal"),
    (DATASET_ROOT / "Myocardial Infarction" / "MI(9).jpg", "MI"),
    (DATASET_ROOT / "Abnormal Heartbeat" / "HB(1).jpg", "Abnormal"),
    (DATASET_ROOT / "History of MI" / "PMI(1).jpg", "HistoryMI"),
]

print("\n" + "─"*80)
print("IMAGE PREDICTION TESTS")
print("─"*80)

for file_path, expected in image_paths:
    if file_path.exists():
        test_name = f"Image: {file_path.name}"
        print(f"\n[TEST] {test_name}")
        try:
            with open(file_path, "rb") as f:
                files = {"file": f}
                resp = requests.post(f"{BASE_URL}/analyze/image", files=files, timeout=10)
            
            status = "✅" if resp.status_code == 200 else "❌"
            print(f"  {status} Status: {resp.status_code}")
            
            if resp.status_code == 200:
                data = resp.json()
                pred = data.get("prediction", {})
                conf = pred.get("confidence", 0)
                cls = pred.get("class", "N/A")
                print(f"    Predicted: {cls}")
                print(f"    Confidence: {conf:.1%}")
                print(f"    Expected: {expected}")
                results.append({
                    "test": test_name,
                    "status": "PASS",
                    "predicted": cls,
                    "confidence": f"{conf:.1%}"
                })
            else:
                print(f"    Error: {resp.text[:200]}")
                results.append({"test": test_name, "status": "FAIL", "code": resp.status_code})
        except Exception as e:
            print(f"  ❌ Error: {str(e)[:100]}")
            results.append({"test": test_name, "status": "ERROR"})
    else:
        print(f"\n[SKIP] {file_path.name} — File not found")

# Summary
print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)

pass_count = sum(1 for r in results if r.get("status") == "PASS")
fail_count = sum(1 for r in results if r.get("status") == "FAIL")
error_count = sum(1 for r in results if r.get("status") == "ERROR")
total = len(results)

print(f"\n✅ Passed:  {pass_count}/{total}")
print(f"❌ Failed:  {fail_count}/{total}")
print(f"⚠️  Errors:  {error_count}/{total}")
print(f"\nSuccess Rate: {100*pass_count//total}%")

print("\n" + "="*80)
print("DETAILED RESULTS")
print("="*80)
for r in results:
    test_name = r.get("test", "Unknown")
    status = r.get("status", "UNKNOWN")
    if status == "PASS":
        icon = "✅"
    elif status == "FAIL":
        icon = "❌"
    else:
        icon = "⚠️"
    
    print(f"{icon} {test_name}")
    if "predicted" in r:
        print(f"   → {r['predicted']} ({r.get('confidence', 'N/A')})")
