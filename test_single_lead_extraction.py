#!/usr/bin/env python3
"""
Test script for single-lead extraction on multi-lead ECG images.
Tests the hybrid strategy against actual images from the dataset.
"""

import os
import sys
import json
import requests
from pathlib import Path

# Configuration
API_BASE = "http://localhost:8000"
DATASET_ROOT = r"C:\Users\HARSHA VARDHAN\Desktop\ECG\dataset"
OUTPUT_FILE = "single_lead_test_results.json"

def test_single_lead_extraction():
    """Test single-lead extraction on all dataset images."""
    
    print("\n" + "="*80)
    print("SINGLE-LEAD EXTRACTION TEST")
    print("="*80)
    
    results = {
        "test_timestamp": str(os.popen('powershell "[DateTime]::Now"').read().strip()),
        "api_url": API_BASE,
        "dataset_root": DATASET_ROOT,
        "results": {
            "Normal": [],
            "MI": [],
            "Abnormal": [],
            "HistoryMI": []
        },
        "summary": {}
    }
    
    # Verify API is running
    try:
        health = requests.get(f"{API_BASE}/health", timeout=5)
        if health.status_code != 200:
            print(f"❌ API health check failed: {health.status_code}")
            return results
        print(f"✓ API is running at {API_BASE}")
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print("   Start the API with: python main.py")
        return results
    
    # Test each category
    for category in results["results"].keys():
        category_path = Path(DATASET_ROOT) / category
        if not category_path.exists():
            print(f"\n⚠️  {category} folder not found: {category_path}")
            continue
        
        images = list(category_path.glob("*.jpg")) + list(category_path.glob("*.png"))
        print(f"\n{'─'*80}")
        print(f"Testing {category}: {len(images)} images")
        print(f"{'─'*80}")
        
        for i, img_path in enumerate(images[:3]):  # Test first 3 of each category
            try:
                print(f"\n  [{i+1}] {img_path.name}")
                
                # Upload image
                with open(img_path, "rb") as f:
                    files = {"file": f}
                    response = requests.post(
                        f"{API_BASE}/analyze",
                        files=files,
                        timeout=10
                    )
                
                if response.status_code != 200:
                    print(f"      ❌ Upload failed: {response.status_code}")
                    results["results"][category].append({
                        "file": img_path.name,
                        "status": "failed",
                        "error": response.text[:100]
                    })
                    continue
                
                data = response.json()
                prediction = data.get("prediction", {})
                
                # Extract key metrics
                test_result = {
                    "file": img_path.name,
                    "status": "success",
                    "predicted_class": prediction.get("class", "unknown"),
                    "confidence": prediction.get("confidence", 0),
                    "digitization_quality": data.get("digitization_quality", "unknown"),
                    "signal_length": len(data.get("signal", [])),
                    "notes": data.get("notes", "")
                }
                
                # Log results
                print(f"      ✓ Predicted: {test_result['predicted_class']} ({test_result['confidence']:.1%})")
                print(f"        Quality: {test_result['digitization_quality']}, Signal length: {test_result['signal_length']}")
                
                results["results"][category].append(test_result)
                
            except Exception as e:
                print(f"      ❌ Error: {str(e)[:100]}")
                results["results"][category].append({
                    "file": img_path.name,
                    "status": "error",
                    "error": str(e)[:100]
                })
    
    # Calculate summary
    total_tested = sum(len(v) for v in results["results"].values())
    total_success = sum(sum(1 for r in v if r.get("status") == "success") for v in results["results"].values())
    
    results["summary"] = {
        "total_tested": total_tested,
        "total_success": total_success,
        "success_rate": f"{100*total_success/total_tested if total_tested > 0 else 0:.1f}%"
    }
    
    # Print summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    print(f"✓ Total tested: {results['summary']['total_tested']}")
    print(f"✓ Successful: {results['summary']['total_success']}")
    print(f"✓ Success rate: {results['summary']['success_rate']}")
    
    # Save results
    try:
        with open(OUTPUT_FILE, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {OUTPUT_FILE}")
    except Exception as e:
        print(f"⚠️  Could not save results: {e}")
    
    return results

if __name__ == "__main__":
    test_single_lead_extraction()
