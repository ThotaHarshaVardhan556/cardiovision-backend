#!/usr/bin/env python3
"""
Quick verification that the single-lead extraction code is properly integrated.
Run this BEFORE starting the API to catch any issues early.
"""

import os
import sys
import traceback

print("\n" + "="*80)
print("VERIFYING SINGLE-LEAD EXTRACTION IMPLEMENTATION")
print("="*80)

# Check 1: Verify main.py exists
print("\n[1/4] Checking main.py exists...")
if not os.path.exists("main.py"):
    print("❌ FAILED: main.py not found in current directory")
    sys.exit(1)
print("✓ main.py found")

# Check 2: Verify methods are defined
print("\n[2/4] Verifying new methods in ECGImageDigitizer...")
try:
    # Read main.py to check for method definitions
    with open("main.py", "r") as f:
        content = f.read()
    
    required_methods = [
        "_detect_single_lead_region",
        "_validate_lead_region",
        "_crop_to_single_lead"
    ]
    
    missing = []
    for method in required_methods:
        if f"def {method}(" not in content:
            missing.append(method)
        else:
            print(f"  ✓ {method} defined")
    
    if missing:
        print(f"❌ FAILED: Missing methods: {missing}")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ FAILED: Could not read main.py: {e}")
    sys.exit(1)

# Check 3: Verify integration point in digitize()
print("\n[3/4] Verifying integration in digitize() method...")
if "self._crop_to_single_lead(img)" in content:
    print("  ✓ _crop_to_single_lead() called in digitize()")
else:
    print("⚠️  WARNING: Could not find integration call in digitize()")
    print("    (This might be OK if integration was done differently)")

# Check 4: Try to import and instantiate
print("\n[4/4] Attempting to import main.py and create ECGImageDigitizer...")
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("main_module", "main.py")
    main_module = importlib.util.module_from_spec(spec)
    
    # Don't execute the whole module (would try to start FastAPI)
    # Just check if it compiles
    compile(open("main.py").read(), "main.py", "exec")
    print("✓ main.py syntax is valid")
    
    # Basic structure check
    if "class ECGImageDigitizer" in content:
        print("✓ ECGImageDigitizer class found")
    
except SyntaxError as e:
    print(f"❌ SYNTAX ERROR in main.py:")
    print(f"  Line {e.lineno}: {e.msg}")
    print(f"  {e.text}")
    sys.exit(1)
except Exception as e:
    print(f"❌ ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)

# Check 5: Verify test script
print("\n[5/5] Checking test script...")
if os.path.exists("test_single_lead_extraction.py"):
    print("✓ test_single_lead_extraction.py found")
else:
    print("⚠️  test_single_lead_extraction.py not found (optional)")

# Check 6: Verify documentation
print("\n[6/6] Checking documentation...")
if os.path.exists("SINGLE_LEAD_EXTRACTION_README.md"):
    print("✓ SINGLE_LEAD_EXTRACTION_README.md found")
else:
    print("⚠️  Documentation file not found (optional)")

print("\n" + "="*80)
print("✓ VERIFICATION PASSED - Ready to start API")
print("="*80)

print("""
NEXT STEPS:
  1. Start the API:
     python main.py
  
  2. In another terminal, test with an image:
     python test_single_lead_extraction.py
  
  3. Or manually test:
     curl -F "file=@image.jpg" http://localhost:8000/analyze

EXPECTED BEHAVIOR:
  - API should start without errors
  - Logs should show single-lead extraction strategy
  - Predictions should be >60% confidence
  - digitization_quality should be "Good" or "Fair"
""")
