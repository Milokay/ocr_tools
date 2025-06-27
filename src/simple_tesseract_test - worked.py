"""
TESSERACT PATH FINDER AND IMAGE TEST
===================================

Finds correct Tesseract installation path and tests Price_Enter.png
Handles common Windows installation locations

Author: Tesseract Path Finder
Date: 2025-06-24
"""

import cv2
import numpy as np
from pathlib import Path
import os
import sys

class TesseractPathFinder:
    """Find and configure Tesseract installation"""
    
    def __init__(self):
        # Common Tesseract installation paths on Windows
        self.possible_paths = [
            r"C:\Users\ykim\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe", 
            r"C:\Users\ykim\AppData\Local\Tesseract-OCR\tesseract.exe",
            r"C:\tesseract\tesseract.exe",
            r"C:\OCR\Tesseract-OCR\tesseract.exe"
        ]
        
        self.tesseract_path = None
        self.image_path = Path(r"C:\Users\ykim\OneDrive - Venture Metals\Desktop\Trading Invoices\Automation Project - Excel Files\python\Cable Automation\Price_Enter.png")
    
    def find_tesseract(self):
        """Find working Tesseract installation"""
        print("🔍 SEARCHING FOR TESSERACT INSTALLATION...")
        print("-" * 50)
        
        for path in self.possible_paths:
            print(f"Checking: {path}")
            if Path(path).exists():
                print(f"✅ Found Tesseract at: {path}")
                self.tesseract_path = path
                return True
            else:
                print(f"❌ Not found")
        
        print("\n❌ Tesseract not found in common locations!")
        print("\n💡 INSTALLATION OPTIONS:")
        print("1. Download from: https://github.com/UB-Mannheim/tesseract/wiki")
        print("2. Install to: C:\\Program Files\\Tesseract-OCR\\")
        print("3. Or install via conda: conda install -c conda-forge tesseract")
        
        return False
    
    def test_tesseract_import(self):
        """Test if pytesseract can import with found path"""
        if not self.tesseract_path:
            return False
        
        try:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_path
            
            # Test with simple image
            test_img = np.ones((100, 100, 3), dtype=np.uint8) * 255
            cv2.putText(test_img, "TEST", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            result = pytesseract.image_to_string(test_img)
            print(f"✅ Tesseract working! Test result: '{result.strip()}'")
            return True
            
        except Exception as e:
            print(f"❌ Tesseract import failed: {e}")
            return False
    
    def simple_image_test(self):
        """Test OCR on Price_Enter.png"""
        if not self.tesseract_path:
            print("❌ Cannot test - Tesseract path not found")
            return False
        
        print(f"\n📸 TESTING IMAGE: {self.image_path.name}")
        print("-" * 50)
        
        # Check if image exists
        if not self.image_path.exists():
            print(f"❌ Image not found: {self.image_path}")
            return False
        
        try:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_path
            
            # Load image
            image = cv2.imread(str(self.image_path))
            if image is None:
                print(f"❌ Cannot load image: {self.image_path}")
                return False
            
            print(f"✅ Image loaded: {image.shape}")
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Extract text
            print("\n🔍 EXTRACTING TEXT...")
            text = pytesseract.image_to_string(gray)
            
            print("Raw OCR Output:")
            print("-" * 30)
            print(text)
            print("-" * 30)
            
            # Look for expected values
            expected_weights = ["14475", "16664", "11532"]
            weights_found = []
            
            for weight in expected_weights:
                if weight in text:
                    weights_found.append(weight)
                    print(f"✅ Found weight: {weight}")
            
            # Look for prices
            prices_found = []
            lines = text.split('\n')
            for line in lines:
                if '$' in line or '1.0000' in line:
                    prices_found.append(line.strip())
                    print(f"💰 Found price line: {line.strip()}")
            
            # Summary
            print(f"\n📊 RESULTS:")
            print(f"Expected weights found: {len(weights_found)}/3")
            print(f"Price lines detected: {len(prices_found)}")
            
            success = len(weights_found) >= 2  # At least 2 weights found
            print(f"🎯 TEST RESULT: {'✅ PASSED' if success else '❌ FAILED'}")
            
            return success
            
        except Exception as e:
            print(f"❌ OCR test failed: {e}")
            return False
    
    def run_complete_test(self):
        """Run complete test sequence"""
        print("🚀 TESSERACT PATH FINDER AND IMAGE TEST")
        print("=" * 60)
        
        # Step 1: Find Tesseract
        if not self.find_tesseract():
            return False
        
        # Step 2: Test Tesseract import
        print(f"\n🔧 TESTING TESSERACT IMPORT...")
        if not self.test_tesseract_import():
            return False
        
        # Step 3: Test image OCR
        success = self.simple_image_test()
        
        print("\n" + "=" * 60)
        if success:
            print("🎉 ALL TESTS PASSED!")
            print(f"✅ Tesseract found at: {self.tesseract_path}")
            print("✅ Image OCR working correctly")
            print("\n💡 USE THIS PATH IN YOUR CODE:")
            print(f"pytesseract.pytesseract.tesseract_cmd = r'{self.tesseract_path}'")
        else:
            print("❌ TESTS FAILED!")
            print("Check the error messages above for troubleshooting")
        
        return success

def quick_fix():
    """Quick fix function to find and test Tesseract"""
    finder = TesseractPathFinder()
    return finder.run_complete_test()

if __name__ == "__main__":
    print("🔧 Finding Tesseract and testing your image...")
    
    try:
        success = quick_fix()
        
        if success:
            print(f"\n✅ SUCCESS: Ready to use OCR!")
        else:
            print(f"\n❌ FAILED: Check installation steps above")
            
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
    
    input("\nPress Enter to exit...")
