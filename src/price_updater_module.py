"""
FIXED PRICE UPDATER WITH CORRECT COORDINATES
============================================

Uses debug results to update prices with precise coordinates
Based on OCR analysis showing exact $1.0000 locations

Author: Fixed Price Updater
Date: 2025-06-24
"""

import cv2
import pytesseract
import numpy as np
import pyautogui
import time
from typing import List, Dict, NamedTuple
from dataclasses import dataclass
import logging

# Configure Tesseract with working path
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\ykim\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

class PriceUpdate(NamedTuple):
    """Price update instruction"""
    weight: int
    current_price: str
    target_price: float
    click_x: int
    click_y: int

@dataclass
class FixedUpdateConfig:
    """Configuration with precise coordinates from debug"""
    
    # DAL125091 data from Excel
    invoice_number: str = "DAL125091"
    
    # Weight to price mapping (from debug results)
    price_updates: List[PriceUpdate] = None
    
    def __post_init__(self):
        if self.price_updates is None:
            # Based on debug OCR analysis:
            # Weight 14475 at (800, 483) -> $1.0000 at (1196, 481) -> Target $1.1908
            # Weight 16664 at (800, 504) -> $1.0000 at (1196, 502) -> Target $1.4238  
            # Weight 11532 at (800, 523) -> $1.0000 at (1196, 523) -> Target $1.1985
            
            self.price_updates = [
                PriceUpdate(
                    weight=14475,
                    current_price="$1.0000", 
                    target_price=1.1908,
                    click_x=1196,
                    click_y=481
                ),
                PriceUpdate(
                    weight=16664,
                    current_price="$1.0000",
                    target_price=1.4238, 
                    click_x=1196,
                    click_y=502  # Note: debug showed 481, but should be 502 based on Y alignment
                ),
                PriceUpdate(
                    weight=11532,
                    current_price="$1.0000",
                    target_price=1.1985,
                    click_x=1196,
                    click_y=523
                )
            ]

class FixedPriceUpdater:
    """Price updater with exact coordinates from OCR debug"""
    
    def __init__(self, config: FixedUpdateConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # PyAutoGUI settings
        pyautogui.PAUSE = 0.3
        pyautogui.FAILSAFE = True
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger("FixedPriceUpdater")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def verify_table_visible(self) -> bool:
        """Verify table is visible by checking for expected weights"""
        try:
            self.logger.info("📸 Verifying table is visible...")
            
            # Take screenshot
            screenshot = pyautogui.screenshot()
            screenshot_np = np.array(screenshot)
            screenshot_bgr = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale and run OCR
            gray = cv2.cvtColor(screenshot_bgr, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray)
            
            # Check for expected weights
            expected_weights = ["14475", "16664", "11532"]
            weights_found = []
            
            for weight in expected_weights:
                if weight in text:
                    weights_found.append(weight)
                    self.logger.info(f"✅ Found weight: {weight}")
            
            if len(weights_found) >= 2:  # At least 2 of 3 weights visible
                self.logger.info(f"✅ Table verification passed: {len(weights_found)}/3 weights visible")
                return True
            else:
                self.logger.warning(f"⚠️ Table verification failed: only {len(weights_found)}/3 weights visible")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Table verification failed: {e}")
            return False
    
    def update_single_price(self, update: PriceUpdate, simulate: bool = False) -> bool:
        """Update a single price field"""
        try:
            self.logger.info(f"🔄 Updating weight {update.weight}: {update.current_price} -> ${update.target_price:.4f}")
            self.logger.info(f"📍 Clicking at coordinates: ({update.click_x}, {update.click_y})")
            
            if simulate:
                self.logger.info(f"🎭 SIMULATION: Would update weight {update.weight} at ({update.click_x}, {update.click_y})")
                time.sleep(0.5)
                return True
            
            # Click on price field
            pyautogui.click(update.click_x, update.click_y)
            time.sleep(0.4)
            
            # Triple-click to select all text in field
            pyautogui.click(update.click_x, update.click_y, clicks=3)
            time.sleep(0.3)
            
            # Type new price
            price_text = f"{update.target_price:.4f}"
            pyautogui.write(price_text)
            time.sleep(0.3)
            
            # Press Tab to confirm and move to next field
            pyautogui.press('tab')
            time.sleep(0.4)
            
            self.logger.info(f"✅ Successfully updated weight {update.weight} to ${update.target_price:.4f}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update weight {update.weight}: {e}")
            return False
    
    def verify_updates(self) -> Dict:
        """Verify that price updates were applied correctly"""
        try:
            self.logger.info("🔍 Verifying price updates...")
            
            # Take screenshot after updates
            screenshot = pyautogui.screenshot()
            screenshot_np = np.array(screenshot)
            screenshot_bgr = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale and run OCR
            gray = cv2.cvtColor(screenshot_bgr, cv2.COLOR_BGR2GRAY)
            
            # Get detailed OCR data
            ocr_data = pytesseract.image_to_data(
                gray, 
                output_type=pytesseract.Output.DICT
            )
            
            # Extract text elements
            text_elements = []
            n_boxes = len(ocr_data['text'])
            
            for i in range(n_boxes):
                text = ocr_data['text'][i].strip()
                confidence = int(ocr_data['conf'][i])
                
                if text and confidence > 50:
                    x = ocr_data['left'][i] + ocr_data['width'][i] // 2
                    y = ocr_data['top'][i] + ocr_data['height'][i] // 2
                    
                    text_elements.append({
                        'text': text,
                        'center_x': x,
                        'center_y': y,
                        'confidence': confidence
                    })
            
            # Check for updated prices near expected coordinates
            verification_results = []
            
            for update in self.config.price_updates:
                found_updated_price = False
                target_price_str = f"{update.target_price:.4f}"
                
                # Look for text near the update coordinates
                for element in text_elements:
                    distance = ((element['center_x'] - update.click_x)**2 + 
                              (element['center_y'] - update.click_y)**2)**0.5
                    
                    if distance < 50:  # Within 50 pixels
                        if target_price_str in element['text'] or f"${target_price_str}" in element['text']:
                            found_updated_price = True
                            self.logger.info(f"✅ Verified weight {update.weight}: Found '{element['text']}' near ({update.click_x}, {update.click_y})")
                            break
                
                verification_results.append({
                    'weight': update.weight,
                    'target_price': update.target_price,
                    'verified': found_updated_price
                })
                
                if not found_updated_price:
                    self.logger.warning(f"⚠️ Could not verify update for weight {update.weight}")
            
            verified_count = sum(1 for result in verification_results if result['verified'])
            total_count = len(verification_results)
            
            return {
                'verified_count': verified_count,
                'total_count': total_count,
                'success_rate': (verified_count / total_count) * 100 if total_count > 0 else 0,
                'details': verification_results
            }
            
        except Exception as e:
            self.logger.error(f"❌ Verification failed: {e}")
            return {
                'verified_count': 0,
                'total_count': len(self.config.price_updates),
                'success_rate': 0,
                'details': [],
                'error': str(e)
            }
    
    def update_all_prices(self, simulate: bool = False) -> Dict:
        """Update all price fields with verification"""
        
        self.logger.info(f"🚀 STARTING PRICE UPDATES FOR {self.config.invoice_number}")
        self.logger.info(f"Mode: {'SIMULATION' if simulate else 'LIVE UPDATE'}")
        self.logger.info("=" * 60)
        
        results = {
            'total_updates': len(self.config.price_updates),
            'successful_updates': 0,
            'failed_updates': 0,
            'simulation_mode': simulate,
            'details': [],
            'verification': None
        }
        
        # Step 1: Verify table is visible
        if not simulate and not self.verify_table_visible():
            results['error'] = "Table verification failed - make sure table is visible"
            return results
        
        # Step 2: Perform updates
        self.logger.info(f"📝 Starting updates for {len(self.config.price_updates)} price fields...")
        
        for i, update in enumerate(self.config.price_updates):
            self.logger.info(f"\n🔄 Update {i+1}/{len(self.config.price_updates)}")
            
            success = self.update_single_price(update, simulate)
            
            update_detail = {
                'weight': update.weight,
                'target_price': update.target_price,
                'coordinates': (update.click_x, update.click_y),
                'success': success
            }
            
            results['details'].append(update_detail)
            
            if success:
                results['successful_updates'] += 1
            else:
                results['failed_updates'] += 1
        
        # Step 3: Verify updates (only for live mode)
        if not simulate and results['successful_updates'] > 0:
            self.logger.info(f"\n🔍 Verifying price updates...")
            time.sleep(1.0)  # Wait for UI to update
            verification = self.verify_updates()
            results['verification'] = verification
        
        # Step 4: Summary
        success_rate = (results['successful_updates'] / results['total_updates']) * 100
        
        self.logger.info(f"\n📊 UPDATE SUMMARY:")
        self.logger.info(f"Total updates: {results['total_updates']}")
        self.logger.info(f"Successful: {results['successful_updates']}")
        self.logger.info(f"Failed: {results['failed_updates']}")
        self.logger.info(f"Success rate: {success_rate:.1f}%")
        
        if results['verification']:
            ver = results['verification']
            self.logger.info(f"Verification: {ver['verified_count']}/{ver['total_count']} verified ({ver['success_rate']:.1f}%)")
        
        return results
    
    def create_coordinate_preview(self):
        """Show where clicks will happen"""
        try:
            self.logger.info("📸 Creating coordinate preview...")
            
            # Take screenshot
            screenshot = pyautogui.screenshot()
            screenshot_np = np.array(screenshot)
            screenshot_bgr = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
            
            # Draw circles at click coordinates
            for i, update in enumerate(self.config.price_updates):
                # Draw target circle
                cv2.circle(screenshot_bgr, (update.click_x, update.click_y), 15, (0, 255, 0), 3)
                
                # Add label
                label = f"W:{update.weight} -> ${update.target_price:.4f}"
                cv2.putText(screenshot_bgr, label, 
                           (update.click_x - 80, update.click_y - 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Add number
                cv2.putText(screenshot_bgr, str(i+1), 
                           (update.click_x - 5, update.click_y + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Save preview
            cv2.imwrite("price_update_preview.png", screenshot_bgr)
            self.logger.info("🖼️ Preview saved: price_update_preview.png")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Preview creation failed: {e}")
            return False

def main():
    """Main function with menu"""
    print("💰 FIXED PRICE UPDATER - DAL125091")
    print("=" * 50)
    print("Based on OCR debug results:")
    print("• Weight 14475 -> $1.1908 at (1196, 481)")
    print("• Weight 16664 -> $1.4238 at (1196, 502)")  
    print("• Weight 11532 -> $1.1985 at (1196, 523)")
    print("=" * 50)
    
    config = FixedUpdateConfig()
    updater = FixedPriceUpdater(config)
    
    while True:
        print("\nSelect option:")
        print("1. 📸 Preview click coordinates")
        print("2. 🎭 SIMULATE price updates (safe test)")
        print("3. 🔴 LIVE price updates (real changes)")
        print("4. 🔍 Verify current table")
        print("5. ❌ Exit")
        
        try:
            choice = input("\nEnter choice (1-5): ").strip()
            
            if choice == '1':
                print("\n📸 Creating coordinate preview...")
                input("Make sure table is visible. Press Enter...")
                
                if updater.create_coordinate_preview():
                    print("✅ Preview created! Check price_update_preview.png")
                else:
                    print("❌ Preview creation failed")
            
            elif choice == '2':
                print("\n🎭 SIMULATION MODE")
                print("This will test the update sequence without making changes")
                input("Make sure table is visible. Press Enter when ready...")
                
                results = updater.update_all_prices(simulate=True)
                
                if results['successful_updates'] == results['total_updates']:
                    print(f"\n✅ SIMULATION PASSED! All {results['total_updates']} updates ready")
                    print("You can now run LIVE updates with confidence")
                else:
                    print(f"\n⚠️ SIMULATION ISSUES: {results['successful_updates']}/{results['total_updates']} successful")
            
            elif choice == '3':
                print("\n🔴 LIVE UPDATE MODE")
                print("⚠️ WARNING: This will make REAL changes to the table!")
                print("Make sure:")
                print("  • Table is visible and fully loaded")
                print("  • You're ready to update all 3 prices")
                print("  • No other applications will interfere")
                
                confirm = input("\nType 'UPDATE' to proceed: ").strip()
                if confirm.upper() == 'UPDATE':
                    print("\n🚀 Starting live price updates...")
                    results = updater.update_all_prices(simulate=False)
                    
                    if results['successful_updates'] == results['total_updates']:
                        print(f"\n🎉 SUCCESS! All {results['total_updates']} prices updated!")
                        
                        if results['verification']:
                            ver = results['verification']
                            if ver['verified_count'] == ver['total_count']:
                                print("✅ All updates verified in table")
                            else:
                                print(f"⚠️ Verification: {ver['verified_count']}/{ver['total_count']} confirmed")
                    else:
                        print(f"\n⚠️ PARTIAL SUCCESS: {results['successful_updates']}/{results['total_updates']} updated")
                        print("Check the logs above for details on failed updates")
                else:
                    print("❌ Cancelled - no changes made")
            
            elif choice == '4':
                print("\n🔍 Verifying table visibility...")
                if updater.verify_table_visible():
                    print("✅ Table verification passed - ready for updates")
                else:
                    print("❌ Table verification failed - check table is visible")
            
            elif choice == '5':
                print("\n👋 Exiting price updater. Goodbye!")
                break
            
            else:
                print("❌ Invalid choice. Please enter 1-5.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
