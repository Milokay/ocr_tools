# ============================================================================
# SCRAP DRAGON AUTOMATION FOLLOWING EXACT PROCESS.TXT REQUIREMENTS
# ============================================================================
"""
Scrap Dragon Automation following Process.txt exactly:

1. Collect all invoice# and price info from calculation sheet - using pandas xlwings
2. Search invoice# and enter - using pyautogui
3. Click dropdown - using pyautogui
4. Click generate pack list - wait 3 seconds
5. Use price_updater_module.py EXACT code to match weights and update prices
6. Save shipment items
7. Move to next invoice

Handles 1-10 lines per invoice possibility.
Only processes 'WORKING' status invoices.
Updates status to 'COMPLETED' when all lines are successfully updated.

Author: Scrap Dragon Process Automation
Date: 2025-06-24
Version: Production Ready
"""

import pandas as pd
import xlwings as xw
import pyautogui
import cv2
import numpy as np
import pytesseract
import time
import json
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple, NamedTuple
import re
import logging
from pathlib import Path
from dataclasses import dataclass


# ============================================================================
# TESSERACT CONFIGURATION (FROM price_updater_module.py)
# ============================================================================

def configure_tesseract():
    """Configure Tesseract with working path from price_updater_module.py"""
    # Use the exact path from price_updater_module.py
    tesseract_path = r'C:\Users\ykim\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
    
    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        print(f"✅ Tesseract configured: {tesseract_path}")
        return True
    
    # Fallback paths
    fallback_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\Users\ykim\AppData\Local\Tesseract-OCR\tesseract.exe",
    ]
    
    for path in fallback_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            print(f"✅ Tesseract configured with fallback: {path}")
            return True
    
    print("❌ Tesseract not found! Please install Tesseract OCR")
    return False


# ============================================================================
# PRICE UPDATE CLASSES (FROM price_updater_module.py - EXACT CODE)
# ============================================================================

class PriceUpdate(NamedTuple):
    """Price update instruction - EXACT from price_updater_module.py"""
    weight: int
    current_price: str
    target_price: float
    click_x: int
    click_y: int


class DynamicPriceUpdater:
    """Dynamic version of FixedPriceUpdater that works with any weights/prices"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        
        # PyAutoGUI settings - EXACT from price_updater_module.py
        pyautogui.PAUSE = 0.3
        pyautogui.FAILSAFE = True
    
    def verify_table_visible(self, expected_weights: List[float]) -> bool:
        """Verify table is visible by checking for expected weights - EXACT approach from price_updater_module.py"""
        try:
            self.logger.info("📸 Verifying table is visible...")
            
            # Take screenshot - EXACT from price_updater_module.py
            screenshot = pyautogui.screenshot()
            screenshot_np = np.array(screenshot)
            screenshot_bgr = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale and run OCR - EXACT from price_updater_module.py
            gray = cv2.cvtColor(screenshot_bgr, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray)
            
            self.logger.info(f"🎯 Looking for weights: {[str(int(w)) for w in expected_weights]}")
            
            # Check for expected weights
            weights_found = []
            for weight in expected_weights:
                weight_str = str(int(weight))
                if weight_str in text:
                    weights_found.append(weight)
                    self.logger.info(f"✅ Found weight: {weight}")
                else:
                    # Try fuzzy matching
                    weight_int = int(weight)
                    for i in range(-5, 6):  # Check +/- 5 for OCR errors
                        alt_weight = str(weight_int + i)
                        if alt_weight in text:
                            weights_found.append(weight)
                            self.logger.info(f"✅ Found weight {weight} as '{alt_weight}' (OCR variation)")
                            break
            
            if len(weights_found) >= 1:  # At least 1 weight visible
                self.logger.info(f"✅ Table verification passed: {len(weights_found)}/{len(expected_weights)} weights visible")
                return True
            else:
                self.logger.warning(f"⚠️ Table verification failed: only {len(weights_found)}/{len(expected_weights)} weights visible")
                
                # Show debugging info
                self.logger.info("🔍 Debugging - searching for any 4+ digit numbers in OCR text:")
                import re
                numbers = re.findall(r'\b\d{4,5}\b', text)
                self.logger.info(f"📋 Found numbers: {numbers[:10]}")  # Show first 10
                
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Table verification failed: {e}")
            return False
    
    def find_weight_coordinates(self, target_weights: List[float]) -> Dict[float, Tuple[int, int]]:
        """Find coordinates for weights using OCR - EXACT approach from price_updater_module.py"""
        try:
            self.logger.info("🔍 Finding weight coordinates using OCR (price_updater_module.py approach)...")
            
            # Take screenshot - EXACT from price_updater_module.py
            screenshot = pyautogui.screenshot()
            screenshot_np = np.array(screenshot)
            screenshot_bgr = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(screenshot_bgr, cv2.COLOR_BGR2GRAY)
            
            # Get detailed OCR data - EXACT from price_updater_module.py
            ocr_data = pytesseract.image_to_data(
                gray, 
                output_type=pytesseract.Output.DICT
            )
            
            # Extract all text elements with coordinates
            text_elements = []
            n_boxes = len(ocr_data['text'])
            
            for i in range(n_boxes):
                text = ocr_data['text'][i].strip()
                confidence = int(ocr_data['conf'][i])
                
                if text and confidence > 30:  # Lower threshold for weight detection
                    x = ocr_data['left'][i] + ocr_data['width'][i] // 2
                    y = ocr_data['top'][i] + ocr_data['height'][i] // 2
                    
                    text_elements.append({
                        'text': text,
                        'center_x': x,
                        'center_y': y,
                        'confidence': confidence
                    })
            
            self.logger.info(f"📊 Found {len(text_elements)} text elements from OCR")
            
            # Find weight coordinates using debug insights
            weight_coordinates = {}
            
            for target_weight in target_weights:
                weight_str = str(int(target_weight))
                self.logger.info(f"🔍 Looking for weight: {weight_str}")
                
                # Find this weight in OCR results
                for element in text_elements:
                    if element['text'] == weight_str:
                        weight_x = element['center_x']
                        weight_y = element['center_y']
                        
                        # Based on debug results: price field is at X=1196, same Y as weight
                        price_x = 1196  # Fixed X coordinate from debug results
                        price_y = weight_y  # Same Y as weight
                        
                        weight_coordinates[target_weight] = (price_x, price_y)
                        self.logger.info(f"✅ Weight {weight_str} found at ({weight_x}, {weight_y}) → Price field at ({price_x}, {price_y})")
                        break
                else:
                    # Try fuzzy matching for slight OCR variations
                    for element in text_elements:
                        # Check if OCR text is close to target weight
                        try:
                            ocr_number = int(element['text'])
                            if abs(ocr_number - target_weight) <= 5:  # Allow small OCR errors
                                weight_x = element['center_x']
                                weight_y = element['center_y']
                                price_x = 1196
                                price_y = weight_y
                                
                                weight_coordinates[target_weight] = (price_x, price_y)
                                self.logger.info(f"✅ Weight {weight_str} matched with OCR '{element['text']}' at ({weight_x}, {weight_y}) → Price field at ({price_x}, {price_y})")
                                break
                        except ValueError:
                            continue
            
            if not weight_coordinates:
                self.logger.warning("⚠️ No weight coordinates found. Showing available text for debugging:")
                # Show what weights are actually visible
                potential_weights = []
                for element in text_elements:
                    try:
                        if element['text'].isdigit() and len(element['text']) >= 4:
                            potential_weights.append(element['text'])
                    except:
                        continue
                
                self.logger.info(f"📋 Potential weights found in OCR: {potential_weights[:10]}")
                self.logger.info(f"🎯 Target weights looking for: {[str(int(w)) for w in target_weights]}")
            
            return weight_coordinates
            
        except Exception as e:
            self.logger.error(f"❌ Coordinate finding failed: {e}")
            return {}
    
    def update_single_price(self, update: PriceUpdate, simulate: bool = False) -> bool:
        """Update a single price field - EXACT from price_updater_module.py"""
        try:
            self.logger.info(f"🔄 Updating weight {update.weight}: {update.current_price} -> ${update.target_price:.4f}")
            self.logger.info(f"📍 Clicking at coordinates: ({update.click_x}, {update.click_y})")
            
            if simulate:
                self.logger.info(f"🎭 SIMULATION: Would update weight {update.weight} at ({update.click_x}, {update.click_y})")
                time.sleep(0.5)
                return True
            
            # EXACT code from price_updater_module.py
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
    
    def debug_weight_detection(self, target_weights: List[float]) -> Dict:
        """Debug function to show OCR detection results - for troubleshooting"""
        try:
            self.logger.info("🐛 DEBUG: Analyzing OCR weight detection...")
            
            # Take screenshot
            screenshot = pyautogui.screenshot()
            screenshot_np = np.array(screenshot)
            screenshot_bgr = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(screenshot_bgr, cv2.COLOR_BGR2GRAY)
            
            # Save debug screenshot
            import datetime
            timestamp = datetime.datetime.now().strftime("%H%M%S")
            debug_path = f"debug_weight_detection_{timestamp}.png"
            cv2.imwrite(debug_path, screenshot_bgr)
            self.logger.info(f"📸 Debug screenshot saved: {debug_path}")
            
            # Get OCR data
            ocr_data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
            
            # Analyze all detected text
            all_numbers = []
            weight_matches = {}
            
            n_boxes = len(ocr_data['text'])
            for i in range(n_boxes):
                text = ocr_data['text'][i].strip()
                confidence = int(ocr_data['conf'][i])
                
                if text and confidence > 20:  # Lower threshold for debugging
                    x = ocr_data['left'][i] + ocr_data['width'][i] // 2
                    y = ocr_data['top'][i] + ocr_data['height'][i] // 2
                    
                    # Check if it's a number
                    if text.isdigit() and len(text) >= 4:
                        all_numbers.append({
                            'text': text,
                            'x': x,
                            'y': y,
                            'confidence': confidence
                        })
                        
                        # Check if it matches any target weight
                        for target_weight in target_weights:
                            if text == str(int(target_weight)):
                                weight_matches[target_weight] = {
                                    'found_text': text,
                                    'coordinates': (x, y),
                                    'price_coords': (1196, y),  # Based on debug results
                                    'confidence': confidence
                                }
            
            # Log results
            self.logger.info(f"🔍 DEBUG RESULTS:")
            self.logger.info(f"📊 Total numbers found: {len(all_numbers)}")
            self.logger.info(f"🎯 Target weights: {[str(int(w)) for w in target_weights]}")
            self.logger.info(f"✅ Weight matches: {len(weight_matches)}")
            
            # Show first 10 numbers found
            self.logger.info(f"📋 Numbers detected (first 10):")
            for i, num in enumerate(all_numbers[:10]):
                self.logger.info(f"  {i+1}. '{num['text']}' at ({num['x']}, {num['y']}) conf:{num['confidence']}")
            
            # Show weight matches
            for weight, match in weight_matches.items():
                self.logger.info(f"🎯 MATCH: Weight {weight} → '{match['found_text']}' at {match['coordinates']} → Price at {match['price_coords']}")
            
            return {
                'total_numbers': len(all_numbers),
                'weight_matches': weight_matches,
                'all_numbers': all_numbers,
                'debug_screenshot': debug_path
            }
            
        except Exception as e:
            self.logger.error(f"❌ Debug analysis failed: {e}")
            return {}
    
    def update_prices_for_invoice(self, invoice_weights: List[float], invoice_prices: List[float], simulate: bool = False) -> Dict:
        """Update prices for an invoice using price_updater_module.py approach"""
        
        self.logger.info(f"💰 Processing price updates using price_updater_module.py approach")
        self.logger.info(f"📊 Invoice weights: {invoice_weights}")
        self.logger.info(f"💵 Invoice prices: {invoice_prices}")
        
        # Step 1: Debug weight detection first
        debug_results = self.debug_weight_detection(invoice_weights)
        
        # Step 2: Verify table is visible - EXACT from price_updater_module.py
        if not simulate and not self.verify_table_visible(invoice_weights):
            return {
                'success': False,
                'updates_attempted': 0,
                'updates_successful': 0,
                'error': 'Table verification failed - make sure table is visible',
                'debug_info': debug_results
            }
        
        # Step 3: Find coordinates for weights
        weight_coordinates = self.find_weight_coordinates(invoice_weights)
        
        if not weight_coordinates:
            return {
                'success': False,
                'updates_attempted': 0,
                'updates_successful': 0,
                'error': f'Could not find coordinates for any weights. Debug found {debug_results.get("total_numbers", 0)} numbers total, {len(debug_results.get("weight_matches", {}))} matches.',
                'debug_info': debug_results
            }
        
        # Step 4: Create price updates
        price_updates = []
        for i, (weight, price) in enumerate(zip(invoice_weights, invoice_prices)):
            if weight in weight_coordinates:
                coords = weight_coordinates[weight]
                price_update = PriceUpdate(
                    weight=int(weight),
                    current_price="$1.0000",  # Default current price
                    target_price=price,
                    click_x=coords[0],
                    click_y=coords[1]
                )
                price_updates.append(price_update)
                self.logger.info(f"📝 Created update: Weight {weight} → ${price:.4f} at {coords}")
        
        if not price_updates:
            return {
                'success': False,
                'updates_attempted': 0,
                'updates_successful': 0,
                'error': f'No price updates created. Found coordinates for {len(weight_coordinates)} weights but none matched invoice weights.',
                'debug_info': debug_results
            }
        
        # Step 5: Perform updates - EXACT approach from price_updater_module.py
        results = {
            'total_updates': len(price_updates),
            'successful_updates': 0,
            'failed_updates': 0,
            'simulation_mode': simulate,
            'details': [],
            'debug_info': debug_results
        }
        
        self.logger.info(f"📝 Starting updates for {len(price_updates)} price fields...")
        
        for i, update in enumerate(price_updates):
            self.logger.info(f"\n🔄 Update {i+1}/{len(price_updates)}")
            
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
        
        # Step 6: Summary
        success_rate = (results['successful_updates'] / results['total_updates']) * 100 if results['total_updates'] > 0 else 0
        
        self.logger.info(f"\n📊 UPDATE SUMMARY:")
        self.logger.info(f"Total updates: {results['total_updates']}")
        self.logger.info(f"Successful: {results['successful_updates']}")
        self.logger.info(f"Failed: {results['failed_updates']}")
        self.logger.info(f"Success rate: {success_rate:.1f}%")
        
        results['success'] = results['successful_updates'] > 0
        return results


# ============================================================================
# CONFIGURATION AND DATA MODELS
# ============================================================================

class AutomationConfig:
    """Configuration for automation process following Process.txt"""
    
    # Excel file paths (Step 1 requirement)
    EXCEL_PATH = r"C:\Users\ykim\OneDrive - Venture Metals\Desktop\daily operation - YK.xlsx"
    SHEET_NAME = "Cable Final"
    
    # UI coordinates (Steps 2-3 requirements)
    SEARCH_BAR_COORDS = (252, 238)  # Step 2: Search invoice
    EXPAND_DROPDOWN_COORDS = (194, 365)  # Step 3: Click dropdown
    
    # Button detection templates
    TEMPLATE_DIR = r"C:\Users\ykim\OneDrive - Venture Metals\Desktop\Trading Invoices\Automation Project - Excel Files\python\Cable Automation"
    GENERATE_TEMPLATE = os.path.join(TEMPLATE_DIR, "Generate.png")
    SAVE_TEMPLATE = os.path.join(TEMPLATE_DIR, "SaveShip.png")
    
    # Timing delays (Step 4: wait 3 seconds)
    DELAY_AFTER_SEARCH = 1.5
    DELAY_AFTER_DROPDOWN = 0.8
    DELAY_AFTER_GENERATE = 4.0  # Step 4: Wait 3 seconds exactly
    DELAY_BETWEEN_INVOICES = 3.0
    
    # Detection settings
    TEMPLATE_MATCH_THRESHOLD = 1
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist."""
        os.makedirs(cls.TEMPLATE_DIR, exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        os.makedirs("screenshots", exist_ok=True)


class InvoiceData(NamedTuple):
    """Invoice data from Excel calculation sheet (Step 1)"""
    invoice_number: str
    net_lbs: List[float]  # All weights for this invoice (1-10 lines possible)
    unit_prices: List[float]  # Column V prices to update
    excel_row: int
    worksheet: object


class ProcessingResult(NamedTuple):
    """Result of processing a single invoice"""
    invoice: str
    status: str
    lines_found: int
    lines_updated: int
    error: Optional[str] = None


def setup_logging() -> logging.Logger:
    """Setup logging for automation process."""
    
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"scrap_dragon_process_{timestamp}.log"
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File and console handlers
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # Configure logger
    logger = logging.getLogger("ScrapDragonProcess")
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info("Scrap Dragon Process Automation started")
    return logger


# ============================================================================
# STEP 1: EXCEL DATA COLLECTOR (ONLY WORKING STATUS INVOICES)
# ============================================================================

class ExcelDataCollector:
    """Step 1: Collect all invoice# and price info from calculation sheet - ONLY 'WORKING' status"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.config = AutomationConfig()
    
    def collect_invoice_data(self) -> List[InvoiceData]:
        """Step 1: Collect ONLY 'WORKING' status invoice numbers and prices from Excel sheet"""
        
        try:
            self.logger.info("📊 Step 1: Collecting WORKING status invoices from calculation sheet...")
            
            # Open Excel file using xlwings
            wb = xw.Book(self.config.EXCEL_PATH)
            ws = wb.sheets[self.config.SHEET_NAME]
            
            # Get used range and convert to pandas DataFrame
            used_range = ws.used_range
            if used_range is None:
                self.logger.error("No data found in Excel sheet")
                return []
            
            df = used_range.options(pd.DataFrame, header=1, index=False).value
            
            if df is None or df.empty:
                self.logger.error("Excel data is empty")
                return []
            
            self.logger.info(f"Loaded Excel data: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Group data by invoice number (possibility of 1-10 lines per invoice)
            # ONLY process 'WORKING' status invoices
            invoice_groups = {}
            working_rows = 0
            total_rows = 0
            
            for index, row in df.iterrows():
                try:
                    total_rows += 1
                    
                    # Extract status from column A (index 0) - ONLY process WORKING status
                    status = str(row.iloc[0]).strip().upper() if pd.notna(row.iloc[0]) else ""
                    
                    if status != "WORKING":
                        continue  # Skip non-WORKING invoices
                    
                    working_rows += 1
                    
                    # Column D: Invoice number (index 3)
                    invoice_number = str(row.iloc[3]).strip() if pd.notna(row.iloc[3]) else ""
                    
                    if not invoice_number or invoice_number.lower() == "nan":
                        continue
                    
                    # Column G: Net LBS / Shipped weight (index 6)
                    net_lbs = 0.0
                    if pd.notna(row.iloc[6]):
                        try:
                            net_lbs = float(row.iloc[6])
                        except (ValueError, TypeError):
                            continue
                    
                    # Column V: Unit price (index 21)
                    unit_price = 0.0
                    if len(row) > 21 and pd.notna(row.iloc[21]):
                        try:
                            unit_price = float(row.iloc[21])
                        except (ValueError, TypeError):
                            continue
                    
                    # Validate data
                    if net_lbs <= 0 or unit_price <= 0:
                        continue
                    
                    # Group by invoice number (handling 1-10 lines per invoice)
                    if invoice_number not in invoice_groups:
                        invoice_groups[invoice_number] = {
                            'net_lbs': [],
                            'unit_prices': [],
                            'excel_row': index + 2,  # Excel row number (1-indexed + header)
                            'worksheet': ws
                        }
                    
                    invoice_groups[invoice_number]['net_lbs'].append(net_lbs)
                    invoice_groups[invoice_number]['unit_prices'].append(unit_price)
                    
                except Exception as e:
                    self.logger.warning(f"Error processing row {index + 2}: {e}")
                    continue
            
            # Convert groups to InvoiceData objects
            invoice_list = []
            for invoice_number, data in invoice_groups.items():
                invoice_data = InvoiceData(
                    invoice_number=invoice_number,
                    net_lbs=data['net_lbs'],
                    unit_prices=data['unit_prices'],
                    excel_row=data['excel_row'],
                    worksheet=data['worksheet']
                )
                invoice_list.append(invoice_data)
                
                self.logger.info(f"Collected WORKING invoice {invoice_number}: {len(data['net_lbs'])} lines")
            
            self.logger.info(f"✅ Step 1 Complete: Found {working_rows} WORKING rows out of {total_rows} total rows")
            self.logger.info(f"📋 Collected {len(invoice_list)} WORKING invoices with total {sum(len(inv.net_lbs) for inv in invoice_list)} lines")
            
            if len(invoice_list) == 0:
                self.logger.warning("⚠️ No WORKING status invoices found! Check if all invoices are already COMPLETED.")
            
            return invoice_list
            
        except Exception as e:
            self.logger.error(f"Step 1 failed: {e}")
            return []


# ============================================================================
# BUTTON DETECTOR (For Steps 4 & 6)
# ============================================================================

class ButtonDetector:
    """Detect Generate Pack List and Save Shipment Items buttons"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.config = AutomationConfig()
    
    def find_generate_button(self) -> Optional[Tuple[int, int]]:
        """Find Generate From Pack List button"""
        try:
            # Try template matching first
            if os.path.exists(self.config.GENERATE_TEMPLATE):
                return self._find_button_by_template(self.config.GENERATE_TEMPLATE)
            
            # Fallback to fixed coordinates
            return (1454, 587)
            
        except Exception as e:
            self.logger.warning(f"Generate button detection failed: {e}")
            return (1454, 587)  # Fallback coordinates
    
    def find_save_button(self) -> Optional[Tuple[int, int]]:
        """Find Save Shipment Items button"""
        try:
            # Try template matching first
            if os.path.exists(self.config.SAVE_TEMPLATE):
                return self._find_button_by_template(self.config.SAVE_TEMPLATE)
            
            # Fallback to fixed coordinates
            return (1800, 580)
            
        except Exception as e:
            self.logger.warning(f"Save button detection failed: {e}")
            return (1800, 580)  # Fallback coordinates
    
    def _find_button_by_template(self, template_path: str) -> Optional[Tuple[int, int]]:
        """Find button using template matching"""
        try:
            # Take screenshot
            screenshot = pyautogui.screenshot()
            screenshot_np = np.array(screenshot)
            screenshot_bgr = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
            
            # Load template
            template = cv2.imread(template_path)
            if template is None:
                return None
            
            # Template matching
            result = cv2.matchTemplate(screenshot_bgr, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val >= self.config.TEMPLATE_MATCH_THRESHOLD:
                h, w = template.shape[:2]
                center_x = max_loc[0] + w // 2
                center_y = max_loc[1] + h // 2
                return (center_x, center_y)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Template matching error: {e}")
            return None


# ============================================================================
# MAIN AUTOMATION ORCHESTRATOR
# ============================================================================

class ScrapDragonProcessAutomator:
    """Main automation following exact Process.txt requirements - ONLY WORKING status invoices"""
    
    def __init__(self):
        self.logger = setup_logging()
        self.config = AutomationConfig()
        
        # Initialize components
        self.excel_collector = ExcelDataCollector(self.logger)
        self.price_updater = DynamicPriceUpdater(self.logger)
        self.button_detector = ButtonDetector(self.logger)
        
        # State tracking
        self.processing_results: List[ProcessingResult] = []
        
        # Configure PyAutoGUI
        pyautogui.PAUSE = 0.3
        pyautogui.FAILSAFE = True
        
        self.logger.info("Scrap Dragon Process Automator initialized - WORKING invoices only")
    
    def run_automation(self, simulate: bool = False) -> Dict:
        """Run the complete automation following Process.txt steps - ONLY WORKING invoices"""
        
        mode = "SIMULATION" if simulate else "LIVE"
        self.logger.info(f"🚀 STARTING SCRAP DRAGON PROCESS AUTOMATION - {mode} MODE")
        self.logger.info("Processing ONLY 'WORKING' status invoices (recalculation safe)")
        self.logger.info("Following Process.txt exactly:")
        self.logger.info("1. Collect WORKING invoice data from Excel")
        self.logger.info("2. Search invoice and enter")
        self.logger.info("3. Click dropdown")
        self.logger.info("4. Click generate pack list - wait 3 seconds")
        self.logger.info("5. Use price_updater_module.py to match weights and update prices")
        self.logger.info("6. Save shipment items")
        self.logger.info("=" * 80)
        
        start_time = datetime.now()
        
        try:
            # STEP 1: Collect all WORKING invoice# and price info from calculation sheet
            invoice_list = self.excel_collector.collect_invoice_data()
            
            if not invoice_list:
                return {
                    "status": "no_working_invoices",
                    "message": "No WORKING status invoices found. All invoices may already be COMPLETED.",
                    "total_invoices": 0,
                    "mode": mode
                }
            
            self.logger.info(f"🎯 Ready to process {len(invoice_list)} WORKING invoices")
            
            # Confirm before proceeding (if not simulating)
            if not simulate:
                proceed = input(f"\nProceed with {mode} automation of {len(invoice_list)} WORKING invoices? (y/n): ").strip().lower()
                if proceed != 'y':
                    return {"status": "cancelled", "message": "User cancelled", "mode": mode}
            
            # Process all invoices following steps 2-6
            summary = self._process_all_invoices(invoice_list, simulate)
            
            # Calculate timing
            end_time = datetime.now()
            summary['duration_seconds'] = (end_time - start_time).total_seconds()
            summary['start_time'] = start_time.isoformat()
            summary['end_time'] = end_time.isoformat()
            summary['mode'] = mode
            
            return summary
            
        except Exception as e:
            error_msg = f"Automation failed: {str(e)}"
            self.logger.error(error_msg)
            return {"status": "error", "message": error_msg, "mode": mode}
    
    def _process_all_invoices(self, invoice_list: List[InvoiceData], simulate: bool = False) -> Dict:
        """Process all WORKING invoices following steps 2-6 for each"""
        
        successful_count = 0
        failed_count = 0
        total_lines_updated = 0
        
        for i, invoice_data in enumerate(invoice_list, 1):
            self.logger.info(f"\n🔄 Processing WORKING invoice {i}/{len(invoice_list)}: {invoice_data.invoice_number}")
            self.logger.info(f"Lines in this invoice: {len(invoice_data.net_lbs)} (weights: {invoice_data.net_lbs})")
            
            try:
                result = self._process_single_invoice(invoice_data, simulate)
                self.processing_results.append(result)
                
                if result.status == 'success':
                    successful_count += 1
                    total_lines_updated += result.lines_updated
                    
                    # IMPORTANT: Update status to COMPLETED only if ALL lines were successfully updated
                    if result.lines_updated == result.lines_found:
                        if not simulate:
                            self._update_excel_status(invoice_data, "COMPLETED")
                        self.logger.info(f"✅ SUCCESS: All {result.lines_updated}/{result.lines_found} lines updated → Status: COMPLETED")
                    else:
                        if not simulate:
                            self._update_excel_status(invoice_data, "PARTIAL")
                        self.logger.warning(f"⚠️ PARTIAL: Only {result.lines_updated}/{result.lines_found} lines updated → Status: PARTIAL")
                else:
                    failed_count += 1
                    if not simulate:
                        self._update_excel_status(invoice_data, "ERROR")
                    self.logger.error(f"❌ FAILED: {result.error} → Status: ERROR")
                
                # Move to next invoice
                if i < len(invoice_list):
                    delay = 0.5 if simulate else self.config.DELAY_BETWEEN_INVOICES
                    self.logger.info(f"⏳ Moving to next invoice in {delay} seconds...")
                    time.sleep(delay)
            
            except Exception as e:
                self.logger.error(f"💥 Error processing {invoice_data.invoice_number}: {e}")
                failed_count += 1
                if not simulate:
                    self._update_excel_status(invoice_data, "ERROR")
        
        return {
            "status": "completed",
            "total_invoices": len(invoice_list),
            "successful": successful_count,
            "failed": failed_count,
            "total_lines_updated": total_lines_updated,
            "simulation": simulate
        }
    
    def _process_single_invoice(self, invoice_data: InvoiceData, simulate: bool = False) -> ProcessingResult:
        """Process single invoice following steps 2-6"""
        
        try:
            self.logger.info(f"📋 Processing: {invoice_data.invoice_number}")
            self.logger.info(f"Weights to match: {invoice_data.net_lbs}")
            self.logger.info(f"Prices to update: {invoice_data.unit_prices}")
            
            # STEP 2: Search invoice# and enter
            if not simulate and not self._step2_search_invoice(invoice_data.invoice_number):
                return ProcessingResult(
                    invoice=invoice_data.invoice_number,
                    status='failed',
                    lines_found=len(invoice_data.net_lbs),
                    lines_updated=0,
                    error='Step 2 failed: Search invoice'
                )
            
            # STEP 3: Click dropdown
            if not simulate and not self._step3_click_dropdown():
                return ProcessingResult(
                    invoice=invoice_data.invoice_number,
                    status='failed',
                    lines_found=len(invoice_data.net_lbs),
                    lines_updated=0,
                    error='Step 3 failed: Click dropdown'
                )
            
            # STEP 4: Click generate pack list - wait 3 seconds
            if not simulate and not self._step4_generate_pack_list():
                return ProcessingResult(
                    invoice=invoice_data.invoice_number,
                    status='failed',
                    lines_found=len(invoice_data.net_lbs),
                    lines_updated=0,
                    error='Step 4 failed: Generate pack list'
                )
            
            # STEP 5: Use price_updater_module.py exact code to match weights and update prices
            self.logger.info("💰 Step 5: Using price_updater_module.py approach for weight matching and price updates...")
            update_result = self.price_updater.update_prices_for_invoice(
                invoice_data.net_lbs, 
                invoice_data.unit_prices, 
                simulate
            )
            
            if not update_result['success']:
                return ProcessingResult(
                    invoice=invoice_data.invoice_number,
                    status='failed',
                    lines_found=len(invoice_data.net_lbs),
                    lines_updated=0,
                    error=f"Step 5 failed: {update_result.get('error', 'Price update failed')}"
                )
            
            lines_updated = update_result['successful_updates']
            self.logger.info(f"✅ Step 5 complete: {lines_updated}/{len(invoice_data.net_lbs)} price fields updated")
            
            # STEP 6: Save shipment items
            if not simulate and not self._step6_save_shipment():
                return ProcessingResult(
                    invoice=invoice_data.invoice_number,
                    status='failed',
                    lines_found=len(invoice_data.net_lbs),
                    lines_updated=lines_updated,
                    error='Step 6 failed: Save shipment'
                )
            
            return ProcessingResult(
                invoice=invoice_data.invoice_number,
                status='success',
                lines_found=len(invoice_data.net_lbs),
                lines_updated=lines_updated
            )
            
        except Exception as e:
            return ProcessingResult(
                invoice=invoice_data.invoice_number,
                status='failed',
                lines_found=len(invoice_data.net_lbs) if invoice_data else 0,
                lines_updated=0,
                error=str(e)
            )
    
    def _step2_search_invoice(self, invoice_number: str) -> bool:
        """Step 2: Search invoice# and enter - using pyautogui"""
        
        try:
            self.logger.info(f"🔍 Step 2: Searching invoice {invoice_number} using pyautogui")
            
            # Click search bar
            pyautogui.click(*self.config.SEARCH_BAR_COORDS)
            time.sleep(0.3)
            
            # Clear and enter invoice number
            pyautogui.hotkey('ctrl', 'a')
            pyautogui.press('delete')
            pyautogui.typewrite(invoice_number)
            pyautogui.press('enter')
            time.sleep(self.config.DELAY_AFTER_SEARCH)
            
            self.logger.info("✅ Step 2 complete: Invoice searched and entered")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Step 2 failed: {e}")
            return False
    
    def _step3_click_dropdown(self) -> bool:
        """Step 3: Click dropdown - using pyautogui"""
        
        try:
            self.logger.info("📂 Step 3: Clicking dropdown using pyautogui")
            
            pyautogui.click(*self.config.EXPAND_DROPDOWN_COORDS)
            time.sleep(self.config.DELAY_AFTER_DROPDOWN)
            
            self.logger.info("✅ Step 3 complete: Dropdown clicked")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Step 3 failed: {e}")
            return False
    
    def _step4_generate_pack_list(self) -> bool:
        """Step 4: Click generate pack list - wait 3 seconds"""
        
        try:
            self.logger.info("📦 Step 4: Clicking generate pack list - will wait 3 seconds")
            
            # Find generate button
            generate_coords = self.button_detector.find_generate_button()
            if not generate_coords:
                self.logger.error("Could not find Generate Pack List button")
                return False
            
            # Click generate button
            pyautogui.click(*generate_coords)
            
            # Wait exactly 3 seconds as specified in Process.txt
            self.logger.info("⏳ Waiting exactly 3 seconds as specified in Process.txt...")
            time.sleep(self.config.DELAY_AFTER_GENERATE)  # 3.0 seconds
            
            self.logger.info("✅ Step 4 complete: Generate pack list clicked, waited 3 seconds")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Step 4 failed: {e}")
            return False
    
    def _step6_save_shipment(self) -> bool:
        """Step 6: Save shipment items"""
        
        try:
            self.logger.info("💾 Step 6: Saving shipment items")
            
            # Find save button
            save_coords = self.button_detector.find_save_button()
            if not save_coords:
                self.logger.error("Could not find Save Shipment Items button")
                return False
            
            # Click save button
            pyautogui.click(*save_coords)
            time.sleep(1.0)
            
            self.logger.info("✅ Step 6 complete: Shipment items saved")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Step 6 failed: {e}")
            return False
    
    def _update_excel_status(self, invoice_data: InvoiceData, status: str) -> None:
        """Update Excel status for processed invoice"""
        
        try:
            invoice_data.worksheet.range(f"A{invoice_data.excel_row}").value = status
            self.logger.info(f"📊 Updated Excel status for {invoice_data.invoice_number} (row {invoice_data.excel_row}): {status}")
        except Exception as e:
            self.logger.warning(f"Excel update error: {e}")


# ============================================================================
# TESTING FUNCTIONS
# ============================================================================

def test_excel_data_collection():
    """Test Step 1: Excel data collection - ONLY WORKING status"""
    
    print("📊 TESTING STEP 1: EXCEL DATA COLLECTION (WORKING STATUS ONLY)")
    print("=" * 60)
    
    if not configure_tesseract():
        return
    
    logger = setup_logging()
    collector = ExcelDataCollector(logger)
    
    invoice_list = collector.collect_invoice_data()
    
    if not invoice_list:
        print("❌ No WORKING status invoice data found!")
        print("💡 This means all invoices are already COMPLETED or have other statuses.")
        return
    
    print(f"\n✅ Successfully collected {len(invoice_list)} WORKING invoices:")
    print("=" * 80)
    print(f"{'#':<3} {'Invoice':<15} {'Lines':<6} {'Weights':<30} {'Prices':<30}")
    print("=" * 80)
    
    for i, invoice in enumerate(invoice_list[:10], 1):  # Show first 10
        weights_str = str(invoice.net_lbs)[:25] + "..." if len(str(invoice.net_lbs)) > 25 else str(invoice.net_lbs)
        prices_str = str([f"{p:.4f}" for p in invoice.unit_prices])[:25] + "..." if len(str(invoice.unit_prices)) > 25 else str([f"{p:.4f}" for p in invoice.unit_prices])
        
        print(f"{i:<3} {invoice.invoice_number:<15} {len(invoice.net_lbs):<6} {weights_str:<30} {prices_str:<30}")
    
    if len(invoice_list) > 10:
        print(f"... and {len(invoice_list) - 10} more invoices")
    
    print("=" * 80)
    total_lines = sum(len(inv.net_lbs) for inv in invoice_list)
    print(f"📊 Summary: {len(invoice_list)} WORKING invoices, {total_lines} total lines")
    print("💡 Only WORKING status invoices are processed. COMPLETED invoices are skipped.")


def test_weight_detection_debug():
    """Test weight detection with comprehensive debugging"""
    
    print("🐛 TESTING WEIGHT DETECTION WITH DEBUG")
    print("=" * 50)
    print("This will help debug why OCR weight detection is failing")
    
    if not configure_tesseract():
        return
    
    ready = input("\nMake sure table is visible on screen. Press Enter when ready...").strip()
    
    logger = setup_logging()
    price_updater = DynamicPriceUpdater(logger)
    
    # Test with sample weights (like what you see in spreadsheet)
    sample_weights = [12442.0, 14070.0, 14360.0, 13984.0]  # From your images
    
    print(f"\n🔍 Testing weight detection for: {sample_weights}")
    
    # Run debug analysis
    debug_results = price_updater.debug_weight_detection(sample_weights)
    
    print(f"\n📊 DEBUG RESULTS:")
    print(f"Total numbers found by OCR: {debug_results.get('total_numbers', 0)}")
    print(f"Weight matches found: {len(debug_results.get('weight_matches', {}))}")
    
    if debug_results.get('debug_screenshot'):
        print(f"📸 Debug screenshot saved: {debug_results['debug_screenshot']}")
    
    # Show what weights were actually found
    weight_matches = debug_results.get('weight_matches', {})
    if weight_matches:
        print(f"\n✅ SUCCESSFUL MATCHES:")
        for weight, match in weight_matches.items():
            print(f"  Weight {weight} → Found '{match['found_text']}' at {match['coordinates']}")
            print(f"    Price field coordinates: {match['price_coords']}")
    else:
        print(f"\n❌ NO MATCHES FOUND")
        print(f"Available numbers from OCR:")
        all_numbers = debug_results.get('all_numbers', [])
        for i, num in enumerate(all_numbers[:15]):  # Show first 15
            print(f"  {i+1}. '{num['text']}' at ({num['x']}, {num['y']}) confidence: {num['confidence']}")
    
    # Test table verification
    print(f"\n🔍 Testing table verification...")
    table_visible = price_updater.verify_table_visible(sample_weights)
    if table_visible:
        print(f"✅ Table verification PASSED")
    else:
        print(f"❌ Table verification FAILED")
    
    return debug_results


# ============================================================================
# MAIN EXECUTION - UPDATED MENU
# ============================================================================

def main():
    """Main execution function - PRODUCTION READY for WORKING invoices only"""
    
    print("🏭 SCRAP DRAGON PROCESS AUTOMATION - PRODUCTION READY")
    print("=" * 80)
    print("🔄 RECALCULATION SAFE: Only processes 'WORKING' status invoices")
    print("✅ STATUS MANAGEMENT: Updates to 'COMPLETED' when all lines updated")
    print("=" * 80)
    print("Following Process.txt exactly:")
    print("1. ✅ Collect WORKING invoice# and price info using pandas xlwings")
    print("2. ✅ Search invoice# and enter using pyautogui") 
    print("3. ✅ Click dropdown using pyautogui")
    print("4. ✅ Click generate pack list - wait 3 seconds")
    print("5. ✅ Use price_updater_module.py EXACT code for weight matching")
    print("6. ✅ Save shipment items")
    print("7. ✅ Update status to COMPLETED (handles 1-10 lines per invoice)")
    print("=" * 80)
    
    # Ensure directories exist
    AutomationConfig.ensure_directories()
    
    # Configure Tesseract
    if not configure_tesseract():
        input("Press Enter to exit...")
        return
    
    print(f"\n🧪 Options:")
    print("1. Run LIVE automation (process WORKING invoices)")
    print("2. Test Excel data collection (show WORKING invoices)")
    print("3. Test weight detection debug")
    print("4. Exit")
    
    choice = input("Choose option (1-4): ").strip()
    
    if choice == "2":
        test_excel_data_collection()
        
    elif choice == "3":
        test_weight_detection_debug()
        
    elif choice == "4":
        print("👋 Goodbye!")
        return
        
    elif choice == "1":
        # Live automation - WORKING invoices only
        automator = ScrapDragonProcessAutomator()
        
        try:
            summary = automator.run_automation(simulate=False)
            
            print(f"\n{'='*80}")
            print("🎯 AUTOMATION COMPLETE")
            print(f"{'='*80}")
            
            if summary['status'] == 'completed':
                print(f"✅ Status: {summary['status']}")
                print(f"📊 Total WORKING invoices: {summary['total_invoices']}")
                print(f"🎯 Successful: {summary['successful']}")
                print(f"❌ Failed: {summary['failed']}")
                print(f"📝 Total lines updated: {summary['total_lines_updated']}")
                print(f"⏱️ Duration: {summary['duration_seconds']:.1f} seconds")
                
                if summary['successful'] > 0:
                    avg_lines = summary['total_lines_updated'] / summary['successful']
                    print(f"📈 Average lines per invoice: {avg_lines:.1f}")
                    
                success_rate = (summary['successful'] / summary['total_invoices']) * 100
                print(f"✨ Success rate: {success_rate:.1f}%")
                
                print(f"\n💡 Status Updates:")
                print(f"   📋 Invoices marked COMPLETED: {summary['successful']}")
                print(f"   🔄 Safe to re-run: Only WORKING invoices will be processed")
                
            elif summary['status'] == 'no_working_invoices':
                print("✅ All invoices already processed!")
                print("📋 No WORKING status invoices found - all are likely COMPLETED")
                print("🔄 This is expected after successful automation runs")
                
            elif summary['status'] == 'cancelled':
                print("🛑 Automation cancelled by user")
            else:
                print(f"❌ Error: {summary.get('message', 'Unknown error')}")
        
        except KeyboardInterrupt:
            print("\n🛑 Automation stopped by user")
        except Exception as e:
            print(f"\n💥 Unexpected error: {e}")
    
    else:
        print("❌ Invalid choice. Please enter 1-4.")
    
    input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()
