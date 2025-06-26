# 🏭 Scrap Dragon Automation Complete Guide

> **A comprehensive automation system for updating invoice line items with enhanced OCR, template matching, and table parsing capabilities**

This document provides everything needed to set up, run, and troubleshoot the Scrap Dragon automation system for processing invoice line items.

---

## 📋 **Table of Contents**

1. [System Overview](#system-overview)
2. [Prerequisites & Setup](#prerequisites--setup)
3. [File Structure & Configuration](#file-structure--configuration)
4. [Automation Features](#automation-features)
5. [Usage Instructions](#usage-instructions)
6. [Testing & Validation](#testing--validation)
7. [Troubleshooting Guide](#troubleshooting-guide)
8. [Technical Implementation](#technical-implementation)

---

## 🎯 **System Overview**

### **What It Does:**
The Scrap Dragon automation processes invoice line items by:
- Loading **WORKING** invoices from Excel (Column A = "WORKING")
- Searching each invoice in the web application
- Detecting **"Generate From Pack List"** and **"Save Shipment Items"** buttons
- Parsing table structure to find **shipped weights** and **price fields**
- Updating **ALL matching line items** with new prices from Excel
- Verifying updates before moving to next invoice
- Updating Excel status to **"COMPLETED"** or **"ERROR"**

### **Key Capabilities:**
- ✅ **Template Matching** - Uses saved PNG images for reliable button detection
- ✅ **Multi-Scale OCR** - Enhanced image preprocessing for better text recognition
- ✅ **Table Structure Analysis** - Understands column layouts and finds price fields
- ✅ **Multiple Line Updates** - Processes ALL matching items per invoice
- ✅ **Verification System** - Confirms price updates were successful
- ✅ **Comprehensive Logging** - Detailed logs for debugging and monitoring

---

## 🔧 **Prerequisites & Setup**

### **Required Software:**
```bash
# Python packages
pip install pandas xlwings pyautogui opencv-python pytesseract numpy pathlib

# Tesseract OCR (Windows)
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
# Install to: C:\Program Files\Tesseract-OCR\
```

### **Required Files:**
```
Cable Automation/
├── enhanced_scrap_dragon_automation.py    # Main automation script
├── Generate.png                           # Template image for Generate button
├── SaveShip.png                          # Template image for Save button
├── logs/                                  # Auto-created log directory
├── screenshots/                           # Auto-created screenshot directory
└── manual_coordinates.json               # Optional manual button coordinates
```

### **Excel Configuration:**
- **File Path**: `C:\Users\ykim\OneDrive - Venture Metals\Desktop\daily operation - YK.xlsx`
- **Sheet Name**: `Cable Final`
- **Column A**: Status ("WORKING" for invoices to process)
- **Column D**: Invoice numbers
- **Column G**: Shipped weights (target for matching)
- **Column V**: Unit prices (new prices to apply)

---

## 📁 **File Structure & Configuration**

### **AutomationConfig Class:**
```python
# Key configuration paths and settings
EXCEL_PATH = r"C:\Users\ykim\OneDrive - Venture Metals\Desktop\daily operation - YK.xlsx"
SHEET_NAME = "Cable Final"
TEMPLATE_DIR = r"C:\Users\ykim\OneDrive - Venture Metals\Desktop\Trading Invoices\Automation Project - Excel Files\python\Cable Automation"

# Detection thresholds
TEMPLATE_MATCH_THRESHOLD = 0.35  # Lowered for better detection
OCR_CONFIDENCE_THRESHOLD = 50
WEIGHT_TOLERANCE = 1.0           # Exact weight matching

# Timing delays
DELAY_AFTER_SEARCH = 1.5
DELAY_AFTER_GENERATE = 2.0
DELAY_BETWEEN_LINE_UPDATES = 0.5
DELAY_BETWEEN_INVOICES = 2.0
```

### **Template Images:**
- **Generate.png**: Screenshot of "Generate From Pack List" button
- **SaveShip.png**: Screenshot of "Save Shipment Items" button

**Template Creation Tips:**
- Include full button text
- Add 5-10 pixel padding around button
- Capture in normal (not hovered) state
- Use consistent UI state (same dropdown expanded)

---

## ⚡ **Automation Features**

### **1. Enhanced Button Detection**
```python
# Multi-method detection hierarchy:
1. Template Matching (Primary) - Uses PNG files
2. OCR Text Detection (Fallback) - Reads button text
3. Color Detection (Last Resort) - Finds blue button regions
4. Manual Coordinates (Override) - User-entered coordinates
```

### **2. Intelligent Table Parsing**
```python
# Table structure detection:
- Finds "Shipped Weight" and "Price" column headers
- Groups data by table rows using Y-coordinates
- Identifies weight values (4-5 digits: 14475, 16664, 11532)
- Locates price fields ($X.XXXX format)
- Calculates clickable coordinates for each price field
```

### **3. Comprehensive Line Item Processing**
```python
# For each invoice:
1. Parse entire table structure
2. Find ALL line items with matching shipped weights
3. Update each matching price field individually
4. Retry failed updates up to 3 times
5. Verify all updates before proceeding
6. Log detailed results for each line item
```

### **4. Robust Error Handling**
```python
# Error recovery mechanisms:
- Template matching with multiple scales (0.8x to 1.2x)
- OCR preprocessing with noise reduction and contrast enhancement
- Fallback coordinate systems
- Retry logic for failed price updates
- Comprehensive logging for debugging
```

---

## 🚀 **Usage Instructions**

### **Running the Automation:**

```bash
# Navigate to automation directory
cd "C:\Users\ykim\OneDrive - Venture Metals\Desktop\Trading Invoices\Automation Project - Excel Files\python\Cable Automation"

# Run the automation
python enhanced_scrap_dragon_automation.py
```

### **Menu Options:**

#### **Option 1: Run Full Automation**
- Processes all WORKING invoices from Excel
- Updates Excel status to COMPLETED/ERROR
- Generates comprehensive logs and results

#### **Option 2: Test Template Detection**
- Tests button detection using PNG templates
- Creates debug image showing detected locations
- Validates coordinates before full automation

#### **Option 3: Help Create New Template Images**
- Guided process for creating new button templates
- Takes screenshot for cropping reference
- Provides detailed cropping instructions

#### **Option 4: Check Excel Data Loading**
- Validates Excel file and sheet access
- Shows all detected WORKING invoices
- Identifies missing or invalid data

#### **Option 5: Manual Coordinate Entry**
- Allows manual button coordinate input
- Creates visual verification of coordinates
- Saves coordinates for future use

#### **Option 6: Test Table Parsing**
- **CRITICAL FOR PRICE UPDATES** - Tests table structure detection
- Shows detected weights and price coordinates
- Creates debug image with price field markers

---

## 🧪 **Testing & Validation**

### **Pre-Automation Testing Sequence:**

#### **Step 1: Validate Excel Data (Option 4)**
```bash
Expected Output:
✅ Found 25 WORKING invoices:
#   Invoice         Weight     Price      Row
1   DAL125097      14070.0    $1.1608    5
2   DAL789123      16664.0    $1.2500    8
...
🎯 Perfect! Found all 25 invoices ready for processing.
```

#### **Step 2: Test Template Detection (Option 2)**
```bash
Expected Output:
✅ Template detection successful!
  generate_pack_list: (1454, 587)
  save_shipment: (1800, 625)
🎯 Debug image saved: debug_template_detection.png
```

#### **Step 3: Test Table Parsing (Option 6)**
```bash
Expected Output:
✅ Table parsing successful!
Found 3 line items:
Row  Weight   Price      Coordinates     Confidence
0    14475    $1.0000    (450, 120)     85.2
1    16664    $1.0000    (450, 145)     87.1  
2    11532    $1.0000    (450, 170)     82.9
🎯 Debug image created: debug_table_parsing.png
```

### **Validation Criteria:**
- ✅ All 25 invoices detected from Excel
- ✅ Button templates match with >35% confidence
- ✅ Table parsing finds shipped weights and price coordinates
- ✅ Debug images show correct detection locations

---

## 🔍 **Troubleshooting Guide**

### **Common Issues & Solutions:**

#### **❌ Template Detection Failed**
```bash
# Problem: Template match confidence < 35%
Solutions:
1. Try Option 2 with different UI state
2. Recreate templates using Option 3
3. Use manual coordinates (Option 5)
4. Lower threshold in config (0.2)
```

#### **❌ No Line Items Detected**
```bash
# Problem: Table parsing returns 0 items
Solutions:
1. Ensure table is visible and expanded
2. Check "Shipped Weight" and "Price" headers exist
3. Verify weights are 4-5 digits (14475, not 1447)
4. Confirm prices in $X.XXXX format
5. Use Option 6 to debug table structure
```

#### **❌ Excel Loading Issues**
```bash
# Problem: Found 0 WORKING invoices
Solutions:
1. Check Excel file path in config
2. Verify "Cable Final" sheet exists
3. Ensure Column A contains exactly "WORKING"
4. Check Columns D, G, V have valid data
5. Close Excel file before running automation
```

#### **❌ Price Updates Not Working**
```bash
# Problem: 0 line items updated
Solutions:
1. Check weight tolerance (currently 1.0)
2. Verify target weights match table weights exactly
3. Ensure price coordinates are clickable
4. Test with Option 6 to see detected coordinates
5. Check if price fields are read-only
```

### **Debug Information Locations:**
```bash
logs/scrap_dragon_automation_YYYYMMDD_HHMMSS.log     # Detailed processing logs
screenshots/screenshot_YYYYMMDD_HHMMSS_*.png          # All screenshots taken
debug_template_detection.png                          # Button detection results
debug_table_parsing.png                              # Table parsing results
manual_coordinates.json                               # Saved manual coordinates
```

---

## ⚙️ **Technical Implementation**

### **Core Classes & Architecture:**

#### **AutomationConfig**
```python
# Centralized configuration management
- File paths and directories
- OCR and detection thresholds  
- Timing delays and tolerances
- Template image locations
```

#### **EnhancedUIDetector**
```python
# Multi-method button detection
- Template matching with multi-scale support
- OCR text recognition with preprocessing
- Color-based detection fallback
- Manual coordinate override system
```

#### **TableParser**
```python
# Intelligent table structure analysis
- Column header detection and mapping
- Row-based data grouping
- Weight and price pattern recognition
- Clickable coordinate calculation
```

#### **ScrapDragonAutomator**
```python
# Main orchestration and process control
- Invoice processing workflow
- Excel integration and status updates
- Comprehensive error handling and logging
- Results tracking and verification
```

### **Processing Workflow:**

#### **Per Invoice Processing:**
```python
1. Search invoice number → Expand dropdown
2. Detect UI buttons → Take screenshot for analysis
3. Click "Generate From Pack List" → Wait for table to load
4. Parse table structure → Find all line items
5. Match shipped weights → Identify price coordinates
6. Update ALL matching prices → Verify each update
7. Click "Save Shipment Items" → Update Excel status
8. Move to next invoice → Repeat process
```

#### **Data Flow:**
```python
Excel (WORKING invoices) 
    ↓
Web Application (Search & Expand)
    ↓  
UI Detection (Buttons & Table)
    ↓
Price Updates (All matching lines)
    ↓
Verification (Confirm changes)
    ↓
Excel Status Update (COMPLETED/ERROR)
```

---

## 📊 **Expected Results**

### **Successful Processing Output:**
```bash
🎯 AUTOMATION SUMMARY
================================================================================
✅ Status: completed
📊 Total invoices: 25
🎯 Successful: 24
❌ Failed: 1
📝 Total line items updated: 47
⏱️ Duration: 890.3 seconds
📈 Average lines per invoice: 1.96
✨ Success rate: 96.0%
```

### **Individual Invoice Processing:**
```bash
📋 Processing: DAL125097
Target Weight: 14070.0 → New Price: $1.1608

🔄 Starting line item updates - processing ALL matching items...
📋 Found 3 line items in table
  Row 0: Weight=14070, Price=$1.0000
  Row 1: Weight=16664, Price=$1.0000  
  Row 2: Weight=11532, Price=$1.0000

🎯 Found 1 matching line items to update
📝 Updating line 0 (1/1)
✅ Successfully updated line 0

🔍 Verifying price updates...
✅ Verification passed: All 1 line items updated correctly
💾 All line items processed, proceeding to save...
✅ Success: 1 lines updated
```

### **Log File Structure:**
```bash
2025-06-24 11:45:23,123 - ScrapDragonAutomation - INFO - 🚀 STARTING ENHANCED SCRAP DRAGON AUTOMATION
2025-06-24 11:45:23,456 - ScrapDragonAutomation - INFO - ✅ Found 25 WORKING invoices
2025-06-24 11:45:45,789 - ScrapDragonAutomation - INFO - 📋 Processing: DAL125097
2025-06-24 11:45:47,234 - ScrapDragonAutomation - INFO - [SUCCESS] Template match found - generate_pack_list: (1454, 587)
2025-06-24 11:45:52,567 - ScrapDragonAutomation - INFO - ✅ Successfully updated line 0
```

---

## 🎯 **Quick Reference Commands**

### **For New Claude Chat Sessions:**

```markdown
I have a Scrap Dragon automation system for processing invoice line items. 

**Current Status**: Working automation with enhanced table parsing for Price_Enter.png structure

**Key Features**:
- Template matching for button detection (Generate.png, SaveShip.png)
- Table parsing that finds shipped weights (14475, 16664, 11532) and price fields
- Updates ALL matching line items before moving to next invoice
- Processes 25 WORKING invoices from Excel Column A

**File Location**: 
C:\Users\ykim\OneDrive - Venture Metals\Desktop\Trading Invoices\Automation Project - Excel Files\python\Cable Automation\enhanced_scrap_dragon_automation.py

**Testing Sequence**:
1. Option 4: Check Excel data (verify 25 invoices)
2. Option 6: Test table parsing (verify Price_Enter.png structure)  
3. Option 2: Test template detection (verify button coordinates)
4. Option 1: Run full automation

**Common Issues**:
- Template confidence < 35% → Recreate PNG templates
- 0 line items detected → Check table headers and format
- Price updates failing → Verify clickable coordinates

Please help me with [specific issue/enhancement needed].
```

### **Configuration File Template:**
```python
# Quick config reference for new implementations
TEMPLATE_MATCH_THRESHOLD = 0.35  # Button detection sensitivity
WEIGHT_TOLERANCE = 1.0           # Exact weight matching
OCR_CONFIDENCE_THRESHOLD = 50    # Text recognition minimum
EXCEL_PATH = "daily operation - YK.xlsx"
SHEET_NAME = "Cable Final"
```

---

## 📝 **Maintenance & Updates**

### **Regular Maintenance Tasks:**
- [ ] **Weekly**: Clear old screenshots and logs
- [ ] **Monthly**: Verify Excel data structure hasn't changed
- [ ] **As Needed**: Update template images if UI changes
- [ ] **Before Major Use**: Run full testing sequence (Options 4, 6, 2)

### **Version Control:**
- Keep backup copies of working template images
- Save manual_coordinates.json if using manual coordinates
- Document any configuration changes in comments

### **Performance Optimization:**
- Monitor processing times per invoice (target: <60 seconds)
- Check OCR accuracy in logs (target: >80% confidence)
- Verify template matching success rate (target: >90%)

---

**📞 Support**: Use this document as reference for troubleshooting. All technical details, error codes, and expected outputs are documented above for quick resolution of common issues.

**🔄 Updates**: This system follows clean code structure principles and can be easily enhanced with additional features like multi-monitor support, alternative OCR engines, or database integration.