import asyncio
from pydoll.browser import Chrome
import pyautogui
import pandas as pd
from datetime import datetime
import time
import os
import re

# Configure PyAutoGUI
pyautogui.PAUSE = 0.3
pyautogui.FAILSAFE = True

# EXACT COORDINATES for HMM search bar (when browser is maximized)
HMM_SEARCH_X = 196
HMM_SEARCH_Y = 120

# Save path for results
SAVE_PATH = r"C:\Users\ykim\OneDrive - Venture Metals\Desktop\Trading Invoices\Automation Project - Excel Files\python\Shipment Tracking"

# CONFIGURATION OPTIONS
TAKE_SCREENSHOTS = False
SAVE_PAGE_SOURCE = False
VERBOSE_LOGGING = True

async def search_multiple_bookings_fixed(booking_numbers):
    """
    Multi-booking search with fixed location extraction
    """
    all_results = []
    
    try:
        os.makedirs(SAVE_PATH, exist_ok=True)
        
        async with Chrome() as browser:
            tab = await browser.start()
            
            if VERBOSE_LOGGING:
                print(f"🚀 Fixed location extraction for {len(booking_numbers)} bookings")
                print(f"📦 Bookings: {', '.join(booking_numbers)}")
            
            # Navigate and setup
            await tab.go_to('https://www.hmm21.com/e-service/general/trackNTrace/TrackNTrace.do')
            await asyncio.sleep(0.5)
            
            try:
                await tab.maximize()
            except:
                pyautogui.hotkey('win', 'up')
            await asyncio.sleep(0.5)
            
            # Process each booking
            for booking_index, booking_number in enumerate(booking_numbers):
                if VERBOSE_LOGGING:
                    print(f"\n📦 Processing {booking_index + 1}/{len(booking_numbers)}: {booking_number}")
                
                booking_results = await search_single_booking_fixed(tab, booking_number, booking_index)
                
                if booking_results:
                    all_results.extend(booking_results)
                    if VERBOSE_LOGGING:
                        print(f"✅ Success: {booking_number}")
                else:
                    if VERBOSE_LOGGING:
                        print(f"❌ No data: {booking_number}")
                
                if booking_index < len(booking_numbers) - 1:
                    await asyncio.sleep(0.5)
    
    except Exception as e:
        print(f"❌ Error: {e}")
    
    return all_results

async def search_single_booking_fixed(tab, booking_number, booking_index):
    """
    Single booking search with improved location extraction
    """
    results = []
    
    try:
        # Click and enter booking number
        pyautogui.click(HMM_SEARCH_X, HMM_SEARCH_Y)
        time.sleep(0.3)
        pyautogui.hotkey('ctrl', 'a')
        time.sleep(0.2)
        pyautogui.write(booking_number)
        time.sleep(0.5)
        
        # Optional screenshot
        if TAKE_SCREENSHOTS:
            screenshot_path = os.path.join(SAVE_PATH, f"typed_{booking_index+1:02d}_{booking_number}.png")
            await tab.take_screenshot(path=screenshot_path)
        
        # Search
        pyautogui.press('enter')
        time.sleep(0.3)
        await asyncio.sleep(1.2)
        
        # Optional screenshot
        if TAKE_SCREENSHOTS:
            screenshot_path = os.path.join(SAVE_PATH, f"results_{booking_index+1:02d}_{booking_number}.png")
            await tab.take_screenshot(path=screenshot_path)
        
        # Try table extraction first
        table_results = await extract_table_data_fixed(tab, booking_number)
        if table_results:
            results.extend(table_results)
        
        # If table fails, try improved content extraction
        if not results:
            content_results = await extract_content_data_fixed(tab, booking_number)
            if content_results:
                results.extend(content_results)
    
    except Exception as e:
        results.append({
            'booking_number': booking_number,
            'error': str(e),
            'search_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    
    return results

async def extract_table_data_fixed(tab, booking_number):
    """
    Fixed table extraction focusing on vessel movement table
    """
    results = []
    
    try:
        tables = await tab.query('table', find_all=True, timeout=5)
        if not tables:
            return results
        
        for table in tables:
            rows = await table.query('tr', find_all=True)
            if len(rows) < 2:
                continue
            
            for row_idx, row in enumerate(rows[1:], 1):
                try:
                    cells = await row.query('td, th', find_all=True)
                    if not cells or len(cells) < 6:
                        continue
                    
                    cell_texts = []
                    for cell in cells:
                        try:
                            text = await cell.text
                            cell_texts.append(text.strip() if text else "")
                        except:
                            cell_texts.append("")
                    
                    # Check for vessel movement data
                    has_dates = any('2025' in str(cell) for cell in cell_texts)
                    has_ports = any(keyword in ' '.join(cell_texts).upper() 
                                  for keyword in ['LOS ANGELES', 'GWANGYANG', 'KOREA', 'CA'])
                    
                    if has_dates and has_ports:
                        result = {
                            'booking_number': booking_number,
                            'vessel_voyage': cell_texts[0],
                            'route': cell_texts[1],
                            'loading_port': cell_texts[2],
                            'departure_date': cell_texts[3],
                            'discharging_port': cell_texts[4],
                            'arrival_date': cell_texts[5],
                            'search_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'extraction_method': 'table_extraction'
                        }
                        results.append(result)
                        
                        if VERBOSE_LOGGING:
                            print(f"✅ Table extraction:")
                            print(f"   🚢 Vessel: {result['vessel_voyage']}")
                            print(f"   📍 Loading: {result['loading_port']}")
                            print(f"   🛫 Departure: {result['departure_date']}")
                            print(f"   📍 Discharge: {result['discharging_port']}")
                            print(f"   🛬 Arrival: {result['arrival_date']}")
                        
                        return results  # Return first successful match
                
                except:
                    continue
    except:
        pass
    
    return results

async def extract_content_data_fixed(tab, booking_number):
    """
    Fixed content extraction with better location parsing
    """
    results = []
    
    try:
        body = await tab.query('body')
        page_text = await body.text
        
        if VERBOSE_LOGGING:
            print("🔍 Extracting with improved location parsing...")
        
        # Extract locations with better cleaning
        clean_locations = extract_clean_locations(page_text)
        
        # Extract dates
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}',
            r'\d{4}-\d{2}-\d{2}'
        ]
        
        all_dates = []
        for pattern in date_patterns:
            dates = re.findall(pattern, page_text)
            all_dates.extend(dates)
        
        unique_dates = list(dict.fromkeys(all_dates))
        
        if VERBOSE_LOGGING:
            print(f"📍 Clean locations found: {len(clean_locations)}")
            for i, loc in enumerate(clean_locations[:10], 1):
                print(f"   {i}. {loc}")
            
            print(f"📅 Dates found: {len(unique_dates)}")
            for i, date in enumerate(unique_dates[:10], 1):
                print(f"   {i}. {date}")
        
        # Smart location matching for shipping ports
        departure_port, arrival_port = find_shipping_ports(clean_locations, booking_number)
        
        # Smart date extraction
        departure_date = unique_dates[0] if len(unique_dates) > 0 else "Not found"
        arrival_date = unique_dates[1] if len(unique_dates) > 1 else "Not found"
        
        if departure_port != "Not found" or arrival_port != "Not found":
            result = {
                'booking_number': booking_number,
                'departure_port': departure_port,
                'arrival_port': arrival_port,
                'departure_date': departure_date,
                'arrival_date': arrival_date,
                'all_locations': ' | '.join(clean_locations[:10]),
                'all_dates': ' | '.join(unique_dates[:10]),
                'search_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'extraction_method': 'smart_content_extraction',
                'locations_found': len(clean_locations),
                'dates_found': len(unique_dates)
            }
            
            results.append(result)
            
            if VERBOSE_LOGGING:
                print(f"✅ Smart extraction successful:")
                print(f"   🛫 Departure Port: {departure_port}")
                print(f"   🛬 Arrival Port: {arrival_port}")
                print(f"   📅 Departure Date: {departure_date}")
                print(f"   📅 Arrival Date: {arrival_date}")
    
    except Exception as e:
        if VERBOSE_LOGGING:
            print(f"❌ Content extraction error: {e}")
    
    return results

def extract_clean_locations(text):
    """
    Extract clean location pairs from messy concatenated text
    """
    clean_locations = []
    
    # Known shipping locations - expand this list as needed
    known_locations = [
        "LOS ANGELES, CA",
        "GWANGYANG, KOREA", 
        "BUSAN, KOREA",
        "HOUSTON, TX",
        "LONG BEACH, CA",
        "NEW YORK, NY",
        "SEATTLE, WA",
        "OAKLAND, CA",
        "TACOMA, WA",
        "CHARLESTON, SC",
        "SAVANNAH, GA",
        "NORFOLK, VA",
        "BALTIMORE, MD",
        "MIAMI, FL",
        "HASLET, TX",
        "DETROIT, MI"
    ]
    
    # First, try to find exact matches for known locations
    for location in known_locations:
        if location.upper() in text.upper():
            if location not in clean_locations:
                clean_locations.append(location)
    
    # Then use improved regex patterns to find other locations
    location_patterns = [
        r'\b([A-Z][A-Z\s]{2,15},\s*[A-Z]{2})\b',      # City, ST format
        r'\b([A-Z][A-Z\s]{2,15},\s*[A-Z]{3,8})\b',    # City, COUNTRY format
    ]
    
    for pattern in location_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            match_clean = match.strip()
            if (len(match_clean) > 6 and 
                not re.search(r'\d{4}', match_clean) and  # No years
                match_clean not in clean_locations and
                not any(bad in match_clean.upper() for bad in ['DATE', 'TIME', 'TYPE', 'STATUS'])):
                clean_locations.append(match_clean)
    
    return clean_locations

def find_shipping_ports(locations, booking_number):
    """
    Smart logic to identify departure and arrival ports for shipping
    """
    departure_port = "Not found"
    arrival_port = "Not found"
    
    # Known departure ports (typically Asian ports)
    asian_ports = ["GWANGYANG, KOREA", "BUSAN, KOREA", "SHANGHAI, CHINA", "NINGBO, CHINA"]
    
    # Known arrival ports (typically US ports)
    us_ports = ["LOS ANGELES, CA", "LONG BEACH, CA", "OAKLAND, CA", "SEATTLE, WA", "HOUSTON, TX"]
    
    # Look for specific patterns based on booking or common routes
    for location in locations:
        location_upper = location.upper()
        
        # Check for Asian departure ports
        if any(port.upper() in location_upper for port in asian_ports):
            if arrival_port == "Not found":
                arrival_port = location



        # Check for US arrival ports  
        if any(port.upper() in location_upper for port in us_ports):
            if departure_port == "Not found":
                departure_port = location
    
    # If we didn't find specific matches, use heuristics
    if departure_port == "Not found" or arrival_port == "Not found":
        for location in locations:
            if "KOREA" in location.upper() and departure_port == "Not found":
                arrival_port = location
            elif ("CA" in location or "TX" in location) and arrival_port == "Not found":
                departure_port = location
    
    return departure_port, arrival_port

async def main():
    """
    Main function with fixed location extraction
    """
    # Your booking numbers ----- replace
    booking_numbers = [
    'DALA19629300',
    'DALA23334300',
    'DALA50507700',
    'DALA55478601',
    'DALA55478603',
    'DALA58731600',
    'DALA6896200',
    'DALA88088200',
    'DALA95501500'




        # 'ADD_MORE_BOOKINGS_HERE'
    ]
    
    print("🔧 FIXED LOCATION EXTRACTION HMM EXTRACTOR")
    print("=" * 50)
    print(f"📦 Bookings: {len(booking_numbers)}")
    print(f"🎯 Target: Gwangyang, Korea → Los Angeles, CA")
    print("=" * 50)
    
    # Ensure directory exists
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    # Process all bookings
    start_time = datetime.now()
    all_results = await search_multiple_bookings_fixed(booking_numbers)
    end_time = datetime.now()
    
    # Save and display results
    if all_results:
        successful_results = [r for r in all_results if 'error' not in r]
        
        if successful_results:
            df = pd.DataFrame(successful_results)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            excel_file = os.path.join(SAVE_PATH, f"hmm_fixed_results_{timestamp}.xlsx")
            df.to_excel(excel_file, index=False)
            
            csv_file = os.path.join(SAVE_PATH, f"hmm_fixed_results_{timestamp}.csv")
            df.to_csv(csv_file, index=False)
            
            print(f"\n📊 FINAL RESULTS:")
            print("=" * 50)
            for result in successful_results:
                print(f"📦 Booking: {result['booking_number']}")
                
                if 'vessel_voyage' in result:
                    print(f"   🚢 Vessel: {result['vessel_voyage']}")
                    print(f"   📍 Loading: {result['loading_port']}")
                    print(f"   🛫 Departure: {result['departure_date']}")
                    print(f"   📍 Discharge: {result['discharging_port']}")
                    print(f"   🛬 Arrival: {result['arrival_date']}")
                else:
                    print(f"   🛫 Departure Port: {result['departure_port']}")
                    print(f"   🛬 Arrival Port: {result['arrival_port']}")
                    print(f"   📅 Departure Date: {result['departure_date']}")
                    print(f"   📅 Arrival Date: {result['arrival_date']}")
                
                print(f"   🔧 Method: {result['extraction_method']}")
                print("-" * 30)
            
            print(f"✅ Successful: {len(successful_results)}")
            print(f"⏱️  Total time: {(end_time - start_time).total_seconds():.1f} seconds")
            print(f"💾 Saved: {excel_file}")
        else:
            print("❌ No successful extractions")
    else:
        print("❌ No results")

if __name__ == "__main__":
    print("🔧 FIXED VERSION - Better location parsing for shipping data!")
    print("🎯 Specifically targets: Gwangyang, Korea → Los Angeles, CA")
    print()
    
    asyncio.run(main())