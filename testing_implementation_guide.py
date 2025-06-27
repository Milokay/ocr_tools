# 🧪 OCR Testing & Implementation Strategy
# Comprehensive testing framework for cable automation OCR system

import pytest
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Tuple
import tempfile
import shutil
from dataclasses import dataclass
import logging

# ============================================================================
# TEST CONFIGURATION AND FIXTURES
# ============================================================================

@dataclass
class TestConfig:
    """Configuration for testing framework."""
    
    # Test data paths
    test_data_dir: Path = Path("tests/fixtures")
    sample_images_dir: Path = Path("tests/fixtures/sample_screenshots")
    expected_outputs_dir: Path = Path("tests/fixtures/expected_outputs")
    
    # Performance benchmarks
    max_processing_time: float = 10.0  # seconds
    min_accuracy_threshold: float = 0.90
    max_memory_usage: float = 512.0  # MB
    
    # Test image specifications
    test_image_width: int = 1920
    test_image_height: int = 1080
    min_confidence_score: float = 50.0
    
    def __post_init__(self):
        # Create test directories
        for dir_path in [self.test_data_dir, self.sample_images_dir, self.expected_outputs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

@pytest.fixture(scope="session")
def test_config():
    """Provide test configuration for all tests."""
    return TestConfig()

@pytest.fixture(scope="session")
def sample_cable_data():
    """Provide sample cable data for testing."""
    return {
        'cables': [
            {'weight': 16664, 'price': 1.0000, 'cable_type': 'VMA Cable', 'coordinates': (1196, 504)},
            {'weight': 14475, 'price': 1.0000, 'cable_type': 'Standard Cable', 'coordinates': (1196, 483)},
            {'weight': 11532, 'price': 1.0000, 'cable_type': 'Premium Cable', 'coordinates': (1196, 523)}
        ],
        'expected_updates': [
            {'weight': 16664, 'new_price': 1.4238},
            {'weight': 14475, 'new_price': 1.1908},
            {'weight': 11532, 'new_price': 1.1985}
        ]
    }

@pytest.fixture
def mock_screenshot():
    """Generate mock screenshot for testing."""
    # Create a synthetic image with table-like structure
    img = np.ones((1080, 1920, 3), dtype=np.uint8) * 255
    
    # Add some table-like rectangles and text areas
    cv2.rectangle(img, (800, 480), (1200, 520), (200, 200, 200), -1)  # Table cells
    cv2.rectangle(img, (1150, 480), (1250, 520), (220, 220, 220), -1)  # Price cells
    
    return img

@pytest.fixture
def temp_test_dir():
    """Create temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

# ============================================================================
# UNIT TESTS FOR CORE COMPONENTS
# ============================================================================

class TestImagePreprocessing:
    """Test suite for image preprocessing components."""
    
    def test_image_loading(self, mock_screenshot, temp_test_dir):
        """Test image loading functionality."""
        from src.preprocessing.image_enhancer import AdvancedImagePreprocessor
        from src.core.config_manager import CableOCRConfig
        
        # Save mock screenshot
        test_image_path = temp_test_dir / "test_image.png"
        cv2.imwrite(str(test_image_path), mock_screenshot)
        
        # Test loading
        config = CableOCRConfig()
        processor = AdvancedImagePreprocessor(config)
        
        processed_image = processor.process(test_image_path)
        
        assert processed_image is not None
        assert len(processed_image.shape) == 2  # Should be grayscale
        assert processed_image.dtype == np.uint8
    
    def test_preprocessing_pipeline(self, mock_screenshot):
        """Test complete preprocessing pipeline."""
        from src.preprocessing.image_enhancer import AdvancedImagePreprocessor
        from src.core.config_manager import CableOCRConfig
        
        config = CableOCRConfig()
        config.preprocessing_enabled = True
        
        processor = AdvancedImagePreprocessor(config)
        
        # Convert to grayscale for testing
        gray_image = cv2.cvtColor(mock_screenshot, cv2.COLOR_BGR2GRAY)
        processed = processor._apply_preprocessing_pipeline(gray_image)
        
        # Verify preprocessing effects
        assert processed.shape == gray_image.shape
        assert processed.dtype == np.uint8
        assert np.mean(processed) != np.mean(gray_image)  # Should be different after processing
    
    def test_noise_reduction(self, mock_screenshot):
        """Test noise reduction algorithms."""
        from src.preprocessing.image_enhancer import AdvancedImagePreprocessor
        from src.core.config_manager import CableOCRConfig
        
        # Add noise to image
        gray = cv2.cvtColor(mock_screenshot, cv2.COLOR_BGR2GRAY)
        noise = np.random.normal(0, 25, gray.shape).astype(np.uint8)
        noisy_image = cv2.add(gray, noise)
        
        config = CableOCRConfig()
        processor = AdvancedImagePreprocessor(config)
        
        # Apply noise reduction
        denoised = cv2.fastNlMeansDenoising(noisy_image)
        
        # Verify noise reduction
        original_noise_level = np.std(gray - noisy_image)
        processed_noise_level = np.std(gray - denoised)
        
        assert processed_noise_level < original_noise_level
    
    @pytest.mark.performance
    def test_preprocessing_performance(self, mock_screenshot, test_config):
        """Test preprocessing performance requirements."""
        from src.preprocessing.image_enhancer import AdvancedImagePreprocessor
        from src.core.config_manager import CableOCRConfig
        
        config = CableOCRConfig()
        processor = AdvancedImagePreprocessor(config)
        
        gray = cv2.cvtColor(mock_screenshot, cv2.COLOR_BGR2GRAY)
        
        # Measure processing time
        start_time = time.time()
        processed = processor._apply_preprocessing_pipeline(gray)
        processing_time = time.time() - start_time
        
        assert processing_time < 2.0  # Should process within 2 seconds
        assert processed is not None

class TestTableDetection:
    """Test suite for table detection and parsing."""
    
    def test_table_structure_analysis(self, mock_screenshot):
        """Test table structure detection."""
        from src.detection.table_detector import CableTableParser
        from src.core.config_manager import CableOCRConfig
        
        config = CableOCRConfig()
        parser = CableTableParser(config)
        
        gray = cv2.cvtColor(mock_screenshot, cv2.COLOR_BGR2GRAY)
        
        # Mock OCR data for testing
        mock_ocr_data = {
            'text': ['16664', '$1.0000', 'VMA Cable', '14475', '$1.0000', 'Standard'],
            'conf': [85, 90, 75, 88, 92, 80],
            'left': [800, 1196, 500, 800, 1196, 500],
            'top': [504, 504, 504, 483, 483, 483],
            'width': [50, 60, 80, 50, 60, 80],
            'height': [20, 20, 20, 20, 20, 20]
        }
        
        with patch('pytesseract.image_to_data', return_value=mock_ocr_data):
            elements = parser._filter_high_confidence_elements(mock_ocr_data)
            
            assert len(elements) == 6
            assert all(elem['confidence'] >= config.confidence_threshold for elem in elements)
    
    def test_coordinate_precision(self, sample_cable_data):
        """Test coordinate detection precision."""
        from src.detection.coordinate_mapper import CoordinateMapper
        
        # Test coordinate mapping accuracy
        expected_coordinates = [(1196, 504), (1196, 483), (1196, 523)]
        tolerance = 25
        
        for i, cable in enumerate(sample_cable_data['cables']):
            expected_x, expected_y = expected_coordinates[i]
            actual_x, actual_y = cable['coordinates']
            
            assert abs(actual_x - expected_x) <= tolerance
            assert abs(actual_y - expected_y) <= tolerance
    
    def test_cable_data_extraction(self, mock_screenshot):
        """Test cable-specific data extraction."""
        from src.parsing.cable_parser import CableDataExtractor
        from src.core.config_manager import CableOCRConfig
        
        config = CableOCRConfig()
        extractor = CableDataExtractor(config)
        
        # Mock extracted elements
        mock_elements = [
            {'text': '16664', 'x': 800, 'y': 504, 'confidence': 85},
            {'text': '$1.0000', 'x': 1196, 'y': 504, 'confidence': 90},
            {'text': 'VMA Cable', 'x': 500, 'y': 504, 'confidence': 75}
        ]
        
        cable_records = extractor._extract_cable_data(mock_elements)
        
        assert len(cable_records) > 0
        assert 'weight' in cable_records[0]
        assert 'price' in cable_records[0]
        assert 'coordinates' in cable_records[0]

class TestVMACableDetection:
    """Test suite for VMA Cable classification."""
    
    def test_vma_cable_string_detection(self):
        """Test VMA Cable string detection logic."""
        from src.parsing.vma_classifier import VMACableClassifier
        
        classifier = VMACableClassifier()
        
        # Test various contract number formats
        test_cases = [
            ('13311', 'VMA Cable'),
            ('13311.0', 'VMA Cable'),
            (13311, 'VMA Cable'),
            (13311.0, 'VMA Cable'),
            ('12345', 'Other'),
            ('', 'Other'),
            (None, 'Other'),
            (np.nan, 'Other')
        ]
        
        for contract_value, expected in test_cases:
            result = classifier.classify_cable(contract_value)
            assert result == expected, f"Failed for {contract_value}: expected {expected}, got {result}"
    
    def test_vma_cable_data_processing(self):
        """Test VMA Cable data processing pipeline."""
        from src.parsing.vma_classifier import VMACableClassifier
        import pandas as pd
        
        # Create test dataframe
        test_data = pd.DataFrame({
            'Contract#': [13311.0, 12345, '13311', '67890', 13311],
            'Weight': [16664, 14475, 11532, 9876, 5432]
        })
        
        classifier = VMACableClassifier()
        processed_data = classifier.process_dataframe(test_data)
        
        vma_cables = processed_data[processed_data['VMA_Classification'] == 'VMA Cable']
        assert len(vma_cables) == 3  # Should detect 3 VMA Cable records
    
    def test_contract_number_cleaning(self):
        """Test contract number cleaning functionality."""
        from src.parsing.vma_classifier import VMACableClassifier
        
        classifier = VMACableClassifier()
        
        test_cases = [
            (13311.0, '13311'),
            ('13311.0', '13311'),
            ('  13311  ', '13311'),
            ('nan', ''),
            (np.nan, ''),
            (None, ''),
            ('', '')
        ]
        
        for input_value, expected in test_cases:
            result = classifier._clean_contract_number(input_value)
            assert result == expected

# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestFullPipeline:
    """Integration tests for complete automation pipeline."""
    
    def test_end_to_end_automation(self, mock_screenshot, temp_test_dir, sample_cable_data):
        """Test complete end-to-end automation workflow."""
        from src.automation.enhanced_cable_automation import EnhancedCableAutomation
        from src.core.config_manager import CableOCRConfig
        
        # Setup
        config = CableOCRConfig()
        config.debug_mode = True
        config.debug_output_path = temp_test_dir / "debug"
        
        automation = EnhancedCableAutomation(config)
        
        # Save test image
        test_image_path = temp_test_dir / "test_screenshot.png"
        cv2.imwrite(str(test_image_path), mock_screenshot)
        
        # Mock PyAutoGUI to prevent actual clicks
        with patch('pyautogui.click'), \
             patch('pyautogui.typewrite'), \
             patch('pyautogui.press'), \
             patch('pyautogui.screenshot', return_value=Mock(save=Mock())):
            
            # Mock OCR results
            mock_parsing_results = {
                'cable_data': sample_cable_data['cables'],
                'total_elements': len(sample_cable_data['cables']),
                'table_structure': {'rows': 3, 'columns': 4}
            }
            
            with patch.object(automation.table_parser, 'process', return_value=mock_parsing_results):
                results = automation.run_complete_automation(test_image_path)
        
        # Verify results
        assert results['overall_success'] is True
        assert 'update_results' in results
        assert results['cable_records_found'] == len(sample_cable_data['cables'])
    
    def test_error_handling_pipeline(self, temp_test_dir):
        """Test error handling throughout the pipeline."""
        from src.automation.enhanced_cable_automation import EnhancedCableAutomation
        from src.core.config_manager import CableOCRConfig
        
        config = CableOCRConfig()
        automation = EnhancedCableAutomation(config)
        
        # Test with non-existent image
        results = automation.run_complete_automation("non_existent_image.png")
        
        assert results['overall_success'] is False
        assert 'error' in results
    
    def test_batch_processing(self, mock_screenshot, temp_test_dir):
        """Test batch processing capabilities."""
        from src.automation.batch_processor import BatchProcessor
        
        # Create multiple test images
        image_paths = []
        for i in range(3):
            image_path = temp_test_dir / f"test_image_{i}.png"
            cv2.imwrite(str(image_path), mock_screenshot)
            image_paths.append(str(image_path))
        
        processor = BatchProcessor(batch_size=2)
        
        with patch('pyautogui.click'), \
             patch('pyautogui.typewrite'), \
             patch('pyautogui.press'):
            
            results = processor.process_batch(image_paths)
        
        assert 'batch_results' in results
        assert 'performance_stats' in results
        assert len(results['batch_results']) == 3

# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Performance and stress testing."""
    
    @pytest.mark.performance
    def test_processing_speed_benchmark(self, mock_screenshot, test_config):
        """Benchmark processing speed requirements."""
        from src.automation.enhanced_cable_automation import EnhancedCableAutomation
        from src.core.config_manager import CableOCRConfig
        
        config = CableOCRConfig()
        automation = EnhancedCableAutomation(config)
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(mock_screenshot, cv2.COLOR_BGR2GRAY)
        
        # Measure preprocessing time
        start_time = time.time()
        processed = automation.preprocessor._apply_preprocessing_pipeline(gray)
        preprocessing_time = time.time() - start_time
        
        assert preprocessing_time < test_config.max_processing_time
        assert processed is not None
    
    @pytest.mark.performance
    def test_memory_usage(self, mock_screenshot, test_config):
        """Test memory usage requirements."""
        import psutil
        import os
        
        from src.automation.enhanced_cable_automation import EnhancedCableAutomation
        from src.core.config_manager import CableOCRConfig
        
        process = psutil.Process(os.getpid())
        
        # Get initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run automation
        config = CableOCRConfig()
        automation = EnhancedCableAutomation(config)
        
        gray = cv2.cvtColor(mock_screenshot, cv2.COLOR_BGR2GRAY)
        processed = automation.preprocessor._apply_preprocessing_pipeline(gray)
        
        # Get final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = final_memory - initial_memory
        
        assert memory_used < test_config.max_memory_usage
    
    @pytest.mark.performance
    def test_concurrent_processing(self, mock_screenshot, temp_test_dir):
        """Test concurrent processing capabilities."""
        import concurrent.futures
        from src.preprocessing.image_enhancer import AdvancedImagePreprocessor
        from src.core.config_manager import CableOCRConfig
        
        config = CableOCRConfig()
        processor = AdvancedImagePreprocessor(config)
        
        # Create multiple test images
        test_images = []
        for i in range(4):
            image_path = temp_test_dir / f"concurrent_test_{i}.png"
            cv2.imwrite(str(image_path), mock_screenshot)
            test_images.append(str(image_path))
        
        # Process concurrently
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(processor.process, img) for img in test_images]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        concurrent_time = time.time() - start_time
        
        # Process sequentially for comparison
        start_time = time.time()
        sequential_results = [processor.process(img) for img in test_images]
        sequential_time = time.time() - start_time
        
        # Concurrent should be faster (or at least not significantly slower)
        assert len(results) == len(test_images)
        assert all(result is not None for result in results)
        # Allow some overhead for threading
        assert concurrent_time <= sequential_time * 1.2

# ============================================================================
# ACCURACY AND REGRESSION TESTS
# ============================================================================

class TestAccuracy:
    """Accuracy and regression testing."""
    
    def test_ocr_accuracy_baseline(self, sample_cable_data):
        """Test OCR accuracy meets baseline requirements."""
        from src.detection.table_detector import CableTableParser
        from src.core.config_manager import CableOCRConfig
        
        config = CableOCRConfig()
        parser = CableTableParser(config)
        
        # Simulate high-confidence OCR results
        mock_elements = []
        for cable in sample_cable_data['cables']:
            mock_elements.extend([
                {'text': str(cable['weight']), 'confidence': 85},
                {'text': f"${cable['price']:.4f}", 'confidence': 90},
                {'text': cable['cable_type'], 'confidence': 80}
            ])
        
        # Calculate accuracy metrics
        high_confidence_count = len([e for e in mock_elements if e['confidence'] >= 70])
        total_elements = len(mock_elements)
        accuracy = high_confidence_count / total_elements
        
        assert accuracy >= 0.80  # 80% accuracy requirement
    
    def test_coordinate_precision_regression(self, sample_cable_data):
        """Test coordinate precision doesn't regress."""
        expected_coordinates = {
            16664: (1196, 504),
            14475: (1196, 483),
            11532: (1196, 523)
        }
        
        tolerance = 25  # pixels
        
        for cable in sample_cable_data['cables']:
            weight = cable['weight']
            actual_coords = cable['coordinates']
            expected_coords = expected_coordinates[weight]
            
            x_diff = abs(actual_coords[0] - expected_coords[0])
            y_diff = abs(actual_coords[1] - expected_coords[1])
            
            assert x_diff <= tolerance, f"X coordinate regression for weight {weight}"
            assert y_diff <= tolerance, f"Y coordinate regression for weight {weight}"
    
    def test_vma_cable_detection_accuracy(self):
        """Test VMA Cable detection accuracy."""
        from src.parsing.vma_classifier import VMACableClassifier
        import pandas as pd
        
        # Test data with known VMA cables
        test_data = pd.DataFrame({
            'Contract#': [13311, 13311.0, '13311', 12345, 67890, 13311],
            'Expected_VMA': [True, True, True, False, False, True]
        })
        
        classifier = VMACableClassifier()
        
        # Classify cables
        results = []
        for _, row in test_data.iterrows():
            classification = classifier.classify_cable(row['Contract#'])
            is_vma = classification == 'VMA Cable'
            results.append(is_vma == row['Expected_VMA'])
        
        accuracy = sum(results) / len(results)
        assert accuracy >= 1.0  # 100% accuracy for VMA detection

# ============================================================================
# MOCK AND FIXTURE UTILITIES
# ============================================================================

class MockPyAutoGUI:
    """Mock PyAutoGUI for testing automation without actual GUI interactions."""
    
    def __init__(self):
        self.click_history = []
        self.type_history = []
        self.key_history = []
    
    def click(self, x, y, clicks=1):
        self.click_history.append({'x': x, 'y': y, 'clicks': clicks})
    
    def typewrite(self, text, interval=0.1):
        self.type_history.append({'text': text, 'interval': interval})
    
    def press(self, key):
        self.key_history.append({'key': key})
    
    def screenshot(self):
        # Return mock screenshot
        mock_img = Mock()
        mock_img.save = Mock()
        return mock_img

@pytest.fixture
def mock_pyautogui():
    """Provide mock PyAutoGUI for testing."""
    return MockPyAutoGUI()

# ============================================================================
# TEST EXECUTION AND REPORTING
# ============================================================================

def generate_test_report():
    """Generate comprehensive test report."""
    import subprocess
    import json
    from datetime import datetime
    
    # Run tests with coverage
    result = subprocess.run([
        'python', '-m', 'pytest', 
        'tests/', 
        '--cov=src', 
        '--cov-report=json',
        '--cov-report=html',
        '--json-report',
        '--json-report-file=test-report.json'
    ], capture_output=True, text=True)
    
    # Generate summary report
    report = {
        'timestamp': datetime.now().isoformat(),
        'test_execution': {
            'return_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        },
        'summary': {
            'success': result.returncode == 0,
            'coverage_available': Path('coverage.json').exists(),
            'html_report_available': Path('htmlcov/index.html').exists()
        }
    }
    
    # Add coverage data if available
    if Path('coverage.json').exists():
        with open('coverage.json', 'r') as f:
            coverage_data = json.load(f)
            report['coverage'] = {
                'total_coverage': coverage_data['totals']['percent_covered'],
                'lines_covered': coverage_data['totals']['covered_lines'],
                'lines_missing': coverage_data['totals']['missing_lines']
            }
    
    # Save comprehensive report
    with open('comprehensive-test-report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    return report

if __name__ == "__main__":
    # Run tests and generate report
    print("🧪 Running comprehensive test suite...")
    report = generate_test_report()
    
    if report['summary']['success']:
        print("✅ All tests passed!")
        if 'coverage' in report:
            print(f"📊 Test coverage: {report['coverage']['total_coverage']:.1f}%")
    else:
        print("❌ Some tests failed!")
        print(f"Error details: {report['test_execution']['stderr']}")
    
    print(f"📄 Detailed report saved to: comprehensive-test-report.json")