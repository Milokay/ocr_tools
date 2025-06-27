# 🚀 OCR Performance Optimization Tools

import time
import functools
from typing import Callable, Any
import cv2
import numpy as np
import concurrent.futures
from dataclasses import dataclass
import psutil
import threading

# ============================================================================
# PERFORMANCE MONITORING DECORATORS
# ============================================================================

def timing_decorator(func: Callable) -> Callable:
    """Decorator to measure function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"⏱️ {func.__name__} completed in {execution_time:.2f} seconds")
        
        # Store timing in result if it's a dict
        if isinstance(result, dict):
            result['execution_time'] = execution_time
        
        return result
    return wrapper

def memory_monitor(func: Callable) -> Callable:
    """Decorator to monitor memory usage."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        result = func(*args, **kwargs)
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before
        
        print(f"🧠 {func.__name__} memory usage: {memory_used:.1f} MB")
        
        return result
    return wrapper

# ============================================================================
# PARALLEL PROCESSING OPTIMIZATION
# ============================================================================

class ParallelOCRProcessor:
    """Parallel processing for multiple image regions."""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
    
    def process_image_regions(self, image: np.ndarray, regions: list) -> dict:
        """Process multiple image regions in parallel."""
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks for each region
            future_to_region = {}
            for i, region in enumerate(regions):
                x, y, w, h = region
                roi = image[y:y+h, x:x+w]
                future = executor.submit(self._process_single_region, roi, i)
                future_to_region[future] = region
            
            # Collect results
            results = {}
            for future in concurrent.futures.as_completed(future_to_region):
                region = future_to_region[future]
                try:
                    result = future.result()
                    results[f"region_{result['region_id']}"] = result
                except Exception as e:
                    print(f"❌ Region processing failed: {e}")
        
        return results
    
    def _process_single_region(self, roi: np.ndarray, region_id: int) -> dict:
        """Process a single image region."""
        import pytesseract
        
        # OCR processing
        text = pytesseract.image_to_string(roi)
        data = pytesseract.image_to_data(roi, output_type=pytesseract.Output.DICT)
        
        return {
            'region_id': region_id,
            'text': text.strip(),
            'element_count': len([t for t in data['text'] if t.strip()]),
            'confidence_avg': np.mean([c for c in data['conf'] if c > 0])
        }

# ============================================================================
# IMAGE CACHING AND PREPROCESSING OPTIMIZATION
# ============================================================================

class ImageCache:
    """LRU cache for preprocessed images."""
    
    def __init__(self, max_size: int = 10):
        self.cache = {}
        self.access_order = []
        self.max_size = max_size
    
    def get(self, image_hash: str) -> np.ndarray:
        """Get cached preprocessed image."""
        if image_hash in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(image_hash)
            self.access_order.append(image_hash)
            return self.cache[image_hash]
        return None
    
    def put(self, image_hash: str, processed_image: np.ndarray):
        """Cache preprocessed image."""
        if len(self.cache) >= self.max_size:
            # Remove least recently used
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        
        self.cache[image_hash] = processed_image
        self.access_order.append(image_hash)
    
    def get_image_hash(self, image: np.ndarray) -> str:
        """Generate hash for image caching."""
        return str(hash(image.tobytes()))

class OptimizedPreprocessor:
    """Optimized image preprocessing with caching."""
    
    def __init__(self, cache_size: int = 10):
        self.cache = ImageCache(cache_size)
        self.processing_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_processing_time': 0
        }
    
    @timing_decorator
    @memory_monitor
    def preprocess_with_cache(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image with caching optimization."""
        
        # Check cache first
        image_hash = self.cache.get_image_hash(image)
        cached_result = self.cache.get(image_hash)
        
        if cached_result is not None:
            self.processing_stats['cache_hits'] += 1
            print(f"🎯 Cache hit! Using cached preprocessing result")
            return cached_result
        
        # Cache miss - process image
        self.processing_stats['cache_misses'] += 1
        print(f"🔄 Cache miss - processing image...")
        
        start_time = time.time()
        processed = self._advanced_preprocessing(image)
        processing_time = time.time() - start_time
        
        self.processing_stats['total_processing_time'] += processing_time
        
        # Cache the result
        self.cache.put(image_hash, processed)
        
        return processed
    
    def _advanced_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """Advanced preprocessing pipeline."""
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply optimized preprocessing steps
        # (Same as in the main enhancement, but optimized)
        
        # 1. Noise reduction (optimized parameters)
        denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # 2. Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # 3. Sharpening filter
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # 4. Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        return binary
    
    def get_cache_stats(self) -> dict:
        """Get caching performance statistics."""
        total_requests = self.processing_stats['cache_hits'] + self.processing_stats['cache_misses']
        cache_hit_rate = (self.processing_stats['cache_hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'cache_hits': self.processing_stats['cache_hits'],
            'cache_misses': self.processing_stats['cache_misses'],
            'cache_hit_rate': f"{cache_hit_rate:.1f}%",
            'total_processing_time': f"{self.processing_stats['total_processing_time']:.2f}s",
            'avg_processing_time': f"{self.processing_stats['total_processing_time'] / max(self.processing_stats['cache_misses'], 1):.2f}s"
        }

# ============================================================================
# BATCH PROCESSING OPTIMIZATION
# ============================================================================

class BatchProcessor:
    """Optimized batch processing for multiple screenshots."""
    
    def __init__(self, batch_size: int = 5):
        self.batch_size = batch_size
        self.preprocessor = OptimizedPreprocessor()
        self.parallel_processor = ParallelOCRProcessor()
    
    def process_batch(self, image_paths: list) -> dict:
        """Process multiple images in optimized batches."""
        
        results = {
            'batch_results': [],
            'performance_stats': {},
            'total_processing_time': 0
        }
        
        start_time = time.time()
        
        # Process in batches
        for i in range(0, len(image_paths), self.batch_size):
            batch = image_paths[i:i + self.batch_size]
            print(f"🔄 Processing batch {i//self.batch_size + 1}: {len(batch)} images")
            
            batch_results = self._process_single_batch(batch)
            results['batch_results'].extend(batch_results)
        
        # Calculate performance stats
        total_time = time.time() - start_time
        results['total_processing_time'] = total_time
        results['performance_stats'] = {
            'total_images': len(image_paths),
            'total_time': f"{total_time:.2f}s",
            'avg_time_per_image': f"{total_time / len(image_paths):.2f}s",
            'images_per_second': f"{len(image_paths) / total_time:.2f}",
            'cache_stats': self.preprocessor.get_cache_stats()
        }
        
        return results
    
    def _process_single_batch(self, image_paths: list) -> list:
        """Process a single batch of images."""
        batch_results = []
        
        for image_path in image_paths:
            try:
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    continue
                
                # Preprocess with caching
                processed = self.preprocessor.preprocess_with_cache(image)
                
                # Define regions for parallel processing
                h, w = processed.shape
                regions = [
                    (0, 0, w//2, h//2),        # Top-left
                    (w//2, 0, w//2, h//2),     # Top-right
                    (0, h//2, w//2, h//2),     # Bottom-left
                    (w//2, h//2, w//2, h//2)   # Bottom-right
                ]
                
                # Process regions in parallel
                region_results = self.parallel_processor.process_image_regions(processed, regions)
                
                batch_results.append({
                    'image_path': image_path,
                    'regions': region_results,
                    'total_elements': sum(r.get('element_count', 0) for r in region_results.values()),
                    'avg_confidence': np.mean([r.get('confidence_avg', 0) for r in region_results.values()])
                })
                
            except Exception as e:
                print(f"❌ Error processing {image_path}: {e}")
                batch_results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return batch_results

# ============================================================================
# ADAPTIVE THRESHOLDING OPTIMIZATION
# ============================================================================

class AdaptiveOCROptimizer:
    """Adaptive optimization based on image characteristics."""
    
    def __init__(self):
        self.optimization_history = []
    
    def optimize_for_image(self, image: np.ndarray) -> dict:
        """Automatically optimize OCR parameters for specific image."""
        
        # Analyze image characteristics
        characteristics = self._analyze_image_characteristics(image)
        
        # Determine optimal parameters
        optimal_params = self._calculate_optimal_parameters(characteristics)
        
        # Test different configurations
        best_config = self._test_configurations(image, optimal_params)
        
        # Store optimization history
        self.optimization_history.append({
            'characteristics': characteristics,
            'best_config': best_config,
            'timestamp': time.time()
        })
        
        return best_config
    
    def _analyze_image_characteristics(self, image: np.ndarray) -> dict:
        """Analyze image characteristics for optimization."""
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        return {
            'mean_brightness': np.mean(gray),
            'contrast': np.std(gray),
            'sharpness': cv2.Laplacian(gray, cv2.CV_64F).var(),
            'noise_level': np.std(gray - cv2.GaussianBlur(gray, (5,5), 0)),
            'text_density': self._estimate_text_density(gray)
        }
    
    def _estimate_text_density(self, gray: np.ndarray) -> float:
        """Estimate text density in image."""
        
        # Simple edge-based text density estimation
        edges = cv2.Canny(gray, 50, 150)
        return np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
    
    def _calculate_optimal_parameters(self, characteristics: dict) -> dict:
        """Calculate optimal OCR parameters based on characteristics."""
        
        base_config = '--oem 3 --psm 6'
        
        # Adjust based on characteristics
        if characteristics['contrast'] < 50:
            # Low contrast - use different preprocessing
            preprocessing = 'enhance_contrast'
        elif characteristics['noise_level'] > 20:
            # High noise - use noise reduction
            preprocessing = 'noise_reduction'
        else:
            # Normal image
            preprocessing = 'standard'
        
        # Adjust PSM based on text density
        if characteristics['text_density'] > 0.3:
            psm = '6'  # Uniform text block
        elif characteristics['text_density'] > 0.1:
            psm = '7'  # Single text line
        else:
            psm = '8'  # Single word
        
        return {
            'ocr_config': f'--oem 3 --psm {psm}',
            'preprocessing': preprocessing,
            'confidence_threshold': 50 if characteristics['sharpness'] > 100 else 30
        }
    
    def _test_configurations(self, image: np.ndarray, base_params: dict) -> dict:
        """Test different configurations and return the best one."""
        
        import pytesseract
        
        configurations = [
            {**base_params, 'confidence_threshold': 30},
            {**base_params, 'confidence_threshold': 50},
            {**base_params, 'confidence_threshold': 70},
        ]
        
        best_config = base_params
        best_score = 0
        
        for config in configurations:
            try:
                # Test OCR with this configuration
                data = pytesseract.image_to_data(
                    image, 
                    output_type=pytesseract.Output.DICT,
                    config=config['ocr_config']
                )
                
                # Calculate quality score
                confidences = [c for c in data['conf'] if c > config['confidence_threshold']]
                texts = [t for t in data['text'] if t.strip()]
                
                if confidences and texts:
                    avg_confidence = np.mean(confidences)
                    text_count = len(texts)
                    score = avg_confidence * (text_count / 10)  # Weighted score
                    
                    if score > best_score:
                        best_score = score
                        best_config = config
                        
            except Exception as e:
                print(f"⚠️ Configuration test failed: {e}")
        
        best_config['quality_score'] = best_score
        return best_config

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def demo_performance_optimization():
    """Demonstrate performance optimization features."""
    
    print("🚀 OCR Performance Optimization Demo")
    print("=" * 50)
    
    # Example batch processing
    image_paths = [
        "screenshot1.png",
        "screenshot2.png", 
        "screenshot3.png"
    ]
    
    # Initialize batch processor
    batch_processor = BatchProcessor(batch_size=2)
    
    # Process batch (if images exist)
    if all(Path(p).exists() for p in image_paths):
        results = batch_processor.process_batch(image_paths)
        
        print("📊 Batch Processing Results:")
        print(f"  Total images: {results['performance_stats']['total_images']}")
        print(f"  Total time: {results['performance_stats']['total_time']}")
        print(f"  Images/second: {results['performance_stats']['images_per_second']}")
        print(f"  Cache hit rate: {results['performance_stats']['cache_stats']['cache_hit_rate']}")
    
    # Example adaptive optimization
    if Path("test_image.png").exists():
        image = cv2.imread("test_image.png")
        optimizer = AdaptiveOCROptimizer()
        
        optimal_config = optimizer.optimize_for_image(image)
        print(f"\n🎯 Optimal Configuration:")
        print(f"  OCR Config: {optimal_config['ocr_config']}")
        print(f"  Preprocessing: {optimal_config['preprocessing']}")
        print(f"  Confidence Threshold: {optimal_config['confidence_threshold']}")
        print(f"  Quality Score: {optimal_config['quality_score']:.2f}")

if __name__ == "__main__":
    demo_performance_optimization()