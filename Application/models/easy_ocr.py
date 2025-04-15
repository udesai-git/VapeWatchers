import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

# Import libraries
import os
import cv2
import numpy as np
import torch
import easyocr
import time
import re
# Removed matplotlib import as we no longer need visualizations
from pathlib import Path
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import logging
import gc
from tqdm.auto import tqdm
import io
import boto3
from PIL import Image
from botocore.exceptions import ClientError
from IPython import get_ipython


def load_image_from_s3(s3_bucket, s3_key, *, aws_access_key=None, aws_secret_key=None, aws_session_token=None):

    s3_session = boto3.Session(
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        aws_session_token=aws_session_token,
        region_name='us-east-1' 
        )

    s3_client = s3_session.client('s3')
    bucket_name = s3_bucket
    key = s3_key
    print(f"bucketname:{bucket_name} and key: {key}")
    

    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        img_data = response['Body'].read()
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        img = np.array(img)
        img = img[:, :, ::-1]  # RGB to BGR for OpenCV
        return img
    except Exception as e:
        logger.error(f"Failed to load image from S3: {e}")
        return None  


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("vape_ocr.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("VapeOCR")

# GPU Check and Configuration
def check_gpu():
    """Verify CUDA is available and return GPU details"""
    logger.info("Checking GPU status...")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        logger.info(f"GPU detected: {gpu_name}")
        logger.info(f"CUDA version: {cuda_version}")
        logger.info(f"GPU memory: {gpu_memory:.2f} GB")

        # Set default tensor type to CUDA to force GPU usage
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        return True, gpu_memory
    else:
        logger.error("No CUDA GPU detected! Make sure PyTorch is properly installed with CUDA support.")
        return False, 0


class EasyVapeImageOCR:
    # def __init__(self, dataset_path, output_dir="output", num_threads=4,
    #              resize_factor=None, log_level=logging.INFO):
    #     """
    #     Initialize the OCR processor with EasyOCR and CUDA acceleration

    #     Args:
    #         dataset_path: Path to the dataset directory containing brand folders
    #         output_dir: Directory to save results
    #         num_threads: Number of threads for parallel processing
    #         resize_factor: Optional factor to resize images (e.g., 0.5 for half size)
    #         log_level: Logging level (default: INFO)
    #     """
    #     self.dataset_path = dataset_path
    #     self.output_dir = Path(output_dir)
    #     self.output_dir.mkdir(exist_ok=True)

    #     # Configure logging
    #     self.logger = logger
    #     self.logger.setLevel(log_level)

    #     # Force GPU usage for EasyOCR
    #     self.use_cuda, self.gpu_memory_gb = check_gpu()
    #     if not self.use_cuda:
    #         raise RuntimeError("CUDA GPU is required but not available")

    #     self.num_threads = num_threads
    #     self.resize_factor = resize_factor

    #     # Initialize EasyOCR reader with forced GPU usage
    #     self.logger.info("Initializing EasyOCR reader with GPU acceleration...")
    #     try:
    #         self.reader = easyocr.Reader(['en'], gpu=True,
    #                                      quantize=False,  # Disable quantization for better accuracy
    #                                      cudnn_benchmark=True)  # Enable cuDNN benchmarking for speed
    #         self.logger.info("EasyOCR initialized successfully with GPU")
    #     except Exception as e:
    #         self.logger.error(f"Error initializing EasyOCR: {e}")
    #         raise

    #     # Enhanced vape-related keywords for text correction/matching
    #     self.vape_keywords = [
    #         "vape", "vapor", "smoke", "smok", "juul", "puff", "pod", "nicotine",
    #         "e-cigarette", "e-liquid", "e-juice", "check", "out", "these", "new",
    #         "with", "me", "flavor", "cloud", "coil", "tank", "mod", "battery",
    #         "charger", "disposable", "refill", "cartridge", "salt", "nic", "hit",
    #         "device", "ml", "mg", "tobacco", "menthol", "fruity", "dessert", "vaper",
    #         "ohm", "watt", "voltage", "mah", "rechargeable", "draw", "throat"
    #     ]

    #     # Check if dataset path is an S3 path
    #     if self.dataset_path.startswith("s3://"):
    #         self.s3_client = boto3.client('s3')
    #         self.bucket_name, self.prefix = self.extract_bucket_and_prefix(self.dataset_path)
    #         if not self.check_s3_path_exists(self.s3_client, self.dataset_path):
    #             self.logger.error(f"ERROR: S3 path {self.dataset_path} does not exist.")
    #             raise FileNotFoundError(f"S3 path {self.dataset_path} does not exist.")
    #     else:
    #         self.dataset_path = Path(self.dataset_path)

    #     self.logger.info(f"CUDA acceleration: {'Enabled' if self.use_cuda else 'Disabled'}")
    #     self.logger.info(f"Using {self.num_threads} threads for parallel processing")
    
    def __init__(self, dataset_path,s3_client, output_dir="output", num_threads=4,
                resize_factor=None, log_level=logging.INFO, require_cuda=False):
        """
        Initialize the OCR processor with EasyOCR and optional CUDA acceleration

        Args:
            dataset_path: Path to the dataset directory or S3 path
            output_dir: Directory to save results
            num_threads: Number of threads for parallel processing
            resize_factor: Optional factor to resize images (e.g., 0.5 for half size)
            log_level: Logging level (default: INFO)
            require_cuda: If True, raises an error if CUDA is not available
        """
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Configure logging
        self.logger = logger
        self.logger.setLevel(log_level)

        # Check for GPU
        self.use_cuda, self.gpu_memory_gb = check_gpu()

        if require_cuda and not self.use_cuda:
            raise RuntimeError("CUDA GPU is required but not available.")
        elif not self.use_cuda:
            self.logger.warning("CUDA not available. Proceeding with CPU inference.")

        self.num_threads = num_threads
        self.resize_factor = resize_factor

        # Initialize EasyOCR reader
        self.logger.info("Initializing EasyOCR reader...")
        try:
            self.reader = easyocr.Reader(
                ['en'],
                gpu=self.use_cuda,
                quantize=False,
                cudnn_benchmark=True
            )
            self.logger.info(f"EasyOCR initialized successfully with {'GPU' if self.use_cuda else 'CPU'}")
        except Exception as e:
            self.logger.error(f"Error initializing EasyOCR: {e}")
            raise

        # Keywords to assist text correction and filtering
        self.vape_keywords = [
            "vape", "vapor", "smoke", "smok", "juul", "puff", "pod", "nicotine",
            "e-cigarette", "e-liquid", "e-juice", "check", "out", "these", "new",
            "with", "me", "flavor", "cloud", "coil", "tank", "mod", "battery",
            "charger", "disposable", "refill", "cartridge", "salt", "nic", "hit",
            "device", "ml", "mg", "tobacco", "menthol", "fruity", "dessert", "vaper",
            "ohm", "watt", "voltage", "mah", "rechargeable", "draw", "throat"
        ]

        # S3 support
        if self.dataset_path.startswith("s3://"):
            self.s3_client = s3_client
            self.bucket_name, self.prefix = self.extract_bucket_and_prefix(self.dataset_path)
            if not self.check_s3_path_exists(self.s3_client, self.dataset_path):
                self.logger.error(f"ERROR: S3 path {self.dataset_path} does not exist.")
                raise FileNotFoundError(f"S3 path {self.dataset_path} does not exist.")
        else:
            self.dataset_path = Path(self.dataset_path)

        self.logger.info(f"CUDA acceleration: {'Enabled' if self.use_cuda else 'Disabled'}")
        self.logger.info(f"Using {self.num_threads} threads for parallel processing")


    def extract_bucket_and_prefix(self, s3_path):
        """Helper method to extract the bucket and prefix from the s3 path"""
        path_parts = s3_path.replace("s3://", "").split("/", 1)
        return path_parts[0], path_parts[1] if len(path_parts) > 1 else ""

    def check_s3_path_exists(self, s3_client, s3_path):
        """Check if the S3 path exists"""
        bucket_name, prefix = self.extract_bucket_and_prefix(s3_path)
        try:
            response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix, MaxKeys=1)
            return 'Contents' in response
        except Exception as e:
            self.logger.error(f"Error checking S3 path: {e}")
            return False


    def get_brand_folders(self):
        brand_folders = []

        if hasattr(self, 's3_client'):  # Check if we're using S3
            try:
                # List objects in the S3 bucket with the given prefix (dataset path)
                response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=self.prefix, Delimiter='/')
                if 'CommonPrefixes' in response:
                    brand_folders = [prefix['Prefix'] for prefix in response['CommonPrefixes']]
                    self.logger.info(f"Found {len(brand_folders)} brand folders in S3.")
                else:
                    self.logger.warning(f"No folders found in S3 path {self.dataset_path}")
            except Exception as e:
                self.logger.error(f"Error retrieving brand folders from S3: {e}")
        else:
            # Local path handling (for backward compatibility or testing locally)
            folders = [f for f in self.dataset_path.iterdir() if f.is_dir()]
            if not folders:
                self.logger.warning(f"No folders found in {self.dataset_path}")
            brand_folders = folders

        return brand_folders

    def process_dataset(self,aws_access_key=None,aws_secret_key=None,aws_session_token=None, brand_limit=None, batch_size=10, save_intermediate=True):
        """
        Process all images in all brand folders with parallel execution

        Args:
            brand_limit: Optional limit on number of brands to process
            batch_size: Number of images to process in each batch
            save_intermediate: Whether to save results after each brand

        Returns:
            DataFrame with results, processing time, and total image count
        """
        # Get all brand folders (now from S3)
        brand_folders = self.get_brand_folders()
        if brand_limit and brand_limit < len(brand_folders):
            self.logger.info(f"Limiting processing to {brand_limit} brands out of {len(brand_folders)}")
            brand_folders = brand_folders[:brand_limit]

        # self.logger.info(f"Found {len(brand_folders)} brand folders: {[folder.name for folder in brand_folders]}")
        self.logger.info(f"Found {len(brand_folders)} brand folders: {[Path(folder).name for folder in brand_folders]}")

        start_time = time.time()
        results = []
        best_method_stats = {}
        total_images = 0

        # Process each brand folder
        for brand_idx, brand_folder in enumerate(brand_folders):
            if isinstance(brand_folder, str):
                brand_name = Path(brand_folder).name
            else:
                brand_name = brand_folder.name
            self.logger.info(f"\nProcessing brand {brand_idx+1}/{len(brand_folders)}: {brand_name}")

            # Get all image files in the brand folder from S3
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                response = self.s3_client.list_objects_v2(
                    Bucket=self.bucket_name, 
                    Prefix=f"{self.prefix}{brand_name}/"  # Adjust for brand folder structure
                )
                for obj in response.get('Contents', []):
                    if obj['Key'].endswith(ext) or obj['Key'].endswith(ext.upper()):
                        image_files.append(obj['Key'])

            self.logger.info(f"Found {len(image_files)} images in {brand_name} folder")
            total_images += len(image_files)

            # Adaptive thread usage based on GPU memory
            if self.gpu_memory_gb < 6:
                effective_threads = 1
            elif self.gpu_memory_gb < 10:
                effective_threads = 2
            else:
                effective_threads = min(self.num_threads, 4)  # Cap at 4 for GPU processing

            # Adaptive batch size based on GPU memory
            if self.gpu_memory_gb < 4:
                effective_batch_size = min(5, len(image_files))
            elif self.gpu_memory_gb < 8:
                effective_batch_size = min(10, len(image_files))
            else:
                effective_batch_size = min(batch_size, len(image_files))

            self.logger.info(f"Using {effective_threads} threads and batch size of {effective_batch_size}")

            brand_results = []  # Store results for this brand

            # Process images in batches with progress tracking
            batch_count = (len(image_files) + effective_batch_size - 1) // effective_batch_size
            for i in range(0, len(image_files), effective_batch_size):
                batch = image_files[i:i+effective_batch_size]
                self.logger.info(f"Processing batch {i//effective_batch_size + 1}/{batch_count}")

                # Process images in parallel using ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=effective_threads) as executor:
                    # Create a progress bar for this batch
                    batch_progress = tqdm(total=len(batch), desc=f"Batch {i//effective_batch_size + 1}",
                                        leave=False, position=0)

                    # Submit batch for processing
                    future_to_img = {
                        executor.submit(self.process_single_image_wrapper, img_key, brand_name,
                                        aws_access_key, aws_secret_key, aws_session_token): img_key
                        for img_key in batch
                    }


                    # Collect results as they complete
                    for future in future_to_img:
                        img_key = future_to_img[future]
                        try:
                            result = future.result()
                            if result:
                                img_name, brand, text, best_method = result
                                brand_results.append((img_name, brand, text))

                                # Track which methods worked best
                                if best_method in best_method_stats:
                                    best_method_stats[best_method] += 1
                                else:
                                    best_method_stats[best_method] = 1

                                # Update progress bar
                                batch_progress.update(1)
                        except Exception as e:
                            img_name = img_key.split("/")[-1]
                            self.logger.error(f"Error processing {img_name}: {e}")
                            brand_results.append((img_name, brand_name, f"ERROR: {str(e)}"))
                            batch_progress.update(1)

                    # Close progress bar
                    batch_progress.close()

                # Clean up GPU memory after each batch
                torch.cuda.empty_cache()
                gc.collect()  # Explicitly run garbage collection

                # Log memory usage
                allocated = torch.cuda.memory_allocated() / (1024**2)
                max_allocated = torch.cuda.max_memory_allocated() / (1024**2)
                self.logger.info(f"GPU memory: {allocated:.1f}MB / {max_allocated:.1f}MB")

            # Add brand results to overall results
            results.extend(brand_results)

            # Save intermediate results after each brand if requested
            if save_intermediate and brand_results:
                brand_df = pd.DataFrame(brand_results, columns=['image_name', 'brand', 'extracted_text'])
                intermediate_file = self.output_dir / f"{brand_name}_results.csv"
                brand_df.to_csv(intermediate_file, index=False)
                self.logger.info(f"Saved intermediate results for {brand_name} to {intermediate_file}")

                # Also save as pickle for compatibility
                pickle_file = self.output_dir / f"{brand_name}_results.pkl"
                brand_df.to_pickle(pickle_file)

            # Report progress
            elapsed = time.time() - start_time
            brands_left = len(brand_folders) - (brand_idx + 1)
            if brands_left > 0 and (brand_idx + 1) > 0:
                est_time_remaining = (elapsed / (brand_idx + 1)) * brands_left
                time_per_brand = elapsed / (brand_idx + 1)
                self.logger.info(f"Progress: {brand_idx+1}/{len(brand_folders)} brands processed")
                self.logger.info(f"Time per brand: {time_per_brand/60:.1f} minutes")
                self.logger.info(f"Estimated time remaining: {est_time_remaining/60:.1f} minutes")

        # Print statistics about which methods worked best
        self.logger.info("\nBest method statistics:")
        for method, count in sorted(best_method_stats.items(), key=lambda x: x[1], reverse=True):
            self.logger.info(f"{method}: {count} images ({count/max(total_images, 1)*100:.1f}%)")

        # Calculate processing time
        end_time = time.time()
        processing_time = end_time - start_time
        self.logger.info(f"\nProcessed {total_images} images in {processing_time:.2f} seconds")
        self.logger.info(f"Average time per image: {processing_time/max(total_images, 1):.2f} seconds")

        # Convert results to DataFrame with brand information
        results_df = pd.DataFrame(results, columns=['image_name', 'brand', 'extracted_text'])

        # Save final results in both CSV and pickle formats
        final_csv = self.output_dir / "vape_ocr_results.csv"
        final_pkl = self.output_dir / "vape_ocr_results.pkl"

        results_df.to_csv(final_csv, index=False)
        results_df.to_pickle(final_pkl)

        self.logger.info(f"Saved complete results to {final_csv} and {final_pkl}")
        
        csv_buffer = io.BytesIO()
        results_df.to_csv(csv_buffer, index=False)

        s3_session = boto3.Session(
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        aws_session_token=aws_session_token,
        region_name='us-east-1' 
        )

        s3_client = s3_session.client('s3')
        bucket_name = "vapewatchers-2025"
        key_name = "vape_ocr_results.csv"

        s3_client.put_object(Bucket=bucket_name, Key=key_name, Body=csv_buffer.getvalue(), ContentType="text/csv")
        print(f"✅ OCR results saved to s3://{bucket_name}/{key_name}")
        
        # Count duplicates in image_name
        image_name_duplicates = results_df['image_name'].duplicated().sum()
        print(f"Number of duplicate image_name entries: {image_name_duplicates}")

        # Count how many image_name values appear more than once
        duplicate_image_names = results_df['image_name'].value_counts()
        duplicate_image_names = duplicate_image_names[duplicate_image_names > 1]
        print(f"Number of image_name values that have duplicates: {len(duplicate_image_names)}")
        print(f"Total duplicate instances: {duplicate_image_names.sum() - len(duplicate_image_names)}")

        # Count duplicates in the combination (image_name, brand)
        image_brand_duplicates = results_df.duplicated(subset=['image_name', 'brand']).sum()
        print(f"Number of duplicate (image_name, brand) entries: {image_brand_duplicates}")

        # Count how many (image_name, brand) combinations appear more than once
        duplicate_image_brands = results_df.groupby(['image_name', 'brand']).size()
        duplicate_image_brands = duplicate_image_brands[duplicate_image_brands > 1]
        print(f"Number of (image_name, brand) combinations that have duplicates: {len(duplicate_image_brands)}")
        print(f"Total duplicate instances: {duplicate_image_brands.sum() - len(duplicate_image_brands)}")

        return results_df, processing_time, total_images
    
    def get_s3_bucket_and_key(self,s3_uri):
        
        if not s3_uri.startswith("s3://"):
            raise ValueError("Invalid S3 URI: Must start with 's3://'")

        parts = s3_uri.replace("s3://", "").split("/", 1)
        if len(parts) != 2:
            raise ValueError("Invalid S3 URI format")

        bucket = parts[0]
        key = parts[1]
        return bucket, key

    def process_single_image_wrapper(self, img_path, brand_name, aws_access_key=None, aws_secret_key=None, aws_session_token=None):
        try:
            print(f"Received image path: {img_path}")
            img_name = os.path.basename(img_path)

            # If it's from S3 (not a local file), build full S3 URI
            if not os.path.exists(str(img_path)):
                s3_uri = f"s3://{self.bucket_name}/{img_path}"  # Construct full S3 URI
                s3_bucket, s3_key = self.get_s3_bucket_and_key(s3_uri)
                img = load_image_from_s3(
                    s3_bucket, s3_key,
                    aws_access_key=aws_access_key,
                    aws_secret_key=aws_secret_key,
                    aws_session_token=aws_session_token
                )
            else:
                img = cv2.imread(str(img_path))

            if img is None:
                self.logger.error(f"Error: Could not read image {img_name}")
                return None

            if self.resize_factor and self.resize_factor != 1.0:
                h, w = img.shape[:2]
                img = cv2.resize(img, (int(w * self.resize_factor), int(h * self.resize_factor)))

            text, best_method = self.process_single_image(img_name, img)
            return img_name, brand_name, text, best_method

        except Exception as e:
            img_name = os.path.basename(img_path)
            self.logger.error(f"Error in wrapper for {img_name}: {e}")
            return None


    # def process_single_image_wrapper(self,img_path, brand_name,aws_access_key=None,aws_secret_key=None,aws_session_token=None):
    #     try:
    #         print(img_path)
    #         img_name = os.path.basename(img_path)
    #         if img_path.startswith("s3://") or not os.path.exists(str(img_path)):
    #             # Handle S3
    #             s3_bucket, s3_key = self.get_s3_bucket_and_key(img_path)
    #             img = load_image_from_s3(s3_bucket, s3_key,aws_access_key=aws_access_key,aws_secret_key=aws_secret_key,aws_session_token=aws_session_token)
    #         else:
    #             # Handle local image
    #             img = cv2.imread(str(img_path))


    #         if img is None:
    #             self.logger.error(f"Error: Could not read image {img_name}")
    #             return None

    #         # Apply resizing if specified
    #         if self.resize_factor and self.resize_factor != 1.0:
    #             h, w = img.shape[:2]
    #             img = cv2.resize(img, (int(w * self.resize_factor), int(h * self.resize_factor)))

    #         # Process the image
    #         text, best_method = self.process_single_image(img_name, img)
    #         return img_name, brand_name, text, best_method
    #     except Exception as e:
    #         img_name = os.path.basename(img_path)
    #         self.logger.error(f"Error in wrapper for {img_name}: {e}")
    #         return None


    def process_single_image(self, filename, img):
        """
        Process a single image with EasyOCR

        Args:
            filename: Name of the image file
            img: OpenCV image object

        Returns:
            Cleaned OCR text and the method that produced the best result
        """
        # Save original for visualization
        original_img = img.copy()

        # Image size
        height, width = img.shape[:2]

        # Store all results with their respective methods
        all_results = {}
        all_raw_results = {}

        # ----------------------------------------
        # 1. Basic preprocessing
        # ----------------------------------------

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filter for noise reduction
        bilateral = cv2.bilateralFilter(gray, 11, 17, 17)

        # ----------------------------------------
        # 2. Enhanced image preprocessing techniques
        # ----------------------------------------

        # 2.1 Basic thresholding
        _, binary = cv2.threshold(bilateral, 127, 255, cv2.THRESH_BINARY)

        # 2.2 Adaptive thresholding
        adaptive = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)

        # 2.3 Otsu's thresholding
        _, otsu = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 2.4 CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray)

        # 2.5 Additional preprocessing: Edge enhancement
        # Canny edge detection for text boundaries
        edges = cv2.Canny(bilateral, 50, 150)
        # Dilate to connect nearby edges
        kernel = np.ones((3,3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)

        # 2.6 Morphological operations to improve text regions
        # Create a morphological gradient to highlight text regions
        gradient_kernel = np.ones((3,3), np.uint8)
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, gradient_kernel)

        # 2.7 Denoising for cleaner images
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

        # Create a collection of processed images
        processed_images = {
            "original": img,
            "grayscale": gray,
            "bilateral": bilateral,
            "binary": binary,
            "adaptive": adaptive,
            "otsu": otsu,
            "clahe": clahe_img,
            "edges": dilated_edges,
            "gradient": gradient,
            "denoised": denoised,
        }

        # ----------------------------------------
        # 3. Apply EasyOCR with various preprocessing methods
        # ----------------------------------------

        # Define confidence thresholds
        high_conf_threshold = 0.5
        mid_conf_threshold = 0.3
        low_conf_threshold = 0.1

        # Try original image first
        try:
            results = self.reader.readtext(img)
            method_key = "original"
            all_raw_results[method_key] = results

            # Extract high confidence text only
            high_conf_text = self._extract_text_from_results(results, high_conf_threshold)
            if high_conf_text:
                all_results[method_key] = high_conf_text

            # If high confidence gave minimal results, try with mid threshold
            if len(high_conf_text.split()) < 3:
                mid_conf_text = self._extract_text_from_results(results, mid_conf_threshold)
                if mid_conf_text and len(mid_conf_text.split()) > len(high_conf_text.split()):
                    all_results[f"{method_key}_mid_conf"] = mid_conf_text
        except Exception as e:
            self.logger.error(f"Error with original image: {e}")

        # ----------------------------------------
        # 4. Try advanced processing methods
        # ----------------------------------------
        # For each processed image, apply EasyOCR
        for img_name, proc_img in processed_images.items():
            # Skip original image (already processed) and color images
            if img_name == "original" or (len(proc_img.shape) > 2 and img_name != "original"):
                continue

            # Apply EasyOCR
            try:
                results = self.reader.readtext(proc_img)
                method_key = f"{img_name}"
                all_raw_results[method_key] = results

                # Extract text with high confidence
                high_conf_text = self._extract_text_from_results(results, high_conf_threshold)
                if high_conf_text:
                    all_results[method_key] = high_conf_text

                # If high confidence gave no or minimal results, try with mid threshold
                if len(high_conf_text.split()) < 3:
                    mid_conf_text = self._extract_text_from_results(results, mid_conf_threshold)
                    if mid_conf_text and len(mid_conf_text.split()) > len(high_conf_text.split()):
                        all_results[f"{img_name}_mid_conf"] = mid_conf_text

                # If still minimal text, try with lower threshold as last resort
                if img_name in ["clahe", "denoised", "gradient"] and len(high_conf_text.split()) < 2:
                    low_conf_text = self._extract_text_from_results(results, low_conf_threshold)
                    if low_conf_text and len(low_conf_text.split()) > 3:  # Only if got substantial text
                        all_results[f"{img_name}_low_conf"] = low_conf_text
            except Exception as e:
                self.logger.error(f"Error with {img_name}: {e}")

        # ----------------------------------------
        # 5. Try resizing for difficult images
        # ----------------------------------------
        # If still no results or minimal results, try resizing
        if not all_results or all(len(text.split()) <= 2 for text in all_results.values()):
            resize_factors = [1.5, 2.0, 3.0]
            for factor in resize_factors:
                # Resize the image
                resized = cv2.resize(img, (int(width * factor), int(height * factor)))

                try:
                    results = self.reader.readtext(resized)
                    method_key = f"original_resized_{factor}"
                    all_raw_results[method_key] = results

                    # Try with mid confidence threshold for resized images
                    text = self._extract_text_from_results(results, mid_conf_threshold)
                    if text and len(text.split()) > 2:  # Only keep if got substantial text
                        all_results[method_key] = text
                        break  # If we found good results, stop trying bigger sizes
                except Exception as e:
                    self.logger.error(f"Error with resized_{factor}: {e}")

        # ----------------------------------------
        # 6. Find the best result using improved scoring
        # ----------------------------------------

        # Score each result based on keyword matches, text coherence, and length
        scored_results = {}

        for method, text in all_results.items():
            # Skip empty results
            if not text:
                continue

            # Calculate score based on recognized vape keywords and text quality
            score = self._calculate_text_score(text)
            scored_results[method] = (text, score)

        # If no results with text, return empty string
        if not scored_results:
            self.logger.warning(f"No text detected in {filename}")
            return "", "no_text_detected"

        # Sort by score (highest first)
        sorted_results = sorted(scored_results.items(), key=lambda x: x[1][1], reverse=True)

        # Get the best result
        best_method, (best_text, best_score) = sorted_results[0]

        self.logger.debug(f"Best method for {filename}: {best_method} (score: {best_score:.2f})")

        # Skip visualization and text file output - just return the results
        # Clean up GPU memory
        del processed_images

        return best_text, best_method

    def _extract_text_from_results(self, results, conf_threshold=0.5):
        """Extract text from EasyOCR results with confidence above threshold"""
        if not results:
            return ""

        text_fragments = []

        # Sort results by vertical position (top to bottom)
        # This ensures text is read in the correct order
        sorted_results = sorted(results, key=lambda x: x[0][0][1])  # Sort by y-coordinate

        # Group text by lines (similar y-coordinates)
        line_tolerance = 15  # Maximum y-difference to consider texts on same line
        lines = []
        current_line = []
        current_y = None

        for bbox, text, prob in sorted_results:
            if prob < conf_threshold:
                continue

            # Get the y-coordinate (vertical position)
            y_coord = sum(point[1] for point in bbox) / 4  # Average y-coordinate

            if current_y is None:
                current_y = y_coord
                current_line.append((bbox, text, prob))
            elif abs(y_coord - current_y) <= line_tolerance:
                # Same line
                current_line.append((bbox, text, prob))
            else:
                # New line
                lines.append(current_line)
                current_line = [(bbox, text, prob)]
                current_y = y_coord

        # Add the last line
        if current_line:
            lines.append(current_line)

        # For each line, sort by x-coordinate (left to right)
        for line in lines:
            # Sort by average x-coordinate
            sorted_line = sorted(line, key=lambda x: sum(point[0] for point in x[0]) / 4)
            # Join text in this line
            line_text = " ".join(text for _, text, _ in sorted_line)
            text_fragments.append(line_text)

        # Join all lines
        combined_text = " ".join(text_fragments)

        # Clean the combined text
        return self._clean_ocr_text(combined_text)

    def _calculate_text_score(self, text):
        """Calculate a score for OCR text based on keyword matches and text coherence"""
        # Start with base score
        score = 0

        # Lowercase for comparison
        text_lower = text.lower()
        word_count = len(text.split())

        # Count keyword matches with bonus for multiple matches
        keyword_matches = 0
        for keyword in self.vape_keywords:
            if keyword in text_lower:
                keyword_matches += 1

        # Add scaled score based on keyword matches (diminishing returns)
        if keyword_matches > 0:
            score += min(5, keyword_matches + (keyword_matches / 10))

        # Add points for longer text (usually more information) with diminishing returns
        if word_count > 0:
            length_score = min(4, 2 * (1 - (1 / word_count)))
            score += length_score

        # Add points for complete sentences and indicators of professional text
        sentence_score = 0
        if re.search(r'[A-Z][^.!?]*[.!?]', text):
            sentence_score += 2

        # Check for URLs or product identifiers that might be relevant
        if re.search(r'www\.|http|\.com|product|model|id', text_lower):
            sentence_score += 1.5

        # Check for numbers that might be significant (percentages, volumes)
        if re.search(r'\d+%|\d+ml|\d+mg', text_lower):
            sentence_score += 1

        score += sentence_score

        # Penalize for excessive special characters (generally lower quality text)
        special_char_ratio = len(re.findall(r'[^\w\s]', text)) / max(len(text), 1)
        score += (1 - min(special_char_ratio, 1)) * 2

        # Bonus for text that looks like natural language
        words = text_lower.split()
        if word_count >= 3:
            # Check for common stop words that indicate natural language
            stop_words = ['the', 'and', 'a', 'an', 'to', 'in', 'for', 'with', 'of', 'on', 'at', 'by']
            stop_word_count = sum(1 for word in words if word in stop_words)
            if stop_word_count >= 2:
                score += 1.5

        return score

    def _clean_ocr_text(self, text):
        """Clean OCR text to improve quality"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Replace common OCR errors
        replacements = {
            '|': 'I',       # vertical bar to I
            '[': '(',       # bracket to parenthesis
            ']': ')',       # bracket to parenthesis
            '{': '(',       # brace to parenthesis
            '}': ')',       # brace to parenthesis
            '0': 'O',       # in some contexts
            'l': 'I',       # in some contexts
            'rnl': 'ml',    # common for volume units
            '°/o': '%',     # percent sign
            'rng': 'mg',    # milligrams
        }

        # Apply replacements
        for old, new in replacements.items():
            # Only replace standalone characters to avoid breaking words
            if len(old) == 1 and len(new) == 1:
                pattern = r'\b' + re.escape(old) + r'\b'
                text = re.sub(pattern, new, text)
            else:
                text = text.replace(old, new)

        # Try to reconstruct common vape phrases
        common_phrases = {
            r'[Cc]h.?ck\s*o.?t': 'Check out',
            r'[Vv]ap.?\s*w[il].?h': 'Vape with',
            r'[Tt]h.?s.?\s*n.?w': 'These new',
            r'[Nn][il]c[0o]t[il]n[ce]': 'Nicotine',
            r'[Dd][il]sp[0o]s[a4]b[il][ce]': 'Disposable',
            r'[Ee]-?[Cc][il]g[a4]r[ce]tt[ce]': 'E-cigarette',
            r'[Ff]l[a4]v[o0]r': 'Flavor',
            r'[Bb][a4]tt[ce]ry': 'Battery',
        }

        # Apply replacements
        for pattern, replacement in common_phrases.items():
            text = re.sub(pattern, replacement, text)

        # Fix mL and mg units
        text = re.sub(r'(\d+)\s*[mM][lL]', r'\1ml', text)
        text = re.sub(r'(\d+)\s*[mM][gG]', r'\1mg', text)

        # Remove repeated punctuation
        text = re.sub(r'([.!?,;:]){2,}', r'\1', text)

        # Fix capitalization issues
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s[0].upper() + s[1:] if len(s) > 0 else s for s in sentences]
        text = ' '.join(sentences)

        return text

    # Removed visualization method as it's no longer needed

# Determine optimal batch size based on GPU memory
def get_optimal_batch_size():
    """Determine optimal batch size based on GPU memory"""
    if not torch.cuda.is_available():
        return 5  # Default safe value

    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    # Use smaller batch sizes for smaller GPU memory
    if gpu_memory_gb < 4:
        return 5
    elif gpu_memory_gb < 8:
        return 8
    elif gpu_memory_gb < 12:
        return 12
    else:
        return 16

def s3_path_exists(s3_path,aws_access_key,aws_secret_key,aws_session_token):
    """Check if an S3 path exists."""
    s3_session = boto3.Session(
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        aws_session_token=aws_session_token,
        region_name='us-east-1' 
    )

    s3_client = s3_session.client('s3')
    bucket_name = s3_path.split('/')[2]  # Extract bucket name from the S3 path
    prefix = '/'.join(s3_path.split('/')[3:])  # Extract prefix (folder path)

    try:
        result = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        return 'Contents' in result  # If there's content in the path, it's valid
    except ClientError:
        return False



# def main(dataset_path, output_dir="output", brand_limit=None):
#     """Main function to run the OCR pipeline"""
#     # First, check if the dataset path exists
#     if not os.path.exists(dataset_path):
#         logger.error(f"ERROR: Dataset path {dataset_path} does not exist. Please update the path.")
#         return

#     # Determine optimal batch size based on GPU memory
#     optimal_batch_size = get_optimal_batch_size()
#     logger.info(f"Optimal batch size for your GPU: {optimal_batch_size}")

#     # Create output directory
#     os.makedirs(output_dir, exist_ok=True)

#     # Initialize OCR with GPU settings
#     ocr = EasyVapeImageOCR(
#         dataset_path=dataset_path,
#         output_dir=output_dir,
#         num_threads=2,  # Using 2 threads works well with GPU
#         resize_factor=None  # Set to 0.5 if you have memory issues
#     )

#     # Process the dataset
#     results_df, processing_time, num_images = ocr.process_dataset(
#         brand_limit=brand_limit,
#         batch_size=optimal_batch_size,
#         save_intermediate=True
#     )

#     # Display summary of results
#     logger.info(f"\nProcessed {num_images} images in {processing_time:.2f} seconds")
#     logger.info(f"Average time per image: {processing_time/max(num_images, 1):.2f} seconds")

#     # Display DataFrame preview
#     logger.info("\nResults preview:")
#     if len(results_df) > 0:
#         print(results_df.head())
#     else:
#         logger.warning("No results were obtained")

#     # Show brand summary
#     logger.info("\nResults by brand:")
#     brand_counts = results_df.groupby('brand').size()
#     print(brand_counts)

#     # Basic text analysis
#     if len(results_df) > 0:
#         logger.info("\nText length statistics:")
#         results_df['text_length'] = results_df['extracted_text'].str.len()
#         print(results_df['text_length'].describe())

#         # Count empty results
#         empty_count = results_df[results_df['extracted_text'] == ''].shape[0]
#         logger.info(f"Empty results: {empty_count} ({empty_count/len(results_df)*100:.1f}%)")

#     logger.info("\nOCR processing complete.")
#     return results_df
def main(dataset_path, output_dir="output",*,
         aws_access_key=None, aws_secret_key=None, aws_session_token=None):

    s3_session = boto3.Session(
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        aws_session_token=aws_session_token,
        region_name='us-east-1' 
    )

    s3_client = s3_session.client('s3')

    bucket_name = 'vapewatchers-2025'
    prefix = 'MarketingImages'
    
    """Main function to run the OCR pipeline"""
    # First, check if the dataset path exists
    if not s3_path_exists(dataset_path,aws_access_key,aws_secret_key,aws_session_token):
        logger.error(f"ERROR: Dataset path {dataset_path} does not exist. Please update the path.")
        return

    # Determine optimal batch size based on GPU memory
    optimal_batch_size = 32  # Adjust this based on your GPU
    logger.info(f"Optimal batch size for your GPU: {optimal_batch_size}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize OCR with GPU settings
    # ocr = easyocr.Reader(['en'], gpu=True, quantize=False, cudnn_benchmark=True)
    logger.info("OCR initialized with GPU support.")
    
    ocr = EasyVapeImageOCR(
        dataset_path=dataset_path,
        output_dir=output_dir,
        num_threads=2,
        resize_factor=None,
        s3_client=s3_client
    )

    # Process the dataset
    results_df, processing_time, num_images = ocr.process_dataset(
        brand_limit=10,
        batch_size=optimal_batch_size,
        save_intermediate=True,
        aws_access_key=aws_access_key,
        aws_secret_key=aws_secret_key,
        aws_session_token=aws_session_token
    )

# Determine if we're running in a Jupyter environment
def is_jupyter():
    try:
        shell = get_ipython().__class__.__name__
        return shell in ['ZMQInteractiveShell', 'TerminalInteractiveShell']
    except (NameError, ImportError):
        return False
