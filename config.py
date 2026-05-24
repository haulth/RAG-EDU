"""
Configuration file cho ChatBot
Định nghĩa các đường dẫn và biến cần thiết
"""

import os
from pathlib import Path

# ==============================================================================
# PATHS CONFIGURATION
# ==============================================================================

# Thư mục gốc của project
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# Thư mục chứa models (local)
MODEL_CACHE_PATH = os.path.join(BASE_PATH, "models")
os.makedirs(MODEL_CACHE_PATH, exist_ok=True)

# Thư mục chứa data đã xử lý
PROCESSED_DATA_PATH = os.path.join(BASE_PATH, "processed_data")
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

# Thư mục chứa documents gốc
DOCUMENTS_PATH = os.path.join(BASE_PATH, "documents")
os.makedirs(DOCUMENTS_PATH, exist_ok=True)


# Biến này được Cell 2, 3 yêu cầu
model_cache_path = MODEL_CACHE_PATH

print("="*70)
print("CONFIGURATION LOADED")
print("="*70)
print(f"Base path: {BASE_PATH}")
print(f"Model cache: {MODEL_CACHE_PATH}")
print(f"Processed data: {PROCESSED_DATA_PATH}")
print(f"Documents: {DOCUMENTS_PATH}")
print("="*70)
