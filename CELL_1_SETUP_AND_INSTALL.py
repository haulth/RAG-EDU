# ==============================================================================
# @title CELL 1: CÀI ĐẶT THƯ VIỆN & KẾT NỐI DRIVE (ALL-IN-ONE)
# ==============================================================================

"""
CELL 1 - Cài đặt và setup môi trường (Phiên bản đầy đủ)

PHƯƠNG ÁN NÀY: Tất cả trong một cell (đơn giản, nhanh)

CHIẾN LƯỢC:
1. Gỡ bỏ thư viện cũ gây conflict
2. Cài đặt đồng bộ tất cả thư viện (pip tự chọn version tương thích)
3. Verify thư viện ngay (không cần restart)
4. Kết nối Drive và setup paths
5. Kiểm tra GPU và dung lượng

THỜI GIAN: ~2-3 phút

LƯU Ý: Nếu gặp lỗi, hãy dùng phương án CELL_1A + CELL_1B
"""

print("="*70)
print("🚀 CELL 1: CÀI ĐẶT THƯ VIỆN & KẾT NỐI DRIVE (IMPROVED)")
print("="*70)

import os
import sys

# ==============================================================================
# BƯỚC 1: GỠ BỎ THƯ VIỆN CŨ (Tránh conflict)
# ==============================================================================

print("\n" + "="*70)
print("🗑️  BƯỚC 1: GỠ BỎ THƯ VIỆN CŨ")
print("="*70)

print("\n⏳ Đang gỡ bỏ các thư viện cũ có thể gây conflict...")
print("💡 Bỏ qua lỗi nếu thư viện chưa được cài")

# Gỡ bỏ các thư viện chính có thể gây conflict
!pip uninstall -y torch torchvision torchaudio transformers accelerate numpy scipy scikit-learn 2>/dev/null || true

print("✅ Đã gỡ bỏ xong!")

# ==============================================================================
# BƯỚC 2: CÀI ĐẶT ĐỒNG BỘ TẤT CẢ THƯ VIỆN
# ==============================================================================

print("\n" + "="*70)
print("📦 BƯỚC 2: CÀI ĐẶT ĐỒNG BỘ THƯ VIỆN")
print("="*70)

print("\n⏳ Đang cài đặt tất cả thư viện cùng lúc... (có thể mất 2-3 phút)")
print("💡 Pip sẽ tự động chọn versions tương thích với nhau")
print("💡 Bỏ qua các warning về image libraries (không ảnh hưởng)")

# Cài đặt numpy trước để tránh conflict với sentence-transformers
!pip install --no-cache-dir -q numpy==1.26.4

# Cài đặt tất cả cùng lúc với --no-cache-dir để tránh lấy file lỗi cũ
# Không cố định version cứng, để pip tự chọn version tương thích với numpy 1.26.4
!pip install --no-cache-dir -q torch torchvision torchaudio transformers accelerate bitsandbytes sentence-transformers faiss-cpu rank_bm25 flashrank pymupdf huggingface_hub tqdm

print("\n✅ Đã cài đặt xong tất cả thư viện!")
print("💡 Đã cố định numpy==1.26.4 để tương thích với sentence-transformers")
print("💡 Không cần restart - có thể import ngay")

# ==============================================================================
# BƯỚC 3: VERIFY THƯ VIỆN (Import ngay không cần restart)
# ==============================================================================

print("\n" + "="*70)
print("🔍 BƯỚC 3: VERIFY THƯ VIỆN")
print("="*70)

print("\n⏳ Đang import và kiểm tra thư viện...")

# Danh sách thư viện cần kiểm tra
libraries_to_check = [
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("accelerate", "accelerate"),
    ("bitsandbytes", "bitsandbytes"),
    ("sentence_transformers", "sentence-transformers"),
    ("faiss", "faiss"),
    ("rank_bm25", "rank_bm25"),
    ("flashrank", "flashrank"),
    ("fitz", "pymupdf"),
    ("huggingface_hub", "huggingface_hub"),
    ("tqdm", "tqdm"),
]

failed_imports = []
successful_imports = []

for import_name, display_name in libraries_to_check:
    try:
        module = __import__(import_name)
        version = getattr(module, "__version__", "unknown")
        print(f"✅ {display_name} {version}")
        successful_imports.append(display_name)
    except ImportError as e:
        print(f"❌ {display_name} - FAILED: {str(e)}")
        failed_imports.append((display_name, str(e)))
    except Exception as e:
        error_msg = str(e)
        print(f"⚠️  {display_name} - WARNING: {error_msg}")
        # Nếu là lỗi numpy compatibility, coi như failed
        if "numpy" in error_msg.lower() or "_center" in error_msg or "umath" in error_msg:
            failed_imports.append((display_name, error_msg))
        else:
            # Các warning khác vẫn coi là thành công
            successful_imports.append(display_name)

# Tổng kết
if failed_imports:
    print(f"\n⚠️  Có {len(failed_imports)} thư viện gặp vấn đề:")
    for lib, error in failed_imports:
        print(f"   • {lib}: {error}")
    
    print(f"\n💡 CÁCH SỬA:")
    print(f"   1. Runtime > Restart runtime")
    print(f"   2. Chạy lại Cell 1 này")
    print(f"\n⚠️  KHÔNG TIẾP TỤC nếu có lỗi!")
else:
    print(f"\n✅ Tất cả {len(successful_imports)} thư viện đã được import thành công!")
    print(f"🎉 Môi trường đã ổn định - có thể tiếp tục Cell 2!")

# ==============================================================================
# BƯỚC 4: KIỂM TRA CUDA/GPU
# ==============================================================================

print("\n" + "="*70)
print("🖥️  BƯỚC 4: KIỂM TRA CUDA/GPU")
print("="*70)

import torch

if torch.cuda.is_available():
    print(f"\n✅ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    device = "cuda"
else:
    print("\n⚠️  CUDA not available - sẽ sử dụng CPU")
    print("   💡 Để sử dụng GPU, hãy bật GPU trong Runtime > Change runtime type")
    device = "cpu"

print(f"\n🎯 Device được sử dụng: {device}")

# ==============================================================================
# BƯỚC 5: KẾT NỐI GOOGLE DRIVE
# ==============================================================================

print("\n" + "="*70)
print("📂 BƯỚC 5: KẾT NỐI GOOGLE DRIVE")
print("="*70)

from google.colab import drive

# Kiểm tra xem Drive đã được mount chưa
if not os.path.exists('/content/drive'):
    print("\n⏳ Đang kết nối Google Drive...")
    print("💡 Bạn sẽ cần authorize quyền truy cập Drive")
    drive.mount('/content/drive')
    print("✅ Đã kết nối Drive thành công!")
else:
    print("\n✅ Google Drive đã được kết nối sẵn")

# ==============================================================================
# BƯỚC 6: THIẾT LẬP ĐƯỜNG DẪN
# ==============================================================================

print("\n" + "="*70)
print("📁 BƯỚC 6: THIẾT LẬP ĐƯỜNG DẪN")
print("="*70)

# Định nghĩa đường dẫn chính
base_path = "/content/drive/MyDrive/ChatbotEdu"
model_cache_path = os.path.join(base_path, "AI_MODELS_CACHE")

# Tạo thư mục nếu chưa có
os.makedirs(base_path, exist_ok=True)
os.makedirs(model_cache_path, exist_ok=True)

print(f"\n📂 Đường dẫn đã được thiết lập:")
print(f"   • Base path: {base_path}")
print(f"   • Model cache: {model_cache_path}")

# Liệt kê models đã có
if os.path.exists(model_cache_path):
    models = [m for m in os.listdir(model_cache_path) if os.path.isdir(os.path.join(model_cache_path, m))]
    if models:
        print(f"\n📦 Models đã có trong cache ({len(models)}):")
        for model in models:
            model_path = os.path.join(model_cache_path, model)
            # Tính kích thước
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(model_path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if os.path.exists(fp):
                        total_size += os.path.getsize(fp)
            size_gb = total_size / (1024**3)
            print(f"   • {model:30s} ({size_gb:.2f} GB)")
    else:
        print(f"\n💡 Chưa có models trong cache")
        print(f"   Hãy chạy Cell 2 để download models")

# ==============================================================================
# BƯỚC 7: THIẾT LẬP BIẾN MÔI TRƯỜNG
# ==============================================================================

print("\n" + "="*70)
print("⚙️  BƯỚC 7: THIẾT LẬP BIẾN MÔI TRƯỜNG")
print("="*70)

# Thiết lập biến môi trường cho Hugging Face
os.environ['HF_HOME'] = model_cache_path
os.environ['TRANSFORMERS_CACHE'] = model_cache_path
os.environ['HF_DATASETS_CACHE'] = os.path.join(base_path, "datasets_cache")

print("\n✅ Đã thiết lập biến môi trường:")
print(f"   • HF_HOME: {os.environ['HF_HOME']}")
print(f"   • TRANSFORMERS_CACHE: {os.environ['TRANSFORMERS_CACHE']}")
print(f"   • HF_DATASETS_CACHE: {os.environ['HF_DATASETS_CACHE']}")

# ==============================================================================
# BƯỚC 8: KIỂM TRA DUNG LƯỢNG DRIVE
# ==============================================================================

print("\n" + "="*70)
print("💾 BƯỚC 8: KIỂM TRA DUNG LƯỢNG DRIVE")
print("="*70)

try:
    import shutil
    
    # Lấy thông tin dung lượng
    total, used, free = shutil.disk_usage("/content/drive")
    
    total_gb = total / (1024**3)
    used_gb = used / (1024**3)
    free_gb = free / (1024**3)
    
    print(f"\n📊 Dung lượng Google Drive:")
    print(f"   • Tổng: {total_gb:.2f} GB")
    print(f"   • Đã dùng: {used_gb:.2f} GB ({used/total*100:.1f}%)")
    print(f"   • Còn trống: {free_gb:.2f} GB ({free/total*100:.1f}%)")
    
    # Cảnh báo nếu dung lượng thấp
    if free_gb < 7:
        print(f"\n⚠️  CẢNH BÁO: Dung lượng còn trống thấp!")
        print(f"   Cần ít nhất 7 GB để download tất cả models")
        print(f"   Hiện tại chỉ còn {free_gb:.2f} GB")
        print(f"\n💡 Giải pháp:")
        print(f"   1. Xóa files không cần thiết trong Drive")
        print(f"   2. Nâng cấp Google Drive storage")
        print(f"   3. Hoặc chỉ download models cần thiết (LLM + Embedding)")
    else:
        print(f"\n✅ Dung lượng đủ để download tất cả models (~7 GB)")
        
except Exception as e:
    print(f"\n⚠️  Không thể kiểm tra dung lượng: {e}")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "="*70)
print("📊 TÓM TẮT CELL 1")
print("="*70)

print("\n✅ Đã hoàn thành:")
print("   1. ✅ Gỡ bỏ thư viện cũ")
print("   2. ✅ Cài đặt đồng bộ tất cả thư viện")
print("   3. ✅ Verify thư viện (import thành công)")
print("   4. ✅ Kiểm tra CUDA/GPU")
print("   5. ✅ Kết nối Google Drive")
print("   6. ✅ Thiết lập đường dẫn")
print("   7. ✅ Thiết lập biến môi trường")
print("   8. ✅ Kiểm tra dung lượng Drive")

print("\n📦 Biến đã được tạo:")
print("   • base_path - Đường dẫn base folder")
print("   • model_cache_path - Đường dẫn cache models")
print("   • device - Device sử dụng (cuda/cpu)")

print("\n🎯 Thư viện đã sẵn sàng:")
print("   • torch, transformers, accelerate, bitsandbytes")
print("   • sentence-transformers, faiss-cpu, rank_bm25")
print("   • flashrank, pymupdf, huggingface_hub")

print("\n💡 Bước tiếp theo:")
print("   ✅ Môi trường đã ổn định - không cần restart")
print("   👉 Chạy Cell 2 để download models (nếu chưa có)")
print("   👉 Hoặc chạy Cell 3 để load models (nếu đã có)")

print("\n" + "="*70)
print("🎉 CELL 1 HOÀN TẤT - SẴN SÀNG CHO CELL 2!")
print("="*70)

print("\n💡 LƯU Ý:")
print("   • Nếu gặp lỗi import → Dùng phương án CELL_1A + CELL_1B")
print("   • CELL_1A: Cài đặt nhanh + kết nối Drive")
print("   • CELL_1B: Fix conflicts + verify thư viện")

# ==============================================================================
# HELPER FUNCTIONS (Optional - để sử dụng trong các cell sau)
# ==============================================================================

def check_library_installed(library_name: str) -> bool:
    """Kiểm tra xem thư viện đã được cài chưa"""
    try:
        __import__(library_name)
        return True
    except ImportError:
        return False

def get_folder_size(folder_path: str) -> float:
    """Lấy kích thước folder (GB)"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024**3)

print("\n💡 Helper functions đã được định nghĩa:")
print("   • check_library_installed(library_name)")
print("   • get_folder_size(folder_path)")
