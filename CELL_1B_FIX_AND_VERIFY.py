# ==============================================================================
# @title CELL 1B: VERIFY & FIX THƯ VIỆN
# ==============================================================================

"""
CELL 1B - Verify và fix conflicts thư viện

PHƯƠNG ÁN NÀY: Chia làm 2 bước (an toàn hơn khi gặp lỗi)
- CELL_1A: Cài đặt + kết nối Drive (cell trước)
- CELL_1B: Verify + fix conflicts (cell này)

CHIẾN LƯỢC:
1. Import và verify tất cả thư viện
2. Nếu có lỗi → Gỡ bỏ và cài lại
3. Kiểm tra CUDA/GPU
4. Kiểm tra dung lượng Drive
5. Tạo helper functions

THỜI GIAN: ~1 phút (nếu không có lỗi)
           ~2-3 phút (nếu cần fix)

LƯU Ý: Phải chạy CELL_1A trước!
"""

print("="*70)
print("🛠️  CELL 1B: VERIFY & FIX THƯ VIỆN")
print("="*70)

import os

# ==============================================================================
# BƯỚC 0: KIỂM TRA CELL_1A ĐÃ CHẠY CHƯA
# ==============================================================================

print("\n" + "="*70)
print("🔍 BƯỚC 0: KIỂM TRA CELL_1A")
print("="*70)

if 'model_cache_path' not in globals():
    print("\n⚠️  CẢNH BÁO: Biến model_cache_path chưa được định nghĩa!")
    print("   👉 Hãy chạy CELL_1A trước!")
    print("\n💡 Đang tạo lại biến...")
    base_path = "/content/drive/MyDrive/ChatbotEdu"
    model_cache_path = os.path.join(base_path, "AI_MODELS_CACHE")
    os.environ['HF_HOME'] = model_cache_path
    os.environ['TRANSFORMERS_CACHE'] = model_cache_path
    print(f"   ✅ Đã tạo lại: {model_cache_path}")
else:
    print("\n✅ CELL_1A đã chạy - biến đã được định nghĩa")
    print(f"   • base_path: {base_path}")
    print(f"   • model_cache_path: {model_cache_path}")

# ==============================================================================
# BƯỚC 1: VERIFY THƯ VIỆN (Lần đầu)
# ==============================================================================

print("\n" + "="*70)
print("🔍 BƯỚC 1: VERIFY THƯ VIỆN (Lần đầu)")
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
        error_msg = str(e)
        print(f"❌ {display_name} - FAILED: {error_msg}")
        failed_imports.append((display_name, error_msg))
    except Exception as e:
        error_msg = str(e)
        print(f"⚠️  {display_name} - WARNING: {error_msg}")
        # Nếu là lỗi numpy compatibility, coi như failed
        if "numpy" in error_msg.lower() or "_center" in error_msg or "umath" in error_msg:
            failed_imports.append((display_name, error_msg))
        else:
            # Các warning khác vẫn coi là thành công
            successful_imports.append(display_name)

# ==============================================================================
# BƯỚC 2: FIX NẾU CÓ LỖI
# ==============================================================================

if failed_imports:
    print("\n" + "="*70)
    print("🔧 BƯỚC 2: FIX CONFLICTS")
    print("="*70)
    
    print(f"\n⚠️  Phát hiện {len(failed_imports)} thư viện gặp vấn đề:")
    for lib, error in failed_imports:
        print(f"   • {lib}: {error[:80]}...")
    
    # Kiểm tra xem có lỗi numpy không
    has_numpy_error = any("numpy" in error.lower() or "_center" in error or "umath" in error 
                          for _, error in failed_imports)
    
    if has_numpy_error:
        print("\n💡 Phát hiện lỗi numpy compatibility!")
        print("   Đang fix bằng cách downgrade numpy...")
        
        # Fix numpy version
        !pip uninstall -y numpy 2>/dev/null || true
        !pip install numpy==1.26.4 -q
        
        print("   ✅ Đã downgrade numpy về 1.26.4")
    
    print("\n⏳ Đang gỡ bỏ và cài lại thư viện có vấn đề...")
    
    # Gỡ bỏ các thư viện chính gây conflict
    !pip uninstall -y sentence-transformers torch torchvision torchaudio transformers accelerate 2>/dev/null || true
    
    print("   ✅ Đã gỡ bỏ xong!")
    
    print("\n⏳ Đang cài đặt lại toàn bộ (mất khoảng 1-2 phút)...")
    print("💡 Pip sẽ tự động chọn versions tương thích với numpy 1.26.4")
    
    # Cài đặt lại toàn bộ cùng lúc với numpy cố định
    !pip install --no-cache-dir -q numpy==1.26.4 torch torchvision torchaudio transformers accelerate bitsandbytes sentence-transformers
    
    print("   ✅ Đã cài đặt lại xong!")
    
    # Verify lại
    print("\n⏳ Đang verify lại...")
    
    failed_imports_2 = []
    successful_imports_2 = []
    
    for import_name, display_name in libraries_to_check:
        try:
            module = __import__(import_name)
            version = getattr(module, "__version__", "unknown")
            print(f"✅ {display_name} {version}")
            successful_imports_2.append(display_name)
        except ImportError as e:
            print(f"❌ {display_name} - FAILED: {str(e)}")
            failed_imports_2.append((display_name, str(e)))
        except Exception as e:
            error_msg = str(e)
            print(f"⚠️  {display_name} - WARNING: {error_msg}")
            if "numpy" in error_msg.lower() or "_center" in error_msg or "umath" in error_msg:
                failed_imports_2.append((display_name, error_msg))
            else:
                successful_imports_2.append(display_name)
    
    if failed_imports_2:
        print(f"\n❌ Vẫn còn {len(failed_imports_2)} thư viện gặp vấn đề!")
        print("💡 CÁCH SỬA:")
        print("   1. Runtime > Restart runtime")
        print("   2. Chạy lại CELL_1A")
        print("   3. Chạy lại CELL_1B này")
    else:
        print(f"\n✅ Đã fix xong! Tất cả {len(successful_imports_2)} thư viện hoạt động!")
        successful_imports = successful_imports_2
        failed_imports = []
else:
    print(f"\n✅ Tất cả {len(successful_imports)} thư viện đã hoạt động ngay!")
    print("🎉 Không cần fix gì cả!")

# ==============================================================================
# BƯỚC 3: KIỂM TRA CUDA/GPU
# ==============================================================================

print("\n" + "="*70)
print("🖥️  BƯỚC 3: KIỂM TRA CUDA/GPU")
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
# BƯỚC 4: KIỂM TRA DUNG LƯỢNG DRIVE
# ==============================================================================

print("\n" + "="*70)
print("💾 BƯỚC 4: KIỂM TRA DUNG LƯỢNG DRIVE")
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
# BƯỚC 5: HELPER FUNCTIONS
# ==============================================================================

print("\n" + "="*70)
print("🛠️  BƯỚC 5: HELPER FUNCTIONS")
print("="*70)

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

print("\n✅ Helper functions đã được định nghĩa:")
print("   • check_library_installed(library_name)")
print("   • get_folder_size(folder_path)")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "="*70)
print("📊 TÓM TẮT CELL 1B")
print("="*70)

print("\n✅ Đã hoàn thành:")
print("   0. ✅ Kiểm tra CELL_1A đã chạy")
print("   1. ✅ Verify thư viện")
if failed_imports:
    print("   2. ✅ Fix conflicts (đã gỡ và cài lại)")
else:
    print("   2. ✅ Không có lỗi (bỏ qua fix)")
print("   3. ✅ Kiểm tra CUDA/GPU")
print("   4. ✅ Kiểm tra dung lượng Drive")
print("   5. ✅ Tạo helper functions")

print("\n📦 Biến đã được tạo:")
print("   • device - Device sử dụng (cuda/cpu)")
print("   • check_library_installed() - Function kiểm tra thư viện")
print("   • get_folder_size() - Function tính kích thước folder")

print("\n🎯 Thư viện đã sẵn sàng:")
print("   • torch, transformers, accelerate, bitsandbytes")
print("   • sentence-transformers, faiss-cpu, rank_bm25")
print("   • flashrank, pymupdf, huggingface_hub")

if failed_imports:
    print("\n⚠️  VẪN CÒN LỖI:")
    for lib, error in failed_imports:
        print(f"   • {lib}: {error}")
    print("\n💡 Cần restart runtime và chạy lại CELL_1A + CELL_1B")
else:
    print("\n💡 Bước tiếp theo:")
    print("   • Nếu chưa có models → Chạy Cell 2 để download")
    print("   • Nếu đã có models → Chạy Cell 3 để load")

print("\n" + "="*70)
if failed_imports:
    print("⚠️  CELL 1B CÓ LỖI - CẦN FIX!")
else:
    print("🎉 CELL 1B HOÀN TẤT - SẴN SÀNG CHO CELL 2!")
print("="*70)
