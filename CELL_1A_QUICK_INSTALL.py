# ==============================================================================
# @title CELL 1A: CÀI ĐẶT NHANH & KẾT NỐI DRIVE
# ==============================================================================

"""
CELL 1A - Cài đặt nhanh các thư viện cơ bản

PHƯƠNG ÁN NÀY: Chia làm 2 bước (an toàn hơn khi gặp lỗi)
- CELL_1A: Cài đặt + kết nối Drive (cell này)
- CELL_1B: Fix conflicts + verify thư viện (cell tiếp theo)

CHIẾN LƯỢC:
1. Gỡ bỏ thư viện cũ gây conflict
2. Cài đặt đồng bộ tất cả thư viện (KHÔNG verify ngay)
3. Kết nối Drive và setup paths
4. Thiết lập biến môi trường
5. → Sau đó chạy CELL_1B để verify

THỜI GIAN: ~1-2 phút
"""

print("="*70)
print("🚀 CELL 1A: CÀI ĐẶT NHANH & KẾT NỐI DRIVE")
print("="*70)

import os

# ==============================================================================
# BƯỚC 1: GỠ BỎ THƯ VIỆN CŨ
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
# BƯỚC 2: CÀI ĐẶT ĐỒNG BỘ THƯ VIỆN
# ==============================================================================

print("\n" + "="*70)
print("📦 BƯỚC 2: CÀI ĐẶT ĐỒNG BỘ THƯ VIỆN")
print("="*70)

print("\n⏳ Đang cài đặt thư viện... (có thể mất 1-2 phút)")
print("💡 Pip sẽ tự động chọn versions tương thích với nhau")
print("💡 Bỏ qua các warning về image libraries (không ảnh hưởng)")

# Cài đặt numpy trước để tránh conflict với sentence-transformers
!pip install --no-cache-dir -q numpy==1.26.4

# Cài đặt tất cả cùng lúc với --no-cache-dir
!pip install --no-cache-dir -q torch torchvision torchaudio transformers accelerate bitsandbytes sentence-transformers faiss-cpu rank_bm25 flashrank pymupdf huggingface_hub tqdm

print("\n✅ Đã cài đặt xong!")
print("💡 Đã cố định numpy==1.26.4 để tương thích với sentence-transformers")
print("💡 KHÔNG verify ngay - sẽ verify ở CELL_1B")

# ==============================================================================
# BƯỚC 3: KẾT NỐI GOOGLE DRIVE
# ==============================================================================

print("\n" + "="*70)
print("📂 BƯỚC 3: KẾT NỐI GOOGLE DRIVE")
print("="*70)

from google.colab import drive

# Kết nối Google Drive
if not os.path.exists('/content/drive'):
    print("\n⏳ Đang kết nối Google Drive...")
    print("💡 Bạn sẽ cần authorize quyền truy cập Drive")
    drive.mount('/content/drive')
    print("✅ Đã kết nối Drive thành công!")
else:
    print("\n✅ Google Drive đã được kết nối sẵn")

# ==============================================================================
# BƯỚC 4: THIẾT LẬP ĐƯỜNG DẪN
# ==============================================================================

print("\n" + "="*70)
print("📁 BƯỚC 4: THIẾT LẬP ĐƯỜNG DẪN")
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

# Liệt kê models đã có (nếu có)
if os.path.exists(model_cache_path):
    models = [m for m in os.listdir(model_cache_path) if os.path.isdir(os.path.join(model_cache_path, m))]
    if models:
        print(f"\n📦 Models đã có trong cache ({len(models)}):")
        for model in models:
            print(f"   • {model}")
    else:
        print(f"\n💡 Chưa có models trong cache")
        print(f"   Hãy chạy Cell 2 để download models")

# ==============================================================================
# BƯỚC 5: THIẾT LẬP BIẾN MÔI TRƯỜNG
# ==============================================================================

print("\n" + "="*70)
print("⚙️  BƯỚC 5: THIẾT LẬP BIẾN MÔI TRƯỜNG")
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
# SUMMARY
# ==============================================================================

print("\n" + "="*70)
print("📊 TÓM TẮT CELL 1A")
print("="*70)

print("\n✅ Đã hoàn thành:")
print("   1. ✅ Gỡ bỏ thư viện cũ")
print("   2. ✅ Cài đặt đồng bộ thư viện")
print("   3. ✅ Kết nối Google Drive")
print("   4. ✅ Thiết lập đường dẫn")
print("   5. ✅ Thiết lập biến môi trường")

print("\n📦 Biến đã được tạo:")
print("   • base_path - Đường dẫn base folder")
print("   • model_cache_path - Đường dẫn cache models")

print("\n⚠️  CHƯA VERIFY thư viện!")
print("   → Cần chạy CELL_1B để verify và fix conflicts")

print("\n💡 Bước tiếp theo:")
print("   👉 Chạy CELL_1B để verify thư viện")

print("\n" + "="*70)
print("🎉 CELL 1A HOÀN TẤT - CHẠY CELL 1B TIẾP!")
print("="*70)
