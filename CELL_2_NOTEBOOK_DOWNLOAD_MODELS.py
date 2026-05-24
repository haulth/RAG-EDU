# ==============================================================================
# @title CELL 2: TẢI TẤT CẢ MODELS VỀ DRIVE (Chạy 1 lần duy nhất)
# ==============================================================================

print("="*70)
print("📥 CELL 2: TẢI TẤT CẢ MODELS VỀ DRIVE")
print("="*70)

import os
from model_assets import (
    EMBED_BGE_M3,
    INFOXLM_LARGE_CONFIG,
    LLM_QWEN_7B,
    RERANKER_BGE_BASE,
    RERANKER_VIETNAMESE,
    SEMVIQA_QATC,
    check_asset_exists,
    download_asset,
    get_model_size,
    normalize_cache_root,
)

# ==============================================================================
# CONFIGURATION - Sử dụng biến từ Cell 1
# ==============================================================================

# Kiểm tra xem Cell 1 đã chạy chưa
if 'model_cache_path' not in globals():
    print("❌ LỖI: Biến 'model_cache_path' chưa được định nghĩa!")
    print("👉 Hãy chạy CELL 1 trước khi chạy Cell này")
    raise NameError("Please run CELL 1 first to define 'model_cache_path'")

model_cache_path = str(normalize_cache_root(model_cache_path))
globals()['model_cache_path'] = model_cache_path

# Model paths
LLM_MODEL_PATH = str(LLM_QWEN_7B.resolve_path(model_cache_path))
EMBEDDING_MODEL_PATH = str(EMBED_BGE_M3.resolve_path(model_cache_path))
RERANKER_STAGE2_PATH = str(RERANKER_VIETNAMESE.resolve_path(model_cache_path))
RERANKER_FALLBACK_PATH = str(RERANKER_BGE_BASE.resolve_path(model_cache_path))
MMR_QATC_MODEL_ID = SEMVIQA_QATC.repo_id
MMR_QATC_MODEL_PATH = str(SEMVIQA_QATC.resolve_path(model_cache_path))
MMR_QATC_BASE_CONFIG_PATH = str(INFOXLM_LARGE_CONFIG.resolve_path(model_cache_path))

print(f"\n📂 Models sẽ được lưu vào: {model_cache_path}")

# ==============================================================================
# KIỂM TRA MODELS ĐÃ TỒN TẠI
# ==============================================================================

print("\n" + "="*70)
print("🔍 KIỂM TRA MODELS ĐÃ CÓ TRONG DRIVE")
print("="*70)

models_status = {
    "LLM (Qwen 7B)": (LLM_MODEL_PATH, check_asset_exists(LLM_QWEN_7B, model_cache_path)),
    "Embedding (BGE-M3)": (EMBEDDING_MODEL_PATH, check_asset_exists(EMBED_BGE_M3, model_cache_path)),
    "ViRanker": (RERANKER_STAGE2_PATH, check_asset_exists(RERANKER_VIETNAMESE, model_cache_path)),
    "Reranker Fallback (BGE-Base)": (RERANKER_FALLBACK_PATH, check_asset_exists(RERANKER_BGE_BASE, model_cache_path)),
    "SemViQA QATC": (MMR_QATC_MODEL_PATH, check_asset_exists(SEMVIQA_QATC, model_cache_path)),
    "InfoXLM base config": (MMR_QATC_BASE_CONFIG_PATH, check_asset_exists(INFOXLM_LARGE_CONFIG, model_cache_path)),
}

all_exist = True
for model_name, (model_path, exists) in models_status.items():
    status = "✅ ĐÃ CÓ" if exists else "❌ CHƯA CÓ"
    size = get_model_size(model_path) if exists else "N/A"
    print(f"{status} {model_name:35s} ({size})")
    if not exists:
        all_exist = False

if all_exist:
    print("\n" + "="*70)
    print("✅ TẤT CẢ MODELS ĐÃ CÓ TRONG DRIVE!")
    print("="*70)
    print("\n💡 Bạn có thể BỎ QUA Cell này và chạy thẳng Cell 3")
    print("   (Trừ khi bạn muốn tải lại models)")
    print("\n⚠️  Nếu muốn tải lại, hãy xóa thư mục AI_MODELS_CACHE/ trong Drive trước")
else:
    print("\n" + "="*70)
    print("⚠️  MỘT SỐ MODELS CHƯA CÓ - SẼ TIẾN HÀNH TẢI")
    print("="*70)
    print(f"\n⏱️  Thời gian dự kiến: ~10-15 phút")
    print(f"💾 Dung lượng cần thiết: ~19.2 GB")

# ==============================================================================
# MODEL 1: LLM (Qwen 2.5-7B-Instruct) - UPGRADED
# ==============================================================================

print("\n" + "="*70)
print("📥 MODEL 1/5: LLM (Qwen 2.5-7B-Instruct) - UPGRADED")
print("="*70)

if check_asset_exists(LLM_QWEN_7B, model_cache_path):
    print(f"✅ Đã có sẵn: {LLM_MODEL_PATH}")
    print(f"   Kích thước: {get_model_size(LLM_MODEL_PATH)}")
else:
    print("⏳ Đang tải LLM model...")
    print("   Model: Qwen/Qwen2.5-7B-Instruct")
    print("   Kích thước: ~15.2 GB")
    print("   Mục đích: Tạo câu trả lời")
    print("   Cải tiến: 7B parameters, context 128K, tốt hơn cho tiếng Việt và structured output")
    
    try:
        print("\n   [1/1] Đang tải model và tokenizer...")
        download_asset(LLM_QWEN_7B, model_cache_path)
        
        print(f"   ✅ Đã lưu! Kích thước: {get_model_size(LLM_MODEL_PATH)}")
        
    except Exception as e:
        print(f"   ❌ Lỗi khi tải LLM: {e}")
        print("   💡 Bạn có thể tiếp tục - LLM sẽ được tải trong Cell 3 nếu cần")

# ==============================================================================
# MODEL 2: EMBEDDING (BGE-M3) - UPGRADED
# ==============================================================================

print("\n" + "="*70)
print("📥 MODEL 2/5: EMBEDDING (BGE-M3) - UPGRADED")
print("="*70)

if check_asset_exists(EMBED_BGE_M3, model_cache_path):
    print(f"✅ Đã có sẵn: {EMBEDDING_MODEL_PATH}")
    print(f"   Kích thước: {get_model_size(EMBEDDING_MODEL_PATH)}")
else:
    print("⏳ Đang tải Embedding model...")
    print("   Model: BAAI/bge-m3")
    print("   Kích thước: ~2.3 GB")
    print("   Mục đích: Tìm kiếm vector, semantic similarity")
    print("   Cải tiến: 100+ ngôn ngữ, context 8192, tốt hơn cho tiếng Việt")
    print("   ⚠️  LƯU Ý: Sau khi tải xong, BẮT BUỘC phải re-embed toàn bộ database!")
    
    try:
        print("\n   [1/1] Đang tải model...")
        download_asset(EMBED_BGE_M3, model_cache_path)
        
        print(f"   ✅ Đã lưu! Kích thước: {get_model_size(EMBEDDING_MODEL_PATH)}")
        print("   ⚠️  QUAN TRỌNG: Phải re-embed database với model mới!")
        
    except Exception as e:
        print(f"   ❌ Lỗi khi tải Embedding: {e}")
        print("   💡 Bạn có thể tiếp tục - Embedding sẽ được tải trong Cell 3 nếu cần")

# ==============================================================================
# MODEL 3: RERANKER STAGE 1 (BGE-reranker-v2-m3)
# ==============================================================================

print("\n" + "="*70)
print("📥 MODEL 3/5: VIRANKER (Vietnamese Cross-Encoder)")
print("="*70)

if check_asset_exists(RERANKER_VIETNAMESE, model_cache_path):
    print(f"✅ Đã có sẵn: {RERANKER_STAGE2_PATH}")
    print(f"   Kích thước: {get_model_size(RERANKER_STAGE2_PATH)}")
else:
    print("⏳ Đang tải ViRanker...")
    print("   Model: itdainb/vietnamese-cross-encoder")
    print("   Kích thước: ~440 MB")
    print("   Mục đích: Reranking chuyên biệt cho tiếng Việt")
    
    try:
        print("\n   [1/1] Đang tải model và tokenizer...")
        download_asset(RERANKER_VIETNAMESE, model_cache_path)
        
        print(f"   ✅ Đã lưu! Kích thước: {get_model_size(RERANKER_STAGE2_PATH)}")
        
    except Exception as e:
        print(f"   ❌ Lỗi khi tải ViRanker: {e}")
        print("   💡 Hệ thống sẽ dùng fallback reranker nếu model này lỗi")

# ==============================================================================
# MODEL 4: RERANKER FALLBACK (BGE-reranker-base)
# ==============================================================================

print("\n" + "="*70)
print("📥 MODEL 4/5: RERANKER FALLBACK (BGE-reranker-base)")
print("="*70)

if check_asset_exists(RERANKER_BGE_BASE, model_cache_path):
    print(f"✅ Đã có sẵn: {RERANKER_FALLBACK_PATH}")
    print(f"   Kích thước: {get_model_size(RERANKER_FALLBACK_PATH)}")
else:
    print("⏳ Đang tải Reranker Fallback...")
    print("   Model: BAAI/bge-reranker-base")
    print("   Kích thước: ~1.1 GB")
    print("   Mục đích: Backup reranker nếu ViRanker lỗi")
    
    try:
        print("\n   [1/1] Đang tải model và tokenizer...")
        download_asset(RERANKER_BGE_BASE, model_cache_path)
        
        print(f"   ✅ Đã lưu! Kích thước: {get_model_size(RERANKER_FALLBACK_PATH)}")
        
    except Exception as e:
        print(f"   ❌ Lỗi khi tải Reranker Fallback: {e}")
        print("   💡 Hệ thống sẽ dùng similarity-based reranking nếu fallback này lỗi")

# ==============================================================================
# MODEL 5: SEMVIQA QATC (SEMANTIC HIGHLIGHTING + MMR)
# ==============================================================================

print("\n" + "="*70)
print("📥 MODEL 5/5: SEMVIQA QATC (SEMANTIC HIGHLIGHTING + MMR)")
print("="*70)

if check_asset_exists(SEMVIQA_QATC, model_cache_path):
    print(f"✅ Đã có sẵn: {MMR_QATC_MODEL_PATH}")
    print(f"   Kích thước: {get_model_size(MMR_QATC_MODEL_PATH)}")
else:
    print("⏳ Đang tải SemViQA QATC model...")
    print(f"   Model: {MMR_QATC_MODEL_ID}")
    print("   Kích thước: ~2.0 GB")
    print("   Mục đích: Semantic highlighting + QATC MMR cho tiếng Việt")
    
    try:
        print("\n   [1/1] Đang tải model và tokenizer...")
        download_asset(SEMVIQA_QATC, model_cache_path)
        
        print(f"   ✅ Đã lưu! Kích thước: {get_model_size(MMR_QATC_MODEL_PATH)}")
        
    except Exception as e:
        print(f"   ❌ Lỗi khi tải SemViQA QATC: {e}")
        print("   💡 Hệ thống sẽ dùng fallback pruning/MMR nếu model này lỗi")

if check_asset_exists(INFOXLM_LARGE_CONFIG, model_cache_path):
    print(f"✅ Base config đã có sẵn: {MMR_QATC_BASE_CONFIG_PATH}")
else:
    print("⏳ Đang tải InfoXLM base config cho SemViQA...")
    try:
        download_asset(INFOXLM_LARGE_CONFIG, model_cache_path)
        print(f"   ✅ Đã lưu base config: {MMR_QATC_BASE_CONFIG_PATH}")
    except Exception as e:
        print(f"   ❌ Lỗi khi tải InfoXLM base config: {e}")
        print("   💡 Nếu base config thiếu, SemViQA có thể cần mạng để load lần đầu")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "="*70)
print("📊 TÓM TẮT KẾT QUẢ")
print("="*70)

total_size = 0
success_count = 0

for model_name, (model_path, _) in models_status.items():
    if model_name == "InfoXLM base config":
        exists = check_asset_exists(INFOXLM_LARGE_CONFIG, model_cache_path)
    elif model_name == "SemViQA QATC":
        exists = check_asset_exists(SEMVIQA_QATC, model_cache_path)
    elif model_name == "Reranker Fallback (BGE-Base)":
        exists = check_asset_exists(RERANKER_BGE_BASE, model_cache_path)
    elif model_name == "ViRanker":
        exists = check_asset_exists(RERANKER_VIETNAMESE, model_cache_path)
    elif model_name == "Embedding (BGE-M3)":
        exists = check_asset_exists(EMBED_BGE_M3, model_cache_path)
    else:
        exists = check_asset_exists(LLM_QWEN_7B, model_cache_path)
    status = "✅ THÀNH CÔNG" if exists else "❌ THẤT BẠI"
    size = get_model_size(model_path) if exists else "N/A"
    print(f"{status} {model_name:35s} ({size})")
    
    if exists:
        success_count += 1
        # Parse size to MB for total
        if 'GB' in size:
            total_size += float(size.split()[0]) * 1024
        elif 'MB' in size:
            total_size += float(size.split()[0])

print(f"\n📦 Tổng số assets thành công: {success_count}/{len(models_status)}")
print(f"💾 Tổng dung lượng: {total_size/1024:.2f} GB")

if success_count == len(models_status):
    print("\n" + "="*70)
    print("🎉 HOÀN TẤT! TẤT CẢ MODEL ASSETS ĐÃ ĐƯỢC TẢI VỀ DRIVE")
    print("="*70)
    print("\n✅ Bạn có thể chạy Cell 3 để load models và bắt đầu sử dụng")
    print("💡 Lần sau chỉ cần chạy Cell 3, không cần chạy lại Cell 2")
elif success_count >= 2:
    print("\n" + "="*70)
    print("⚠️  MỘT SỐ MODELS CHƯA TẢI ĐƯỢC")
    print("="*70)
    print("\n💡 Bạn vẫn có thể tiếp tục với các models đã có")
    print("   Hệ thống sẽ tự động fallback nếu thiếu reranker models")
else:
    print("\n" + "="*70)
    print("❌ LỖI: KHÔNG THỂ TẢI MODELS CẦN THIẾT")
    print("="*70)
    print("\n💡 Hãy kiểm tra:")
    print("   1. Kết nối internet")
    print("   2. Dung lượng Drive còn trống (cần ~7 GB)")
    print("   3. Quyền truy cập Hugging Face")
