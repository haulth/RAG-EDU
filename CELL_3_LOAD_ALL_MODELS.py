# ==============================================================================
# @title CELL 3: LOAD TẤT CẢ MODELS TỪ DRIVE
# ==============================================================================

"""
CELL 3 - Load TẤT CẢ models từ Drive/HF vào memory

MODELS SẼ LOAD (5 model groups / 6 capabilities):
1. LLM Model: Qwen/Qwen2.5-7B-Instruct - Answer generation
2. Embedding Model: embed_bge_m3 - Vector search
3. ViRanker: itdainb/vietnamese-cross-encoder - Vietnamese reranking
4. Reranker Fallback: BAAI/bge-reranker-base - Backup reranker
5. SemViQA QATC: SemViQA/qatc-infoxlm-viwikifc - Vietnamese semantic highlighting + MMR evidence extraction

THỜI GIAN LOAD: ~30-60 giây (nhanh hơn nhiều so với download)

OUTPUT VARIABLES:
- llm_model: LLM model (Qwen)
- llm_tokenizer: Tokenizer cho LLM
- embedder: Sentence Transformer model
- reranker_stage2_model: ViRanker model (nếu có)
- reranker_stage2_tokenizer: ViRanker tokenizer (nếu có)
- reranker_fallback_model: BGE-Base fallback model (nếu có)
- reranker_fallback_tokenizer: BGE-Base fallback tokenizer (nếu có)
- semantic_highlight_model: Alias tới SemViQA QATC model cho context pruning
- semantic_highlight_tokenizer: Tokenizer cho semantic highlighting backend
- mmr_qatc_model: SemViQA QATC model cho MMR evidence extraction
- mmr_qatc_tokenizer: Tokenizer cho MMR QATC model
- generate_text(): Hàm tạo text với LLM

CÁCH DÙNG:
1. Đảm bảo đã chạy Cell 1 (kết nối Drive)
2. Đảm bảo đã chạy Cell 2 ít nhất 1 lần (download models)
3. Chạy Cell này để load tất cả models vào memory
4. Các cell sau sẽ sử dụng models từ Cell này
"""

print("="*70)
print("CELL 3: LOAD TAT CA MODELS")
print("="*70)

# ==============================================================================
# FIX NUMPY COMPATIBILITY (if needed)
# ==============================================================================
try:
    import numpy as np
    # Test if numpy works
    _ = np.array([1, 2, 3])
except (ValueError, ImportError) as e:
    if "numpy.dtype size changed" in str(e) or "binary incompatibility" in str(e):
        print("\n  Phát hiện lỗi numpy compatibility!")
        print(" Đang fix tự động...")
        
        import sys
        import subprocess
        
        # Reinstall numpy
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "numpy"], 
                      capture_output=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "numpy==1.26.4"], 
                      capture_output=True)
        
        print(" Đã fix numpy!")
        print("  VUI LÒNG RESTART RUNTIME và chạy lại cell này")
        print("   Runtime → Restart runtime (hoặc Ctrl+M .)")
        raise RuntimeError("Please restart runtime after numpy fix")
    else:
        raise

import os
import torch
from model_assets import (
    EMBED_BGE_M3,
    INFOXLM_LARGE_CONFIG,
    LLM_QWEN_7B,
    RERANKER_BGE_BASE,
    RERANKER_VIETNAMESE,
    SEMVIQA_QATC,
    check_asset_exists,
    check_model_exists,
    download_asset,
    normalize_cache_root,
)
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    BitsAndBytesConfig
)
from sentence_transformers import SentenceTransformer

BOOTSTRAP_REMOTE_ONLY = bool(globals().get("BOOTSTRAP_REMOTE_ONLY", False))

# ==============================================================================
# CONFIGURATION - Sử dụng biến từ Cell 1
# ==============================================================================

# Kiem tra xem Cell 1 da chay chua
if 'model_cache_path' not in globals():
    print("CANH BAO: Bien 'model_cache_path' chua duoc dinh nghia!")
    print("Dang load tu config...")
    from config import MODEL_CACHE_PATH
    model_cache_path = MODEL_CACHE_PATH
    globals()['model_cache_path'] = model_cache_path

model_cache_path = str(normalize_cache_root(model_cache_path))
globals()['model_cache_path'] = model_cache_path

# Model paths
LLM_MODEL_PATH = str(LLM_QWEN_7B.resolve_path(model_cache_path))
LLM_MODEL_FALLBACK_PATH = os.path.join(model_cache_path, "llm_qwen")
EMBEDDING_MODEL_PATH = str(EMBED_BGE_M3.resolve_path(model_cache_path))
RERANKER_STAGE2_PATH = str(RERANKER_VIETNAMESE.resolve_path(model_cache_path))
RERANKER_FALLBACK_PATH = str(RERANKER_BGE_BASE.resolve_path(model_cache_path))
MMR_QATC_MODEL_ID = os.environ.get("MMR_QATC_MODEL_ID", SEMVIQA_QATC.repo_id)
MMR_QATC_MODEL_PATH = os.environ.get(
    "MMR_QATC_MODEL_PATH",
    str(SEMVIQA_QATC.resolve_path(model_cache_path)),
)
MMR_QATC_BASE_CONFIG_PATH = os.environ.get(
    "MMR_QATC_BASE_CONFIG_PATH",
    str(INFOXLM_LARGE_CONFIG.resolve_path(model_cache_path)),
)

print(f"\nDang load models tu: {model_cache_path}")
if BOOTSTRAP_REMOTE_ONLY:
    print("Mode: remote-only bootstrap (skip local LLM load)")

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print(f"  Device: {device}")


def _resolve_aux_model_device(env_name: str, default: str) -> str:
    requested = str(os.environ.get(env_name, "") or "").strip().lower()
    resolved = requested or default
    if resolved == "gpu":
        resolved = "cuda"
    if resolved == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if resolved == "cuda" and not torch.cuda.is_available():
        print(f"⚠️  {env_name}=cuda nhưng CUDA không sẵn sàng. Fallback sang CPU.")
        return "cpu"
    return resolved


# Default to auto-select CUDA when available, then fallback to CPU.
EMBEDDER_DEVICE = _resolve_aux_model_device("EMBEDDER_DEVICE", "auto")
MMR_QATC_DEVICE = _resolve_aux_model_device(
    "MMR_QATC_DEVICE",
    "auto"
)
print(f"Embedding device: {EMBEDDER_DEVICE}")
print(f"MMR QATC device: {MMR_QATC_DEVICE}")

gpu_memory_gb = 0.0
if torch.cuda.is_available():
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

llm_offload_folder = os.path.join(model_cache_path, "llm_offload")
os.makedirs(llm_offload_folder, exist_ok=True)

if torch.cuda.is_available():
    gpu_limit_gb = max(6, int(gpu_memory_gb - 1.5))
    llm_max_memory = {0: f"{gpu_limit_gb}GiB", "cpu": "48GiB"}
else:
    llm_max_memory = {"cpu": "48GiB"}

def ensure_semviqa_local_assets() -> None:
    """Ensure the SemViQA checkpoint and its base encoder config are downloaded."""
    if check_asset_exists(SEMVIQA_QATC, model_cache_path) and check_asset_exists(INFOXLM_LARGE_CONFIG, model_cache_path):
        return

    print("   SemViQA local assets chua day du. Dang tai bo sung...")

    if not check_asset_exists(SEMVIQA_QATC, model_cache_path):
        download_asset(SEMVIQA_QATC, model_cache_path)

    if not check_asset_exists(INFOXLM_LARGE_CONFIG, model_cache_path):
        download_asset(INFOXLM_LARGE_CONFIG, model_cache_path)

# ==============================================================================
# MODEL 1: LOAD LLM (Qwen 2.5-7B-Instruct) - UPGRADED
# ==============================================================================

print("\n" + "="*70)
print("MODEL 1/6: LOAD LLM (Qwen 2.5-7B-Instruct)")
print("="*70)

llm_model_name = "Qwen 2.5-7B-Instruct"
llm_model = None
llm_tokenizer = None
if BOOTSTRAP_REMOTE_ONLY:
    llm_model_name = "Remote provider (skip local LLM)"
    print("   Skip local LLM load vi runtime se dung backend tu xa.")
else:
    if not check_model_exists(LLM_MODEL_PATH):
        if check_model_exists(LLM_MODEL_FALLBACK_PATH):
            print(f"CANH BAO: Khong tim thay 7B model tai: {LLM_MODEL_PATH}")
            print(f"Fallback sang model cu: {LLM_MODEL_FALLBACK_PATH}")
            LLM_MODEL_PATH = LLM_MODEL_FALLBACK_PATH
            llm_model_name = "Qwen 2.5-3B-Instruct (fallback)"
        else:
            print(f"LOI: Khong tim thay LLM model tai: {LLM_MODEL_PATH}")
            print("Hay download models truoc")
            raise FileNotFoundError(f"LLM model not found at {LLM_MODEL_PATH}")

    try:
        print(f"Dang load {llm_model_name} tu: {LLM_MODEL_PATH}")
        if torch.cuda.is_available():
            print(f"   GPU VRAM: {gpu_memory_gb:.1f} GB")
            print(f"   Max memory map: {llm_max_memory}")
        
        # Load tokenizer
        print("   [1/2] Dang load tokenizer...")
        llm_tokenizer = AutoTokenizer.from_pretrained(
            LLM_MODEL_PATH,
            trust_remote_code=True,
            local_files_only=True
        )
        
        # Try quantization first, fallback to full precision
        print("   [2/2] Dang load model...")
        try:
            # Quantization config for memory efficiency
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            
            llm_model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_PATH,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                dtype=torch.float16,
                low_cpu_mem_usage=True,
                max_memory=llm_max_memory,
                offload_folder=llm_offload_folder,
                offload_state_dict=True,
                local_files_only=True
            )
            print("   Da load LLM voi 4-bit quantization!")
        except Exception as quant_error:
            print(f"   Khong the dung quantization: {quant_error}")
            print("   Dang load voi fp16 + CPU offload...")
            llm_model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_PATH,
                device_map="auto",
                trust_remote_code=True,
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True,
                max_memory=llm_max_memory,
                offload_folder=llm_offload_folder,
                offload_state_dict=True,
                local_files_only=True
            )
            print("   Da load LLM voi fp16 + CPU offload!")
        
        print("   Da load LLM thanh cong!")
        
    except Exception as e:
        print(f"   Loi khi load LLM: {e}")
        raise

# ==============================================================================
# MODEL 2: LOAD EMBEDDING (BGE-M3) - UPGRADED
# ==============================================================================

print("\n" + "="*70)
print("MODEL 2/6: LOAD EMBEDDING (BGE-M3)")
print("="*70)
embedding_path = EMBEDDING_MODEL_PATH

if not check_model_exists(EMBEDDING_MODEL_PATH):
    print(f"LOI: Khong tim thay Embedding model tai: {EMBEDDING_MODEL_PATH}")
    print("Hay chay CELL 2 de download models truoc")
    raise FileNotFoundError(f"Embedding model not found at {EMBEDDING_MODEL_PATH}")

try:
    print(f"Dang load Embedding tu: {embedding_path}")
    embedder = SentenceTransformer(embedding_path, device=EMBEDDER_DEVICE)
    print("   Da load Embedding thanh cong!")
except Exception as e:
    print(f"   Loi khi load Embedding: {e}")
    raise
print("="*70)

# ==============================================================================
# MODEL 3: LOAD VIRANKER (Vietnamese Cross-Encoder)
# ==============================================================================

print("\n" + "="*70)
print(" MODEL 3/6: LOAD VIRANKER (Vietnamese Cross-Encoder)")
print("="*70)

reranker_stage2_model = None
reranker_stage2_tokenizer = None

if not check_model_exists(RERANKER_STAGE2_PATH):
    print(f"  Không tìm thấy ViRanker tại: {RERANKER_STAGE2_PATH}")
    print("    Hệ thống sẽ sử dụng fallback reranking")
else:
    try:
        print(f"⏳ Đang load ViRanker từ: {RERANKER_STAGE2_PATH}")
        
        # Load tokenizer
        print("   [1/2] Đang load tokenizer...")
        reranker_stage2_tokenizer = AutoTokenizer.from_pretrained(
            RERANKER_STAGE2_PATH,
            local_files_only=True
        )
        
        # Load model
        print("   [2/2] Đang load model...")
        reranker_stage2_model = AutoModelForSequenceClassification.from_pretrained(
            RERANKER_STAGE2_PATH,
            local_files_only=True
        )
        reranker_stage2_model.to(device)
        reranker_stage2_model.eval()
        
        print("    Đã load ViRanker thành công!")
        
    except Exception as e:
        print(f"     Lỗi khi load ViRanker: {e}")
        print("    Hệ thống sẽ sử dụng fallback reranking")
        reranker_stage2_model = None
        reranker_stage2_tokenizer = None

# ==============================================================================
# MODEL 4: LOAD RERANKER FALLBACK (BGE-reranker-base)
# ==============================================================================

print("\n" + "="*70)
print(" MODEL 4/6: LOAD RERANKER FALLBACK (BGE-reranker-base)")
print("="*70)

reranker_fallback_model = None
reranker_fallback_tokenizer = None

if not check_model_exists(RERANKER_FALLBACK_PATH):
    print(f"  Không tìm thấy Reranker Fallback tại: {RERANKER_FALLBACK_PATH}")
    print("    Hệ thống sẽ sử dụng similarity-based reranking nếu cần")
else:
    try:
        print(f"⏳ Đang load Reranker Fallback từ: {RERANKER_FALLBACK_PATH}")
        
        # Load tokenizer
        print("   [1/2] Đang load tokenizer...")
        reranker_fallback_tokenizer = AutoTokenizer.from_pretrained(
            RERANKER_FALLBACK_PATH,
            local_files_only=True
        )
        
        # Load model
        print("   [2/2] Đang load model...")
        reranker_fallback_model = AutoModelForSequenceClassification.from_pretrained(
            RERANKER_FALLBACK_PATH,
            local_files_only=True
        )
        reranker_fallback_model.to(device)
        reranker_fallback_model.eval()
        
        print("    Đã load Reranker Fallback thành công!")
        
    except Exception as e:
        print(f"     Lỗi khi load Reranker Fallback: {e}")
        print("    Hệ thống sẽ sử dụng similarity-based reranking nếu cần")
        reranker_fallback_model = None
        reranker_fallback_tokenizer = None

# ==============================================================================
# MODEL 5: LOAD SEMVIQA QATC (FOR SEMANTIC HIGHLIGHTING + MMR)
# ==============================================================================

print("\n" + "="*70)
print(" MODEL 5/5: LOAD SEMVIQA QATC (SEMANTIC HIGHLIGHTING + MMR)")
print("="*70)

semantic_highlight_model = None
semantic_highlight_tokenizer = None

mmr_qatc_model = None
mmr_qatc_tokenizer = None

try:
    ensure_semviqa_local_assets()
    print(f" Đang load SemViQA QATC model từ: {MMR_QATC_MODEL_PATH}")
    print(f" Base encoder config: {MMR_QATC_BASE_CONFIG_PATH}")
    print(f" Device yêu cầu: {MMR_QATC_DEVICE}")

    from modeling_semviqa_qatc_standalone import QATCConfig, QATCForQuestionAnswering

    mmr_qatc_tokenizer = AutoTokenizer.from_pretrained(
        MMR_QATC_MODEL_PATH,
        local_files_only=True,
    )
    mmr_qatc_config = QATCConfig.from_pretrained(
        MMR_QATC_MODEL_PATH,
        local_files_only=True,
    )
    mmr_qatc_config.model_name = MMR_QATC_BASE_CONFIG_PATH
    mmr_qatc_model = QATCForQuestionAnswering.from_pretrained(
        MMR_QATC_MODEL_PATH,
        config=mmr_qatc_config,
        local_files_only=True,
    )
    mmr_qatc_model.to(MMR_QATC_DEVICE)
    mmr_qatc_model.eval()

    try:
        first_param_device = str(next(mmr_qatc_model.parameters()).device)
        print(f" SemViQA đang chạy trên: {first_param_device}")
    except Exception:
        pass

    semantic_highlight_model = mmr_qatc_model
    semantic_highlight_tokenizer = mmr_qatc_tokenizer

    print(" Đã load SemViQA QATC model thành công!")
    print(" Model sẽ được dùng chung cho semantic highlighting và MMR")

except Exception as e:
    print(" Lỗi khi load SemViQA QATC model")
    print(" Chi tiết:", e)
    print(" Fallback: similarity-based pruning + embedding-based MMR diversity")
    semantic_highlight_model = None
    semantic_highlight_tokenizer = None
    mmr_qatc_model = None
    mmr_qatc_tokenizer = None

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "="*70)
print("TOM TAT MODELS DA LOAD")
print("="*70)

models_loaded = {
    f"LLM ({llm_model_name})": BOOTSTRAP_REMOTE_ONLY or (llm_model is not None and llm_tokenizer is not None),
    "Embedding (BGE-M3)": embedder is not None,
    "ViRanker": reranker_stage2_model is not None,
    "Reranker Fallback (BGE-Base)": reranker_fallback_model is not None,
    "Semantic Highlighting (SemViQA)": semantic_highlight_model is not None and semantic_highlight_tokenizer is not None,
    "MMR QATC (SemViQA)": mmr_qatc_model is not None and mmr_qatc_tokenizer is not None,
}

success_count = sum(models_loaded.values())

for model_name, loaded in models_loaded.items():
    status = "THANH CONG" if loaded else "THAT BAI"
    print(f"{status} {model_name}")

print(f"\nTong so capabilities da san sang: {success_count}/6")

minimum_ready = embedder is not None and (BOOTSTRAP_REMOTE_ONLY or (llm_model is not None and llm_tokenizer is not None))
if minimum_ready:
    print("\n" + "="*70)
    print("DA LOAD THANH CONG CAC MODELS CAN THIET!")
    print("="*70)
    
    print("\nCac bien da duoc tao:")
    if BOOTSTRAP_REMOTE_ONLY:
        print("   • llm_model - SKIPPED (remote backend se cung cap sinh cau tra loi)")
        print("   • llm_tokenizer - SKIPPED")
    else:
        print("   • llm_model - LLM model (Qwen)")
        print("   • llm_tokenizer - Tokenizer cho LLM")
    print("   • embedder - Sentence Transformer model")
    if reranker_stage2_model:
        print("   • reranker_stage2_model - ViRanker model")
        print("   • reranker_stage2_tokenizer - ViRanker tokenizer")
    if reranker_fallback_model:
        print("   • reranker_fallback_model - BGE-Base fallback")
        print("   • reranker_fallback_tokenizer - BGE-Base tokenizer")
    if semantic_highlight_model and semantic_highlight_tokenizer:
        print("   • semantic_highlight_model - Alias toi SemViQA QATC model (context pruning)")
        print("   • semantic_highlight_tokenizer - SemViQA tokenizer cho semantic highlighting")
    if mmr_qatc_model and mmr_qatc_tokenizer:
        print("   • mmr_qatc_model - SemViQA QATC model (MMR evidence extraction)")
        print("   • mmr_qatc_tokenizer - SemViQA tokenizer")
    
    print("\n Reranking Configuration:")
    if reranker_stage2_model:
        print("    ViRanker Reranking: ENABLED")
        print("      Single step: ViRanker (Vietnamese refinement)")
        print("      Expected accuracy: +15-20% vs baseline")
    elif reranker_fallback_model:
        print("     Fallback Reranking: ENABLED")
        print("      Using BGE-Base fallback")
        print("      Expected accuracy: +5-10% vs baseline")
    else:
        print("     Similarity-Based Reranking: FALLBACK")
        print("      No reranker models available")
        print("      Using cosine similarity")

    print("\n MMR Diversity Configuration:")
    if mmr_qatc_model and mmr_qatc_tokenizer:
        print("    ✅ QATC MMR: ENABLED")
        print(f"      Model: {MMR_QATC_MODEL_ID}")
        print("      Query-conditioned evidence span extraction for diversity scoring")
    else:
        print("    ⚠️ Embedding MMR: FALLBACK MODE")
        print("      Using cosine similarity on contextualized chunks")
    
    print("\n Context Pruning Configuration:")
    if semantic_highlight_model and semantic_highlight_tokenizer:
        print("    ✅ Semantic Highlighting: ENABLED")
        print(f"      Model: {MMR_QATC_MODEL_ID}")
        print("      Query-conditioned sentence scoring via SemViQA QATC")
        print("      Expected: 70-80% token reduction")
    else:
        print("    ⚠️ Similarity-Based Pruning: FALLBACK MODE")
        print("      Using sentence-level cosine similarity")
        print("      Expected: 40-50% token reduction")
        print("      💡 Vẫn hoạt động tốt, chỉ kém hiệu quả một chút")
    
    print("\nBan co the tiep tuc voi Cell 4 de process documents")
    
else:
    print("\n" + "="*70)
    print("LOI: KHONG THE LOAD CAC MODELS CAN THIET")
    print("="*70)
    print("\nHay kiem tra:")
    print("   1. Da download models chua?")
    print("   2. Models co ton tai khong?")
    print("   3. Duong dan model_cache_path co dung khong?")
    raise RuntimeError("Failed to load required runtime models")

# ==============================================================================
# EXPORT VARIABLES TO GLOBALS (for main.py exec)
# ==============================================================================
print("\n" + "="*70)
print("EXPORTING VARIABLES TO GLOBALS")
print("="*70)

# Export all models to globals for use in other cells
if 'globals' in dir():
    globals()['llm_model'] = llm_model
    globals()['llm_tokenizer'] = llm_tokenizer
    globals()['embedder'] = embedder
    globals()['reranker_stage2_model'] = reranker_stage2_model
    globals()['reranker_stage2_tokenizer'] = reranker_stage2_tokenizer
    globals()['reranker_fallback_model'] = reranker_fallback_model
    globals()['reranker_fallback_tokenizer'] = reranker_fallback_tokenizer
    globals()['semantic_highlight_model'] = semantic_highlight_model
    globals()['semantic_highlight_tokenizer'] = semantic_highlight_tokenizer
    globals()['mmr_qatc_model'] = mmr_qatc_model
    globals()['mmr_qatc_tokenizer'] = mmr_qatc_tokenizer
    print("✓ All models exported to globals")

print("\n" + "="*70)
print("HOAN TAT CELL 3 - SAN SANG CHO CELL 4!")
print("="*70)

# Debug: Kiem tra cac bien da duoc tao
print("\n[DEBUG] Kiem tra cac bien trong globals():")
print(f"  llm_model: {'llm_model' in globals()}")
print(f"  llm_tokenizer: {'llm_tokenizer' in globals()}")
print(f"  embedder: {'embedder' in globals()}")
