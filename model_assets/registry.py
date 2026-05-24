"""Central registry for local model cache paths and Hugging Face assets."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Union

from config import MODEL_CACHE_PATH


PathLike = Union[str, os.PathLike[str], Path]


@dataclass(frozen=True)
class ModelAsset:
    key: str
    repo_id: str
    dir_name: str
    size_hint: str
    description: str
    config_only: bool = False
    allow_patterns: tuple[str, ...] = ()

    def resolve_path(self, cache_root: Optional[PathLike] = None) -> Path:
        return normalize_cache_root(cache_root) / self.dir_name


def normalize_cache_root(cache_root: Optional[PathLike] = None) -> Path:
    return Path(cache_root or MODEL_CACHE_PATH).resolve()


def _as_path(path: PathLike) -> Path:
    return Path(path).resolve()


def check_model_exists(model_path: PathLike) -> bool:
    path = _as_path(model_path)
    if not path.exists():
        return False
    files = [item.name for item in path.iterdir()]
    has_config = any(name == "config.json" for name in files)
    has_model = any(
        ("model" in name or "pytorch_model" in name or name.endswith(".safetensors"))
        for name in files
    )
    return has_config and has_model


def check_config_exists(config_path: PathLike) -> bool:
    path = _as_path(config_path)
    return (path / "config.json").exists()


def get_model_size(model_path: PathLike) -> str:
    path = _as_path(model_path)
    if not path.exists():
        return "0 MB"

    total_size = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            total_size += file_path.stat().st_size

    size_mb = total_size / (1024 * 1024)
    if size_mb > 1024:
        return f"{size_mb / 1024:.2f} GB"
    return f"{size_mb:.1f} MB"


LLM_QWEN_7B = ModelAsset(
    key="llm_qwen_7b",
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    dir_name="llm_qwen_7b",
    size_hint="~15.2 GB",
    description="LLM model de tao cau tra loi",
)

EMBED_BGE_M3 = ModelAsset(
    key="embed_bge_m3",
    repo_id="BAAI/bge-m3",
    dir_name="embed_bge_m3",
    size_hint="~2.3 GB",
    description="Embedding model cho vector search",
)

RERANKER_VIETNAMESE = ModelAsset(
    key="reranker_vietnamese",
    repo_id="itdainb/vietnamese-cross-encoder",
    dir_name="reranker_vietnamese",
    size_hint="~440 MB",
    description="ViRanker cho reranking tieng Viet",
)

RERANKER_BGE_BASE = ModelAsset(
    key="reranker_bge_base",
    repo_id="BAAI/bge-reranker-base",
    dir_name="reranker_bge_base",
    size_hint="~1.1 GB",
    description="Reranker fallback",
)

SEMVIQA_QATC = ModelAsset(
    key="semantic_highlight",
    repo_id="SemViQA/qatc-infoxlm-viwikifc",
    dir_name="semviqa_qatc_infoxlm_viwikifc",
    size_hint="~2.0 GB",
    description="SemViQA QATC cho semantic highlighting + MMR",
)

INFOXLM_LARGE_CONFIG = ModelAsset(
    key="infoxlm_large_config",
    repo_id="microsoft/infoxlm-large",
    dir_name="infoxlm_large_config",
    size_hint="config-only",
    description="Base config cho SemViQA standalone wrapper",
    config_only=True,
    allow_patterns=("config.json",),
)


MODEL_DOWNLOAD_ORDER = (
    LLM_QWEN_7B,
    EMBED_BGE_M3,
    RERANKER_VIETNAMESE,
    RERANKER_BGE_BASE,
    SEMVIQA_QATC,
    INFOXLM_LARGE_CONFIG,
)

MODEL_ASSETS: Dict[str, ModelAsset] = {
    asset.key: asset
    for asset in MODEL_DOWNLOAD_ORDER
}


def get_asset(asset: Union[str, ModelAsset]) -> ModelAsset:
    if isinstance(asset, ModelAsset):
        return asset
    key = str(asset or "").strip()
    if key not in MODEL_ASSETS:
        raise KeyError(f"Unknown model asset: {key}")
    return MODEL_ASSETS[key]


def get_asset_path(asset: Union[str, ModelAsset], cache_root: Optional[PathLike] = None) -> Path:
    return get_asset(asset).resolve_path(cache_root)


def check_asset_exists(asset: Union[str, ModelAsset], cache_root: Optional[PathLike] = None) -> bool:
    model_asset = get_asset(asset)
    path = model_asset.resolve_path(cache_root)
    if model_asset.config_only:
        return check_config_exists(path)
    return check_model_exists(path)


def iter_assets() -> Iterable[ModelAsset]:
    return MODEL_DOWNLOAD_ORDER


def download_asset(asset: Union[str, ModelAsset], cache_root: Optional[PathLike] = None) -> Path:
    model_asset = get_asset(asset)
    local_dir = model_asset.resolve_path(cache_root)

    from huggingface_hub import snapshot_download

    snapshot_kwargs = {
        "repo_id": model_asset.repo_id,
        "local_dir": str(local_dir),
        "local_dir_use_symlinks": False,
    }
    if model_asset.allow_patterns:
        snapshot_kwargs["allow_patterns"] = list(model_asset.allow_patterns)
    snapshot_download(**snapshot_kwargs)
    return local_dir
