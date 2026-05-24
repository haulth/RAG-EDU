"""Shared runtime bootstrap for API, UI, and test runners."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Dict


def bootstrap_runtime(project_root: Path, remote_only: bool = False) -> Dict:
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from config import BASE_PATH, DOCUMENTS_PATH, MODEL_CACHE_PATH, PROCESSED_DATA_PATH, model_cache_path

    module_name = "rag_test_runtime"
    runtime_module = types.ModuleType(module_name)
    runtime_module.__file__ = str(project_root / "runtime_bootstrap.py")
    sys.modules[module_name] = runtime_module

    runtime_globals: Dict = {
        "__name__": module_name,
        "__file__": str(project_root / "runtime_bootstrap.py"),
        "BASE_PATH": BASE_PATH,
        "DOCUMENTS_PATH": DOCUMENTS_PATH,
        "MODEL_CACHE_PATH": MODEL_CACHE_PATH,
        "PROCESSED_DATA_PATH": PROCESSED_DATA_PATH,
        "model_cache_path": model_cache_path,
        "BOOTSTRAP_REMOTE_ONLY": bool(remote_only),
    }
    runtime_module.__dict__.update(runtime_globals)
    runtime_globals = runtime_module.__dict__

    cell_files = [
        "CELL_3_LOAD_ALL_MODELS.py",
        "CELL_4_ULTIMATE_COMPLETE_METADATA.py",
        "CELL_5_HYBRID_RETRIEVAL_ENHANCED.py",
        "CELL_6_LLM_SYNTHESIS_WITH_PRUNING.py",
        "CELL_8_END_TO_END_PIPELINE.py",
    ]

    for cell_name in cell_files:
        cell_path = project_root / cell_name
        cell_module_name = cell_path.stem
        cell_module = types.ModuleType(cell_module_name)
        cell_module.__file__ = str(cell_path)
        cell_module.__dict__.update(runtime_globals)
        cell_module.__dict__["__name__"] = cell_module_name
        cell_module.__dict__["__file__"] = str(cell_path)
        sys.modules[cell_module_name] = cell_module

        with open(cell_path, "r", encoding="utf-8") as fh:
            code = fh.read()
        exec(compile(code, str(cell_path), "exec"), cell_module.__dict__)

        for key, value in cell_module.__dict__.items():
            if key in {"__name__", "__file__", "__package__", "__loader__", "__spec__", "__cached__"}:
                continue
            runtime_globals[key] = value

    required = ["rag_pipeline", "RAGConfig"]
    missing = [name for name in required if name not in runtime_globals]
    if missing:
        raise RuntimeError(f"Missing runtime objects after bootstrap: {', '.join(missing)}")

    return runtime_globals
