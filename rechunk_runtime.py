from __future__ import annotations

import argparse
import json

from runtime_service import (
    clear_processed_data_cache,
    get_runtime,
    get_runtime_state,
    release_runtime,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rechunk local processed data and rebuild runtime indexes.")
    parser.add_argument(
        "--provider",
        default=None,
        choices=["local", "groq", "ollama"],
        help="Optional provider to warm after rechunk.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    release_runtime()
    cache_reset = clear_processed_data_cache()
    get_runtime(force_reload=True, provider_name=args.provider)

    payload = {
        "action": "rechunk_local",
        "provider": args.provider or get_runtime_state().get("provider"),
        "cache_reset": cache_reset,
        "runtime_state": get_runtime_state(),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
