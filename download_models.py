"""Download all Hugging Face model assets into the shared local cache."""

from model_assets import (
    MODEL_DOWNLOAD_ORDER,
    check_asset_exists,
    download_asset,
    normalize_cache_root,
)


print("="*70)
print("DOWNLOAD MODELS TU HUGGING FACE")
print("="*70)


def download_model(asset) -> bool:
    """Download one asset from the shared registry."""
    model_path = asset.resolve_path()

    if check_asset_exists(asset):
        print(f"\n[SKIP] {asset.key} - Da ton tai")
        return True

    print(f"\n[DOWNLOAD] {asset.key}")
    print(f"  Repo: {asset.repo_id}")
    print(f"  Size: {asset.size_hint}")
    print(f"  Desc: {asset.description}")

    try:
        download_asset(asset)
        print(f"  [OK] Downloaded successfully -> {model_path}")
        return True
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


if __name__ == "__main__":
    cache_root = normalize_cache_root()
    print(f"\nThu muc luu models: {cache_root}")
    print("Tong dung luong can: ~19.2 GB")
    print("Thoi gian du kien: 20-40 phut (tuy toc do mang)")

    input("\nNhan Enter de bat dau download...")

    success = 0
    failed = 0

    for asset in MODEL_DOWNLOAD_ORDER:
        if download_model(asset):
            success += 1
        else:
            failed += 1

    print("\n" + "="*70)
    print("KET QUA")
    print("="*70)
    print(f"Thanh cong: {success}/{len(MODEL_DOWNLOAD_ORDER)}")
    print(f"That bai: {failed}/{len(MODEL_DOWNLOAD_ORDER)}")

    if success >= 2:
        print("\nCo the chay chatbot voi cac models da download")
    else:
        print("\nCan download it nhat LLM va Embedding models")
