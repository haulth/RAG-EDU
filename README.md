# ChatBot Quy Che Dao Tao

Chatbot hoi dap Quy che dao tao dai hoc su dung RAG (Retrieval-Augmented Generation), dong bo theo mot duong chay duy nhat:
- `FastAPI + web chat UI`
- UI, API va FAQ test deu dung chung `runtime_service.run_query`

## Trang thai dong goi hien tai

Repo da co:
- `pyproject.toml` de cai dat package editable
- `setup_env.bat` de tao moi truong va cai thu vien
- `run.bat` de chay UI thong nhat
- `run_api.bat` de chay API + chat UI
- `run_test.bat` de chay FAQ test theo cung luong giao dien
- `local_app_settings.example.json` de cau hinh chung cho UI / API / terminal test

Repo **khong** commit cac thu muc/artifact local sau:
- `venv/`
- `models/`
- `processed_data/`
- `documents/`
- `test_logs/`
- `local_app_settings.json`

## Yeu cau he thong

- Windows + Python `3.9`
- GPU NVIDIA khuyen nghi `RTX 3060 12GB` tro len
- CUDA 11.8 neu chay GPU
- RAM khuyen nghi `16GB+`

## Cai dat nhanh tren Windows

### Cach 1: dung file bat

```bat
setup_env.bat
```

Script se:
- tao `venv`
- cai PyTorch CUDA 11.8
- cai dependencies tu `requirements.txt`
- tao `local_app_settings.json` tu file mau neu chua co

### Cach 2: cai editable package

```powershell
py -3.9 -m venv venv
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
venv\Scripts\python.exe -m pip install -e .
```

## Cau hinh model / API

Tao file:

```text
local_app_settings.json
```

tu file mau:

```text
local_app_settings.example.json
```

Trong file nay ban co the cau hinh:
- `local`
- `groq`
- `ollama`

Luu y:
- khong commit file `local_app_settings.json`
- `api_key` de trong repo mac dinh la rong

## Download models

Neu chua co models local:

```powershell
venv\Scripts\python.exe download_models.py
```

## Cac cach chay

### 1. UI thong nhat

```bat
run.bat
```

Hoac:

```powershell
venv\Scripts\python.exe main.py
```

Mac dinh mo tai:
- `http://localhost:8000`

`main.py` hien chi la compatibility launcher, va cung se khoi dong FastAPI + web chat UI tai cong tren.

### 2. API + chat UI rieng

```bat
run_api.bat
```

Hoac:

```powershell
venv\Scripts\python.exe api_server.py
```

API chinh:
- `GET /api/health`
- `GET /api/providers`
- `GET /api/settings`
- `POST /api/bootstrap`
- `POST /api/settings`
- `POST /api/chat`

### 3. Chay test FAQ cung luong giao dien

```bat
run_test.bat
```

Hoac:

```powershell
venv\Scripts\python.exe run_random_faq_system_test.py --execution-path service --sample-size 5 --seed 42
```

`--execution-path pipeline` da duoc giu lai nhu alias cu, nhung se tu dong map sang `service` de UI, API va test cung dung mot semantics.

## Cau truc chinh

```text
api_server.py
runtime_service.py
runtime_bootstrap.py
main.py
download_models.py
run_random_faq_system_test.py

CELL_3_LOAD_ALL_MODELS.py
CELL_4_ULTIMATE_COMPLETE_METADATA.py
CELL_5_HYBRID_RETRIEVAL_ENHANCED.py
CELL_6_LLM_SYNTHESIS_WITH_PRUNING.py
CELL_8_END_TO_END_PIPELINE.py

web/
data/
```

## Models chinh

- `Qwen2.5-7B-Instruct` cho answer generation
- `BAAI/bge-m3` cho embeddings
- `BAAI/bge-reranker-v2-m3` cho rerank stage 1
- `itdainb/vietnamese-cross-encoder` cho rerank stage 2

## Goi lenh sau khi cai package

Neu da `pip install -e .`, co the dung:

```powershell
chatbotedu-api
chatbotedu-ui
chatbotedu-gradio
chatbotedu-test --execution-path service --sample-size 5 --seed 42
chatbotedu-download-models
```

`chatbotedu-gradio` hien la compatibility alias va cung se khoi dong UI thong nhat qua FastAPI.

## Chuan bi push GitHub

Truoc khi push:
- kiem tra `local_app_settings.json` khong chua secret
- dam bao `models/`, `venv/`, `processed_data/`, `test_logs/` khong bi add
- neu can, chi commit:
  - source code
  - `requirements.txt`
  - `pyproject.toml`
  - README
  - scripts `.bat`

## License

MIT
