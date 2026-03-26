# chatbot-karbi

간단한 FastAPI 기반 사내 AI 어시스턴트 프로젝트입니다.

## 포함 파일

- `main.py`: 백엔드 API
- `static/index.html`: 프론트엔드 화면
- `requirements.txt`: Python 의존성 목록

## 실행 준비

1. Python 가상환경 생성
2. 패키지 설치
3. Ollama 실행 및 모델 준비
4. FastAPI 서버 실행

예시:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8091 --reload
```

## 환경 변수

필요하면 아래 값을 조정해서 실행할 수 있습니다.

- `OLLAMA_URL` 기본값: `http://127.0.0.1:11434`
- `TEXT_MODEL` 기본값: `qwen2.5:7b`
- `VISION_MODEL` 기본값: `qwen2.5vl:7b`
- `CHROMA_DIR` 기본값: `./chroma_db`
- `EMBED_MODEL` 기본값: `sentence-transformers/all-MiniLM-L6-v2`
- `WHISPER_MODEL` 기본값: `base`

## 참고

- `chatbot-env/`는 로컬 가상환경 폴더라 저장소에 포함하지 않습니다.
- 프론트엔드의 API 주소는 `static/index.html` 내부 `const API` 값을 실제 서버 주소에 맞게 바꿔야 할 수 있습니다.
- 업로드된 PDF 벡터 데이터까지 공유하려면 `chroma_db/` 폴더도 함께 관리해야 합니다.
