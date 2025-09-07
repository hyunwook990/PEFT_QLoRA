# LLM 사용기
- 특정 업무에 특화된 LLM을 만들기 위해 파인튜닝(Fine-Tuning)은 필수적이라고 생각합니다.
- 파인튜닝과 랭체인(LangChain)을 사용하고 올라마(Ollama)로 로컬에서 구동이 가능한 LLM을 위한 실습입니다.

## 이슈
### 2025-09-07
- Ollama를 사용해 로컬에서 구동하기 위해선 허깅페이스(hugging face)에서 저장하는 방식(HF)과 다른 방식(GGUF)을 사용해야 합니다.

#### 🔹 Hugging Face (HF) 포맷

- 형식: pytorch_model.bin (또는 safetensors), config.json, tokenizer.json, special_tokens_map.json 등 여러 파일

주 사용처: Hugging Face Transformers 라이브러리

#### 특징

PyTorch, TensorFlow에서 바로 로드 가능 `AutoModelForCausalLM.from_pretrained(...)`

구조 정보(config.json) + 가중치 파일이 분리되어 있음

학습·추론·파인튜닝에 적합

일반적으로 FP32/FP16 같은 정밀도 높은 형식으로 저장 → 용량 큼 (예: Llama2-7B fp16 ≈ 13~14GB)

🔹 GGUF 포맷

형식: .gguf (파일 하나에 모든 정보 포함)

주 사용처: llama.cpp, Ollama, KoboldCpp, LM Studio 등 경량화 실행기

특징

CPU/GPU에서 빠르고 가볍게 실행 가능하도록 최적화

**양자화(quantization)**가 기본: Q2, Q4, Q5, Q8 등으로 메모리와 속도 절약

하나의 파일에 모델 + 메타데이터 + 토크나이저 정보까지 포함 → 배포·실행 편리

주로 실행/배포용이지, 추가 학습·파인튜닝은 GGUF로 직접 하기 힘듦