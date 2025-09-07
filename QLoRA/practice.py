import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, # 인과적 언어 추론(예: GPT)을 위한 모델을 자동으로 불러오는 클래스
    AutoTokenizer,        # 입력 문장을 토큰 단위로 자동으로 잘라주는 역할
    BitsAndBytesConfig,   # 모델 구성 (벡터의 INT8 최대 절대값 양자화 기법을 사용할 수 있도록 도와주는 Meta AI의 라이브러리)
    HfArgumentParser,     # 파라미터 파싱
    TrainingArguments,    # 훈련 설정
    pipeline,             # 파이프라인 설정 
    logging,              # 로깅을 위한 클래스
)

from peft import LoraConfig, PeftModel
from trl import SFTTrainer

model_name = "NousResearch/Llama-2-7b-chat-hf"

dataset_name = "mlabonne/guanaco-llama2-1k"

new_model = "tuned-llama-2-7b-miniguanaco"

lora_r = 16
lora_alpha = 16
lora_dropout = 0.05
use_4bit = True 
bnb_4bit_compute_dtype = "bfloat16"
bnb_4bit_quant_type = "nf4" 
use_nested_quant = False 
output_dir = "./results" 
num_train_epochs = 1 

fp16 = False   
bf16 = True   

per_device_train_batch_size = 1
per_device_eval_batch_size = 1
gradient_accumulation_steps = 8
gradient_checkpointing = True  
max_grad_norm = 0.3

learning_rate = 2e-6
weight_decay = 0.001

optim = "paged_adamw_32bit"  

lr_scheduler_type = "cosine"

max_steps = -1
warmup_ratio = 0.03  

group_by_length = True   

save_steps = 0

logging_steps = 25

max_seq_length = 512

packing = False

device_map = {"": 0}  

dataset = load_dataset(dataset_name, split="train")

compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

tokenizer.pad_token = tokenizer.eos_token   # padding token을 EOS로 설정 Llama2에는 padding token이 없어서 설정
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training. Padding을 오른쪽 위치에 추가한다.

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",    # bias 파라미터는 수정하지 않는다.
    task_type="CAUSAL_LM", # 파인튜닝할 태스크를 Optional로 지정할 수 있는데, 여기서는 CASUAL_LM을 지정하였다.
)

training_arguments = TrainingArguments(
    output_dir=output_dir,      # 학습 결과 저장 위치
    num_train_epochs=num_train_epochs,  # 학습 반복 횟수
    per_device_train_batch_size=per_device_train_batch_size,    # GPU 하나당 batch 크기
    gradient_accumulation_steps=gradient_accumulation_steps,    # 여러 step에서 grad 쌓아서 효과적으로 큰 batch처럼 학습
    optim=optim,    # optimizer 종류
    save_steps=save_steps,  # 몇 step마다 체크포인트 저장할지
    logging_steps=logging_steps,    # 몇 step마다 로그 출력할지
    learning_rate=learning_rate,    # 학습률
    weight_decay=weight_decay,  # 규제
    fp16=fp16,  # 사용여부
    bf16=bf16,  # 이하동문
    max_grad_norm=max_grad_norm,    # gradient clipping 최대값
    max_steps=max_steps,    # 총 학습 step(epoch대신 step으로 제어 가능)
    warmup_ratio=warmup_ratio,  # 학습 초반에 learning rate를 천천히 올리는 비율
    group_by_length=group_by_length,    # 비슷한 길이의 샘플끼리 묶어서 효율적으로 학습
    lr_scheduler_type=lr_scheduler_type,    # learning rate 스케줄러 종류
    report_to="tensorboard" # 로그 기록을 TensorBoard로 보냄
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,    # LoRA 설정
    dataset_text_field="text",  # 데이터셋에서 텍스트가 들어있는 열 이름
    max_seq_length=max_seq_length,  # 토큰 시퀀스 최대 길이(넘어가면 자름)
    tokenizer=tokenizer,    # 토크나이저
    args=training_arguments,    # 학습 설정 전달
    packing=packing,    # 여러 샘플을 이어 붙여 max seq_length를 채워 학습할지 여부(사용 시 효율 상승)
)

trainer.train()

trainer.model.save_pretrained(new_model)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16
)
model = PeftModel.from_pretrained(base_model, new_model)

model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)  

tokenizer.pad_token = tokenizer.eos_token  

tokenizer.padding_side = "right"