import json
import os

import torch
from datasets import load_dataset
from modelscope import snapshot_download
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    EarlyStoppingCallback,
    IntervalStrategy
)
from transformers.trainer_utils import SaveStrategy

# ==================== 1、定义变量 ====================
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "../../datas/code_alpaca.json")
output_dir = os.path.abspath(os.path.join(current_dir, "./output/qwen2.5-7b-qlora-2"))
os.makedirs(output_dir, exist_ok=True)

assert os.path.exists(data_path), f"数据文件不存在：{data_path}"

model_name = snapshot_download("Qwen/Qwen2.5-Coder-7B-Instruct")

# ==================== 2、量化加载模型（QLoRA） ====================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

print("正在加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ==================== 3、LoRA 配置 ====================
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# ==================== 4、显存监控 ====================
def print_gpu_memory(prefix=""):
    if torch.cuda.is_available():
        print(f"\n{prefix} GPU 显存占用（GB）：")
        print(f"已分配: {torch.cuda.memory_allocated() / 1024 ** 3:.2f}")
        print(f"总预留: {torch.cuda.memory_reserved() / 1024 ** 3:.2f}")
    else:
        print("\n当前没有检测到 GPU！")


print_gpu_memory("训练前")

# ==================== 5、加载与格式化数据 ====================
dataset = load_dataset("json", data_files=data_path)
full_dataset = dataset["train"]

# 训练集和验证集划分
split_dataset = full_dataset.train_test_split(test_size=0.05, seed=42)
train_data = split_dataset["train"]
eval_data = split_dataset["test"]

print(f"\n数据集划分：")
print(f"训练集样本数：{len(train_data)}")
print(f"验证集样本数：{len(eval_data)}")


# 支持批处理、过滤空样本，防止 padding_mask 报错
def format_batch(batch):
    prompts = []
    for inst, inp, outp in zip(batch["instruction"], batch["input"], batch["output"]):
        if not inst or not outp:  # 跳过空样本
            continue

        if inp:
            text = (
                f"### 指令:\n{inst.strip()}\n\n"
                f"### 输入:\n{inp.strip()}\n\n"
                f"### 回复:\n{outp.strip()}"
            )
        else:
            text = (
                f"### 指令:\n{inst.strip()}\n\n"
                f"### 回复:\n{outp.strip()}"
            )

        prompts.append(text)

    tokenized = tokenizer(
        prompts,
        truncation=True,
        max_length=1024,
        padding="max_length",
        return_tensors=None,
    )
    return tokenized


tokenized_train = train_data.map(format_batch, batched=True, remove_columns=train_data.column_names)
tokenized_eval = eval_data.map(format_batch, batched=True, remove_columns=eval_data.column_names)

# 🔧 检查空样本
print(f"Tokenized Train: {len(tokenized_train)}, Eval: {len(tokenized_eval)}")

# ==================== 6、训练配置 ====================
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    warmup_steps=50,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=True,
    eval_strategy=IntervalStrategy.STEPS,
    eval_steps=80,
    save_strategy=SaveStrategy.STEPS,
    save_steps=80,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    logging_steps=50,
    report_to="none",
    prediction_loss_only=True,
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)


# ==================== 7、自定义回调 ====================
class MemoryMonitorCallback(TrainerCallback):
    def on_log(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        step = state.global_step
        if step % 10 == 0 or step == 1:
            print(f"\n========== Step {step} ==========")
            if torch.cuda.is_available():
                print(f"已分配显存 = {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
                print(f"总预留显存 = {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        if metrics:
            eval_loss = metrics.get("eval_loss", 0)
            if len(state.log_history) > 1:
                train_loss = state.log_history[-2].get("loss", 0)
                print(f"\n【过拟合检测】")
                print(f"  训练损失: {train_loss:.4f}")
                print(f"  验证损失: {eval_loss:.4f}")
                if eval_loss - train_loss > 0.5:
                    print(f"警告：可能出现过拟合！")


# ==================== 8、启动训练 ====================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=data_collator,
    callbacks=[
        # 监控训练过程中的内存（显存）使用情况，便于排查 OOM（内存溢出）问题
        MemoryMonitorCallback(),
        # 早停策略
        EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=0.001
        )
    ],
)

print("\n开始训练...")

trainer.train()

# ==================== 9、保存 LoRA 权重 ====================
print("\n保存 LoRA 权重...")
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"✓ LoRA 权重已保存到：{output_dir}")

history_file = os.path.join(output_dir, "training_history.json")
with open(history_file, 'w', encoding='utf-8') as f:
    json.dump(trainer.state.log_history, f, indent=2, ensure_ascii=False)
print(f"✓ 训练历史已保存到：{history_file}")

# ==================== 10、训练结束 ====================
print_gpu_memory("训练结束")
print("\n" + "=" * 50)
print("训练完成！")
print(f"LoRA adapter 保存在: {output_dir}")
print("=" * 50)
