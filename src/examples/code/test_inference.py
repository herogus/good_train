# from modelscope import snapshot_download
import os

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def print_gpu_memory(prefix=""):
    """打印 GPU 显存占用"""
    if torch.cuda.is_available():
        print(f"\n{prefix} GPU 显存占用（GB）：")
        print(f"已分配: {torch.cuda.memory_allocated() / 1024 ** 3:.2f}")
        print(f"总预留: {torch.cuda.memory_reserved() / 1024 ** 3:.2f}")
    else:
        print("\n当前没有检测到 GPU！")


import torch


def generate_response(model, tokenizer, instruction="", input_text="", max_new_tokens=1024):
    if input_text:
        prompt = (
            f"### 指令:\n{instruction.strip()}\n\n"
            f"### 输入:\n{input_text.strip()}\n\n"
            f"### 回复:\n"
        )
    else:
        prompt = (
            f"### 指令:\n{instruction.strip()}\n\n"
            f"### 回复:\n"
        )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.3,
        top_p=0.9,
        do_sample=False, # 启用采样生成模式，False，则使用贪婪解码（每次选概率最高的 token），输出更确定
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "### 回复:" in response:
        response = response.split("### 回复:")[-1].strip()
    return response


def main():
    # 定义路径
    # LoRA 微调后的模型
    current_dir = os.path.dirname(os.path.abspath(__file__))
    merged_dir = os.path.join(current_dir, "./output/qwen2.5-7b-qlora-merged")

    # 原始模型
    # merged_dir = snapshot_download("Qwen/Qwen3-4B-Instruct-2507")

    # 配置4bit量化参数
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 启用4bit量化
        bnb_4bit_quant_type="nf4",  # 使用NF4量化类型
        bnb_4bit_compute_dtype=torch.float16,  # 计算时使用float16
        bnb_4bit_use_double_quant=True,  # 启用双重量化，进一步压缩
    )

    print("\n加载合并后的模型进行推理（4bit量化）...")

    # 使用4bit量化加载模型
    inference_model = AutoModelForCausalLM.from_pretrained(
        merged_dir,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )

    inference_tokenizer = AutoTokenizer.from_pretrained(merged_dir, trust_remote_code=True)

    print_gpu_memory("推理模型加载后")

    # 测试用例
    test_cases = [
        {
            "instruction": "editRow方法的参数有哪些？",
            "input": "编辑某行记录",
        },
        {
            "instruction": "请实现GridView的editRow方法",
            "input": "编辑某行记录",
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'=' * 60}")
        print(f"测试用例 {i}:")
        print(f"{'=' * 60}")
        print(f"指令: {test_case['instruction']}")
        print(f"输入: {test_case['input']}")
        print(f"\n生成中...")

        response = generate_response(
            inference_model,
            inference_tokenizer,
            test_case['instruction'],
            test_case['input']
        )

        print(f"\n模型回复:")
        print("-" * 60)
        print(response)
        print("-" * 60)

    # 最终显存统计
    print_gpu_memory("推理结束")


if __name__ == "__main__":
    main()
