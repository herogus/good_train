import os

import torch
from modelscope import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def print_gpu_memory(prefix=""):
    """打印 GPU 显存占用"""
    if torch.cuda.is_available():
        print(f"\n{prefix} GPU 显存占用（GB）：")
        print(f"已分配: {torch.cuda.memory_allocated() / 1024 ** 3:.2f}")
        print(f"总预留: {torch.cuda.memory_reserved() / 1024 ** 3:.2f}")
    else:
        print("\n当前没有检测到 GPU！")


def main():
    # ==================== 1、定义路径 ====================
    current_dir = os.path.dirname(os.path.abspath(__file__))
    lora_dir = os.path.join(current_dir, "./output/qwen2.5-7b-qlora")
    merged_dir = os.path.join(current_dir, "./output/qwen2.5-7b-qlora-merged")

    # 检查 LoRA 权重是否存在
    assert os.path.exists(lora_dir), f"LoRA 权重目录不存在：{lora_dir}"

    model_name = snapshot_download("Qwen/Qwen2.5-Coder-7B-Instruct")

    # ==================== 2、合并 LoRA 权重 ====================
    print("=" * 60)
    print("开始合并 LoRA 权重到基础模型...")
    print("=" * 60)

    # 加载完整精度的基础模型（不使用量化）
    print("\n[1/4] 加载基础模型（FP16）...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # 使用 FP16，不使用 4-bit 量化
        device_map="auto",
        trust_remote_code=True,
    )
    print("✓ 基础模型加载完成")
    print_gpu_memory("加载基础模型后")

    # 加载 LoRA 权重
    print("\n[2/4] 加载 LoRA 权重...")
    model_with_lora = PeftModel.from_pretrained(base_model, lora_dir)
    print("✓ LoRA 权重加载完成")

    # 合并 LoRA 权重
    print("\n[3/4] 合并 LoRA 权重...")
    merged_model = model_with_lora.merge_and_unload()
    print("✓ 合并完成")
    print_gpu_memory("合并后")

    # 保存合并后的模型
    print(f"\n[4/4] 保存合并后的模型到：{merged_dir}")
    merged_model.save_pretrained(merged_dir)

    # 保存 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.save_pretrained(merged_dir)

    print("✓ 模型保存完成")

    # 删除合并目录中的 adapter 文件，避免被 PEFT 误识别为 LoRA 模型
    adapter_config_path = os.path.join(merged_dir, "adapter_config.json")
    adapter_model_path = os.path.join(merged_dir, "adapter_model.safetensors")

    if os.path.exists(adapter_config_path):
        os.remove(adapter_config_path)
        print("✓ 已删除 adapter_config.json")

    if os.path.exists(adapter_model_path):
        os.remove(adapter_model_path)
        print("✓ 已删除 adapter_model.safetensors")

    # 检查保存的文件
    saved_files = os.listdir(merged_dir)
    print(f"\n保存的文件列表：")
    model_file_size = 0
    for f in sorted(saved_files):
        file_path = os.path.join(merged_dir, f)
        if os.path.isfile(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  - {f} ({size_mb:.2f} MB)")
            if f.startswith("model") and f.endswith(".safetensors"):
                model_file_size += size_mb

    print(f"\n模型总大小: {model_file_size:.2f} MB ({model_file_size / 1024:.2f} GB)")

    print("\n" + "=" * 60)
    print("合并完成！")
    print("=" * 60)

    # 清理显存
    del base_model
    del model_with_lora
    del merged_model
    torch.cuda.empty_cache()

    print(f"\n✓ 合并后的完整模型保存在: {merged_dir}")


if __name__ == "__main__":
    main()
