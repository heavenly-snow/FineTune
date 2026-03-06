import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. 指路牌
base_model_path = "D:/HFCache/hub/qwen/Qwen2___5-1___5B-Instruct"  # 你的原版大模型路径
lora_path = "./my_poetry_lora_final"                          # 你的外挂 U 盘路径

print("1. 正在请出大模型...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

print("2. 正在插上你的专属唐诗 U 盘...")
# 核心大招：把你的 LoRA 权重和原版模型合并！
model = PeftModel.from_pretrained(base_model, lora_path)

print("\n================== 开始对诗 ===================")
# 写一句你想让它续写的诗
prompt_text = "西风雨骤人间晚"

# 按照 Qwen 喜欢的格式组织语言
messages = [
    {"role": "system", "content": "你是一位精通中国传统文化的古代诗人，擅长根据上半句续写古诗。"},
    {"role": "user", "content": f"请续写这句诗：{prompt_text},"}
]

# 翻译成底层代码
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

print("3. 模型正在思考...\n")
# 让模型开始吐字
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=50, # 最多让她续写 50 个字
    temperature=0.7    # 创意度，0.7 比较适合写诗
)

# 把生成的数字翻译回人类的文字（只截取新生成的部分）
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(f"你的上句：{prompt_text}")
print(f"AI的下句：{response}")
