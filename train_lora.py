import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,logging
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import logging as hf_logging
from peft import LoraConfig, get_peft_model

print("===========================1.导入数据===============================")
dataset = load_dataset("json",data_files= {"train":"dataResource/train_poetry_hf.jsonl","test":"dataResource/test_poetry_hf.jsonl"})

model_id = "D:/HFCache/hub/qwen/Qwen2___5-1___5B-Instruct"
print(f"\n==================2.正在载入 ({model_id}) ，配置降维压缩========")

print("=====================3.加载LoRA配置=================================")
lora_config = LoraConfig(
    r=8,               # 外挂的“厚度”/容量 (秩)。通常选 8 或 16。数字越大，模型学得越细，但越吃显存。
    lora_alpha=16,     # 外挂的“嗓门”/影响力。一般设为 r 的 2 倍。决定了这个外挂对大模型原本性格的影响程度。
    target_modules=["q_proj", "v_proj"], # 外挂插在哪里？这里指插在 Transformer 的核心区 (Attention 注意力机制的 Q 和 V 矩阵上)。
    lora_dropout=0.05, # 防过拟合机制：训练时随机让 5% 的神经元“睡觉”，逼着大模型去理解诗词，而不是死记硬背。
    bias="none",       # 不更新偏置项（省算力）
    task_type="CAUSAL_LM" # 我们的任务类型：因果语言建模（说人话就是：顺着上文往下续写文字）
)

print("\n==================4.正在加载分词器(Tokenizer)===================")
# 核心知识点 2：分词器 (Tokenizer)
# 大模型不认识汉字，它只认识数字。分词器就是大模型的“新华字典”。
# 比如把“床前明月光”变成 [243, 8912, 12, 66] 这样的数字编号喂给模型。
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 补丁：Qwen 等很多模型没有专门的 pad_token（用来补齐长短不一的句子的占位符）
# 工业界标准做法：拿 eos_token (句子结束符) 顶替一下
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("\n======================5.正在把压缩后的大模型装载到 GPU 显存里========================")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                  # 开启 4-bit 终极压缩
    bnb_4bit_use_double_quant=True,     # 开启二次量化，省显存，不影响计算
    bnb_4bit_quant_type="nf4",          # 默认用这个，好用
    bnb_4bit_compute_dtype=torch.float16 # 普通显卡用float16就行
)
# 开启日志
hf_logging.set_verbosity_info()
print("\n================================6.模型开始下载================================================")
# 这一步会联网下载模型权重（大约 3 个 G），并根据咱们的配置直接以 4-bit 格式塞进显卡
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"  # 自动调度：显卡装不下就塞点给内存，极其智能
)

# 省显存 关闭kv cache
# Transformer 在推理时会保存：Key Value Attention cache 微调训练是“全量并行”的，不需要一步步预测
model.config.use_cache = False
# 再省显存 前向传播时只在每隔几层的地方设一个“存档点”。等反向传播需要用到中间数据时，从最近的存档点重新计算一遍。
# 显存占用瞬间暴降 50% 以上，代价是：训练速度会变慢大约 20%~30%。典型的时间换空间
model.gradient_checkpointing_enable()
print("\n================================7.模型开始载入================================================")
model = get_peft_model(model, lora_config)

print("\n==================8.成功完成===============================")
model.print_trainable_parameters()

from transformers import TrainingArguments
from trl import SFTTrainer

print("\n==================9. 配置炼丹炉火候 (TrainingArguments)===================")
# 这里面的参数就是所谓的“超参数 (Hyperparameters)”，直接决定了模型聪不聪明、会不会跑爆显存
training_args = TrainingArguments(
    output_dir="./poetry_model_output", # 训练过程中的存档点 (Checkpoints) 会存到这里
    per_device_train_batch_size=2,      # 每次喂给大模型几首诗？显存小就填 1 或者 2，显存大可以填 4 或 8
    gradient_accumulation_steps=8,      # 显存不够的变相大招：攒够 4 次小 batch 再更新一次参数，相当于 batch_size=8
    num_train_epochs=1,                 # 咱们先跑 1 遍全量数据试试水 (测试跑通最重要！)
    learning_rate=2e-4,                 # LoRA 微调的黄金学习率，照抄就行
    logging_steps=50,                   # 每训练 50 步，在屏幕上打印一次进度和 Loss (损失值)
    eval_strategy="steps",              # 咱们不是有 test 集吗？告诉它按步数来考试
    eval_steps=200,                      # 每训练 200 步，就拿 test 集考一次试
    save_strategy="steps",              # 每隔多少步保存一次存档，不然会磁盘
    save_steps=200,
    save_total_limit= 3,                # 最多保存三个多了就删除
    bf16=True,                          # 你的显卡支持 CUDA 13.0，绝对是现代新卡，开启 bf16 混合精度可以大幅提速！(如果报错改 fp16=True)
    report_to="none"                    # 关掉第三方监控面板，保持终端清爽
)

print("\n==================10. 请出教练机 (SFTTrainer)===================")
# SFTTrainer (Supervised Fine-Tuning Trainer) 会自动处理一切复杂的反向传播、梯度下降求导
tokenizer.model_max_length = 256

trainer = SFTTrainer(
    model=model,                        # 挂载了 LoRA 的大模型
    args=training_args,                 # 炼丹火候
    train_dataset=dataset["train"],     # 咱们洗好的 38 万条唐诗训练集
    eval_dataset=dataset["test"],       # 咱们预留的 1710 条考试题
    processing_class=tokenizer,                # 新华字典
    # SFTTrainer 极其智能，只要你的数据列名叫 "messages"，它会自动调取模型自带的 chat_template 把数据拼装好！
)

print("\n==================11. 🔥 正式点火炼丹！===================")
trainer.train()

print("\n==================12. 炼丹结束，打包出炉！===================")
# 训练完成后，把咱们千辛万苦炼出来的 LoRA 外挂 U 盘保存下来！
trainer.save_model("./my_poetry_lora_final")
print("🎉 恭喜兄弟！你的第一个专属微调大模型已彻底完工！")