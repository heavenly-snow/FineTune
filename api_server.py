import torch
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ==========================================
# 1. 启动时的准备工作：把大模型请进内存供起来
# ==========================================
print("🚀 正在启动 AI 微服务，加载基座模型与外挂...")
base_model_path = "D:/LLM_Models/qwen/Qwen2.5-1.5B-Instruct"
# ⚠️ 注意：换成你刚才 eval_loss 最低的那个 checkpoint 文件夹路径！
lora_path = "./poetry_model_output/checkpoint-1600"

tokenizer = AutoTokenizer.from_pretrained(base_model_path)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"  # 自动调度显存
)
# 挂载你的诗词外挂 U 盘
model = PeftModel.from_pretrained(base_model, lora_path)
print("✅ 模型加载完毕，服务器准备接客！")

# ==========================================
# 2. 定义微服务的接口和数据格式
# ==========================================
app = FastAPI(title="Poetry AI Microservice")


# 规定前端/Java端传过来的 JSON 必须长什么样
class ChatRequest(BaseModel):
    prompt: str  # 用户输入的上半句诗
    max_tokens: int = 50  # 最多生成多少字
    temperature: float = 0.7  # 模型的发散程度


# ==========================================
# 3. 核心业务逻辑：暴露出 /generate 接口
# ==========================================
@app.post("/generate")
async def generate_poetry(request: ChatRequest):
    # 组装 Qwen 专属的对话剧本
    messages = [
        {"role": "system", "content": "你是一位精通中国传统文化的古代诗人，擅长根据上半句续写古诗。"},
        {"role": "user", "content": f"请续写这句诗：{request.prompt}"}
    ]

    # 翻译成底层 Token
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 让显卡开始算力狂飙
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature
    )

    # 剥离掉我们输入的提示词，只保留模型新吐出来的话
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # 返回标准的 JSON 响应给 Java 后端
    return {
        "code": 200,
        "msg": "success",
        "data": {
            "prompt": request.prompt,
            "reply": response_text
        }
    }


# ==========================================
# 4. 点火启动服务器
# ==========================================
if __name__ == "__main__":
    # 启动在 8000 端口
    uvicorn.run(app, host="127.0.0.1", port=8000)