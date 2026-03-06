import time
import torch
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ==========================================
# 1. 准备模型 (跟刚才完全一样)
# ==========================================
base_model_path = "D:/HFCache/hub/qwen/Qwen2___5-1___5B-Instruct"
lora_path = "./poetry_model_output/checkpoint-1400"  # 记得换成你跑出来的 Checkpoint 路径

print("正在启动兼容 OpenAI 协议的本地大模型微服务...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, lora_path)
print("模型加载完毕，伪装成 OpenAI 服务器成功！")

app = FastAPI(title="OpenAI-Compatible Local LLM")


# ==========================================
# 2. 完美复刻 OpenAI 的入参数据结构
# ==========================================
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "qwen2.5-1.5b-lora"  # 模型名字随便起
    messages: list[ChatMessage]  # 接收标准的消息列表
    temperature: float = 0.7
    max_tokens: int = 512


# ==========================================
# 3. 核心接口：路径必须是严格的 /v1/chat/completions
# ==========================================
@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    # 把请求里的 Pydantic 对象转换成字典列表
    msgs = [{"role": m.role, "content": m.content} for m in request.messages]

    # 套用 Qwen 的底层对话模板
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 显卡狂飙
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature
    )

    # 截取新生成的部分
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # ==========================================
    # 4. 重点：组装成 OpenAI 一模一样的 JSON 响应骗过 Java 端
    # ==========================================
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"  # 告诉客户端回答完毕
            }
        ],
        "usage": {
            "prompt_tokens": len(model_inputs.input_ids[0]),
            "completion_tokens": len(generated_ids[0]),
            "total_tokens": len(model_inputs.input_ids[0]) + len(generated_ids[0])
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)