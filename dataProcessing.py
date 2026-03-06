from datasets import load_dataset
import json
print("1. ============数据预览=================")

raw_dataset = load_dataset(
    "csv",
    data_files= {"train": "chinese-poetry-collection/train.csv","test" : "chinese-poetry-collection/test.csv"}
)

# 数据骨架
print("==========Structure===========")
print(raw_dataset)
# 具体数据
print("===========Data===================")
print(raw_dataset["train"][0])

print("\n2.=====================数据预处理函数构建=====================")

def format_poetry(example):
    text = example.get("text1","")

    # 有逗号找逗号，没逗号找句号
    split_idx = text.find("，")
    if split_idx == -1:
        split_idx = text.find("。")
    # 句号也没有或者太短了,舍弃掉
    if split_idx == -1 or split_idx + 1 >= len(text):
        return {"messages":[]}
    # 有说法的，messages作为对话，中间有多条消息，所以[]
    # 每条消息，又是一个结构体，包含role和content，所以{}
    return {
        "messages":[
            {"role":"system","content":"根据上半句古诗续写下半句，要求字数一致、风格统一、语义连贯、风格符合中国古典诗词。只输出续写内容，不得包含解释或额外文本。"},
            {"role":"user","content":text[:split_idx+1]},
            {"role":"assistant","content":text[split_idx+1:]},
        ]
    }

print("\n3.=================启动数据处理流水线=====================")

processed_datasets = raw_dataset.map(
    format_poetry,
    remove_columns = ['text1']
)

print("\n4.==================处理后的骨架======================")
print(processed_datasets)

print(json.dumps(processed_datasets["train"][0], indent=2, ensure_ascii=False))

print("\n5.==================质检剔除残次数据======================")
final_datasets = processed_datasets.filter(lambda x: len(x["messages"]) > 0)

print("=== 质检报告 ===")
print(f"清洗前 train: {len(processed_datasets['train'])} 条 -> 清洗后: {len(final_datasets['train'])} 条")
print(f"清洗前 test: {len(processed_datasets['test'])} 条 -> 清洗后: {len(final_datasets['test'])} 条")

print("\n6.==================保存最终数据======================")
final_datasets['train'].to_json("dataResource/train_poetry_hf.jsonl",force_ascii=False)
final_datasets['test'].to_json("dataResource/test_poetry_hf.jsonl",force_ascii=False)

print("\n==================================================")
print("==================保存最终数据======================")
print("===================================================")