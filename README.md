# 中文诗词写作模型微调
### 项目结构
-chinese-poetry-collection 源数据集，来自魔搭社区的大佬

-dataProcessing.py 数据预处理，将源数据集转化为messages格式用于微调

-model_get huggingface模型下得太慢了，建议用这个脚本从魔搭下

-train_lora 用LoRA微调模型

-test_lora 测试微调后结果

-api_server 封装为本地微服务，可使用curl调用

-api_server_Java 封装为openai格式，可使用Java结合langchain4j无缝调用