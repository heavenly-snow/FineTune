from modelscope.hub.snapshot_download import snapshot_download

print("🚀 启动国内满速下载通道...")

# model_id 是魔搭上的名字
# cache_dir 是你想把模型保存在电脑上的哪个绝对路径
model_dir = snapshot_download(
    'qwen/Qwen2.5-1.5B-Instruct',
    cache_dir='D:/HFCache/hub'  # 这里改成你电脑上实际存在的文件夹路径
)

print(f"✅ 下载彻底完成！模型存放在这里：{model_dir}")