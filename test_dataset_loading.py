"""
测试数据集加载
用于诊断为什么只加载了1个episode
"""
from habitat import make_dataset
from vlnce_baselines.config.default import get_config

# 加载配置
config = get_config("vlnce_baselines/config/exp1.yaml")

print("="*60)
print("配置信息:")
print("="*60)
print(f"数据集类型: {config.TASK_CONFIG.DATASET.TYPE}")
print(f"Split: {config.TASK_CONFIG.DATASET.SPLIT}")
print(f"数据路径: {config.TASK_CONFIG.DATASET.DATA_PATH}")
print(f"场景目录: {config.TASK_CONFIG.DATASET.SCENES_DIR}")

# 检查 CONTENT_SCENES
if hasattr(config.TASK_CONFIG.DATASET, 'CONTENT_SCENES'):
    print(f"CONTENT_SCENES: {config.TASK_CONFIG.DATASET.CONTENT_SCENES}")
else:
    print("CONTENT_SCENES: 未设置")

print("\n" + "="*60)
print("加载数据集...")
print("="*60)

dataset = make_dataset(
    id_dataset=config.TASK_CONFIG.DATASET.TYPE,
    config=config.TASK_CONFIG.DATASET
)

print(f"\n✓ 数据集加载完成: {len(dataset.episodes)} episodes")

if len(dataset.episodes) > 0:
    print("\n前10个episodes信息:")
    for i, ep in enumerate(dataset.episodes[:10]):
        scene = ep.scene_id.split('/')[-1].split('.')[0]
        print(f"  {i}: Episode ID={ep.episode_id}, Scene={scene}")
    
    # 统计场景分布
    scene_count = {}
    for ep in dataset.episodes:
        scene = ep.scene_id.split('/')[-1].split('.')[0]
        scene_count[scene] = scene_count.get(scene, 0) + 1
    
    print(f"\n场景分布: {len(scene_count)} 个不同场景")
    if len(scene_count) <= 20:
        for scene, count in sorted(scene_count.items()):
            print(f"  {scene}: {count} episodes")
