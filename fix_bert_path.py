#!/usr/bin/env python3
"""
修改 GroundingDINO 使用本地 BERT 模型

用法：
    在服务器上运行：python fix_bert_path.py
"""

import os
import shutil
from pathlib import Path

# 配置路径
GROUNDINGDINO_DIR = Path("/root/autodl-tmp/model_zoo/bert")
GET_TOKENLIZER_FILE = GROUNDINGDINO_DIR / "groundingdino/util/get_tokenlizer.py"
LOCAL_BERT_PATH = "/root/navid_ws/pretrained_models/bert-base-uncased"

def backup_file(file_path):
    """备份原始文件"""
    backup_path = file_path.with_suffix(file_path.suffix + '.bak')
    if not backup_path.exists():
        shutil.copy2(file_path, backup_path)
        print(f"✅ 已备份: {backup_path}")
    else:
        print(f"⚠️  备份已存在，跳过: {backup_path}")

def modify_get_tokenlizer():
    """修改 get_tokenlizer.py 使用本地 BERT"""
    
    new_content = f'''# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizerFast
import os

# ========================================
# 🔧 修改：使用本地 BERT 模型
# ========================================
LOCAL_BERT_PATH = "{LOCAL_BERT_PATH}"


def get_pretrained_language_model(text_encoder_type):
    """
    加载预训练语言模型（优先使用本地）
    
    Args:
        text_encoder_type: 模型类型，如 "bert-base-uncased"
    
    Returns:
        预训练的语言模型
    """
    if text_encoder_type == "bert-base-uncased":
        # 检查本地模型是否存在
        if os.path.exists(LOCAL_BERT_PATH):
            print(f"✅ [GroundingDINO] 从本地加载 BERT: {{LOCAL_BERT_PATH}}")
            try:
                return BertModel.from_pretrained(LOCAL_BERT_PATH, local_files_only=True)
            except Exception as e:
                print(f"⚠️  [GroundingDINO] 本地模型加载失败: {{e}}")
                print(f"⚠️  [GroundingDINO] 尝试在线下载...")
                return BertModel.from_pretrained(text_encoder_type)
        else:
            print(f"⚠️  [GroundingDINO] 本地模型不存在: {{LOCAL_BERT_PATH}}")
            print(f"⚠️  [GroundingDINO] 尝试在线下载: {{text_encoder_type}}")
            return BertModel.from_pretrained(text_encoder_type)
    
    if text_encoder_type == "roberta-base":
        return RobertaModel.from_pretrained(text_encoder_type)
    
    raise NotImplementedError(f"Unknown text encoder type: {{text_encoder_type}}")


def get_tokenlizer(text_encoder_type):
    """
    加载 tokenizer（优先使用本地）
    
    Args:
        text_encoder_type: 模型类型，如 "bert-base-uncased"
    
    Returns:
        tokenizer
    """
    if text_encoder_type == "bert-base-uncased":
        # 检查本地模型是否存在
        if os.path.exists(LOCAL_BERT_PATH):
            print(f"✅ [GroundingDINO] 从本地加载 Tokenizer: {{LOCAL_BERT_PATH}}")
            try:
                return BertTokenizer.from_pretrained(LOCAL_BERT_PATH, local_files_only=True)
            except Exception as e:
                print(f"⚠️  [GroundingDINO] 本地 tokenizer 加载失败: {{e}}")
                print(f"⚠️  [GroundingDINO] 尝试在线下载...")
                return BertTokenizer.from_pretrained(text_encoder_type)
        else:
            print(f"⚠️  [GroundingDINO] 本地 tokenizer 不存在: {{LOCAL_BERT_PATH}}")
            print(f"⚠️  [GroundingDINO] 尝试在线下载: {{text_encoder_type}}")
            return BertTokenizer.from_pretrained(text_encoder_type)
    
    if text_encoder_type == "roberta-base":
        return RobertaTokenizerFast.from_pretrained(text_encoder_type)
    
    raise NotImplementedError(f"Unknown text encoder type: {{text_encoder_type}}")
'''
    
    return new_content

def verify_bert_model():
    """验证 BERT 模型文件是否完整"""
    bert_path = Path(LOCAL_BERT_PATH)
    
    required_files = [
        "config.json",
        "pytorch_model.bin",
        "tokenizer_config.json",
        "vocab.txt"
    ]
    
    print(f"\n检查 BERT 模型: {bert_path}")
    
    if not bert_path.exists():
        print(f"❌ 模型目录不存在: {bert_path}")
        return False
    
    missing_files = []
    for file_name in required_files:
        file_path = bert_path / file_name
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"  ✅ {file_name} ({size / 1024 / 1024:.1f} MB)")
        else:
            print(f"  ❌ {file_name} (缺失)")
            missing_files.append(file_name)
    
    if missing_files:
        print(f"\n❌ 缺少文件: {', '.join(missing_files)}")
        return False
    
    print(f"\n✅ BERT 模型文件完整")
    return True

def main():
    print("=" * 60)
    print("修改 GroundingDINO 使用本地 BERT 模型")
    print("=" * 60)
    
    # 1. 检查 GroundingDINO 是否存在
    print(f"\n[1/4] 检查 GroundingDINO 目录...")
    if not GROUNDINGDINO_DIR.exists():
        print(f"❌ 错误: GroundingDINO 目录不存在: {GROUNDINGDINO_DIR}")
        return
    print(f"✅ GroundingDINO 目录存在")
    
    # 2. 验证 BERT 模型
    print(f"\n[2/4] 验证 BERT 模型...")
    if not verify_bert_model():
        print(f"\n⚠️  警告: BERT 模型不完整")
        print(f"请确保已下载并解压模型到: {LOCAL_BERT_PATH}")
        print(f"\n需要的文件:")
        print(f"  - config.json")
        print(f"  - pytorch_model.bin (约 440 MB)")
        print(f"  - tokenizer_config.json")
        print(f"  - vocab.txt")
        return
    
    # 3. 备份原始文件
    print(f"\n[3/4] 备份原始文件...")
    if not GET_TOKENLIZER_FILE.exists():
        print(f"❌ 错误: 文件不存在: {GET_TOKENLIZER_FILE}")
        return
    backup_file(GET_TOKENLIZER_FILE)
    
    # 4. 修改文件
    print(f"\n[4/4] 修改 get_tokenlizer.py...")
    new_content = modify_get_tokenlizer()
    
    with open(GET_TOKENLIZER_FILE, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"✅ 已修改: {GET_TOKENLIZER_FILE}")
    
    print(f"\n" + "=" * 60)
    print("✅ 修改完成！")
    print("=" * 60)
    print(f"\n现在可以运行测试程序:")
    print(f"  cd /root/navid_ws/CA-Nav-code")
    print(f"  python minimal_mapping_test.py --exp-config vlnce_baselines/config/exp1.yaml")
    print(f"\n如果需要恢复原始文件:")
    print(f"  mv {GET_TOKENLIZER_FILE}.bak {GET_TOKENLIZER_FILE}")

if __name__ == "__main__":
    main()
