"""
创建可靠性装箱系统发布包
"""
import os
import shutil
import zipfile
from datetime import datetime

def create_release_package():
    """创建发布包，包含所有必要文件"""
    
    # 创建发布目录
    release_name = f"reliable-robot-packing-{datetime.now().strftime('%Y%m%d')}"
    release_dir = f"./{release_name}"
    
    if os.path.exists(release_dir):
        shutil.rmtree(release_dir)
    os.makedirs(release_dir)
    
    # 需要包含的文件和目录
    include_items = [
        # 核心实现
        "acktr/uncertainty.py",
        "acktr/motion_primitive.py",
        "acktr/visual_feedback.py",
        "acktr/performance_optimizer.py",
        "acktr/utils.py",
        "acktr/arguments.py",
        "acktr/model.py",
        "acktr/envs.py",
        "acktr/storage.py",
        "acktr/distributions.py",
        "acktr/reorder.py",
        "acktr/model_loader.py",
        "acktr/box_creators.py",
        "acktr/__init__.py",
        
        # 算法实现
        "acktr/algo/",
        
        # 环境
        "envs/bpp0/bin3D_reliable.py",
        "envs/bpp0/bin3D.py",
        "envs/bpp0/space.py",
        "envs/bpp0/binCreator.py",
        "envs/bpp0/cutCreator.py",
        "envs/bpp0/mdCreator.py",
        "envs/bpp0/__init__.py",
        
        # 主程序
        "main.py",
        "unified_test.py",
        "evaluation.py",
        
        # 测试文件
        "acktr/test_uncertainty.py",
        "acktr/test_motion_primitive.py",
        "acktr/test_visual_feedback.py",
        "acktr/test_candidate_map.py",
        "acktr/test_reliable_packing.py",
        "acktr/test_reward_function.py",
        "acktr/test_performance_metrics.py",
        "acktr/test_performance_optimization.py",
        "test_multi_camera_integration.py",
        
        # 示例脚本
        "example_train_with_uncertainty.py",
        "example_test_with_reliability.py",
        "example_full_pipeline.py",
        
        # 配置文件
        "camera_config_example.json",
        "requirements.txt",
        
        # 文档
        "README.md",
        "RELIABILITY_FEATURES_GUIDE.md",
        "RELIABILITY_QUICK_REFERENCE.md",
        "MULTI_CAMERA_USAGE_GUIDE.md",
        "UNIFIED_TEST_USAGE_GUIDE.md",
        "ENVIRONMENT_SELECTION_GUIDE.md",
        "EXAMPLE_SCRIPTS_README.md",
        "DOCUMENTATION_INDEX.md",
        
        # 规范文档
        ".kiro/specs/reliable-robot-packing/",
        
        # 数据集
        "dataset/",
        
        # 基础库
        "baselines/",
        
        # MCTS（用于对比）
        "MCTS/",
    ]
    
    # 复制文件
    print("正在创建发布包...")
    for item in include_items:
        src = item
        dst = os.path.join(release_dir, item)
        
        if not os.path.exists(src):
            print(f"警告: {src} 不存在，跳过")
            continue
            
        if os.path.isdir(src):
            # 复制目录
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git'))
            print(f"✓ 复制目录: {src}")
        else:
            # 复制文件
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            print(f"✓ 复制文件: {src}")
    
    # 创建README
    readme_content = """# Reliable Robot Packing System

基于论文 "Towards reliable robot packing system based on deep reinforcement learning" 的完整实现

## 快速开始

1. 安装依赖:
```bash
pip install -r requirements.txt
```

2. 训练带不确定性的模型:
```bash
python example_train_with_uncertainty.py
```

3. 测试可靠性功能:
```bash
python example_test_with_reliability.py
```

## 文档

- [可靠性功能指南](RELIABILITY_FEATURES_GUIDE.md)
- [快速参考](RELIABILITY_QUICK_REFERENCE.md)
- [文档索引](DOCUMENTATION_INDEX.md)

## 测试

运行所有测试:
```bash
pytest acktr/test_*.py -v
pytest test_*.py -v
```

## 论文引用

如果使用此代码，请引用原始论文和可靠性改进论文。

详细信息请参阅 README.md
"""
    
    with open(os.path.join(release_dir, "QUICK_START.md"), "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    # 创建ZIP压缩包
    zip_filename = f"{release_name}.zip"
    print(f"\n正在创建压缩包: {zip_filename}")
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(release_dir):
            # 排除__pycache__
            dirs[:] = [d for d in dirs if d != '__pycache__']
            
            for file in files:
                if file.endswith('.pyc'):
                    continue
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, release_dir)
                zipf.write(file_path, arcname)
                
    print(f"\n✓ 发布包创建完成: {zip_filename}")
    print(f"✓ 临时目录: {release_dir}")
    print(f"\n您可以:")
    print(f"1. 解压 {zip_filename} 到新位置")
    print(f"2. 删除临时目录: rmdir /S /Q {release_dir}")
    
    return zip_filename, release_dir

if __name__ == "__main__":
    zip_file, temp_dir = create_release_package()
    
    print("\n" + "="*60)
    print("发布包已创建！")
    print("="*60)
    print(f"\n压缩包: {zip_file}")
    print(f"大小: {os.path.getsize(zip_file) / 1024 / 1024:.2f} MB")
    print("\n下一步:")
    print("1. 将压缩包上传到您的仓库或分享给他人")
    print("2. 或解压到新位置继续开发")
