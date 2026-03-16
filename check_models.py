"""
检查训练的模型

这个脚本会列出所有可用的模型
"""

import os
import glob
from datetime import datetime

def check_models():
    print("="*60)
    print("🔍 检查模型文件")
    print("="*60)
    print()
    
    # 检查预训练模型
    print("📁 预训练模型目录 (pretrained_models/):")
    print("-"*60)
    pretrained_dir = 'pretrained_models'
    if os.path.exists(pretrained_dir):
        models = glob.glob(os.path.join(pretrained_dir, '*.pt'))
        if models:
            for model in models:
                size = os.path.getsize(model) / (1024 * 1024)  # MB
                mtime = os.path.getmtime(model)
                mtime_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
                print(f"  ✓ {os.path.basename(model)}")
                print(f"    大小: {size:.2f} MB")
                print(f"    修改时间: {mtime_str}")
                print()
        else:
            print("  ❌ 没有找到模型文件")
    else:
        print("  ❌ 目录不存在")
    
    print()
    
    # 检查训练的模型
    print("📁 训练模型目录 (saved_models/):")
    print("-"*60)
    saved_dir = 'saved_models'
    if os.path.exists(saved_dir):
        models = glob.glob(os.path.join(saved_dir, '*.pt'))
        if models:
            # 按修改时间排序
            models.sort(key=os.path.getmtime, reverse=True)
            for i, model in enumerate(models, 1):
                size = os.path.getsize(model) / (1024 * 1024)  # MB
                mtime = os.path.getmtime(model)
                mtime_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
                marker = "⭐ 最新" if i == 1 else "  "
                print(f"{marker} {os.path.basename(model)}")
                print(f"    大小: {size:.2f} MB")
                print(f"    修改时间: {mtime_str}")
                print()
        else:
            print("  ❌ 没有找到模型文件")
            print("  💡 提示: 训练时需要使用 --save_model 参数")
    else:
        print("  ❌ 目录不存在")
    
    print()
    print("="*60)
    print("📊 总结")
    print("="*60)
    
    # 检查 unified_test.py 使用的模型
    print("\n🔍 unified_test.py 默认使用的模型:")
    default_model = 'pretrained_models/default_cut_2.pt'
    if os.path.exists(default_model):
        size = os.path.getsize(default_model) / (1024 * 1024)
        mtime = os.path.getmtime(default_model)
        mtime_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        print(f"  ✓ {default_model}")
        print(f"    大小: {size:.2f} MB")
        print(f"    修改时间: {mtime_str}")
        
        # 检查是否是最近训练的
        saved_models = glob.glob('saved_models/*.pt')
        if saved_models:
            latest_saved = max(saved_models, key=os.path.getmtime)
            latest_mtime = os.path.getmtime(latest_saved)
            
            if mtime < latest_mtime:
                print()
                print("  ⚠️  警告: 这个模型比 saved_models/ 中的模型旧")
                print(f"  💡 你可能想使用: {os.path.basename(latest_saved)}")
                print()
                print("  要使用你训练的模型，运行:")
                print(f"  copy \"{latest_saved}\" pretrained_models\\default_cut_2.pt")
    else:
        print(f"  ❌ {default_model} 不存在")
    
    print()

if __name__ == "__main__":
    check_models()
