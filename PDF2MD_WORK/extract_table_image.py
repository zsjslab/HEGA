
# 一
# 1mineru
# 1.1首先现在mineru到环境中(首次安装需要)
"""
conda create -n mineru python=3.10
conda activate mineru
pip install -U magic-pdf[full] --extra-index-url https://wheels.myhloli.com
"""
# 1.2安装模型的权重(首次安装需要)
"""
pip install huggingface_hub
wget https://github.com/opendatalab/MinerU/raw/master/scripts/download_models_hf.py -O download_models_hf.py
python download_models_hf.py

"""
# 1.3下载paddle-gpu使用OCR功能（首次安装需要）  ocr:paddlepaddle-gpu==3.0.0b1
"""
python -m pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
"""
# 1.4 root下，使用CUDA加速(首次安装需要)
"""
在 ~/magic-pdf.json 中 把device-mode调整为cuda
"""
# 1.5 运行，，开启ocr \不开启ocr\自动
# -m, --method [ocr|txt|auto]
"""
magic-pdf -p {输入文件目录或文件名} -o ./output
magic-pdf -p input -o ./output/predata -m auto
"""
# 输出文件会在 ./output 目录下
# 只提取里面的markdown文件和images文件夹

# 2、下面代码为提取predata里面的markdown文件和markdown中索引的图片，，结果保存在sufdata文件夹下
import os
import shutil
import re

def extract_image_paths(markdown_content):
    """从markdown内容中提取图片路径"""
    # 匹配 markdown 的图片语法 ![alt](path)
    image_pattern = r'!\[.*?\]\((.*?)\)'
    return re.findall(image_pattern, markdown_content)

def process_markdown_file(src_dir, dest_dir, file_path):
    """处理单个markdown文件及其图片"""
    # 创建目标目录
    os.makedirs(dest_dir, exist_ok=True)
    
    # 读取markdown文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取图片路径
    image_paths = extract_image_paths(content)
    
    # 复制markdown文件
    markdown_filename = os.path.basename(file_path)
    dest_markdown_path = os.path.join(dest_dir, markdown_filename)
    shutil.copy2(file_path, dest_markdown_path)
    
    # 创建images目录
    images_dir = os.path.join(dest_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    # 复制相关图片
    for img_path in image_paths:
        # 构建源图片的完整路径
        src_img_path = os.path.join(os.path.dirname(file_path), img_path)
        if os.path.exists(src_img_path):
            # 保持与原始markdown中相同的相对路径结构
            dest_img_path = os.path.join(dest_dir, img_path)
            os.makedirs(os.path.dirname(dest_img_path), exist_ok=True)
            shutil.copy2(src_img_path, dest_img_path)
            print(f"Copied image: {img_path}")

def extract_markdown_and_images():
    """主函数：处理所有markdown文件及其图片"""
    # 指定源目录和目标目录
    src_base_dir = "./output/predata"
    dest_base_dir = "./output/sufdata"
    
    # 确保目标目录存在
    os.makedirs(dest_base_dir, exist_ok=True)
    
    # 遍历源目录
    for root, dirs, files in os.walk(src_base_dir):
        for file in files:
            if file.endswith('.md'):
                src_file_path = os.path.join(root, file)
                
                # 计算相对路径，用于在目标目录中创建相同的目录结构
                rel_path = os.path.relpath(root, src_base_dir)
                dest_dir = os.path.join(dest_base_dir, rel_path)
                
                print(f"Processing: {src_file_path}")
                process_markdown_file(src_base_dir, dest_dir, src_file_path)

if __name__ == "__main__":
    extract_markdown_and_images()
