# 二
# 在InternVL2.5-38B环境下进行部署
    # conda activate InternVL2.5-38B
    # lmdeploy serve api_server /workspace/LLMFiles/LLMs/Qwen/Qwen2.5-7B-Instruct/ --model-name qwen-7b
    # lmdeploy serve api_server /workspace/LLMFiles/LLMs/InternVL2.5-8B-Instruct/ --model-name internvl2_5-8b --server-port 23334
    # nohup lmdeploy serve api_server /workspace/LLMFiles/LLMs/Qwen/Qwen2.5-7B-Instruct/ --model-name qwen-7b >md2chunk_qwen-7b.log 2>&1 &
    # nohup lmdeploy serve api_server /workspace/LLMFiles/LLMs/InternVL2-8B/ --model-name internvl2-8b --server-port 23334 >md2chunk_internvl2-8b.log 2>&1
import os
import re
import base64
from typing import List, Dict
import json
import csv
from pathlib import Path
from openai import OpenAI

class ContentUnderstanding:
    def __init__(self, 
                 table_api_key: str, 
                 table_base_url: str, 
                 table_model: str,
                 image_api_key: str,
                 image_base_url: str, 
                 image_model: str):
        """初始化表格和图片的OpenAI客户端"""
        # 表格理解客户端
        self.table_client = OpenAI(
            api_key=table_api_key,
            base_url=table_base_url
        )
        self.table_model = table_model

        # 图片理解客户端
        self.image_client = OpenAI(
            api_key=image_api_key,
            base_url=image_base_url
        )
        self.image_model = image_model
        
    def extract_tables_from_markdown(self, markdown_path: str) -> List[str]:
        """从markdown文件中提取HTML表格"""
        with open(markdown_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 匹配完整的HTML表格
        table_pattern = r'<html>.*?<table>.*?</table>.*?</html>'
        tables = re.findall(table_pattern, content, re.DOTALL)
        return tables
    
    def understand_table(self, table_html: str) -> Dict[str, str]:
        """使用表格专用模型理解表格内容"""
        prompt = f"""请分析这个HTML表格并完成两项任务：
        1. 生成一个简短的表名，概括表格的主要内容（不超过20字）
        2. 生成一段描述性文字，详细说明表格的内容、结构和主要数据点

        表格内容如下：
        {table_html}

        请按以下格式返回：
        表名：[生成的表名]
        描述：[生成的描述]
        """
        
        response = self.table_client.chat.completions.create(
            model=self.table_model,
            messages=[
                {"role": "system", "content": "你是一个专业的表格理解助手，请帮我理解HTML表格并生成恰当的表名和内容描述。"},
                {"role": "user", "content": prompt}
            ]
        )
        content = response.choices[0].message.content
        
        # 解析返回的内容
        table_name = ""
        description = ""
        
        if "表名：" in content:
            parts = content.split("表名：", 1)
            if len(parts) > 1:
                name_parts = parts[1].split("描述：", 1)
                table_name = name_parts[0].strip()
                if len(name_parts) > 1:
                    description = name_parts[1].strip()
                    
        return {
            "table_name": table_name,
            "description": description
        }
    
    def extract_image_refs(self, markdown_path: str) -> List[Dict]:
        """提取markdown中的图片引用"""
        with open(markdown_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        image_pattern = r'!\[(.*?)\]\((.*?)\)'
        matches = re.findall(image_pattern, content)
        
        images = []
        markdown_dir = os.path.dirname(markdown_path)
        for alt, path in matches:
            image_path = os.path.join(markdown_dir, path)
            if os.path.exists(image_path):
                images.append({
                    'alt': alt,
                    'path': image_path
                })
        return images
    
    def image_to_base64(self, image_path: str) -> str:
        """将图片转换为base64编码"""
        with open(image_path, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
    
    def understand_image(self, image_path: str) -> Dict[str, str]:
        """使用图片专用模型理解图片内容"""
        base64_image = self.image_to_base64(image_path)
        
        response = self.image_client.chat.completions.create(
            model=self.image_model,
            messages=[
                {
                    "role": "system",
                    "content": "你是一个专业的图像理解助手，请帮我分析图片并生成合适的图名和详细描述。"
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": """请分析这张图片并完成两项任务：
                            1. 生成一个简短的图名，概括图片的主要内容（不超过20字）
                            2. 生成一段描述性文字，详细说明图片的内容和关键信息
                            
                            请按以下格式返回：
                            图名：[生成的图名]
                            描述：[生成的描述]"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        )
        
        content = response.choices[0].message.content
        
        # 解析返回的内容
        image_name = ""
        description = ""
        
        if "图名：" in content:
            parts = content.split("图名：", 1)
            if len(parts) > 1:
                name_parts = parts[1].split("描述：", 1)
                image_name = name_parts[0].strip()
                if len(name_parts) > 1:
                    description = name_parts[1].strip()
        
        return {
            "image_name": image_name,
            "description": description
        }

    def get_current_heading_path(self, content: str, target_position: int) -> str:
        """获取指定位置的完整标题路径"""
        current_headings = []
        lines = content.split('\n')
        current_position = 0
        
        for line in lines:
            if current_position > target_position:
                break
                
            line = line.strip()
            if line.startswith('#'):
                # 计算标题级别
                level = len(line) - len(line.lstrip('#'))
                heading_text = line.lstrip('#').strip()
                # 更新当前标题层级
                current_headings = current_headings[:level-1] + [heading_text]
            
            current_position += len(line) + 1  # +1 for newline
            
        return current_headings
    
    def find_block_position(self, content: str, target_content: str) -> int:
        """找到目标内容在文档中的位置"""
        return content.find(target_content)
    
    def find_image_position(self, content: str, image_path: str) -> int:
        """找到图片在文档中的实际位置"""
        # 将路径转换为与文档中可能的格式匹配
        image_name = os.path.basename(image_path)
        # 查找所有可能的图片引用格式
        patterns = [
            f'![.*?]\\(.*?{image_name}\\)',  # 标准Markdown格式
            f'<img.*?src=["|\'].*?{image_name}["|\'].*?>',  # HTML格式
            image_name  # 直接查找文件名
        ]
        
        # 尝试所有模式查找位置
        positions = []
        for pattern in patterns:
            matches = list(re.finditer(pattern, content))
            positions.extend(m.start() for m in matches)
        
        # 如果找到多个位置，返回第一个
        return min(positions) if positions else -1
    
    def get_heading_path(self, markdown_path: str, target_content: str, is_image: bool = False) -> str:
        """获取指定内容的完整标题路径"""
        with open(markdown_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 根据内容类型选择不同的定位方法    
        if is_image:
            pos = self.find_image_position(content, target_content)
        else:
            pos = self.find_block_position(content, target_content)
            
        if pos == -1:
            return ""
            
        headings = self.get_current_heading_path(content, pos)
        file_name = os.path.splitext(os.path.basename(markdown_path))[0]
        
        # 组合完整路径：文件名-标题1-标题2-...
        return f"{file_name}-{'-'.join(headings)}" if headings else file_name

# 将process_markdown_understanding函数移到类外部
def process_markdown_understanding(
    markdown_dir: str, 
    output_dir: str, 
    table_api_key: str, 
    table_base_url: str,
    table_model: str,
    image_api_key: str,
    image_base_url: str,
    image_model: str):
    """处理目录下所有markdown文件的表格和图片理解，输出CSV格式"""
    understanding = ContentUnderstanding(
        table_api_key=table_api_key,
        table_base_url=table_base_url,
        table_model=table_model,
        image_api_key=image_api_key,
        image_base_url=image_base_url,
        image_model=image_model
    )
    
    # 准备CSV输出
    output_path = os.path.join(output_dir, 'understanding_results.csv')
    os.makedirs(output_dir, exist_ok=True)
    
    # 更新CSV头部，添加所属文本块列
    headers = ['所属文本块', '图名/表名', '类型', '摘要', '路径']
    
    with open(output_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        
        # 处理所有markdown文件
        for root, _, files in os.walk(markdown_dir):
            for file in files:
                if not file.endswith('.md'):
                    continue
                    
                markdown_path = os.path.join(root, file)
                print(f"Processing {markdown_path}...")
                
                # 处理表格
                tables = understanding.extract_tables_from_markdown(markdown_path)
                for table in tables:
                    result = understanding.understand_table(table)
                    heading_path = understanding.get_heading_path(markdown_path, table)
                    # 写入表格数据：所属文本块、表名、类型、描述、空路径
                    writer.writerow([
                        heading_path,
                        result['table_name'],
                        '表格',
                        result['description'],
                        ''  # 表格没有路径
                    ])
                
                # 处理图片
                images = understanding.extract_image_refs(markdown_path)
                for image in images:
                    result = understanding.understand_image(image['path'])
                    # 使用图片路径进行定位，并标记为图片类型
                    heading_path = understanding.get_heading_path(
                        markdown_path, 
                        image['path'],
                        is_image=True
                    )
                    # 写入图片数据：所属文本块、图名、类型、描述、路径
                    rel_path = os.path.relpath(image['path'], markdown_dir)
                    writer.writerow([
                        heading_path,
                        result['image_name'],
                        '图片',
                        result['description'],
                        rel_path
                    ])
    
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    # 表格理解配置
    TABLE_API_KEY = "YOU"
    TABLE_BASE_URL = "http://127.0.0.1:23333/v1"
    TABLE_MODEL = "qwen-7b"
    
    # 图片理解配置
    IMAGE_API_KEY = "your-image-api-key"
    IMAGE_BASE_URL = "http://127.0.0.1:23334/v1"
    IMAGE_MODEL = "internvl2_5-8b"
    
    MARKDOWN_DIR = "./output/sufdata"
    OUTPUT_DIR = "./output/understanding"
    
    process_markdown_understanding(
        markdown_dir=MARKDOWN_DIR,
        output_dir=OUTPUT_DIR,
        table_api_key=TABLE_API_KEY,
        table_base_url=TABLE_BASE_URL,
        table_model=TABLE_MODEL,
        image_api_key=IMAGE_API_KEY,
        image_base_url=IMAGE_BASE_URL,
        image_model=IMAGE_MODEL
    )
