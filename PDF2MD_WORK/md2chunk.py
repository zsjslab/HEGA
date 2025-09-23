"""
MD2CHUNK: Markdown文档处理工具

主要功能:
1. 解析Markdown文件，提取标题、文本、表格和图片内容
2. 按指定大小(最大512字符)对文本进行分块
3. 使用AI模型对表格和图片内容进行理解，生成名称和描述
4. 构建文档内容的层次结构和三元组关系
5. 输出结构化的CSV和JSON数据，保留文档的层次结构

作者: chenjiehao
最后更新: 2023
"""
# 在InternVL2.5-38B环境下进行部署
    # conda activate InternVL2.5-38B
    # CUDA_VISIBLE_DEVICES=0
    # lmdeploy serve api_server /workspace/Qwen2.5-7B-Instruct/ --model-name qwen-7b
    # lmdeploy serve api_server /workspace/InternVL2.5-8B-Instruct/ --model-name internvl2_5-8b --server-port 23334
    # nohup lmdeploy serve api_server /workspace/Qwen2.5-7B-Instruct/ --model-name qwen-7b >md2chunk_qwen-7b.log 2>&1 &
    # CUDA_VISIBLE_DEVICES=1 nohup lmdeploy serve api_server /workspace/InternVL2-8B/ --model-name internvl2-8b --server-port 23334 >md2chunk_internvl2-8b.log 2>&1

import os
import jsonlines
import pandas as pd
import glob
# from docx import Document
import json
import re
import base64
from typing import List, Dict
from pathlib import Path

# 全局变量声明
input_dir = ""  # 用于存储输入目录路径

# 用于导入OpenAI库，如果不可用则提供模拟实现
try:
    from openai import OpenAI
except ImportError:
    # 提供一个简单的模拟OpenAI实现，以便代码结构保持不变
    class MockOpenAI:
        """如果无法导入OpenAI库，提供一个模拟实现"""
        def __init__(self, **kwargs):
            self.chat = type('obj', (object,), {
                'completions': type('obj', (object,), {
                    'create': self.mock_create
                })
            })
            
        def mock_create(self, **kwargs):
            print("警告: OpenAI API 不可用，使用了模拟返回值")
            return type('obj', (object,), {
                'choices': [type('obj', (object,), {
                    'message': type('obj', (object,), {
                        'content': '模拟响应：表名/图名：示例名称\n描述：这是一个模拟的描述。'
                    })
                })]
            })
    OpenAI = MockOpenAI

# 表格和图片理解功能
class ContentUnderstanding:
    """
    负责理解和处理文档中的表格和图片内容
    
    主要功能:
    1. 从Markdown文件中提取表格和图片
    2. 使用AI模型理解表格内容，生成表名和描述
    3. 使用AI模型理解图片内容，生成图名和描述
    4. 分析文档结构，确定每个表格/图片所属的标题路径
    """
    
    def __init__(self, 
                 table_api_key: str = "demo", 
                 table_base_url: str = "http://127.0.0.1:23333/v1", 
                 table_model: str = "mock-model",
                 image_api_key: str = "demo",
                 image_base_url: str = "http://127.0.0.1:23334/v1", 
                 image_model: str = "mock-model"):
        """
        初始化表格和图片的OpenAI客户端
        
        参数:
            table_api_key: 表格理解API密钥
            table_base_url: 表格理解API基础URL
            table_model: 表格理解使用的模型名称
            image_api_key: 图片理解API密钥
            image_base_url: 图片理解API基础URL
            image_model: 图片理解使用的模型名称
        """
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
        
    # def extract_tables_from_markdown(self, markdown_path: str) -> List[Dict]:
    #     """
    #     从markdown文件中提取HTML表格和其所属路径
        
    #     参数:
    #         markdown_path: Markdown文件的路径
            
    #     返回:
    #         包含表格内容、标题路径和文件名的字典列表
    #     """
    #     with open(markdown_path, 'r', encoding='utf-8') as f:
    #         content = f.read()
            
    #     # 匹配完整的HTML表格
    #     table_pattern = r'<html>.*?<table>.*?</table>.*?</html>'
    #     tables = re.findall(table_pattern, content, re.DOTALL)
        
    #     # 获取文件名（不包含扩展名）
    #     file_name = os.path.splitext(os.path.basename(markdown_path))[0]
        
    #     # 为每个表格创建字典，包含表格内容和位置
    #     table_dicts = []
    #     for table in tables:
    #         # 获取表格在文档中的位置
    #         pos = content.find(table)
    #         heading_path = self.get_current_heading_path(content, pos)
            
    #         table_dicts.append({
    #             'content': table,
    #             'headings': heading_path,
    #             'file_name': file_name
    #         })
            
    #     return table_dicts
    def extract_tables_from_markdown(self, markdown_path: str) -> List[Dict]:
        """
        从markdown文件中提取HTML表格、其所属路径及表格前导文本
        
        参数:
            markdown_path: Markdown文件的路径
            
        返回:
            包含表格内容、标题路径、文件名及前导文本的字典列表
        """
        with open(markdown_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 优化表格匹配逻辑[1](@ref)
        table_pattern = r'(<table>.*?</table>)'  # 简化匹配核心表格标签
        tables_matches = list(re.finditer(table_pattern, content, re.DOTALL))
        
        # 获取文件名（不包含扩展名）
        file_name = os.path.splitext(os.path.basename(markdown_path))[0]
        
        table_dicts = []
        for idx, match in enumerate(tables_matches):
            table_content = match.group(1)
            start_pos = match.start()
            
            # 提取表格前导文本[2,3](@ref)
            prev_start = tables_matches[idx-1].end() if idx>0 else 0
            preceding_text = content[prev_start:start_pos].strip()
            
            # 智能过滤空段落和非文本元素
            if any(char.isalnum() for char in preceding_text):
                clean_text = re.sub(r'[\n]{2,}', '\n', preceding_text)
            else:
                clean_text = "无相关前导文本"
                
            # 获取标题路径
            heading_path = self.get_current_heading_path(content, start_pos)
            
            table_dicts.append({
                'content': table_content,
                'headings': heading_path,
                'file_name': file_name,
                'preceding_text': clean_text  # 新增前导文本字段
            })
            
        return table_dicts
    
    def understand_table(self, table_html: str) -> Dict[str, str]:
        """
        使用表格专用模型理解表格内容
        
        参数:
            table_html: 前一段文本+HTML格式的表格内容
            
        返回:
            包含表名和描述的字典
        """
        try:
            prompt = f"""请分析这个HTML表格并完成两项任务：
            1. 生成一个简短的表名，概括表格的主要内容（不超过20字）
            2. 生成一段描述性文字，详细说明表格的内容、结构和主要数据点

            表格内容如下（其中包括一段文本和表格，文本如果与表格有关就辅助表名和描述生成，文本如果与表格无关就忽略该文本）：
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
        except Exception as e:
            print(f"表格理解出错: {str(e)}")
            table_name = "未命名表格"
            description = "无法解析表格内容"
                    
        return {
            "table_name": table_name,
            "description": description
        }
    
    def extract_image_refs(self, markdown_path: str) -> List[Dict]:
        """
        提取markdown中的图片引用及其所属路径
        
        参数:
            markdown_path: Markdown文件的路径
            
        返回:
            包含图片替代文本、路径、标题路径和文件名的字典列表
        """
        with open(markdown_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        image_pattern = r'!\[(.*?)\]\((.*?)\)'
        matches = re.findall(image_pattern, content)
        
        # 获取文件名（不包含扩展名）
        file_name = os.path.splitext(os.path.basename(markdown_path))[0]
        
        images = []
        markdown_dir = os.path.dirname(markdown_path)
        for alt, path in matches:
            # 解析路径并尝试确定图片的完整路径
            image_path = path
            if not os.path.isabs(path):
                image_path = os.path.join(markdown_dir, path)
                
            # 处理可能的相对路径问题
            if not os.path.exists(image_path) and '../' in path:
                # 尝试处理相对路径
                base_dir = os.path.dirname(markdown_dir)
                relative_path = path.replace('../', '')
                image_path = os.path.join(base_dir, relative_path)
            
            if os.path.exists(image_path):
                # 获取图片所在位置的标题路径
                pos = content.find(f"![{alt}]({path})")
                heading_path = self.get_current_heading_path(content, pos)
                
                images.append({
                    'alt': alt,
                    'path': image_path,
                    'headings': heading_path,
                    'file_name': file_name
                })
            else:
                print(f"警告: 找不到图片文件: {path}")
                
        return images
    
    def image_to_base64(self, image_path: str) -> str:
        """
        将图片转换为base64编码，用于API调用
        
        参数:
            image_path: 图片文件路径
            
        返回:
            图片的base64编码字符串
        """
        try:
            with open(image_path, 'rb') as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_string
        except Exception as e:
            print(f"图片编码出错: {str(e)}")
            return ""
    
    def understand_image(self, image_path: str) -> Dict[str, str]:
        """
        使用图片专用模型理解图片内容
        
        参数:
            image_path: 图片文件路径
            
        返回:
            包含图名和描述的字典
        """
        try:
            base64_image = self.image_to_base64(image_path)
            if not base64_image:
                return {"image_name": "未命名图片", "description": "无法加载图片内容"}
            
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
        except Exception as e:
            print(f"图片理解出错: {str(e)}")
            image_name = "未命名图片"
            description = "无法解析图片内容"
        
        return {
            "image_name": image_name,
            "description": description
        }

    def get_current_heading_path(self, content: str, target_position: int) -> List[str]:
        """
        获取指定位置的完整标题路径
        
        参数:
            content: 文档内容
            target_position: 目标位置在文档中的索引
            
        返回:
            标题路径列表
        """
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

# 定义一个函数来处理段落
def process_paragraph(para, file_name, headings, doc):
    """
    处理文档中的段落文本
    
    参数:
        para: 段落对象
        file_name: 文件名
        headings: 当前标题路径
        doc: 文档对象
    """
       # 替换list连接符
    # headings = '->'.join(headings)
    global module_count
    if para.text.strip():
        module_count += 1
        blocks.append({
            'module_id': module_count,
            'type': 'text',
            'content': para.text.strip(),
            'file_name': file_name,
            'headings': headings
        })

# 定义一个函数来处理标题
def process_headings(para, file_name, headings, doc):
    """
    处理文档中的标题
    
    参数:
        para: 段落对象
        file_name: 文件名
        headings: 当前标题路径
        doc: 文档对象
    """
    # 替换list连接符
    # headings = '->'.join(headings)
    global module_count
    if para.text.strip():
        module_count += 1
        blocks.append({
            'module_id': module_count,
            'type': 'headings',
            'content': para.text.strip(),
            'file_name': file_name,
            'headings': headings
        })        

# 判断段落是否为标题
def is_heading(para, current_headings):
    """
    判断段落是否为标题并更新标题层级
    
    参数:
        para: 段落对象
        current_headings: 当前标题路径
        
    返回:
        (类型, 更新后的标题路径)元组
    """
    if 'Heading' in para.style.name:  # 检查段落是否为标题
        level = int(para.style.name[-1])
        current_headings = current_headings[:level-1] + [para.text]
        return 'Heading',current_headings
    return 'text',current_headings

# 将各级标题构建层级关系三元组保存在output_headings_triples_path
def headings_triples(df, file_name):
    """
    构建标题层级关系的三元组
    
    参数:
        df: 文档DataFrame
        file_name: 文件名
        
    返回:
        标题层级关系的三元组列表
    """
    # 处理标题层级关系
    df_headings = df[df.type=='headings']
    if len(df_headings) == 0:
        return []
        
    # 将列表展开为多列
    df_head = pd.DataFrame([x for x in df_headings['headings']], index=df_headings.index)
    
    # 增加文件名列
    df_head.insert(0, 'file_name', df_headings['file_name'])
    
    # 将 NaN 替换为空字符串 ''
    df_head = df_head.fillna('')
    
    # 初始化用于存储三元组的列表
    triples = []
    
    # 更新非空单元格 aij 为 aij-1 + aij
    for idx, item in enumerate(df_head.columns[1:], start=1):
        prev_item = df_head.columns[idx - 1]  # 获取前一列的列名
        
        # 安全地访问 DataFrame
        for i in df_head.index:
            try:
                current_value = df_head.loc[i, item]
                if isinstance(current_value, str) and current_value.strip() != '':
                    previous_value = df_head.loc[i, prev_item]
                    new_value = previous_value + '-' + current_value
                    df_head.loc[i, item] = new_value
                    
                    # 存储三元组
                    triple = [previous_value.strip(), '包含', new_value]
                    if triple not in triples:
                        triples.append(triple)
            except KeyError:
                continue
                
    return triples

def merge_chunk(df, chunks_dir, file_name, content_understanding=None):
    """
    将文档内容分割成较小的块，并处理表格和图片
    
    核心功能:
    - 将文本按最大512个字符分块
    - 处理表格和图片内容，生成名称和描述
    - 构建内容间的包含和上下位关系
    - 生成CSV文件
    
    参数:
        df: 文档DataFrame
        file_name: 文件名
        content_understanding: 内容理解工具实例(可选)
        
    返回:
        生成的三元组关系列表
    """
    merged_blocks = []
    current_merged_text = ''
    current_tokens = 0
    chunk_triples = []
    last_text_block = None  # 用于跟踪上一个文本块
    current_headings = None
    print("Merging blocks...")
    pre_text = ''
    # 遍历 DataFrame 中的每一行
    for index, row in df.iterrows():
        if row['type'] == 'text':
            para_text = row['content'].replace(' ','')
            tokens = len(para_text)
            if current_tokens + tokens <= 512:
                current_merged_text += para_text + '\n'
                current_tokens += tokens
                current_headings = row['headings']
            else:
                # 保存当前文本块
                current_headings_str = '-'.join(current_headings)
                headings = f"{file_name}-{current_headings_str}".strip()
                if current_merged_text:
                    # 创建当前文本块，添加 file_name 字段
                    current_block = {
                        'type': 'text',
                        'position': len(merged_blocks),
                        'content': current_merged_text,
                        'headings': headings,
                        'file_name': file_name  # 添加文件名字段
                    }
                    # 如果存在上一个文本块且标题相同，添加上下位关系
                    if last_text_block and last_text_block['headings'] == headings:
                        chunk_triples.append([last_text_block['content'], '上下位', current_merged_text])
                    
                    merged_blocks.append(current_block)
                    chunk_triples.append([headings, '包含', current_merged_text])
                    last_text_block = current_block
                    # print('current_block:',current_block)
                current_merged_text = para_text + '\n'
                current_tokens = tokens
                current_headings = row['headings']
                pre_text = para_text
        elif row['type'] in ['table', 'figure']:
            
            if current_tokens > 15:
                # 保存当前文本块
                current_headings_str = '-'.join(current_headings)
                headings = f"{file_name}-{current_headings_str}".strip()
                if current_merged_text:
                    current_block = {
                        'type': 'text',
                        'position': len(merged_blocks),
                        'content': current_merged_text,
                        'headings': headings,
                        # 'headings': headings,
                        'file_name': file_name  # 添加文件名字段
                    }
                    # if last_text_block and last_text_block['headings'] == headings:
                    
                    merged_blocks.append(current_block)
                    # 如果存在上一个文本块且标题相同，添加上下位关系
                    if last_text_block and last_text_block['headings'] == headings:
                        chunk_triples.append([last_text_block['headings'], '包含', current_merged_text])
                        chunk_triples.append([last_text_block['content'], '上下位', current_merged_text])

                    last_text_block = current_block
                    current_merged_text = ''
                    current_tokens = 0
           
            # 直接添加 table 和 figure，添加 file_name 字段
            current_headings_str = '-'.join(row['headings'])
            headings = f"{file_name}-{current_headings_str}".strip()
            
            # 如果是表格并且启用了内容理解，生成表格名称和描述
            name = ""
            description = ""
            path = ""
            
            if content_understanding:
                if row['type'] == 'table':
                    
                    # 为表格生成名称和描述
                    table_result = content_understanding.understand_table(str(pre_text) + '/n' + str(row['content']))
                    name = table_result['table_name']
                    description = table_result['description']
                elif row['type'] == 'figure':
                    print(f"处理图片: {row['content']}")
                    # 尝试从图片引用中提取路径，改进正则表达式以处理更多情况
                    img_match = re.search(r'!\[(.*?)\]\((.*?)\)', row['content'])
                    if img_match:
                        alt_text = img_match.group(1)
                        img_path = img_match.group(2)
                        print(f"  - 图片路径: {img_path}")
                        print(f"  - 图片替代文本: {alt_text}")
                        
                        # 获取当前处理的markdown文件路径
                        # 从row中的file_name值找出对应的原始markdown文件路径
                        # 通过查找在处理目录下的所有满足特定pattern的文件
                        original_md_files = find_markdown_files(input_dir)
                        markdown_file_path = None
                        for md_file in original_md_files:
                            if os.path.splitext(os.path.basename(md_file))[0] == file_name:
                                markdown_file_path = md_file
                                break
                        
                        if markdown_file_path:
                            # 获取markdown文件所在目录
                            md_dir = os.path.dirname(markdown_file_path)
                            
                            # 直接构建图片的完整路径：markdown文件目录/images/图片名
                            img_full_path = os.path.join(md_dir, "images", os.path.basename(img_path))
                            print(f"  - 尝试markdown同级images目录: {img_full_path}")
                            
                            if os.path.exists(img_full_path):
                                print(f"  - 找到图片: {img_full_path}")
                                path = img_path
                                try:
                                    # 为图片生成名称和描述
                                    img_result = content_understanding.understand_image(img_full_path)
                                    name = img_result['image_name']
                                    description = img_result['description']
                                    print(f"  - 生成图名: {name}")
                                    print(f"  - 生成描述: {description[:50]}...")
                                except Exception as e:
                                    print(f"  - 图片理解错误: {str(e)}")
                                    name = "未命名图片"
                                    description = f"图片内容无法解析: {str(e)}"
                            else:
                                # 如果同级images目录找不到，尝试其他可能的路径
                                possible_paths = [
                                    img_path,  # 原始路径
                                    os.path.join(os.getcwd(), img_path),  # 相对于当前工作目录
                                    os.path.join(os.path.dirname(md_dir), "images", os.path.basename(img_path)),  # 父级目录的images
                                    os.path.join(md_dir, img_path)  # 相对于markdown文件所在目录的路径
                                ]
                                
                                # 尝试所有可能的路径
                                for p in possible_paths:
                                    print(f"  - 尝试备选路径: {p}")
                                    if os.path.exists(p):
                                        print(f"  - 找到图片: {p}")
                                        path = img_path
                                        try:
                                            img_result = content_understanding.understand_image(p)
                                            name = img_result['image_name']
                                            description = img_result['description']
                                            print(f"  - 生成图名: {name}")
                                            print(f"  - 生成描述: {description[:50]}...")
                                            break
                                        except Exception as e:
                                            print(f"  - 图片理解错误: {str(e)}")
                        else:
                            print(f"  - 无法找到对应的markdown文件路径")
                            
                        if not name:
                            # 如果无法处理图片，至少使用替代文本作为名称
                            print("  - 无法处理图片，使用替代文本")
                            name = alt_text or "未命名图片"
                            description = f"包含图片: {alt_text}"
                    else:
                        print(f"  - 无法从内容中提取图片路径: {row['content']}")
            tabel_figure_name = file_name + '-' + name
            current_block = {
                'type': row['type'],
                'position': len(merged_blocks),
                'content': tabel_figure_name,    # 添加元素名称（表名/图名）
                'tabel_figure_content': row['content'],
                'headings': headings,
                'file_name': file_name,  # 添加文件名字段
                'description': description, # 添加描述
                'path': path              # 添加路径（仅对图片有效）
                
            }
            merged_blocks.append(current_block)
            # chunk_triples.append([headings, '包含', row['content']])
                 
            if row['type'] == 'table':
                # 标题->表名
                chunk_triples.append([headings, '包含表', tabel_figure_name])
                # 表名->表内容
                chunk_triples.append([tabel_figure_name, '表内容', row['content']])
                # 表名->表描述
                chunk_triples.append([tabel_figure_name, '表描述', description])

            elif row['type'] == 'figure':
                # 标题->图名
                chunk_triples.append([headings, '包含图', tabel_figure_name])
                # 图名->图路径
                chunk_triples.append([tabel_figure_name, '路径', path])
                # 图名、表名->图描述、表描述
                chunk_triples.append([tabel_figure_name, '图描述', description])

            if last_text_block and last_text_block['headings'] == headings:
                chunk_triples.append([last_text_block['content'], '上下位', tabel_figure_name])
            last_text_block = current_block
            # last_text_block = None  # 表格和图片后重置上一个文本块

        elif row['type'] == 'headings':
            if current_tokens > 15:
                # 保存当前文本块
                current_headings_str = '-'.join(current_headings)
                headings = f"{file_name}-{current_headings_str}".strip()
                if current_merged_text:
                    current_block = {
                        'type': 'text',
                        'position': len(merged_blocks),
                        'content': current_merged_text,
                        'headings': headings,
                        # 'headings': headings,
                        'file_name': file_name  # 添加文件名字段
                    }
                    # if last_text_block and last_text_block['headings'] == headings:
                    
                    merged_blocks.append(current_block)
                    # 如果存在上一个文本块且标题相同，添加上下位关系
                    if last_text_block and last_text_block['headings'] == headings:
                        chunk_triples.append([last_text_block['headings'], '包含', current_merged_text])
                        chunk_triples.append([last_text_block['content'], '上下位', current_merged_text])

                    last_text_block = current_block
                # last_text_block = None
            
            current_merged_text = ''
            current_tokens = 0
            current_headings = row['headings']
            pre_text = row['headings']
    
    # 处理最后一个合并段落
    if current_tokens > 15:
        current_headings_str = '-'.join(current_headings)
        headings = f"{file_name}-{current_headings_str}".strip()
        current_block = {
            'type': 'text',
            'position': len(merged_blocks),
            'content': current_merged_text,
            'headings': headings,
            'file_name': file_name  # 添加文件名字段
        }
        if last_text_block and last_text_block['headings'] == headings:
            chunk_triples.append([last_text_block['content'], '上下位', current_merged_text])
        
        merged_blocks.append(current_block)
        chunk_triples.append([headings, '包含', current_merged_text])
        
    # 添加新列到DataFrame
    merged_df = pd.DataFrame(merged_blocks)
    
    # 如果表格/图片的列不存在，添加空列
    if 'element_name' not in merged_df.columns:
        merged_df['element_name'] = ''
    if 'description' not in merged_df.columns:
        merged_df['description'] = ''
    if 'path' not in merged_df.columns:
        merged_df['path'] = ''
        
    # 修改存储路径为 ./output/chunks/ 而非 .output/chunks/
    merged_output_module = os.path.join(chunks_dir, file_name + '_chunk.csv')
    # merged_output_module = chunks_dir + file_name + '_chunk.csv'
    os.makedirs(os.path.dirname(merged_output_module), exist_ok=True)
    merged_df.to_csv(merged_output_module, index=False)
    print(f"Saved final merged DataFrame to: {merged_output_module}")
    return chunk_triples

def process_markdown_file(file_path):
    """
    解析单个Markdown文件，提取其中的标题、文本、表格和图片
    
    参数:
        file_path: Markdown文件路径
        
    返回:
        (包含文档结构的DataFrame, 文件名)元组
    """
    blocks = []
    module_count = 0
    current_headings = []
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
            
        module_count += 1
        
        if line.startswith('#'):
            # Heading processing
            # ...existing heading logic...
            heading_level = len(line) - len(line.lstrip('#'))
            heading_text = line.lstrip('#').strip()
            current_headings = current_headings[:heading_level-1] + [heading_text]
            blocks.append({
                'module_id': module_count,
                'type': 'headings',
                'content': heading_text,
                'file_name': file_name,
                'headings': current_headings.copy()
            })
            i += 1
        elif line.startswith('!['):
            # Figure processing
            blocks.append({
                'module_id': module_count,
                'type': 'figure',
                'content': line,
                'file_name': file_name,
                'headings': current_headings.copy()
            })
            i += 1
        elif '<html>' in line.lower():
            # Table processing - collect complete table
            table_content = []
            while i < len(lines) and '</html>' not in lines[i].lower():
                table_content.append(lines[i])
                i += 1
            if i < len(lines):  # Add the closing tag
                table_content.append(lines[i])
            
            blocks.append({
                'module_id': module_count,
                'type': 'table',
                'content': '\n'.join(table_content),
                'file_name': file_name,
                'headings': current_headings.copy()
            })
            i += 1
        else:
            blocks.append({
                'module_id': module_count,
                'type': 'text',
                'content': line,
                'file_name': file_name,
                'headings': current_headings.copy()
            })
            i += 1
            
    return pd.DataFrame(blocks), file_name

def merge_csv_files(chunks_dir, output_file):
    """
    合并所有生成的CSV文件为一个
    
    参数:
        chunks_dir: CSV文件目录
        output_file: 输出文件路径
        
    返回:
        合并后的DataFrame
    """
    print("\n开始合并CSV文件...")
    
    # 获取所有CSV文件
    csv_files = glob.glob(os.path.join(chunks_dir, "*_chunk.csv"))
    if not csv_files:
        print("没有找到需要合并的CSV文件")
        return
    
    print(f"找到 {len(csv_files)} 个CSV文件待合并")
    
    # 读取并合并所有CSV文件
    dfs = []
    for csv_file in csv_files:
        print(f"正在读取: {os.path.basename(csv_file)}")
        df = pd.read_csv(csv_file)
        dfs.append(df)
    
    # 合并所有数据框
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # 保存合并后的文件
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    combined_df.to_csv(output_file, index=False)
    print(f"合并完成！文件已保存到: {output_file}")
    
    # 返回合并后的数据框，以便进一步处理
    return combined_df

def find_markdown_files(base_dir):
    """
    递归查找目录中的所有Markdown文件
    
    参数:
        base_dir: 基础目录路径
        
    返回:
        Markdown文件路径的列表
    """
    markdown_files = []
    
    # 遍历目录及子目录
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.md'):
                markdown_files.append(os.path.join(root, file))
                
    return markdown_files

if __name__ == '__main__':
    """
    主程序入口
    
    处理流程:
    1. 设置输入输出路径和参数
    2. 初始化内容理解工具(如果启用)
    3. 查找所有Markdown文件
    4. 对每个文件进行处理：解析、分块、生成三元组
    5. 保存三元组到JSON文件
    6. 合并所有CSV文件
    7. 输出处理结果信息
    
    输出文件:
    - CSV文件: 包含文档内容的结构化数据，包括:
      * 文本内容、标题路径、文件名
      * 表格和图片的名称、描述、路径
    - JSON文件: 包含文档结构和内容的三元组关系
    """
    # 修改输入目录为 ./output/sufdata
    input_dir = "./output/test_sufdata"
    chunks_dir = "./output/test_chunk"
    output_dir = "./output"
    
    # 是否启用内容理解功能（表格和图片）
    enable_content_understanding = True
    # 在InternVL2.5-38B环境下进行部署
    # conda activate InternVL2.5-38B
    # CUDA_VISIBLE_DEVICES=0
    # lmdeploy serve api_server /workspace/LLMFiles/LLMs/Qwen/Qwen2.5-7B-Instruct/ --model-name qwen-7b
    # lmdeploy serve api_server /workspace/LLMFiles/LLMs/InternVL/InternVL2.5-8B-Instruct/ --model-name internvl2_5-8b --server-port 23334
    # nohup lmdeploy serve api_server /workspace/LLMFiles/LLMs/Qwen/Qwen2.5-7B-Instruct/ --model-name qwen-7b >md2chunk_qwen-7b.log 2>&1 &
    # nohup CUDA_VISIBLE_DEVICES=1 lmdeploy serve api_server /workspace/LLMFiles/LLMs/InternVL/InternVL2-8B/ --model-name internvl2-8b --server-port 23334 >md2chunk_internvl2-8b.log 2>&1
    
    # 内容理解的API配置
    content_api_config = {
        'table_api_key': "demo",
        'table_base_url': "http://127.0.0.1:23333/v1",
        'table_model': "qwen-7b",
        'image_api_key': "demo",
        'image_base_url': "http://127.0.0.1:23334/v1",
        'image_model': "internvl2-8b"
    }
    
    # 初始化内容理解工具
    content_understanding = None
    if enable_content_understanding:
        try:
            content_understanding = ContentUnderstanding(**content_api_config)
            print("内容理解功能已启用")
        except Exception as e:
            print(f"内容理解功能初始化失败: {str(e)}")
            print("将继续处理，但不包含表格和图片的理解")
    
    # 确保输出目录存在
    os.makedirs(chunks_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # 三元组输出路径
    triples_output = os.path.join(output_dir, "test_md_triples.json")
    # 合并CSV输出路径
    merged_csv_output = os.path.join(output_dir, "test_combined_chunks.csv")
    
    all_triples = []
    
    # 使用递归方式查找所有Markdown文件
    md_files = find_markdown_files(input_dir)
    print(f"找到 {len(md_files)} 个Markdown文件")
    
    # 处理所有Markdown文件
    for file_path in md_files:
        print(f"处理文件: {file_path}")
        df, file_name = process_markdown_file(file_path)
        title_triples = headings_triples(df, file_name)
        
        # 传入内容理解工具到chunk处理函数chunks_dir
        print('chunks_dir:',chunks_dir)
        chunk_triples = merge_chunk(df, chunks_dir,file_name, content_understanding)
        
        file_triples = title_triples + chunk_triples
        all_triples.extend(file_triples)
    
    # 保存三元组
    with open(triples_output, 'w', encoding='utf-8') as f:
        json.dump({"last_triples": [all_triples]}, f, ensure_ascii=False, indent=2)
    print(f"三元组已保存到: {triples_output}")
    
    # 合并所有生成的CSV文件
    merged_df = merge_csv_files(chunks_dir, merged_csv_output)
    
    print("\n处理完成！")
    print(f"- 三元组文件: {triples_output}")
    print(f"- 合并后的CSV文件: {merged_csv_output}")