# 3
def prompt(text, query_triplet, query_relation, query_relation_definition, choices):
    sc_prompt = f"""您作为运营商知识图谱质检专家，请执行以下任务：
给定以下文本以及从中抽取的关系三元组:
Text: {text}
Triplet: {query_triplet}

三元组中的关系'{query_relation}'被定义为'{query_relation_definition}'。
验证步骤：
1. 行业规范检查：对照YD/T 3829-2021《云通信服务技术要求》和《电信业务分类目录》
2. 语义匹配度评估：分析关系谓词与文本上下文的契合程度（0-100分）
3. 替换必要性判断：当且仅当存在更符合行业术语标准的关系时进行替换

在此上下文中，是否有其他合适的关系可以替换它？请仅从以下提供的关系中选择回答！
候选关系集（必须选择且仅选其一）:
{choices}

answer: """
    return sc_prompt