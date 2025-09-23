# 3
def prompt(text, query_triplet, query_relation, query_relation_definition, choices):
    relation_fusion_prompt = f"""给定以下文本以及从中提取出的一个关系三元组：
Text: {text}
Triplet: {query_triplet}

三元组中的关系 '{query_relation}' 被定义为 '{query_relation_definition}'。
在此上下文中，是否有更适合替代它的关系？请仅提供你选择的选项字母作为回答！

Choices:
{choices}

answer: """
    return relation_fusion_prompt