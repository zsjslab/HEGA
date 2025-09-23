# import regex
# import json
# import string
# import unicodedata
# from typing import List
# import numpy as np
# from collections import Counter
# import pickle as pkl
# import itertools
# import argparse
# import ast
import os
import torch
import argparse
import json
import numpy as np
from rouge import Rouge
from bert_score import BERTScorer
from datetime import datetime
import jieba
# 方法1：环境变量控制（推荐）
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # 仅显示物理卡2为逻辑cuda:0
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_json_lines(file_path):
    """读取多行JSON文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

# def load_data(file_path):
#     """增强型数据加载"""
#     data = []
#     with open(file_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             line = line.strip()
#             if not line: continue
#             try:
#                 # 修复不规则JSON格式
#                 line = line.replace('\n', ' ').replace('\\n', ' ')
#                 data.append(json.loads(line))
#             except Exception as e:
#                 print(f"数据解析异常：{str(e)}")
#     return data

def evaluate_single_pair(pred, answer, rouge, bert_scorer):
    """单条数据评估"""
    # ROUGE计算
    rouge_pred = ' '.join(jieba.cut(pred))
    rouge_answer = ' '.join(jieba.cut(answer))
    rouge_scores = rouge.get_scores(rouge_pred, rouge_answer)[0]
    
    # BERTScore计算
    P, R, F1 = bert_scorer.score([pred], [answer])
    
    return {
        "rouge-1": {
            "f": rouge_scores['rouge-1']['f'],
            "p": rouge_scores['rouge-1']['p'],
            "r": rouge_scores['rouge-1']['r']
        },
        "rouge-2": {
            "f": rouge_scores['rouge-2']['f'],
            "p": rouge_scores['rouge-2']['p'],
            "r": rouge_scores['rouge-2']['r']
        },
        "rouge-l": {
            "f": rouge_scores['rouge-l']['f'],
            "p": rouge_scores['rouge-l']['p'],
            "r": rouge_scores['rouge-l']['r']
        },
        "bert_score": {
            "f1": F1.item(),
            "precision": P.item(),
            "recall": R.item()
        }
    }

def batch_evaluate(data):
    """批量评估与结果聚合"""
    # 初始化评估工具
    rouge = Rouge()
    bert_scorer = BERTScorer(
        model_type="/workspace/bert-base-chinese/bert-base-chinese",
        lang="zh",
        num_layers=8,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    
    # 逐条评估
    results = []
    for item in data:
        pred = item['pred'].replace('\n', ' ')
        answer = item['answer'].replace('\n', ' ')
        
        eval_result = evaluate_single_pair(pred, answer, rouge, bert_scorer)
        
        results.append({
            "idx": item['idx'],
            "question": item['question'],
            "true_answer": answer,
            "predicted_answer": pred,
            "scores": eval_result
        })
    
    # 计算整体指标
    aggregate = {
        "rouge-1": {"f": [], "p": [], "r": []},
        "rouge-2": {"f": [], "p": [], "r": []},
        "rouge-l": {"f": [], "p": [], "r": []},
        "bert_score": { "f1": [], "precision": [], "recall": []}
    }
    
    for res in results:
        for metric in ['rouge-1', 'rouge-2', 'rouge-l']:
            for key in ['f', 'p', 'r']:
                aggregate[metric][key].append(res['scores'][metric][key])
        for key in ['precision', 'recall', 'f1']:
            aggregate['bert_score'][key].append(res['scores']['bert_score'][key])
    
    # 生成统计结果
    final_output = {
        "metadata": {
            "evaluation_date": datetime.now().isoformat(),
            "model_used": "bert-base-chinese",
            "total_samples": len(data)
        },
        "overall_scores": {
            "rouge-1": {k: np.mean(v) for k, v in aggregate['rouge-1'].items()},
            "rouge-2": {k: np.mean(v) for k, v in aggregate['rouge-2'].items()},
            "rouge-l": {k: np.mean(v) for k, v in aggregate['rouge-l'].items()},
            "bert_score": {k: np.mean(v) for k, v in aggregate['bert_score'].items()}
        },
        "detailed_results": results
        
    }
    
    return final_output

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='/workspace/term_basev2_1203/KG_RAG_WORK/src/GraphRAG/results/test_0327.json')
    parser.add_argument('--output_path', type=str, default='/workspace/term_basev2_1203/KG_RAG_WORK/src/GraphRAG/rouge/')
    args = parser.parse_args()
    
    # 提取文件名（不带扩展名）
    base_name = os.path.splitext(os.path.basename(args.dataset))[0]  # 提取 'test_0327'
    # 构造 output文件路径
    json_path = f'{args.output_path}{base_name}_rouge_0327.json'


    # 加载数据
    data = load_json_lines(args.dataset)
    
    evaluation_results = batch_evaluate(data)
    print(evaluation_results['overall_scores'])
    # 保存结果
    with open(json_path, "w", encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
        