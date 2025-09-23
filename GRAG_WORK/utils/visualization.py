from pyvis.network import Network
import json
# def KgVisualize(triples_list, idx, type, dataset):
def KgVisualize(triples_list,file_path):
    net = Network(notebook=True)
    entity_dict = {}
    edges = []
    index = 0
    for triples in triples_list:
        for triple in triples:
            if len(triple) != 3:
                print(triple)
            else:
                h, r, t = triple
                if h not in list(entity_dict.keys()):
                    entity_dict[h] = index
                    index += 1
                if t not in list(entity_dict.keys()):
                    entity_dict[t] = index
                    index += 1
                edges.append([entity_dict[h], r, entity_dict[t]])

    for e, i in entity_dict.items():
        net.add_node(i, label=e)

    for edge in edges:
        net.add_edge(edge[0], edge[2], label=edge[1], color="blue", width=2)

    # if type == "ori":
    #     net.show(f'results/{dataset}/extraction/q{idx}/ori.html')
    # else:
    #     net.show(f'results/{dataset}/extraction/q{idx}/refine.html')
    net.show(f"{file_path.split('.')[0]}.html") 
if __name__ == '__main__':

    def extract_triples_from_line(line):
        """从单个JSON数据字符串中提取'triples'"""
        try:
            # 将JSON字符串解析为Python对象
            data = json.loads(line)
            # 返回'triples'键下的值
            return data.get('last_triples', [])
        except json.JSONDecodeError as e:
            print(f"JSON 解析错误: {e}")
            return []
        except Exception as e:
            print(f"处理 JSON 时发生错误: {e}")
            return []
    
    def batch_process_json_file(file_path):
        """批量处理JSON文件中的每一行，返回所有'triples'"""
        all_triples = []
    
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                triples = extract_triples_from_line(line.strip())
                if triples:
                    all_triples.append(triples[0])  # 假设每个JSON只有一个'triples'列表
        # with open(file_path2, 'r', encoding='utf-8') as file:
        #     for line in file:
        #         triples = extract_triples_from_line(line.strip())
        #         if triples:
        #             all_triples.append(triples[0])  # 假设每个JSON只有一个'triples'列表
    
        return all_triples
    
    # 示例：处理文件中的JSON数据
    file_path = '/workspace/term_basev2_1203/KG_RAG_WORK/output/triples_data/test_triples_0326/test_triples_0326_v2.json'
    # file_path2 = '/workspace/term_basev2_1203/output/test2_Triples.json'
    all_triples = batch_process_json_file(file_path)
    KgVisualize(all_triples,file_path)
    # print("所有JSON数据的'triples':", all_triples)
    