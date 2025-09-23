from transformers import AutoModelForCausalLM, AutoTokenizer
# from mistral_inference.transformer import Transformer
# from mistral_inference.generate import generate

# from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
# from mistral_common.protocol.instruct.messages import UserMessage
# from mistral_common.protocol.instruct.request import ChatCompletionRequest
from tqdm import tqdm
import json
# from prompts.WikiMQA import extract_prompt, schema_prompt, sc_prompt
from prompts.TERM_BASE_QA import extract_prompt, schema_prompt, sc_prompt
import ast
import re
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from vllm import LLM, SamplingParams
from openai import OpenAI

class KGC:

    def __init__(self, model, tokenizer, encoder, use_vllm, use_api_models):
        self.model = model
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.use_vllm = use_vllm
        self.use_api_models = use_api_models

    def generating(self, prompt):
        messages = [
            {"role": "system", "content": "你是一个电信行业规范知识图谱三元组抽取专家"},
            {"role": "user", "content": prompt}
        ]
        # text = self.tokenizer.apply_chat_template(
        #     messages,
        #     tokenize=False,
        #     add_generation_prompt=True
        # )
        if self.use_vllm:
            text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
            )
            sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=2048)
            output = self.model.generate([text], sampling_params)
            return output[0].outputs[0].text
        elif self.use_api_models:
            client = OpenAI(
                # #将这里换成你在aihubmix api keys拿到的密钥
                api_key="",
                # 这里将官方的接口访问地址，替换成aihubmix的入口地址
                base_url=""
            )
            
            
            try:
                chat_completion = client.chat.completions.create(
                messages=messages,
                # model="deepseek-ai/DeepSeek-V3",
                    # model="DeepSeek-V3",
                    model="chatgpt-4o-latest",
            )
                first_choice = chat_completion.choices[0]
                message_content = first_choice.message.content
                # print('message_content',message_content)
                return message_content
            except Exception as e:
                print(f"API调用异常: {str(e)}")
                return ""

        else:
            text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to('cuda')

            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=2048,
                pad_token_id=self.tokenizer.eos_token_id
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            return response
        
    def extractBase(self, topic, text):
        prompt = extract_prompt.prompt(topic, text)
        extracted_triples = self.generating(prompt)
        try:
            res = ast.literal_eval(extracted_triples)
            # print('1res:',res)
            # print('origin_res:',res)
            return res
        except:
            res = []
            triples_ = extracted_triples[1:-1].split("], ")
            for tri in triples_:
                try:
                    res.append(ast.literal_eval(tri + ']'))
                except:
                    tri = tri[1:].split(', ')
                    tri_ = []
                    for n in tri:
                        n = n[1:-1].replace("'", "")
                        tri_.append(n)

                    res.append(tri_)
            
            # print('origin_res:',res)
            return res

    def schemaDefine(self, text, extracted_triples, extracted_relations):
        prompt = schema_prompt.prompt(text, extracted_triples, extracted_relations)
        relations_define = self.generating(prompt)
        return relations_define

    def schemaCanonicalization(self, text, extracted_triples, extracted_relation, relation_define, choices):
        prompt = sc_prompt.prompt(text, extracted_triples, extracted_relation, relation_define, choices)
        choice = self.generating(prompt)
        return choice

    def extract(self, text2topic_idx, texts, relations):
        once_triples = []
        triples = []
        alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for i, text in enumerate(texts):
            # print('i, text:',i, text)
            topic = text2topic_idx[text][0]
            extracted_triples = self.extractBase(topic, text)
            # print('extracted_triples:',extracted_triples)
            extracted_triples_ = []
            for triple in extracted_triples:
                if len(triple) == 3:
                    extracted_triples_.append(triple) 
            extracted_relations = []
            relations_define = {}
            schema_triples = []
            for tri in extracted_triples_:
                if tri[1] in extracted_relations:
                    continue
                if tri[1] in relations.keys():
                    relations_define[tri[1]] = relations[tri[1]]
                    continue
                extracted_relations.append(tri[1])
                schema_triples.append(tri)
            relations_define_res = self.schemaDefine(text, schema_triples, extracted_relations)
            # print("relations_define_res: ", relations_define_res)
            relations_define_res = relations_define_res.replace('：', ': ').replace('Answer:\n', '').split('\n')
            
            # print("relations_define: ", relations_define_res)

            for rel_define in relations_define_res:
                if rel_define and ':' in rel_define:
                    try:
                        rel = ast.literal_eval(rel_define.split(': ')[0])
                    except:
                        rel = rel_define.split(': ')[0]
                    # print("rel_define: ", rel_define)
                    define = rel_define.split(': ')[1]
                    relations_define[rel] = define

            if not relations:
                for rel, define in relations_define.items():
                    relations[rel] = define
            else:
                # print("num of relations: ", len(relations.keys()))

                for idx in range(len(extracted_triples_)):
                    triple = extracted_triples_[idx]
                    emb = self.encoder.encode(triple[1])
                    choices = ""
                    for i, (rel, define) in enumerate(relations.items()):
                        rel_emb = self.encoder.encode(rel_define)
                        similarity = cos_sim(emb, rel_emb)
                        sim_dic = {}
                        # print(triple[1], rel, similarity)
                        if similarity > 0.8:
                            sim_dic[(rel, define)] = similarity
                    sorted_sim = sorted(sim_dic.items(), key=lambda x: x[1], reverse=True)
                    num = 10
                    if len(sorted_sim) < num:
                        num = len(sorted_sim)
                    for c_num, ((rel, define), _) in enumerate(sorted_sim[ :num]):
                        choices += alphabet[c_num] + ". '" + rel + "': " + define + '\n'
                        
                    if choices:
                        choices += alphabet[c_num + 1] + ". None of the above.\n"
                        # print("relations define: ", relations_define, '\n\n')
                        print("choices: ", choices)
                        try:
                            choice = self.schemaCanonicalization(text, triple, triple[1], relations_define[triple[1]], choices).split('.')[0].replace('answer: ','')
                        except:
                            print("relation: ", triple[1], "  relations defined: ", relations_define.keys())
                        
                        print("alphabet[num]: ", alphabet[num])
                        print("choice: ", choice)

                        if alphabet[num] == choice:
                            relations[triple[1]] = relations_define[triple[1]]
                        else:
                            
                            pattern = r"^" + choice + r"\. '(.*?)'"
                            print("choice: ", choice)
                            result = re.search(pattern, choices, re.MULTILINE).group(1)
                            # print(triple[1], '\n', choices, choice, '\n')
                            extracted_triples_[idx][1] = result

            # res.append({"text": text, "prediction": extracted_triples, "reference": reference})
            once_triples.append(extracted_triples)
            triples.append(extracted_triples_)
        # print('triples:',triples)
        return triples, relations