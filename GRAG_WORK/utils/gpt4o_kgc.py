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

class gpt4o_kgc:

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
                return "000"

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
        MAX_RETRIES = 3  # 设置最大重试次数
        RETRY_DELAY = 2  # 每次重试间隔（秒）
        for attempt in range(MAX_RETRIES):
            extracted_triples = self.generating(prompt)
            if extracted_triples != "000":
                break  # 成功获取非 1 的结果，跳出循环
            else:
                print(f"Attempt {attempt + 1} returned '000', retrying...")  # 可选：输出日志
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
            return res
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
        return extracted_triples_