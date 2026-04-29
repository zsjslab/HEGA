# HEGA

🌐 **[Project Homepage](#)** Official code implementation for the paper *"HEGA: Hybrid Embedding-Generative Approach for Schema-Free Unsupervised Knowledge Graph Construction"*.

The figure below illustrates the overall architecture of our proposed UKG:

![Model Architecture](框架图.png)

## 🏃‍♂️ Quick Start

### 1. PDF to Markdown
```python
# The configuration file is `magic-pdf.json`, and the output path is `./output/sufdata`
# 1. Activate the virtual environment
conda activate mineru

# 2. Run mineru to generate markdown
# Note: Processing 13 PDFs (approx. 15MB) took mineru about 1 hour and 10 minutes.
magic-pdf -p ./0506data/input/ -o ./0506data/pre_output/ -m auto

# 3. Extract the markdown and its embedded images
# Modify src_base_dir and dest_base_dir in extract_table_image.py
# src_base_dir = "./output/predata"
# dest_base_dir = "./output/sufdata"
python extract_table_image.py
```

###  2. Markdown to Chunk, Title Triples, and Chart Understanding
```python
# 1. Activate the virtual environment
conda activate InternVL2.5-38B

# 2. Start the LLM API services
nohup lmdeploy serve api_server /workspace/LLMFiles/LLMs/Qwen/Qwen2.5-7B-Instruct/ --model-name qwen-7b >md2chunk_qwen-7b.log 2>&1 &
nohup env CUDA_VISIBLE_DEVICES=3 lmdeploy serve api_server /workspace/LLMFiles/LLMs/InternVL/InternVL2-8B/ --model-name internvl2-8b --server-port 23334 > md2chunk_internvl2-8b.log 2>&1 &

# 3. Run the script for chunking, chart naming, and description generation, then save to CSV
python md2chunk.py
```


## 📁 Directory Structure
```python
/workspace/UKG_Tele/KG_WORK
    --dataset               # Datasets, including domain_data, rebel_data, and webnlg_data
    --prompt                # Prompts, containing the 5-step prompts for different datasets
    --result                # Output results
    --utils                 # Utility codes and scripts
    --all_triples_zyq.py    # Main entry script
```

## ⚙️ Parameter Explanations in `all_triples_zyq`
```python
parser.add_argument('--input_path', type=str, default='/workspace/term_basev2_1203/KG_WORK/src/GraphRAG/dataset/domain_data/domain_data_0620.csv') # Path to the input text chunk file
parser.add_argument('--llm_model', type=str, default='chatgpt-4o-latest') # Name of the LLM when calling the API
parser.add_argument('--llm_model_jc', type=str, default='gpt') # Abbreviation of the LLM, used to distinguish different models and concatenate output filenames
parser.add_argument('--encoder', type=str, default='/workspace/bge-large-zh-v1.5') # Path/Name of the embedding/vectorization model
parser.add_argument('--model_path', type=str, default='/workspace/Qwen2.5-7B-Instruct') # Local path when calling a local model
parser.add_argument('--index', type=int, default=-1)
parser.add_argument('--use_vllm', type=bool, default=False)   # Whether to use vLLM acceleration for the model
parser.add_argument('--use_api_models', type=bool, default=True) # Whether to use the API
parser.add_argument('--output_version', type=str, default='0623') # Output version, used to track time versions and concatenate output filenames
parser.add_argument('--label', type=str, default='last') # Label, used to distinguish ablation experiments and concatenate output filenames
```

```bash
python all_triples_zyq.py
```
