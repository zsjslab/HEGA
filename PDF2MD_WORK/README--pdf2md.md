# PDF -> Markdown -> 表格/图片的抽取

## 任务目标

本任务的目标是建立从PDF文件转换到Markdown并将其中的表格和图片内容进行抽取，标注表格/图片内容的全流程工作。

## 容器：

id:1e8406e8dc88

镜像：42a0e9b621e2

容器名：cjh

端口：9824-9826->9824-9826

```json
{
    "bucket_info": {
        "bucket-name-1": [
            "ak",
            "sk",
            "endpoint"
        ],
        "bucket-name-2": [
            "ak",
            "sk",
            "endpoint"
        ]
    },
    "models-dir": "/root/.cache/modelscope/hub/opendatalab/PDF-Extract-Kit-1___0/models",
    "layoutreader-model-dir": "/root/.cache/modelscope/hub/ppaanngggg/layoutreader",
    "device-mode": "cuda",
    "layout-config": {
        "model": "doclayout_yolo"
    },
    "title-config": {
        "enable": true
    },
    "formula-config": {
        "mfd_model": "yolo_v8_mfd",
        "mfr_model": "unimernet_small",
        "enable": true
    },
    "table-config": {
        "model": "rapid_table",	// 默认使用"rapid_table",可以切换为"tablemaster"和"struct_eqtable",效果：rapid_table<struct_eqtable<tablemaster.
        "sub_model": "slanet_plus",	// 当model为"rapid_table"时，可以自选sub_model，可选项为"slanet_plus"和"unitable"
        "enable": true,	// 表格识别功能默认是开启的，如果需要关闭请修改此处的值为"false"
        "max_time": 400
 	},
    "llm-aided-config": {
        "formula_aided": {
            "api_key": "sk-ortmefpwfkwrjcxelwirsqbzgplxsjgmaekqzcpcwjnomybu",
            "base_url": "https://api.siliconflow.cn/v1",
            "model": "deepseek-ai/DeepSeek-V3",
            "enable": true
        },
        "text_aided": {
            "api_key": "sk-ortmefpwfkwrjcxelwirsqbzgplxsjgmaekqzcpcwjnomybu",
            "base_url": "https://api.siliconflow.cn/v1",
            "model": "deepseek-ai/DeepSeek-V3",
            "enable": true
        },
        "title_aided": {
            "api_key": "sk-ortmefpwfkwrjcxelwirsqbzgplxsjgmaekqzcpcwjnomybu",
            "base_url": "https://api.siliconflow.cn/v1",
            "model": "deepseek-ai/DeepSeek-V3",
            "enable": true
    	}
    },
    "config_version": "1.1.1"
}

```



## PDF转换MD

PDF转换使用了开源方案MinerU来做，首先使用深度学习来对每一页进行OCR、表格识别和版面分析，然后使用大模型做后续处理，配置的文件在` magic-pdf.json`中，可配置本地的大模型进行分析。

> 标题识别的准确度依赖于模型的能力，处于**保密性**考虑可以使用本地模型

所有需要的内容都存在 @extract_table_image.py  中，按照其中的指令操作即可，可以生成markdown文件和对应的图片文件夹，存储在` ./output/sufdata`中。

```bash
python extract_table_image.py
```



## Markdown -> chunk
抽取的文件存储在`./output/sufdata`中。
对于生成的markdown文件，进行下一步操作，任务是：

+ 识别markdown中的表格和文件
+ 定位所属的chunk
+ 使用大模型对表格和文件分别生成摘要或者描述文本
+ 存入csv，格式为

| 所属文本块 | 表格/图片名称 | 类型（表格或图片） | 表格/图片摘要 | 存储路径（只针对图片） |
| ---------- | ------------- | ------------------ | ------------- | ---------------------- |

在这一步骤，直接执行 @table_image_understanding.py 文件即可。最后输出为`./output/understanding/*.csv`中。

```bash
python md2chunk.py
```



## 结果

最后能输出为csv为标注结果。


```bash
./chenjiehao/ALL_WORK/
|-- README.md # 说明文档
|-- extract_table_image.py
|-- input # 数据输入
|   |-- xx.pdf
|   |-- xx.pdf
|-- md2chunk.py # 抽取主函数，生成csv和json数据
|-- output
|   |-- chunks # 单文档生成的 chunks
|   |   |-- xx_chunk.csv
|   |   |-- xx_chunk.csv
|   |-- combined_chunks.csv # 多文档融合后的chunk数据
|   |-- md_triples.json  # 三元组json数据
|   |-- predata # 初次处理:使用minerU后的数据
|   |   |-- xx文件 #一个文档对应的文件夹，文件夹下只有auto
|   |   |   -- auto
|   |   |       |-- images#若有表格，也会转成图片，存储在该位置
|   |   |       |   -- xx.jpg
|   |   |       |-- xx.md #pdf->md
|   |   |       |-- 《xx.json #没用
|   |   |       |-- 《xx.pdf #没用
|   |   |       |-- xx.json #没用
|   |   |       |-- xx.pdf #没用

|   |-- sufdata # 二次处理：执行extract_table_image.py后抽取出的文件和图片
|   |   |-- 《关于优化全业务工单管理办法的通知》 中电信黔客服〔2021〕4号
|   |   |   `-- auto
|   |   |       |-- images
|   |   |       |   |-- xx.jpg
|   |   |       |   |-- xx.jpg
|   |   |       |--xx.md
|   |   |-- 《关于规范和统一10000号工单类型的通知》中国电信客服业〔2023〕3号
|   |   |   `-- auto
|   |   |       |-- images
|   |   |       |--xx.md

36 directories, 175 files
```