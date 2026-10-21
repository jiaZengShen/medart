
# 面向医学影像的多模态大模型推理能力评估及修复方法研究

## 各个模块简介
本项目包括3个子模块

| 文件夹名称 | 说明                   |
|------------|----------------------|
| `mml_test` | 第3章 VANK归因 + 15个模型评测 |
| `biobert_fillter` | 对应第4章 语言捷径检测         |
| `medart` | 第5章 MedART修复方法       |
|`requirements.txt'| Python依赖包及版本             |




## mml_test 
“医学影像推理失败的形式化分析”
如需重现实验，需要把api_key中的*****替换为实际的api_key 。
[recognition_export.json](mml_test/input/recognition_export.json)是VANK数据集，
[not_valid_ids.json](mml_test/input/not_valid_ids.json)是无效的数据集数据的id。
数据集的路径是Windows路径，在运行的时候，代码会自动转化成linux路径。
如果需要运行，需要把[MedPix-2.0](https://github.com/CHILab1/MedPix-2.0.git)、[MedXpertQA](https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA)、[HC18](https://hc18.grand-challenge.org/?spm=5176.28103460.0.0.96a07551AeOyQ0)放到replace_image_prefix参数指定的目录中。


### 主要文件夹介绍：

| 文件夹名称          | 说明                                     |
|----------------|----------------------------------------|
| `input`        | VANK 数据集                               |
| `outputs`      | 主实验的结果                                 |
| `outputs_lite` | 使用 doubao-seed-2.0-lite 模型重复 10 次的测试结果 |
| `main.py`      | 入口py文件                                 |
|`utils/constant.py`| 测试使用的promote模版|
| `运行命令.txt`      | 运行命令的示例                                |





## biobert_fillter 
是“语言捷径的检测通用框架”

运行前需要把[biobert-base-cased-v1.2](https://huggingface.co/dmis-lab/biobert-base-cased-v1.2/)放入../../../model/biobert-base-cased-v1.2目录中。

### 主要文件夹介绍：
| 文件夹名称              | 说明               |
|--------------------|------------------|
| `data`             | 数据集              |
| `mml_tset_outputs` | 使用mml_test做的实验结果 |
| `train.py`         | 使用数据集训练的入口文件     |
| `predict.py`       | 测试的入口文件          |



## medart 
“第5章	基于失效归因的医学影像推理修复方法”

MedArt智能体的实验及其结果的目录

### 主要文件夹介绍：
| 文件夹名称                 | 说明                |
|-----------------------|-------------------|
| `input`               | VANK数据集           |
| `outputs_**`          | 实验结果的存放目录         |
| `src/services`        | 测量推理的工具的service目录 |
| `src/single_agent_v1` | 智能体主流程的实现         |
| `src/tools`           | 智能体需要的各个工具        |
| `src/tools/hc_measurement.py`          | 测量核验工具            |
| `src/tools/unified_rag_tool.py`          | 医学知识片段检索工具        |
| `src/tools/rule_rag_tool.py`          | 既往经验规则检索工具       |


### 实验结果：
| 结果类型 | 实验结果文件夹 |
|------------|------|
| 主实验| outputs_2026-09-07_14-10-02 |
| 停用影像证据检索 | outputs_2026-09-18_21-43-46 |
| 停用既往经验规则检索 | outputs_2026-09-19_23-11-14 |
| 停用确定性测量核验 | outputs_2026-09-20_21-34-47 |








感谢 [SciVer](https://github.com/QDRhhhh/SciVer.git) 提供基础代码 
 