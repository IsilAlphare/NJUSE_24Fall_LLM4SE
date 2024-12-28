# RAG实验(必做)

## 题目要求

### 基础要求

- 构建一个针对arXiv的知识问答系统
- 要求如下
    - 给定一个入口，用户可以输入提问
    - 不要求要求构建GUI界面
    - 用户通过对话进行交互
    - 系统寻找与问题相关的论文abstract
    - 使用用户的请求对向量数据库进行请求
    - 寻找与问题最为相关的abstract
    - 系统根据问题和论文abstract回答用户问题，并给出解答问题的信息来源
- 示例

### 进阶要求

- 提示优化
    - 用户给出的问题或陈述不一定能够匹配向量数据库的查询
    - 使用大模型对用户的输入进行润色，提高找到对应文档的概率
    - 思路提示(解决思路不唯一，提示仅作为可能的思路示例)
        - 观察不同输入后向量数据库找到对应文档的概率
        - 总结适用于查询的语句
        - 构建提示(prompt)实现对用户输入的润色
    - 查询迭代
        - 单次的查询可能无法寻找到用户所期望的答案
        - 需要通过多轮的搜索和尝试才能获得较为准确的答案
        - 思路提示
        - 如何将用户的需求拆解，变成可以拆解的逻辑步骤
        - 如何判断已经获得准确的答案并停止迭代
        - 如何再思路偏移后进行修正

## 资源简介

向量数据库可按需自行部署，大模型可选择自有api，下列内容为能完成任务的所需资源。

arxiv数据集可以从这里获取 https://www.kaggle.com/datasets/Cornell-University/arxiv

### 大模型(Qwen2.5-14B)

```python
# Qwen2.5-14B模型已接入LangChain Openai API,调用示例如下
from langchain.llms import OpenAI, OpenAIChat
import os
os.environ["OPENAI_API_KEY"] = "None"
os.environ["OPENAI_API_BASE"] = "http://10.58.0.2:8000/v1"
llm_completion = OpenAI(model_name="Qwen2.5-14B")
llm_chat = OpenAIChat(model_name="Qwen2.5-14B")

```

openai包使用特定版本，避免与langchain不兼容 pip install openai==0.28

### 嵌入模型(sentence-transformers/all-MiniLM-L12-v2)

* 嵌入模型使用huggingface中的all-MiniLM-L12-v2模型

``` python
from langchain.embeddings import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
```

当前由于huggingface被墙，无梯子可以使用镜像，详见 https://hf-mirror.com/

### 向量数据库

arXiv数据存在Milvus中

``` python
from langchain.vectorstores import Milvus
db = Milvus(embedding_function=embedding, collection_name="arXiv",connection_args={"host": "10.58.0.2", "port": "19530"})
```

由于向量数据库与SDK存在强绑定关系，安装milvus包时请检查版本： pip install pymilvus==2.2.6

#### 数据项解释

- vector： 论文abstract的向量化表示
- access_id：论文的唯一id
    - https://arxiv.org/abs/{access_id} 论文的详情页
    - https://arxiv.org/pdf/{access_id} 论文的pdf地址
- authors：论文的作者
- title：论文的题目
- comments：论文的评论，一般为作者的补充信息
- journal_ref：论文的发布信息
- doi：电子发行doi
- text：论文的abstract (为了兼容langchain必须命名为text)
- categories：论文的分类

### LangChain

- langchain官方文档 https://python.langchain.com/
- langchain官方课程 https://learn.deeplearning.ai/langchain

## 提交内容

1. 代码实现
2. 预置题目
    1. 预置由json文件构成，包含10个问题(question项目)
    2. 使用算法回答其中问题，答案存在answer项内
    3. 该文件存储为answer.json，单独提交

## 注意事项

1. Python版本要在3.8和3.11之间。（我使用了3.9.2rc1）因为要求ymilvus=2.2.6，这个库依赖1.49.1到1.53.0的grpcio，1.53.0的grpcio不支持python 3.12。

2. 因为网络原因，运行时无法自动下载 sentence-transformers，需要手动下载：[sentence-transformers](https://hf-mirror.com/sentence-transformers/all-MiniLM-L12-v2/tree/main) 整个目录。对于根目录下 `pytorch_model.bin` 和 `model.safetensors` 两个文件，需要手动下载。

   注：

   **sentence-transformers**是一个基于Python的库，它专门用于句子、文本和图像的嵌入。这个库可以计算100多种语言的文本嵌入，并且这些嵌入可以轻松地用于语义文本相似性、语义搜索和同义词挖掘等任务。sentence-transformers基于PyTorch和Transformers库构建，提供了大量针对各种自然语言处理任务的预训练模型。此外，用户还可以根据自己的需求对模型进行微调。

   a. 在项目根目录下git clone sentence-transformers目录

   ```
   git clone https://hf-mirror.com/sentence-transformers/all-MiniLM-L12-v
   ```

   b 再手动下载将网站中`pytorch_model.bin` 和 `model.safetensors`，将两个文件放入all-MiniLM-L12-v文件夹中。

3. 安装正确版本的依赖：

用管理员模式进行安装：
```bash
pip install openai==0.28
pip install pymilvus==2.2.6
```

其他依赖直接安装最新版即可。(pip install + 需要的包 如langchain/langchain_openai)

4. 如遇已安装pymilvus却无法import的报错，有可能是没有google模块

   ```bash
   pip install --upgrade google-api-python-client
   ```

5. 记得在校园网环境下运行，否则milvus会超时

6. 终端运行结果示例：

   ```bash
   c:\Users\xiaoyu\Desktop\RAG-arVix\main.py:22: LangChainDeprecationWarning: The class `Milvus` was deprecated in LangChain 0.2.0 and will be removed in 1.0. An updated version of the class exists in the :class:`~langchain-milvus package and should be used instead. To use it run `pip install -U :class:`~langchain-milvus` and import as `from :class:`~langchain_milvus import MilvusVectorStore``.
     db = Milvus(embedding_function=embedding, collection_name="arXiv_Back",
   2024-12-27 22:32:33,121 - INFO - Answering question: 什么是大语言模型？
   2024-12-27 22:32:41,158 - INFO - HTTP Request: POST http://10.58.0.2:8000/v1/chat/completions "HTTP/1.1 200 OK"
   2024-12-27 22:32:44,626 - INFO - HTTP Request: POST http://10.58.0.2:8000/v1/chat/completions "HTTP/1.1 200 OK"
   2024-12-27 22:32:44,631 - INFO - Answering question: 形式化软件工程是什么？
   2024-12-27 22:32:52,830 - INFO - HTTP Request: POST http://10.58.0.2:8000/v1/chat/completions "HTTP/1.1 200 OK"
   2024-12-27 22:32:56,517 - INFO - HTTP Request: POST http://10.58.0.2:8000/v1/chat/completions "HTTP/1.1 200 OK"
   2024-12-27 22:32:56,519 - INFO - Answering question: 大语言模型的缩放定理是什么？
   2024-12-27 22:33:04,939 - INFO - HTTP Request: POST http://10.58.0.2:8000/v1/chat/completions "HTTP/1.1 200 OK"
   2024-12-27 22:33:11,058 - INFO - HTTP Request: POST http://10.58.0.2:8000/v1/chat/completions "HTTP/1.1 200 OK"
   2024-12-27 22:33:11,060 - INFO - Answering question: 代码评审的目标是什么？
   2024-12-27 22:33:15,563 - INFO - HTTP Request: POST http://10.58.0.2:8000/v1/chat/completions "HTTP/1.1 200 OK"
   2024-12-27 22:33:18,328 - INFO - HTTP Request: POST http://10.58.0.2:8000/v1/chat/completions "HTTP/1.1 200 OK"
   2024-12-27 22:33:18,330 - INFO - Answering question: 重复的数据会对In-content Learning产生什么影响？
   2024-12-27 22:33:26,725 - INFO - HTTP Request: POST http://10.58.0.2:8000/v1/chat/completions "HTTP/1.1 200 OK"
   2024-12-27 22:33:31,333 - INFO - HTTP Request: POST http://10.58.0.2:8000/v1/chat/completions "HTTP/1.1 200 OK"
   2024-12-27 22:33:31,337 - INFO - Answering question: 软件工程领域如何适应不同领域？
   2024-12-27 22:33:38,194 - INFO - HTTP Request: POST http://10.58.0.2:8000/v1/chat/completions "HTTP/1.1 200 OK"
   2024-12-27 22:33:46,489 - INFO - HTTP Request: POST http://10.58.0.2:8000/v1/chat/completions "HTTP/1.1 200 OK"
   2024-12-27 22:33:46,492 - INFO - Answering question: 区块链如何保证安全？
   2024-12-27 22:33:55,499 - INFO - HTTP Request: POST http://10.58.0.2:8000/v1/chat/completions "HTTP/1.1 200 OK"
   2024-12-27 22:34:01,950 - INFO - HTTP Request: POST http://10.58.0.2:8000/v1/chat/completions "HTTP/1.1 200 OK"
   2024-12-27 22:34:01,953 - INFO - Answering question: 指令微调的目标是什么？
   2024-12-27 22:34:09,939 - INFO - HTTP Request: POST http://10.58.0.2:8000/v1/chat/completions "HTTP/1.1 200 OK"2024-12-27 22:34:14,604 - INFO - HTTP Request: POST http://10.58.0.2:8000/v1/chat/completions "HTTP/1.1 200 OK"2024-12-27 22:34:14,607 - INFO - Answering question: 离子阱计算机的原理是什么？
   2024-12-27 22:34:18,538 - INFO - HTTP Request: POST http://10.58.0.2:8000/v1/chat/completions "HTTP/1.1 200 OK"2024-12-27 22:34:21,816 - INFO - HTTP Request: POST http://10.58.0.2:8000/v1/chat/completions "HTTP/1.1 200 OK"2024-12-27 22:34:21,818 - INFO - Answering question: 人造原子是什么？
   2024-12-27 22:34:29,293 - INFO - HTTP Request: POST http://10.58.0.2:8000/v1/chat/completions "HTTP/1.1 200 OK"2024-12-27 22:34:33,883 - INFO - HTTP Request: POST http://10.58.0.2:8000/v1/chat/completions "HTTP/1.1 200 OK"2024-12-27 22:34:33,887 - INFO - Answers successfully written to answers.json.
   ```

   