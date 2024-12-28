import json
import os
import logging

# 导入必要的模块
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import HumanMessage

# 配置大模型API
os.environ["OPENAI_API_KEY"] = "None"
os.environ["OPENAI_API_BASE"] = "http://10.58.0.2:8000/v1"

# 初始化大模型完成对象
llm_completion = ChatOpenAI(model_name="Qwen2.5-14B")

# 初始化嵌入模型，使用本地模型地址
embedding = HuggingFaceEmbeddings(model_name="./all-MiniLM-L12-v2")

# 连接到Milvus数据库
db = Milvus(embedding_function=embedding, collection_name="arXiv_Back",
            connection_args={"host": "10.58.0.2", "port": "19530"})

# 设置日志配置
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def optimize_query(question):
    """优化用户输入的查询，以便更好地找到相关学术论文摘要"""
    optimized_prompt = f"Please optimize the following query to better find relevant academic paper abstracts: {question}"
    optimized_query = llm_completion.invoke([HumanMessage(content=optimized_prompt)])
    return optimized_query.content

def fetch_documents(query):
    """查询向量数据库，返回最相关的文档"""
    docs = db.similarity_search(query, k=5)
    return docs

def get_answer_from_doc(best_doc, question):
    """根据最相关文档内容生成回答"""
    answer_prompt = f"Based on the following paper abstract, answer the question: {best_doc.page_content}\n\nQuestion: {question}"
    answer = llm_completion.invoke([HumanMessage(content=answer_prompt)])
    return answer.content

def query_abstract(question):
    """主查询函数，执行优化查询、获取文档和生成答案"""
    attempts = 0
    while attempts < 3:
        try:
            # 优化查询
            optimized_query_content = optimize_query(question)

            # 查询数据库
            docs = fetch_documents(optimized_query_content)

            if docs:
                best_doc = docs[0]
                answer = get_answer_from_doc(best_doc, question)
                return {
                    "answer": answer,
                    "source": f"https://arxiv.org/abs/{best_doc.metadata['access_id']}"
                }
            else:
                attempts += 1
                logger.info(f"Attempt {attempts}/3: No relevant documents found, re-optimizing query.")
        except Exception as e:
            logger.error(f"Error during query: {e}")
            attempts += 1
            logger.info(f"Attempt {attempts}/3: Query error, re-optimizing query.")

    # 如果多次尝试仍未成功
    return {"answer": "No relevant documents found, please try another question.", "source": None}

def answer_questions(questions_file, output_file):
    """处理批量问题，并将答案写入输出文件"""
    try:
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions = json.load(f)
    except FileNotFoundError:
        logger.error(f"Question file {questions_file} not found.")
        return
    except json.JSONDecodeError:
        logger.error(f"Question file {questions_file} has incorrect format and cannot be parsed.")
        return

    for item in questions:
        question = item.get('question', '')
        if not question:
            logger.warning(f"Skipping item without question content: {item}")
            continue

        logger.info(f"Answering question: {question}")
        result = query_abstract(question)
        item['answer'] = result['answer']
        item['source'] = result.get('source', '')

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(questions, f, ensure_ascii=False, indent=4)
        logger.info(f"Answers successfully written to {output_file}.")
    except Exception as e:
        logger.error(f"Error writing to file {output_file}: {e}")

if __name__ == "__main__":
    questions_file = './questions.jsonl'
    answers_file = 'answers.json'
    answer_questions(questions_file, answers_file)