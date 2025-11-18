import fitz  # PyMuPDF库，用于处理PDF文件
import os  # 操作系统相关功能
import numpy as np  # NumPy库，用于数值计算
import json  # JSON数据处理
from openai import OpenAI  # OpenAI API客户端


def extract_text_from_pdf(pdf_path):
    """
    从PDF文件中提取文本并打印前`num_chars`个字符。

    参数:
    pdf_path (str): PDF文件的路径。

    返回:
    str: 从PDF中提取的文本。
    """
    # 打开PDF文件
    mypdf = fitz.open(pdf_path)
    all_text = ""  # 初始化一个空字符串用于存储提取的文本

    # 遍历PDF中的每一页
    for page_num in range(mypdf.page_count):
        page = mypdf[page_num]  # 获取页面
        text = page.get_text("text")  # 从页面提取文本
        all_text += text  # 将提取的文本追加到all_text字符串中

    return all_text  # 返回提取的文本

def chunk_text(text, n, overlap):
    """
    将给定的文本分割为长度为 n 的段，并带有指定的重叠字符数。

    参数:
    text (str): 需要分割的文本。
    n (int): 每个片段的字符数量。
    overlap (int): 段与段之间的重叠字符数量。

    返回:
    List[str]: 一个包含文本片段的列表。
    """
    chunks = []  # 初始化一个空列表用于存储片段
    
    # 使用 (n - overlap) 的步长遍历文本
    for i in range(0, len(text), n - overlap):
        # 将从索引 i 到 i + n 的文本片段添加到 chunks 列表中
        chunks.append(text[i:i + n])

    return chunks  # 返回包含文本片段的列表

from dotenv import load_dotenv
import os

load_dotenv()  # 加载.env文件
api_key = os.getenv("OPENAI_API_KEY")  # 读取密钥
print(api_key)
# 初始化 OpenAI 客户端，设置基础 URL 和 API 密钥  
client = OpenAI(  
    base_url="https://api.openai.com/v1/",  
    api_key=os.getenv("OPENAI_API_KEY")  # 从环境变量中获取 API 密钥  
)

# 定义PDF文件的路径
pdf_path = "/Users/eddiex/rag-demo/data/AI_Information.pdf"

# 从PDF文件中提取文本
extracted_text = extract_text_from_pdf(pdf_path)

# 将提取的文本分割为每段1000个字符并带有200个字符重叠的片段
text_chunks = chunk_text(extracted_text, 1000, 200)

# 打印生成的文本片段数量
# print("Number of text chunks:", len(text_chunks))

# # 打印第一个文本片段
# print("\nFirst text chunk:")
# print(text_chunks[0])

def create_embeddings(text, model="text-embedding-ada-002"):
    """
    使用指定的OpenAI模型为给定文本创建嵌入。

    参数:
    text (str): 需要为其创建嵌入的输入文本。
    model (str): 用于创建嵌入的模型。

    返回:
    dict: 包含嵌入结果的OpenAI API回复。
    """
    # 使用指定模型为输入文本创建嵌入
    response = client.embeddings.create(
        model=model,
        input=text
    )

    return response  # 返回包含嵌入结果的回复

# 为文本块创建嵌入
response = create_embeddings(text_chunks)

import json

# 如果 response 是 dict 或有 dict() 方法
# try:
#     # print(json.dumps(response, indent=2, ensure_ascii=False))
#     print(response["data"][0]["embedding"])
# except TypeError:
#     # 如果 response 不是 dict，可以尝试转换
#     # print(json.dumps(response.model_dump(), indent=2, ensure_ascii=False))
#     print("--------------------------------")
#     response_dict = response.model_dump()
#     # print("response的第一项：",json.dumps(response_dict["data"][0], indent=2, ensure_ascii=False))
#     print(response_dict["data"][0].keys())
#     print(response_dict["data"][1]["index"])
#     print(response_dict["data"][0]["object"])
#     print("embedding的数量：",len(response_dict["data"]))
#     print("embedding：",response_dict["data"][0]["embedding"])
#     print("embedding的维度：",len(response_dict["data"][0]["embedding"]))

def cosine_similarity(vec1, vec2):
    """
    计算两个向量之间的余弦相似度。

    参数:
    vec1 (np.ndarray): 第一个向量。
    vec2 (np.ndarray): 第二个向量。

    返回:
    float: 两个向量之间的余弦相似度。
    """
    # 计算两个向量的点积，并除以它们范数的乘积
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def semantic_search(query, text_chunks, embeddings, k=5):
    """
    使用给定的查询和嵌入对文本块执行语义搜索。

    ------索引index+文本text+向量embedding-------
    
    参数:
    query (str): 语义搜索的查询。
    text_chunks (List[str]): 要搜索的文本块列表。
    embeddings (List[dict]): 文本块的嵌入列表。
    k (int): 返回的相关文本块数量。默认值为5。

    返回:
    List[str]: 基于查询的前k个最相关文本块列表。
    """
    # 为查询创建嵌入
    query_embedding = create_embeddings(query).data[0].embedding
    similarity_scores = []  # 初始化一个用于存储相似度分数的列表

    # 计算查询嵌入与每个文本块嵌入之间的相似度分数
    for i, chunk_embedding in enumerate(embeddings):
        similarity_score = cosine_similarity(np.array(query_embedding), np.array(chunk_embedding.embedding))
        similarity_scores.append((i, similarity_score))  # 将索引和相似度分数追加到列表中

    # 按降序对相似度分数进行排序
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    # 获取前k个最相似文本块的索引
    top_indices = [index for index, _ in similarity_scores[:k]]
    # 返回前k个最相关的文本块
    return [text_chunks[index] for index in top_indices]


# 从JSON文件中加载验证数据  
with open('/Users/eddiex/rag-demo/data/val.json') as f:  
    data = json.load(f)  

# 从验证数据中提取第一个查询  
query = data[0]['question']  

# 执行语义搜索以找到与查询最相关的前2个文本片段  
top_chunks = semantic_search(query, text_chunks, response.data, k=2)  

# 打印查询  
print("Query:", query)  

# 打印前2个最相关的文本片段  
# for i, chunk in enumerate(top_chunks):  
#     print(f"Context {i + 1}:\n{chunk}\n=====================================")

# 定义AI助手的系统提示
system_prompt = "You are an AI assistant that strictly answers based on the given context. If the answer cannot be derived directly from the provided context, respond with: 'I do not have enough information to answer that.'"

def generate_response(system_prompt, user_message, model="gpt-4o"):
    """
    根据系统提示和用户消息生成AI模型的回复。

    参数:
    system_prompt (str): 指导AI行为的系统提示。
    user_message (str): 用户的消息或查询。
    model (str): 用于生成回复的模型。

    返回:
    dict: AI模型的回复。
    """
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    )
    return response

# 基于top片段创建用户提示
user_prompt = "\n".join([f"Context {i + 1}:\n{chunk}\n=====================================\n" for i, chunk in enumerate(top_chunks)])
user_prompt = f"{user_prompt}\nQuestion: {query}"

# 生成AI回复
ai_response = generate_response(system_prompt, user_prompt)
print(ai_response.choices[0].message.content)