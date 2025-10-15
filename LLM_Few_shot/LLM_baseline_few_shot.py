import json
import os
import re
import time
from tqdm import tqdm
from openai import OpenAI
from Prompt_content_few_shot import MLLM_base_generator
from utils import encode_image
import api_config
from utils import read_json, str_to_list
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

# 定义线程锁
output_file_lock = Lock()
log_file_lock = Lock()

def completion_with_backoff(in_data, system_prompt, client, retries=2, backoff_factor=0.5):
    # print(api_config.MODEL_NAME)
    for attempt in range(retries):
        try:
            return client.chat.completions.create(
                model=api_config.MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt}, 
                    {"role": "user", "content": in_data}
                    ],
                temperature=0.1,
            )
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(backoff_factor * (2 ** attempt))
    return None

def chat(input_content, system_prompt, client):
    # print(input_content)
    response = completion_with_backoff(input_content, system_prompt, client)
    if response is not None:
        return response.choices[0].message.content
    else:
        return None

def get_system_prompt(dataset_name):

   
    system_prompt = MLLM_base_generator(dataset_name).get_system_prompt()
   
    
    content = [{"type": "text","text": system_prompt},]

    return content

def get_few_shot_example(dataset_name, img_path):
    few_shot_content = [
        {"type": "text", "text": "Here are some examples you can refer to: "},
    ]

    if dataset_name == "GMNER":
        few_shot_data_path = "GMNER_demos.json"
    else:
        few_shot_data_path = "FMNERG_demos.json"
        
    with open(few_shot_data_path, 'r', encoding='utf-8') as file:
        # 直接加载整个 JSON 文件
        few_shot_data = json.load(file)

        for data in few_shot_data:
            img_id = data["img_id"]
            img = encode_image(img_path + img_id)
            few_shot_content.append({"type": "text", "text": "[Input]\n Original Image:"})
            few_shot_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}})
            original_text = "Original Text: {" + " ".join(data["tokens"]) + "}\n"
            entities = ""
            entity_strings = []
            for entity in data["gt_entities"]:
                phrase = entity["phrase"]
                entity_type = entity["entity_type"]
                region_box = entity["region_box"]
                # 构建单个实体的 JSON 字符串
                entity_str = (
                    " {\n"
                    f"  \"phrase\": \"{phrase}\",\n"
                    f"  \"entity_type\": \"{entity_type}\",\n"
                    f"  \"region_box\": {region_box}\n"
                    "  }"
                )
                entity_strings.append(entity_str)
            # 使用逗号连接所有实体字符串
            entities = ",\n".join(entity_strings)
            entities = "[Output]\n```json\n{\n  \"pre_entities\": [\n" + entities + "]\n}```\n"
            few_shot_text = original_text + entities
            few_shot_content.append({"type": "text", "text": few_shot_text})
    
    return few_shot_content

def find_phrase_indices(tokens, phrase):
    """
    根据切分后的 tokens 找出 phrase 在原始文本中的 start 和 end 索引。

    :param tokens: 切分后的 token 列表
    :param phrase: 要查找的实体短语
    :return: 包含 start 和 end 索引的字典，如果未找到则返回 None
    """
    phrase_tokens = phrase.split()
    if not phrase_tokens:
        return None
    first_token = phrase_tokens[0]
    last_token = phrase_tokens[-1]
    first_token_index = 0
    last_token_index = 0

    # 查找第一个 token 的位置
    for i in range(len(tokens)):
        if tokens[i] == first_token:
            first_token_index = i
            break

    if first_token_index is not None:
        # 查找最后一个 token 的位置，从第一个 token 位置开始向后查找
        for j in range(first_token_index, len(tokens)):
            if tokens[j] == last_token:
                last_token_index = j
                break

    return {"start": first_token_index, "end": last_token_index}


def Refinement_generation(prediction, dataset_name, img_path,  client, output_path):
    original_text = "Original Text: {" + " ".join(prediction["tokens"]) + "}\n[Output]:..."
    img_id = prediction["image_id"]
    try:
        few_shot_example = get_few_shot_example(dataset_name, img_path)
    except Exception as e:
        print(f"few shot生成错误: {e}")
        few_shot_example = []
    try:
        img = encode_image(img_path + img_id)
        system_prompt = get_system_prompt(dataset_name)
        input_text =  original_text
        input_content = [
            {"type": "text", "text": "[Input]\nOriginal Image:"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}},
            {"type": "text", "text": input_text},
        ]

    except Exception as e:
        print(f"读取图片错误: {e}")
        system_prompt = get_system_prompt(dataset_name)
    
        input_text =  original_text
        input_content = [
            {"type": "text", "text": "[Input]\n"+input_text},
        ]

    input_content = few_shot_example + input_content
    chat_result = chat(input_content, system_prompt, client)

    # print(f"chat_result: {chat_result}")

    Final_result = json_parsing(chat_result)

    # print(f"\nFinal_result: {Final_result}")


    if Final_result:
        for entity in Final_result["pre_entities"]:
            try:
                phrase = entity["phrase"]
                indices = find_phrase_indices(prediction["tokens"], phrase)
                entity["start"] = indices["start"]
                entity["end"] = indices["end"]
                if isinstance(entity["region_box"], str):
                    entity["region_box"] = str_to_list(entity["region_box"])
            except Exception as e:
                print(f"处理实体时发生错误: {e}")


    # save_result
    Save_result(Final_result, img_id, output_path)

def json_parsing(sample_result):
    # 解析 JSON 结果
    final_result = None
    if sample_result is not None:
        pattern = r'```json(.*?)```'
        match = re.search(pattern, sample_result, re.DOTALL)
        if match:
            json_part = match.group(1).strip()
            try:
                final_result = json.loads(json_part)
            except json.JSONDecodeError as e:
                print(f"LLM输出JSON 解析错误: {e}")
                return None
        else:
            try:
                final_result = json.loads(sample_result)
            except json.JSONDecodeError as e:
                print(f"LLM输出JSON 解析错误: {e}")
                return None
     
    return final_result


def Save_result(final_result, img_id, output_path):
    # 检查 output_path 所在目录是否存在，不存在则创建
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 加锁操作
    with output_file_lock:
        try:
            # 尝试读取现有的 JSON 文件
            if os.path.exists(output_path):
                with open(output_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
            else:
                data = []

            # 处理 final_result
            if final_result is not None:
                pre_entities = final_result["pre_entities"]
            else:
                pre_entities = []
                print(f"LLM输入错误，跳过 image_id 为 {img_id} 的样例！")

            # 追加新数据
            data.append({"image_id": img_id, "pre_entities": pre_entities})

            # 将更新后的数据写回文件
            with open(output_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, ensure_ascii=False, indent=4)

        except Exception as e:
            print(f"写入 JSON 文件时出错: {e}")



def call_LLM(img_path, pred_path, output_path, dataset_name, max_threads=1, model_name="gemma-3-4b-it"):

    output_path = output_path + f"/{dataset_name}/" + f"{os.path.splitext(output_path)[1]}/{dataset_name}_{model_name}_20250511_base_fewshot_result.json"
    
    Local_predictions = read_json(pred_path)

    # Check the processed img_id
    processed_img_ids = set()
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as file:
                # 直接加载整个 JSON 文件
                data_list = json.load(file)
                # 提取所有 image_id
                processed_img_ids = {item["image_id"] for item in data_list}
            print(f"Found {len(processed_img_ids)} processed entities in {output_path}")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON file {output_path}: {e}")
    # Filter out the samples that have been processed.
    remaining_predictions = [pred for pred in Local_predictions if pred["image_id"] not in processed_img_ids]

    # Update the model name
    api_config.MODEL_NAME = model_name
    print(f"LLM4MNER start, using model: {api_config.MODEL_NAME}")
    print(f"LLM4MNER start, using dataset: {dataset_name}")
    client = OpenAI(base_url=api_config.API_BASE, api_key=api_config.API_KEY)

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = [executor.submit(Refinement_generation, prediction, dataset_name, img_path, client, output_path) for prediction in remaining_predictions]
        for future in tqdm(futures, desc="LLM4MNER...", total=len(futures)):
            future.result()

    print("LLM4MNER completed!")

if __name__ == '__main__':
    call_LLM(
        img_path = "../twitter_gmner_image/whole_image/",
        pred_path = "data/FG_pred_best_model_mcdCL_10.json",
        output_path ="output",
        dataset_name = "FMNERG",
        model_name = "qwen2.5-vl-32b-instruct"
    )   


