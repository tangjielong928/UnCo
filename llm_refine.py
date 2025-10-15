import json
import os
import re
import time
from tqdm import tqdm
from openai import OpenAI
from Prompt_content import Span_patch_generator, Region_patch_generator
from utils import encode_image
import api_config
from utils import read_json
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

# thread lock
output_file_lock = Lock()
log_file_lock = Lock()

def completion_with_backoff(in_data, system_prompt, client, retries=2, backoff_factor=0.5):

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

def get_system_prompt(dataset_name = "GMNER", refine_type = "Entity_span_refine", threshold = 0.5):

    assert refine_type in ["Entity_span_refine", "Entity_type_refine", "Entity_region_refine"]
    if refine_type == "Entity_span_refine":
        system_prompt = Span_patch_generator(threshold).get_system_prompt()
    # elif refine_type == "Entity_type_refine":
    #     system_prompt = Type_patch_generator(threshold).get_system_prompt()
    elif refine_type == "Entity_region_refine":
        system_prompt = Region_patch_generator(threshold, dataset_name).get_system_prompt()
    else:
        raise ValueError("refine_type must be one of ['Entity_span_refine', 'Entity_type_refine', 'Entity_region_refine']")
    
    content = [{"type": "text","text": system_prompt},]

    return content


def Entity_span_refine(under_refine_entity, client, img, original_text):
    
    system_prompt = get_system_prompt(refine_type = "Entity_span_refine")
    input_text =  original_text + "The pre-detected entity: " + under_refine_entity["entity"] +"\n"

    input_content = [
        {"type": "text", "text": input_text},
    ]

    chat_result = chat(input_content, system_prompt, client)

    if chat_result is not None:
        span_refine_result = json_parsing(chat_result)

        if span_refine_result is not None:
            under_refine_entity["corrected_entity"] = span_refine_result['corrected_entity']
            under_refine_entity["span_refine_reasoning"] = span_refine_result['reasoning_process']
        else:
            print("LLM output is None")
            return under_refine_entity
        
    return under_refine_entity

# def Entity_type_refine(under_refine_entity, client, img, original_text):
#     system_prompt = get_system_prompt(type = "Entity_type_refine")
#     input_text =  original_text + "Span of entities that need to be refined: " + under_refine_entity["entity"] 

#     input_content = [
#         {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}},
#         {"type": "text", "text": input_text},
#     ]

#     chat_result = chat(input_content, system_prompt, client)

#     return chat_result

def Entity_region_refine(under_refine_entity, client, dataset_name, img_path, original_text):
    system_prompt = get_system_prompt(dataset_name, refine_type = "Entity_region_refine")
    input_text =  original_text + "The pre-detected entity: " + under_refine_entity["corrected_entity"] + "\n"+\
    " The type of the pre-detected entity: " + under_refine_entity["type"] + " \nUncertainty of type: " + str(under_refine_entity["type_uncertainty"]) + \
    " \nThe bounding box of the pre-detected entity: " + str(under_refine_entity["region"]) + " \nUncertainty of bounding box: " + str(under_refine_entity["region_uncertainty"])
    try:
        img = encode_image(img_path)
        input_content = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}},
            {"type": "text", "text": input_text},
        ]
    except Exception as e:
        print(f"读取图片错误: {e}")
        input_content = [
            {"type": "text", "text": input_text},
        ]

    chat_result = chat(input_content, system_prompt, client)

    
    if chat_result is not None:
        span_refine_result = json_parsing(chat_result)
        if span_refine_result is not None:
            under_refine_entity["type_and_region_refine_reasoning"] = span_refine_result.get("reasoning_process", "")
            under_refine_entity["corrected_type"] = span_refine_result.get("corrected_type", "")
            under_refine_entity["corrected_region"] = span_refine_result.get("corrected_bounding_box", "")
        else:
            print("LLM output is None")
            return under_refine_entity
        
    return under_refine_entity

def Refinement_generation(prediction, dataset_name, img_path, span_threshold, type_threshold, region_threshold, client, output_path, save_process_file):
    original_text = "Original Text: {" + " ".join(prediction["tokens"]) + "}\n"
    pre_entities = prediction["pre_entities"]
    img_id = prediction["image_id"]
    # img = encode_image(img_path + img_id)
    img_path = img_path + img_id

    # The first step is to refine the span of entities with uncertainty exceeding the threshold.
    span_refinement = []
    for pre_entity in pre_entities:
        pre_entity_f = {
            "orginal_text": " ".join(prediction["tokens"]),
            "entity": pre_entity["phrase"],
            "corrected_entity": pre_entity["phrase"],
            "span_uncertainty": pre_entity["span_uncertainty"],

            "region": [round(i,2) for i in pre_entity["region_box"]] if len(pre_entity["region_box"])>0 else [],
            "corrected_region": [round(i,2) for i in pre_entity["region_box"]] if len(pre_entity["region_box"])>0 else [],
            "region_uncertainty": pre_entity["region_uncertainty"],

            "type": pre_entity["entity_type"],
            "corrected_type": pre_entity["entity_type"],
            "type_uncertainty": pre_entity["type_uncertainty"],

            "span_refine_reasoning": "",
            "type_and_region_refine_reasoning": "",
        }
        if pre_entity["span_uncertainty"] > span_threshold: 
            span_refinement.append(Entity_span_refine(pre_entity_f, client, img_path, original_text))
            # span_refinement.append(pre_entity_f)
        else:
            span_refinement.append(pre_entity_f)

    # # The second step is to refine the type of entities with uncertainty exceeding the threshold.
    # type_refinement = []
    # for span_and_type_entity in span_refinement:
    #     if span_and_type_entity["type_uncertainty"] > type_threshold:
    #         type_refinement.append(Entity_type_refine(span_and_type_entity, client, img, original_text))
    #     else:
    #         type_refinement.append(span_and_type_entity)

    # The second step is to refine the region and type of entities with uncertainty exceeding the threshold.
    region_refinement = []
    for span_refine_entity in span_refinement:
        if span_refine_entity["corrected_entity"] not in [None, "Null", "null",""]:
            region_refinement.append(Entity_region_refine(span_refine_entity, client, dataset_name, img_path, original_text))
        else:
            region_refinement.append(span_refine_entity)

    Final_result = region_refinement
    # save_result
    Save_result(Final_result, img_id, output_path, save_process_file)

def json_parsing(sample_result):
    # 解析 JSON 结果
    refine_result = None
    if sample_result is not None:
        pattern = r'```json(.*?)```'
        match = re.search(pattern, sample_result, re.DOTALL)
        if match:
            json_part = match.group(1).strip()
            try:
                refine_result = json.loads(json_part)
                # refine_result = list_to_dict(refine_result)
            except json.JSONDecodeError as e:
                print(f"LLM输出JSON 解析错误: {e}")
                return None
        else:
            try:
                refine_result = json.loads(sample_result)
                # refine_result = list_to_dict(refine_result)
            except json.JSONDecodeError as e:
                print(f"LLM输出JSON 解析错误: {e}")
                return None
     
    return refine_result


def Save_result(refine_result, img_id, output_path, save_process_file):
    
    # 加锁并追加写入解析后的 JSON 结果
    if refine_result is not None:
        # 检查 output_path 所在目录是否存在，不存在则创建
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        with output_file_lock:
            with open(output_path, 'a') as file:
                json.dump({"img_id": img_id, "pre_entities": refine_result}, file)
                file.write('\n')
    else:
        print(f"LLM输入错误，跳过 img_id 为 {img_id} 的样例！")



def call_LLM(span_threshold, type_threshold, region_threshold,  img_path, pred_path, output_path, dataset_name, save_process_file, max_threads=1, model_name="Qwen2___5-VL-7B-Instruct"):

    output_path = output_path + f"/{dataset_name}/" + f"{os.path.splitext(output_path)[1]}/refine_result_{dataset_name}_{model_name}_20250501.jsonl"
    save_process_file = save_process_file + f"{os.path.splitext(save_process_file)[1]}/MLLM_gen_log.txt"
    
    Local_predictions = read_json(pred_path)

    # Check the processed img_id
    processed_img_ids = set()
    if os.path.exists(output_path):
        with open(output_path, 'r') as file:
            for line in tqdm(file, desc="Checking exist processed entities..."):
                try:
                    data = json.loads(line)
                    processed_img_ids.add(data["img_id"])
                except json.JSONDecodeError:
                    continue
            print(f"Found {len(processed_img_ids)} processed entities in {output_path}")
    # Filter out the samples that have been processed.
    remaining_predictions = [pred for pred in Local_predictions if pred["image_id"] not in processed_img_ids]

    # Update the model name
    api_config.MODEL_NAME = model_name
    print(f"LLM_refine start, using model: {api_config.MODEL_NAME}")
    print(f"LLM_refine start, using dataset: {dataset_name}")
    client = OpenAI(base_url=api_config.API_BASE, api_key=api_config.API_KEY)

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = [executor.submit(Refinement_generation, prediction, dataset_name, img_path, span_threshold, type_threshold, region_threshold, client, output_path, save_process_file) for prediction in remaining_predictions]
        for future in tqdm(futures, desc="LLM correcting...", total=len(futures)):
            future.result()

    print("LLM correction completed!")

if __name__ == '__main__':
    call_LLM(
        span_threshold = 0.4, 
        type_threshold = 0.3, 
        region_threshold = 0.5,
        img_path = "../twitter_gmner_image/whole_image/",
        pred_path = "./data/FG_pred_best_model_mcdCL_10.json",
        output_path ="./refine_results",
        dataset_name = "FMNERG",
        save_process_file="./refine_process_logs",
        model_name = "InternVL3-9B"
    )
