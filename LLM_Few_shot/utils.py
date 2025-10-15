import json
import base64
import torch
from torchvision.ops import box_iou

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
# 字符串转 JSON
def string_to_json(json_string):
    try:
        json_obj = json.loads(json_string)
        return json_obj
    except json.JSONDecodeError:
        print("错误: 输入的字符串不是有效的 JSON 格式!")
        return None

# JSON 转字符串
def json_to_string(json_obj):
    try:
        json_string = json.dumps(json_obj)
        return json_string
    except TypeError:
        print("错误: 输入的对象无法转换为 JSON 字符串!")
        return None

def str_to_list(input_str):
    if isinstance(input_str, list):
        return input_str
    try:
        input_str = input_str.strip('[]')
        elements = input_str.split(',')
        if len(elements) == 4:
            return [int(i) for i in elements]
        else:
            return []
    except (AttributeError, ValueError, IndexError):
        return []

def find_sublist(main_list, sub_list):
    if not sub_list: 
        return None
    if not main_list: 
        return None
        
    sub_len = len(sub_list)
    main_len = len(main_list)
    
    if sub_len > main_len:
        return None
    
    for i in range(main_len - sub_len + 1):
        match = True
        
        for j in range(sub_len):
            if main_list[i + j] != sub_list[j]:
                match = False
                break
        if match:
            return (i, i + sub_len - 1)
    return None

def calculate_iou(box1, box2):
    if len(box1) == len(box2) == 0:
        return 100.0
    elif len(box1) == 0 or len(box2) == 0:
        return 0.0
    elif len(box1) != 4 or len(box2) != 4:
        return 0.0
    # 转换为PyTorch张量
    box1_tensor = torch.tensor([box1], dtype=torch.float)
    box2_tensor = torch.tensor([box2], dtype=torch.float)
    iou_matrix = box_iou(box1_tensor, box2_tensor)
    return iou_matrix[0, 0].item()

def collect(union, sample_pred, sample_gt):
    fn = [x for x in union if x not in sample_gt]
    fp = []
    for i in range(len(sample_pred)):
        if sample_pred[i] in fn:
            for tp in sample_gt:
                if sample_pred[i][:3] == tp[:3] and sample_pred[i][3] != tp[3] and (sample_pred[i][0], sample_pred[i][1], sample_pred[i][2], 0):
                    fp.append((sample_pred[i][0], sample_pred[i][1], sample_pred[i][2], 0))
    union.update(fp)
    sample_pred.extend(fp)
    return sample_pred, union

def read_json_to_list(file_path):
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                obj = json.loads(line)
                # 检查解析结果是否为字典
                if isinstance(obj, dict):
                    # 检查是否存在 pre_entities 键且其值为列表
                    if 'pre_entities' in obj and isinstance(obj['pre_entities'], list):
                        data_list.append(obj)
            except json.JSONDecodeError:
                # 若解析失败，跳过当前行
                continue
    return data_list

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def img_2_image(input_file, output_file):
    data_list = []
    # 读取 JSONL 文件
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            data = json.loads(line)
            if 'img_id' in data:
                data['image_id'] = data.pop('img_id')
            data_list.append(data)

    # 将数据列表以 JSON 格式写入文件
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(data_list, outfile, ensure_ascii=False, indent=4)

    print("字段重命名完成，结果已保存到", output_file)


def reorder_json_by_ground_truth(processed_json_path, ground_truth_path, output_path):
    """
    根据 ground truth 文件中的 image_id 顺序对处理后的 JSON 文件进行重排序

    :param processed_json_path: 处理后的 JSON 文件路径
    :param ground_truth_path: ground truth 文件路径
    :param output_path: 重排序后 JSON 文件的输出路径
    """
    # 读取 ground truth 文件，提取 image_id 顺序
    with open(ground_truth_path, 'r', encoding='utf-8') as gt_file:
        ground_truth_data = json.load(gt_file)
        # 假设 ground_truth_data 是列表，每个元素包含 image_id 字段
        gt_image_ids = [item.get('image_id') for item in ground_truth_data]

    # 读取处理后的 JSON 文件
    with open(processed_json_path, 'r', encoding='utf-8') as processed_file:
        processed_data = json.load(processed_file)

    # 创建一个 image_id 到数据项的映射
    image_id_to_data = {item.get('image_id'): item for item in processed_data}

    # 按照 ground truth 中的 image_id 顺序重新排序
    reordered_data = [image_id_to_data.get(image_id) for image_id in gt_image_ids if image_id in image_id_to_data]

    # 将重排序后的数据写入新的 JSON 文件
    with open(output_path, 'w', encoding='utf-8') as output_file:
        json.dump(reordered_data, output_file, ensure_ascii=False, indent=4)

    print(f"文件已按 ground truth 顺序重排序，结果已保存到 {output_path}")

if __name__ == '__main__':
    processed_json_path = 'output/FMNERG/qwen2.5-vl-72b-instruct_20250427_base_result.json'
    ground_truth_path = 'annotations/twitter_fmnerg_gt.json'
    output_path = 'output/FMNERG/qwen2.5-vl-72b-instruct_20250427_base_result-rerank.json'
    reorder_json_by_ground_truth(processed_json_path, ground_truth_path, output_path)

