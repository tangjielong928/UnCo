import json
from copy import deepcopy
from tqdm import tqdm
import os
import torch
import torchvision
from PIL import Image
from utils import read_json
import xml.etree.ElementTree as ET
import numpy as np
import cv2

def mapping_region(xml_path='../GMNER/Twitter10000_v2.0/xml', vinvl_path='../GMNER/Twitter10000_v2.0/Vinvl_detection_path', max_num = 10, iou_value = 0.5, normalize = True):
    xmls = os.listdir(xml_path)
    res_dict = {}
    for xml in tqdm(xmls,desc="Mapping the candidate regions with gt"):
        img_id = xml.split('.')[0]
        tree = ET.parse(os.path.join(xml_path,xml))
        root = tree.getroot()
        res_dict[img_id] = {"bbox":[],"aspect":[]}
        aspects = []
        gt_boxes = []
        for object_container in root.findall('object'):
            for names in object_container.findall('name'):
                box_name = names.text
                box_container = object_container.findall('bndbox')
                if len(box_container) > 0:
                    xmin = int(box_container[0].findall('xmin')[0].text)
                    ymin = int(box_container[0].findall('ymin')[0].text)
                    xmax = int(box_container[0].findall('xmax')[0].text)
                    ymax = int(box_container[0].findall('ymax')[0].text)
                aspects.append(box_name)
                gt_boxes.append([xmin, ymin, xmax, ymax])
        assert len(aspects) == len(gt_boxes)
        bounding_boxes = np.zeros((max_num, 4), dtype=np.float32)
        image_feature = np.zeros((max_num, 2048), dtype=np.float32)
        img_path = os.path.join(vinvl_path, img_id+'.jpg.npz')
        crop_img = np.load(img_path)
        image_num = crop_img['num_boxes']
        final_num = min(image_num, max_num)
        bounding_boxes[:final_num] = crop_img['bounding_boxes'][:final_num]
        for aspect,gt_box in zip(aspects,gt_boxes):
            IoUs = (torchvision.ops.box_iou(torch.tensor([gt_box]), torch.tensor(bounding_boxes))).numpy() #(1,x)
            IoU = IoUs[0]
            flag = 0 # 记录是否检测到
            for i,iou in enumerate(IoU):
                if len(res_dict[img_id]["bbox"])==0 or not np.any(np.all(bounding_boxes[i] == res_dict[img_id]["bbox"], axis=1)):
                    res_dict[img_id]["bbox"].append(bounding_boxes[i].tolist())
                    res_dict[img_id]["aspect"].append("N")
                if iou>=iou_value:
                    res_dict[img_id]["aspect"][i] = aspect
                    flag = 1
            # if flag == 0:
            #     res_dict[img_id]["bbox"].append(gt_box)
            #     res_dict[img_id]["aspect"].append(aspect)
        assert len(res_dict[img_id]["bbox"]) == len(res_dict[img_id]["aspect"])

    # preprocess these img ungroundable
    imags_list = [x.split('.')[0] for x in os.listdir(vinvl_path)]
    xml_img_list = [x.split('.')[0] for x in xmls]
    for img_id in tqdm(imags_list, desc="process these ungroundable image"):
        if img_id not in xml_img_list:
            res_dict[img_id] = {"bbox":[],"aspect":[]}
            bounding_boxes = np.zeros((max_num, 4), dtype=np.float32)
            image_feature = np.zeros((max_num, 2048), dtype=np.float32)
            img_path = os.path.join(vinvl_path, img_id + '.jpg.npz')
            crop_img = np.load(img_path)
            image_num = crop_img['num_boxes']
            final_num = min(image_num, max_num)
            bounding_boxes[:final_num] = crop_img['bounding_boxes'][:final_num]
            for i in range(final_num):
                res_dict[img_id]["bbox"].append(bounding_boxes[i].tolist())
                res_dict[img_id]["aspect"].append("N")
        assert len(res_dict[img_id]["bbox"]) == len(res_dict[img_id]["aspect"])
    return res_dict

def extract(data_path = '../GMNER/Twitter10000_v2.0/txt/test.txt'):
    mapping_regions = mapping_region()
    with open(data_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        raw_words, raw_targets = [], []
        raw_word, raw_target = [], []
        imgs = []
        for line in lines:
            if line.startswith("IMGID:"):
                img_id = line.strip().split('IMGID:')[1] + '.jpg'
                imgs.append(img_id)
                continue
            if line != "\n":
                raw_word.append(line.split('\t')[0])
                label = line.split('\t')[1][:-1]
                if 'OTHER' in label:
                    label = label[:2] + 'MISC'
                raw_target.append(label)
            else:
                raw_words.append(raw_word)
                raw_targets.append(raw_target)
                raw_word, raw_target = [], []

    assert len(raw_words) == len(raw_targets) == len(imgs), "{}, {}, {}".format(len(raw_words), len(raw_targets),
                                                                                len(imgs))
    res = []
    for sentence, target, img_id in zip(raw_words, raw_targets, imgs):
        entities = []
        start, end = 0, 0
        while start<len(target) and end<len(target):
            if 'B-' in target[start]:
                end = start
                end +=1
                while end<len(target) and 'I-' in target[end]:
                    end +=1
                entity = " ".join(sentence[start:end])
                entities.append({
                    "type": target[start][2:],
                    "entity": entity,
                    "region": [f"<Region-{index}>" for index in range(len(mapping_regions[img_id.split('.')[0]]['aspect']))
                               if mapping_regions[img_id.split('.')[0]]['aspect'][index] == entity]
                })
                start = end
            start +=1
        res.append({
            "tokens": " ".join(sentence),
            "img_id": img_id,
            "entities": entities,
            "candidate_region": mapping_regions[img_id.split('.')[0]]['bbox']
        })

    return  res

def demo_to_prompt(res, json_out = './demos_test.json'):
    prompt_list = []
    for sample in res:
        raw_txt_prompt = "\n[Original social media post]: " + sample["tokens"] + "\n"
        region_prompt = "[Candidate regions in the image]: "
        for i in range(len(sample["candidate_region"])):
            x1 = sample["candidate_region"][i][0]
            y1 = sample["candidate_region"][i][1]
            x2 = sample["candidate_region"][i][2]
            y2 = sample["candidate_region"][i][3]
            region_prompt += f"<Region-{i}>: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}], "
        region_prompt += "\n"
        output_prompt = "[Pre-extracted GMNER results]: "
        for i in range(len(sample["entities"])):
            entity = sample["entities"][i]["entity"]
            type = sample["entities"][i]["type"]
            region = ", ".join(sample["entities"][i]["region"])
            output_prompt += f"<Output-{i}>: {{entity: {entity}, region: [], type: {type}}}, "
        output_prompt += "\n"
        prompt = raw_txt_prompt+region_prompt+output_prompt
        prompt_list.append({
            "img_id": sample["img_id"],
            "prompt": prompt
        })

    with open(json_out, "w") as f:
        json.dump(prompt_list, f)
    return prompt_list

def prediction_to_prompt(pred_json):
    prediction = read_json(pred_json)
    prompt_list = []
    for sample in tqdm(prediction, total=len(prediction), desc="Preprocess the previously predicted data...."):
        if len(sample["pre_entities"])>0:
            raw_txt_prompt = "Original social media post: " + " ".join(sample["tokens"]) + "\n"
            region_prompt = "Candidate regions (bounding box) in the image: {"
            for i in range(len(sample["candidate_regions"])):
                x1 = sample["candidate_regions"][i][0]
                y1 = sample["candidate_regions"][i][1]
                x2 = sample["candidate_regions"][i][2]
                y2 = sample["candidate_regions"][i][3]
                region_prompt += f"\"region-{i+1}\": [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}], "
            region_prompt += "}\n"
            output_prompt = "Pre-detected results: "
            output_list = []
            for i in range(len(sample["pre_entities"])):
                entity = sample["pre_entities"][i]["phrase"]
                type = sample["pre_entities"][i]["entity_type"]
                region = sample["pre_entities"][i]["region"]
                if region == 0:
                    output_list.append({"entity": entity, "region": None, "type": type})
                else:
                    output_list.append({"entity": entity, "region":f"region-{region}","type":type})
            output_prompt += json.dumps(output_list)
            prompt = raw_txt_prompt + region_prompt + output_prompt
            prompt_list.append({
                "img_id": sample["image_id"],
                "prompt": prompt
            })
    # with open('./prediction_test.json', "w") as f:
    #     json.dump(prompt_list, f)
    return prompt_list


if __name__ == '__main__':
    # res = extract()
    # prompt_list = demo_to_prompt(res)
    prediction_to_prompt(pred_json="./data/checkpoints/twitterGMNER/twitter10k_eval/202y5-03-12_12:02:39.077556/predictions_test_epoch_0.json")