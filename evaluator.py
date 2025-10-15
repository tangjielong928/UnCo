from typing import List, Tuple, Dict
from sklearn.metrics import precision_recall_fscore_support as prfs
from utils import collect, read_json_to_list, read_json, save_json, calculate_iou, str_to_list, find_sublist
# from fg_label import coarse_to_label_dict, fine_to_label_dict
import re
import sys



class EntityEvaluator:
    def __init__(self, label_dict, gt_path):
        '''
        @ param label_dict: 标签字典
        @ param gt_path: 标注文件路径
        '''
        self.label_dict = label_dict
        self.gt_path = gt_path
        self.gt_entities = read_json(gt_path)

    def filter_with_threshold(self, threshold, local_prediction_json, filter_field = "region_uncertainty", ):
        """
        根据阈值拆分预测JSON为两部分：
        1. 一部分包含filter_field小于等于阈值threshold的实体（保留部分）
        2. 一部分包含filter_field大于阈值threshold的实体（过滤部分）
        
        Args:
            threshold (float): 阈值
            local_prediction_json (list): 原始预测JSON
            filter_field (str): 用于过滤的字段名称
            
        Returns:
            tuple: (保留部分JSON, 过滤部分JSON)
        """
        below_threshold_json = []
        above_threshold_json = []
        
        for item in local_prediction_json:
            below_item = item.copy()
            above_item = item.copy()
            below_entities = []
            above_entities = []
            
            for entity in item.get('pre_entities', []):
                if filter_field in entity:
                    if entity[filter_field] <= threshold:
                        below_entities.append(entity.copy())
                    else:
                        above_entities.append(entity.copy())
                else:
                    below_entities.append(entity.copy())

            below_item['pre_entities'] = below_entities
            above_item['pre_entities'] = above_entities

            below_threshold_json.append(below_item)
            above_threshold_json.append(above_item)
        
        return below_threshold_json, above_threshold_json

    def evaluate(self, pred_json):
        '''
        @ param pred_json: 待评估的预测结果，格式为：
        [{'image_id': '1', 'pre_entities': [{'start': 1, 'end': 2, 'entity_type': 'entity_type', 'region_box': [1, 2, 3, 4]}]}
        其中 region_box 为 [x1, y1, x2, y2]为边界框的坐标, start为实体在句子中的起始位置, end为实体在句子中的结束位置. 
        1. 提供region_box; 或者2.提供mapping_regions与region (index), 0表示没有region.
        @ return: 评估指标
        '''
        assert len(pred_json) == len(self.gt_entities)
        pred_entities, gt_entities = [], []
        for pred_item, gt_item in zip(pred_json, self.gt_entities):
            assert pred_item['image_id'] == gt_item['image_id']
            pred_items, gt_items = [], []
            for pred_entity in pred_item['pre_entities']:
                if pred_entity["entity_type"] not in self.label_dict:
                    continue
                if 'region_box' in pred_entity:
                    pred_items.append((pred_entity['start'], pred_entity['end'], pred_entity['entity_type'], pred_entity['region_box']))
                elif 'region' and 'mapping_regions' in pred_entity: 
                    pred_items.append((pred_entity['start'], pred_entity['end'], pred_entity['entity_type'], [] if pred_entity['region'] == 0 else pred_entity['mapping_regions'][pred_entity['region']-1]))
                else:
                    pred_items.append((pred_entity['start'], pred_entity['end'], pred_entity['entity_type'], []))
            for gt_entity in gt_item['gt_entities']:
                gt_items.append((gt_entity['start'], gt_entity['end'], gt_entity['entity_type'], gt_entity['region_box']))
            pred_entities.append(pred_items)
            gt_entities.append(gt_items)

        return self.compute_scores(gt_entities, pred_entities)


    def correct_entities(self, llm_entities, local_entities, src_tokens, correct_filed):
        '''
        修正实体
        @ param llm_entities: LLM预测的实体
        @ param local_entities: 本地预测的实体
        @ return: 修正后的实体
        '''
        # region_to_index = {}
        # llm_phrase_to_entity = {
        #     entity['entity']: {"type": entity['type'], "region": int(re.search(r'region-(\d+)', entity['region']).group(1)) if entity['region'] and 'region-' in entity['region'] else 0} for
        #     entity in llm_entities}
        llm_phrase_to_entity = {
            entity['entity']: {
                "type": entity['corrected_type'] if "corrected_type" in entity else entity["type"],
                "corrected_entity": entity["corrected_entity"] if "corrected_entity" in entity else None,
                "region_box": str_to_list(entity['corrected_region']) if entity.get('corrected_region') not in [None, "Null", "null"] else []
            } for entity in llm_entities
        }

        for local_entity in local_entities:
            phrase = local_entity['phrase']
            if phrase in llm_phrase_to_entity:                
                if correct_filed == "entity_type":
                    local_entity['entity_type'] = llm_phrase_to_entity[phrase]['type'] if llm_phrase_to_entity[phrase]['type'] in self.label_dict else local_entity['entity_type']   
                if correct_filed == "region":
                    local_entity['region_box'] = llm_phrase_to_entity[phrase]['region_box']
                    local_entity['region'] = 0 if local_entity['region_box'] == [] else None 
                if correct_filed == "entity_span":
                    if llm_phrase_to_entity[phrase]['corrected_entity'] not in [None, "Null", "null",""] and find_sublist(src_tokens, llm_phrase_to_entity[phrase]['corrected_entity']) is not None:
                    # local_entity['phrase'] = llm_phrase_to_entity[phrase]['corrected_entity']
                        local_entity['start'], local_entity['end'] = find_sublist(src_tokens, llm_phrase_to_entity[phrase]['corrected_entity'])
                    else:
                        local_entity['phrase'] = None
                # print(f"*************** correct {phrase}, type: {local_entity['entity_type']}, region: {local_entity['region']} ***************")
        return local_entities

    def refine_to_entities(self, llm_refine, local_pre, local_refine, correct_filed, merge = False):
        '''
        提炼并输出修正后的metrics
        @ param llm_refine_path: LLM修正后的实体路径
        @ param local_pre_path: 本地不需修正的实体路径 (uncertainty filtering)
        @ param local_refine_path: 本地待修正的实体路径
        @ param correct_filed: 待修改的内容：["entity_span", "region", "entity_type"]
        @ param merge: 是否合并实体
        @ return: 修正后的实体
        '''
        img_id_to_llm_entities = {item['img_id']: item['pre_entities'] for item in llm_refine if len(item['pre_entities'])>0}
        for local_item in local_refine:
            image_id = local_item['image_id']
            if image_id in img_id_to_llm_entities:
                local_entities = local_item['pre_entities']
                llm_entities = img_id_to_llm_entities[image_id]
                local_item['pre_entities'] = self.correct_entities(llm_entities, local_entities,local_item["tokens"],correct_filed)
        # save_json(local_refine, "./output/local_refine.json")
        merged = []
        if merge:
            # Merge pre_entities for each corresponding object
            assert len(local_pre) == len(local_refine)
            for obj1, obj2 in zip(local_pre, local_refine):
                merged_obj = obj1.copy()
                pre_entities1 = obj1.get('pre_entities', [])
                pre_entities2 = obj2.get('pre_entities', [])
                merged_obj['pre_entities'] = pre_entities1 + pre_entities2
                merged.append(merged_obj)
        else:
            merged = local_refine
        metrics = self.evaluate(merged)
        return merged, metrics
        
    def _convert_by_setting(self, gt: List[List[Tuple]], pred: List[List[Tuple]],
                          include_entity_types: bool = True, include_score: bool = False, include_region: bool = False, only_eeg: bool = False):
        assert len(gt) == len(pred)

        # pred: [(start, end, entity_type, match_region, cls_score)]
        # either include or remove entity types based on setting
        def convert(t):
            if only_eeg and len(t) >3:
                c = [t[0], t[1], 'None-type', t[3]]
                return tuple(c)
            if not include_entity_types:
                # remove entity type and score for evaluation
                c = [t[0], t[1], 'None-type']
            else:
                c = list(t[:3])

            if include_region and len(t) > 3:
                # include prediction scores
                c.append(t[3])

            return tuple(c)
        
        def convert_for_bbox(tupe1, tupe2, IoU_threshold = 0.5):
            if not include_region and not only_eeg:
                return tupe1, tupe2
            if len(tupe1) == 0 or len(tupe2) == 0:
                return [], []
            if not (len(tupe1[-1]) > 3 and len(tupe2[-1]) > 3):
                return tupe1, tupe2
            if not (isinstance(tupe1[-1][3], list) and isinstance(tupe2[-1][3], list)):
                return tupe1, tupe2

            tupe1_list = [list(t) for t in tupe1]
            tupe2_list = [list(t) for t in tupe2]
          
            box_to_index = [([],0)]
            for i, t1 in enumerate(tupe1_list):
                if len(t1) > 3 and isinstance(t1[3], list) and len(t1[3]) > 0:
                    box_to_index.append((t1[3], i+1))
                    t1[3] = i+1  
                elif len(t1) > 3 and isinstance(t1[3], list) and len(t1[3]) == 0:
                    t1[3] = 0

            for i, t2 in enumerate(tupe2_list):
                if len(t2) <= 3 or not isinstance(t2[3], list):
                    continue
                match_found = False
                for box, index in box_to_index:
                    if isinstance(t2[3], list) and len(t2[3]) == 0:
                        t2[3] = 0   
                    if isinstance(t2[3], list) and len(t2[3]) > 0 and calculate_iou(t2[3], box) >= IoU_threshold \
                    and tupe1_list[index-1][0] == t2[0] and tupe1_list[index-1][1] == t2[1]: 
                        t2[3] = index  
                        match_found = True
                        break
                if not match_found and isinstance(t2[3], list):
                    t2[3] = -100  # 未找到匹配时的默认值
            new_tupe1 = [tuple(t) for t in tupe1_list]
            new_tupe2 = [tuple(t) for t in tupe2_list]
            return new_tupe1, new_tupe2
                
        converted_gt, converted_pred = [], []

        for sample_gt, sample_pred in zip(gt, pred):
            gt, pred = convert_for_bbox([convert(t) for t in sample_gt], [convert(t) for t in sample_pred])
            converted_gt.append(gt)
            converted_pred.append(pred)

        return converted_gt, converted_pred


    def _score(self, gt: List[List[Tuple]], pred: List[List[Tuple]], print_results: bool = False, cls_metric = False, only_text = True):
        assert len(gt) == len(pred)
        # import pdb;pdb.set_trace()
        gt_flat = []
        pred_flat = []
        types = set()

        for (sample_gt, sample_pred) in zip(gt, pred):
            union = set()
            if cls_metric:
                union.update(sample_gt)
                loc_gt = list(map(lambda x:(x[0],x[1]), sample_gt))
                sample_loc_true_pred =  list(filter(lambda x:(x[0], x[1]) in  loc_gt, sample_pred))
                union.update(sample_loc_true_pred)
            else:
                union.update(sample_gt)
                union.update(sample_pred)
            if not only_text:
                sample_pred, union = collect(union, sample_pred, sample_gt)
            for s in union:
                if s in sample_gt:
                    t = s[2]
                    gt_flat.append(self.label_dict[t])
                    types.add(t)
                else:
                    gt_flat.append(0)

                if s in sample_pred:
                    t = s[2]
                    pred_flat.append(self.label_dict[t])
                    types.add(t)
                else:
                    pred_flat.append(0)
        metrics = self._compute_metrics(gt_flat, pred_flat, types, print_results, only_text = only_text)
        return metrics

    def _compute_metrics(self, gt_all, pred_all, types, print_results: bool = False, only_text = True):
        labels = [self.label_dict[t] for t in types]
        per_type = prfs(gt_all, pred_all, labels=labels, average=None)
        micro = prfs(gt_all, pred_all, labels=labels, average='micro')[:-1]
        macro = prfs(gt_all, pred_all, labels=labels, average='macro')[:-1]
        total_support = sum(per_type[-1])

        if print_results:
            self._print_results(per_type, list(micro) + [total_support], list(macro) + [total_support], types, only_text = only_text)

        return [m * 100 for m in micro + macro]


    def _print_results(self, per_type: List, micro: List, macro: List, types: List, only_text = True):
        columns = ('type', 'precision', 'recall', 'f1-score', 'support')

        row_fmt = "%20s" + (" %12s" * (len(columns) - 1))
        print(row_fmt % columns)

        metrics_per_type = []
        for i, t in enumerate(types):
            metrics = []
            for j in range(len(per_type)):
                metrics.append(per_type[j][i])
            metrics_per_type.append(metrics)

        for m, t in zip(metrics_per_type, types):
            print(row_fmt % self._get_row(m, t))

        print('')
        print(row_fmt % self._get_row(micro, 'micro'))
        print(row_fmt % self._get_row(macro, 'macro'))


    def _get_row(self, data, label):
        row = [label]
        for i in range(len(data) - 1):
            row.append("%.2f" % (data[i] * 100))
        row.append(data[3])
        return tuple(row)


    def compute_scores(self, _gt_entities, _pred_entities):
        print("Evaluation")
        print("")
        print("--- GMNER ---")
        print("")
        gt, pred = self._convert_by_setting(_gt_entities, _pred_entities, include_entity_types=True, include_region=True)
        gmner_eval = self._score(gt, pred, print_results=True, only_text=False)

        print("")
        print("--- EEG ---")
        print("")
        gt, pred = self._convert_by_setting(_gt_entities, _pred_entities, only_eeg=True)
        eeg_eval = self._score(gt, pred, print_results=True, only_text=False)

        print("")
        print("--- MNER ---")
        # print("An entity is considered correct if the entity type and span is predicted correctly")
        print("")
        gt, pred = self._convert_by_setting(_gt_entities, _pred_entities, include_entity_types=True)
        ner_eval = self._score(gt, pred, print_results=True)

        print("")
        print("--- MNER on Localization ---")
        print("")
        gt_wo_type, pred_wo_type = self._convert_by_setting(_gt_entities, _pred_entities, include_entity_types=False)
        ner_loc_eval = self._score(gt_wo_type, pred_wo_type, print_results=True)

        print("")
        print("--- MNER on Classification ---")
        # print("An entity is considered correct if the entity type and span is predicted correctly")
        print("")
        # gt, pred = _convert_by_setting(_gt_entities, _pred_entities, include_entity_types=True)
        ner_cls_eval = self._score(gt, pred, print_results=True, cls_metric=True)

        return gmner_eval, eeg_eval, ner_eval, ner_loc_eval, ner_cls_eval



if __name__ == '__main__':
    evaluator = EntityEvaluator(label_dict= {"LOC": 1, "PER": 2, "ORG": 3, "MISC": 4,"None-type": 5}, gt_path="./annotations/twitter_gmner_gt.json")
    prediction_data = read_json("data/pred_best_model_mcd_num10_t30.json")
    llm_refine = read_json_to_list("refine_results/refine_result_T5-cl-qwen2.5-vl-72B-20250424_193322.jsonl")
    evaluator.evaluate(prediction_data)
    LLM_base_predcition = read_json("output/qwen_vl_72B_20250425_base_result-rerank.json")
    # evaluator.evaluate(LLM_base_predcition)


    # low_unc_data, high_unc_data = evaluator.filter_with_threshold(
    #     threshold=0.4,
    #     local_prediction_json=merged_pred,
    #     filter_field="span_uncertainty"  
    # )

    # merged_pred, metrics = evaluator.refine_to_entities(
    #     llm_refine = llm_refine,
    #     local_pre = low_unc_data,
    #     local_refine = high_unc_data,
    #     merge = True,
    #     correct_filed = "entity_span"
    # )
    low_unc_data, high_unc_data = evaluator.filter_with_threshold(
        threshold=0.5,
        local_prediction_json=merged_pred,
        filter_field="region_uncertainty"  
    )

    merged_pred, _ = evaluator.refine_to_entities(
        llm_refine = llm_refine,
        local_pre = low_unc_data,
        local_refine = high_unc_data,
        merge = True,
        correct_filed = "region"
    )
    
    low_unc_data, high_unc_data = evaluator.filter_with_threshold(
        threshold=0.3,
        local_prediction_json=prediction_data,
        filter_field="type_uncertainty"  
    )

    merged_pred, _ = evaluator.refine_to_entities(
        llm_refine = llm_refine,
        local_pre = low_unc_data,
        local_refine = high_unc_data,
        merge = True,
        correct_filed = "entity_type"
    )
        # result_list.append(metrics[1][2])
    # print(result_list)
    


