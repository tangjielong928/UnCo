from evaluator import EntityEvaluator
from utils import read_json
coarse_fine_tree = {
    'location': ['city',
                 'country',
                 'state',
                 'continent',
                 'location_other',
                 'park',
                 'road'],
    'building': ['building_other',
                 'cultural_place',
                 'entertainment_place',
                 'sports_facility'],
    'organization': ['company',
                     'educational_institution',
                     'band',
                     'government_agency',
                     'news_agency',
                     'organization_other',
                     'political_party',
                     'social_organization',
                     'sports_league',
                     'sports_team'],
    'person': ['politician',
               'musician',
               'actor',
               'artist',
               'athlete',
               'author',
               'businessman',
               'character',
               'coach',
               'director',
               'intellectual',
               'journalist',
               'person_other'],
    'other': ['animal',
              'award',
              'medical_thing',
              'website',
              'ordinance'],
    'art': ['art_other',
            'film_and_television_works',
            'magazine',
            'music',
            'written_work'],
    'event': ['event_other',
              'festival',
              'sports_event'],
    'product': ['brand_name_products',
                'game',
                'product_other',
                'software']}

def fine_to_label_dict(coarse_fine_tree = coarse_fine_tree):
    label_dict = {}
    index = 1
    for coarse in coarse_fine_tree:
        for fine_label in coarse_fine_tree[coarse]:
            label_dict[fine_label] = index
            index+=1
    label_dict["None-type"] = index
    return label_dict

def coarse_to_label_dict(coarse_fine_tree = coarse_fine_tree):
    label_dict = {}
    index = 1
    for keys in coarse_fine_tree:
        label_dict[keys] = index
        index+=1
    label_dict["None-type"] = index
    return label_dict

def print_fg_type_prompt(coarse_fine_tree = coarse_fine_tree):
    str = r""
    for coarse in coarse_fine_tree:
        for fine_label in coarse_fine_tree[coarse]:
            str += "- " + fine_label + "\n"
    print(repr(str))

if __name__ == '__main__':
    evaluator = EntityEvaluator(label_dict= fine_to_label_dict(), gt_path="./annotations/twitter_fmnerg_gt.json")
    # local model输出结果
    prediction_data = read_json("output/FMNERG/FMNERG_Qwen2___5-VL-7B-Instruct_20250510_base_fewshot_result-rerank.json")
    # 评估原始结果
    evaluator.evaluate(prediction_data)
    # print_fg_type_prompt()