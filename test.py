import sys
sys.path.append('../')
import os
if 'p' in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['p']
    # os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import warnings
                                                 
warnings.filterwarnings('ignore')


from model.data_pipe import BartNERPipe
from model.bart_multi_concat import BartSeq2SeqModel  
from model.generater_multi_concat import SequenceGeneratorModel     
from model.metrics import Seq2SeqSpanMetric 
from model.losses import get_loss, compute_uncertainty

import fitlog
import datetime
from fastNLP import Trainer

from torch import optim
from fastNLP import BucketSampler, GradientClipCallback, cache_results, EarlyStopCallback, SequentialSampler

from model.callbacks import WarmupCallback
from fastNLP.core.sampler import SortedSampler
# from fastNLP.core.sampler import  ConstTokenNumSampler
from model.callbacks import FitlogCallback
from fastNLP import DataSetIter
from tqdm import tqdm, trange
from fastNLP.core.utils import _move_dict_value_to_device
import random
import torch
import json
import numpy as np

fitlog.debug()
fitlog.set_log_dir('logs')




import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--bart_name', default='facebook/bart-large', type=str)
parser.add_argument('--datapath', default='./Twitter_GMNER/txt/', type=str)
parser.add_argument('--image_feature_path',default='./data/Twitter_GMNER_vinvl', type=str)
parser.add_argument('--image_annotation_path',default='./Twitter_GMNER/xml/', type=str)
parser.add_argument('--box_num',default='16', type=int)
parser.add_argument('--model_weight',default= None, type = str)
parser.add_argument('--normalize',default=False, action = "store_true")
parser.add_argument('--use_edl',default=False,action ="store_true")
parser.add_argument('--with_kl',default=True,action ="store_false", help='evidential loss kl penalty')
parser.add_argument('--use_kl',default=False,action ="store_true", help='region_loss')
parser.add_argument('--etrans_func',default='softplus',type=str,choices=['softplus','exp','relu','softmax'],help='evidential transform function')
parser.add_argument('--annealing_coef',default=1.0,type=float,help='annealing coefficient for KL divergence in evidential loss')
parser.add_argument('--max_len', default=30, type=int)
parser.add_argument('--batch_size',default=16,type=int)
parser.add_argument("--log",default='./logs',type=str)
parser.add_argument('--mc_dropout', action='store_true', help='MC Dropout')
parser.add_argument('--mc_times', type=int, default=10, help='MC Dropout sampling time')
args= parser.parse_args()


model_path = args.model_weight.rsplit('/')
args.pred_output_file = '/'.join(model_path[:-1])+'/pred_'+model_path[-1]+'.txt'


dataset_name = 'twitter-ner'
args.length_penalty = 1
args.target_type = 'word'
args.schedule = 'linear'
args.decoder_type = 'avg_feature'
args.num_beams = 1   
args.use_encoder_mlp = 1
args.warmup_ratio = 0.01
eval_start_epoch = 0
spt_to_type= {'<<location>>': 'LOC',
              '<<person>>': 'PER',
              '<<others>>': 'MISC',
              '<<organization>>': 'ORG'}

if 'twitter' in dataset_name:  
    max_len, max_len_a = args.max_len, 0.6
else:
    print("Error dataset_name!")


if isinstance(args.decoder_type, str) and args.decoder_type.lower() == 'none':
    args.decoder_type = None
demo = False

def get_data():

    pipe = BartNERPipe(image_feature_path=args.image_feature_path, 
                       image_annotation_path=args.image_annotation_path,
                       max_bbox =args.box_num,
                       normalize=args.normalize,
                       tokenizer=args.bart_name, 
                       target_type=args.target_type)
    if dataset_name == 'twitter-ner': 
        paths ={
            'train': os.path.join(args.datapath,'train.txt'),
            'dev': os.path.join(args.datapath,'dev.txt'),
            'test': os.path.join(args.datapath,'test.txt') }

        data_bundle = pipe.process_from_file(paths, demo=demo)
        
    return data_bundle, pipe.tokenizer, pipe.mapping2id

data_bundle, tokenizer, mapping2id = get_data()

print(f'max_len_a:{max_len_a}, max_len:{max_len}')

print(data_bundle)
print("The number of tokens in tokenizer ", len(tokenizer.decoder))  

bos_token_id = 0
eos_token_id = 1
label_ids = list(mapping2id.values())


model = BartSeq2SeqModel.build_model(args.bart_name, tokenizer, label_ids=label_ids, decoder_type=args.decoder_type,
                                     use_encoder_mlp=args.use_encoder_mlp,box_num = args.box_num)


vocab_size = len(tokenizer)

model = SequenceGeneratorModel(model, bos_token_id=bos_token_id,
                               eos_token_id=eos_token_id, 
                               max_length=max_len, max_len_a=max_len_a,num_beams=args.num_beams, do_sample=False,
                               repetition_penalty=1, length_penalty=args.length_penalty, pad_token_id=eos_token_id,
                               restricter=None, top_k = 1
                               )

model.load_state_dict(torch.load(args.model_weight))



if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


metric = Seq2SeqSpanMetric(eos_token_id, num_labels=len(label_ids), region_num =args.box_num, target_type=args.target_type,print_mode = False )


test_dataset = data_bundle.get_dataset('test')
print(test_dataset[:3])

test_dataset.set_target('raw_words', 'raw_target')

device = torch.device(device)
model.to(device)

def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def mc_dropout_predict(model, batch_x, T=20, device='cuda'):
    token_logits_list = []
    region_logits_list = []
    model.eval()
    enable_dropout(model)
    for _ in range(T):
        with torch.no_grad():
            results = model.predict_with_logits(
                batch_x['src_tokens'],
                batch_x['image_feature'],
                src_seq_len=batch_x['src_seq_len'],
                first=batch_x['first']
            )
            token_logits_list.append(results['pred_logits'].detach().cpu().numpy())
            region_logits_list.append(results['region_logits'].detach().cpu().numpy())
    token_logits = np.stack(token_logits_list, axis=0)  # [T, B, L, V]
    region_logits = np.stack(region_logits_list, axis=0)  # [T, B, L, box_num]
    token_probs = torch.softmax(torch.tensor(token_logits), dim=-1).numpy()  # [T, B, L, V]
    region_probs = torch.softmax(torch.tensor(region_logits), dim=-1).numpy()  # [T, B, L, box_num]
    token_probs_mean = np.mean(token_probs, axis=0)  # [B, L, V]
    region_probs_mean = np.mean(region_probs, axis=0)  # [B, L, box_num]
    token_entropy = -np.sum(token_probs_mean * np.log(token_probs_mean + 1e-12), axis=-1)  # [B, L]
    region_entropy = -np.sum(region_probs_mean * np.log(region_probs_mean + 1e-12), axis=-1)  # [B, L]
    token_ids = np.argmax(token_probs_mean, axis=-1)
    region_ids = np.argmax(region_probs_mean, axis=-1)
    return token_entropy.tolist(), region_entropy.tolist(), token_ids.tolist(), region_ids.tolist()


def Predict(args,eval_data, model, device, metric,tokenizer,ids2label):
    data_iterator = DataSetIter(eval_data, batch_size=args.batch_size * 2, sampler=SequentialSampler())
    res_json_list = []
    with open (args.pred_output_file,'w') as fw:
        for batch_x, batch_y in tqdm(data_iterator, total=len(data_iterator)):
            _move_dict_value_to_device(batch_x, batch_y, device=device)
            src_tokens = batch_x['src_tokens']
            image_feature = batch_x['image_feature']
            tgt_tokens = batch_x['tgt_tokens']
            src_seq_len = batch_x['src_seq_len']
            tgt_seq_len = batch_x['tgt_seq_len']
            first = batch_x['first']
            region_label = batch_y['region_label']
            target_span = batch_y['target_span']
            cover_flag = batch_y['cover_flag']

            if args.mc_dropout:
                token_uncertainty, region_uncertainty, token_ids, region_ids = mc_dropout_predict(model, batch_x, T=args.mc_times, device=device)
            else:
                results = model.predict_with_logits(src_tokens, image_feature, src_seq_len=src_seq_len, first=first)
                pred, region_pred, pred_logits, region_logits = results['pred'], results['region_pred'], results['pred_logits'], results['region_logits']
                if args.use_edl:
                    token_uncertainty, token_ids = compute_uncertainty(pred_logits, 'softmax')
                    region_uncertainty, region_ids = compute_uncertainty(region_logits, 'softmax')
                else:
                    token_uncertainty, token_ids = compute_uncertainty(pred_logits, 'softmax')
                    region_uncertainty, region_ids = compute_uncertainty(region_logits, 'softmax')
            
            pred_pairs, target_pairs, uncertainty_pairs = metric.evaluate(target_span, pred, tgt_tokens, region_pred, region_label, cover_flag, token_uncertainty, region_uncertainty, predict_mode=True)
            
            word_start_index = 8  # 2 + 2 + 4
            entity_positions = {}
            for i in range(pred.size(0)):  
                entity_positions[i] = {}
                ps = pred[i].tolist()
                k = 0
                cur_pair = []
                cur_positions = []
                
                while k < len(ps)-2:
                    if ps[k] < metric.word_start_index:  
                        if len(cur_pair) > 0 and all([cur_pair[j]<cur_pair[j+1] for j in range(len(cur_pair)-1)]):
                            entity_positions[i][tuple(cur_pair)] = k 
                        cur_pair = []
                        cur_positions = []
                        k += 2
                    else:  
                        cur_pair.append(ps[k])
                        cur_positions.append(k)
                        k += 1
                
                if len(cur_pair) > 0 and all([cur_pair[j]<cur_pair[j+1] for j in range(len(cur_pair)-1)]):
                    entity_positions[i][tuple(cur_pair)] = k  
            
            raw_words = batch_y['raw_words']
            assert len(pred_pairs) == len(target_pairs)
            for i in range(len(pred_pairs)):
                res_json = {
                    "tokens": raw_words[i],
                    "pre_entities": [],
                    "gt_entities": [],
                    "image_id": str(batch_x['img_id'][i]),
                    "candidate_regions": batch_x['candidate_regions'][i].cpu().numpy().tolist(),
                }
                cur_src_token = src_tokens[i].cpu().numpy().tolist()
                fw.write(' '.join(raw_words[i])+'\n')
                fw.write('Pred: ')
                for (k,v),(k1,v1) in zip(pred_pairs[i].items(), uncertainty_pairs[i].items()):
                    entity_span_ind_list =[]
                    for kk in k:
                        entity_span_ind_list.append(cur_src_token[kk-word_start_index])
                    entity_span = tokenizer.decode(entity_span_ind_list)

                    entity_span_unc_list = []
                    for kk1 in k1:
                        entity_span_unc_list.append(kk1)
                    original_start_idx = None
                    original_end_idx = None
                    
                    if len(k) > 0:
                        src_start_idx = k[0] - word_start_index
                        src_end_idx = k[-1] - word_start_index

                        if 'bpes_to_raw_index' in batch_x and src_start_idx < len(batch_x['bpes_to_raw_index'][i]):
                            bpes_to_raw_mapping = batch_x['bpes_to_raw_index'][i].cpu().tolist()
                            if 0 <= src_start_idx < len(bpes_to_raw_mapping):
                                original_start_idx = bpes_to_raw_mapping[src_start_idx]
                            if 0 <= src_end_idx < len(bpes_to_raw_mapping):
                                original_end_idx = bpes_to_raw_mapping[src_end_idx]
                    original_entity_text = None
                    if original_start_idx is not None and original_end_idx is not None and original_start_idx >= 0 and original_end_idx >= 0:
                        if original_start_idx < len(raw_words[i]) and original_end_idx < len(raw_words[i]):
                            original_entity_text = ' '.join(raw_words[i][original_start_idx:original_end_idx+1])
                    
                    display_entity = original_entity_text if original_entity_text else entity_span
                    
                    region_pred, entity_type_ind = v
                    region_unc, entity_type_unc = v1
                    try:
                        entity_type = ids2label[entity_type_ind[0]]
                        
                        for region_item in region_pred:
                            res_json['pre_entities'].append({
                                "entity_type": spt_to_type[entity_type],
                                "phrase": display_entity,
                                "span_uncertainty": round(sum(entity_span_unc_list), 2),
                                "region_uncertainty": round(sum(region_unc), 2),
                                "type_uncertainty": round(entity_type_unc[0],2),
                                "start": original_start_idx,
                                "end": original_end_idx,
                                "bpe_start_idx": src_start_idx if 'src_start_idx' in locals() else None,
                                "bpe_end_idx": src_end_idx if 'src_end_idx' in locals() else None,
                                "region": region_item + 1 if region_item < args.box_num else 0,
                                "region_box": batch_x['candidate_regions'][i][region_item].cpu().numpy().tolist() if region_item < args.box_num else []
                            })
                        fw.write('('+display_entity+' , '+ str(region_pred)+' , '+entity_type+' ) ')
                    except:
                        continue

                fw.write('\n')
                fw.write(' UNC : ')
                for k,v in uncertainty_pairs[i].items():
                    entity_span_unc_list = []
                    for kk in k:
                        entity_span_unc_list.append(kk)

                    region_unc, entity_type_unc = v
                    fw.write('(' + str([round(i,2) for i in entity_span_unc_list]) + ' , ' + str(
                        [round(i,2) for i in region_unc]) + ' , ' + str(round(entity_type_unc[0],2))  + ' ) ')
                    

                fw.write('\n')
                fw.write(' GT : ')
                for k,v in target_pairs[i].items():
                    entity_span_ind_list =[]
                    for kk in k:
                        entity_span_ind_list.append(cur_src_token[kk-word_start_index])
                    entity_span = tokenizer.decode(entity_span_ind_list)
        
                    original_start_idx = None
                    original_end_idx = None
                    

                    if len(k) > 0:
                        src_start_idx = k[0] - word_start_index
                        src_end_idx = k[-1] - word_start_index
                        

                        if 'bpes_to_raw_index' in batch_x and src_start_idx < len(batch_x['bpes_to_raw_index'][i]):
                            bpes_to_raw_mapping = batch_x['bpes_to_raw_index'][i].cpu().tolist()

                            if 0 <= src_start_idx < len(bpes_to_raw_mapping):
                                original_start_idx = bpes_to_raw_mapping[src_start_idx]
                            if 0 <= src_end_idx < len(bpes_to_raw_mapping):
                                original_end_idx = bpes_to_raw_mapping[src_end_idx]
                    

                    original_entity_text = None
                    if original_start_idx is not None and original_end_idx is not None and original_start_idx >= 0 and original_end_idx >= 0:
                        if original_start_idx < len(raw_words[i]) and original_end_idx < len(raw_words[i]):
                            original_entity_text = ' '.join(raw_words[i][original_start_idx:original_end_idx+1])
                    
                    display_entity = original_entity_text if original_entity_text else entity_span
                    
                    region_pred, entity_type_ind = v
                    entity_type = ids2label[entity_type_ind[0]]
                    for region_item in region_pred:
                        res_json['gt_entities'].append({
                            "entity_type": spt_to_type[entity_type],
                            "phrase": display_entity,
                            "start": original_start_idx,
                            "end": original_end_idx,
                            "bpe_start_idx": src_start_idx if 'src_start_idx' in locals() else None,
                            "bpe_end_idx": src_end_idx if 'src_end_idx' in locals() else None,
                            "region": region_item + 1 if region_item < args.box_num else 0,
                            "region_box": batch_x['candidate_regions'][i][
                                region_item].cpu().numpy().tolist() if region_item < args.box_num else []
                        })
                    fw.write('('+display_entity+' , '+ str(region_pred)+' , '+entity_type+' ) ')
                fw.write('\n\n')
                res_json_list.append(res_json)
        res = metric.get_metric()  
        fw.write(str(res))
        print(len(res_json_list))
        print(len(eval_data))
        assert len(res_json_list) == len(eval_data)
    return res, res_json_list





ids2label = {2+i:l for i,l in enumerate(mapping2id.keys())}
model.eval()
test_res, res_json = Predict(args,eval_data=test_dataset, model=model, device=device, metric = metric,tokenizer=tokenizer,ids2label=ids2label)
test_f = test_res['f']
print("test: "+str(test_res))
# 将结果写入JSON文件
output_dir = '/'.join(model_path[:-1])
output_filename = 'pred_' + model_path[-1] + '.json'
output_path = os.path.join(output_dir, output_filename)
os.makedirs(output_dir, exist_ok=True)
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(res_json, f, ensure_ascii=False, indent=4)
