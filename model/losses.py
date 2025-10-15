# from fastNLP import LossBase
import torch.nn.functional as F
from fastNLP import seq_len_to_mask
import torch


def evidence_trans(y, mode):
    if mode == 'softplus':
        return F.softplus(y)
    elif mode == 'exp':
        return torch.exp(y)
    elif mode == 'relu':
        return F.relu(y)
    elif mode == 'softmax':
        classifier = torch.nn.Softmax(dim=-1)
        return classifier(y)

def get_one_hot(label, num_classes, off_value, on_value, device):
    size = list(label.size())
    size.append(num_classes)
    
    label_flat = label.view(-1)
    valid_mask = (label_flat >= 0)
    valid_label = label_flat[valid_mask]
    
    ones = torch.sparse.torch.eye(num_classes) * on_value
    
    if valid_label.size(0) > 0:  
        ones = ones.index_select(0, valid_label.cpu())
    ones += off_value
    result = ones.new_full((label_flat.size(0), num_classes), off_value)
    if valid_label.size(0) > 0:
        result[valid_mask] = ones
    
    result = result.view(*size)
    return result.to(device)

def kl_divergence(alpha, beta):
    S_alpha = torch.sum(alpha, dim=-1, keepdim=False)
    S_beta = torch.sum(beta, dim=-1, keepdim=False)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=-1, keepdim=False)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=-1, keepdim=False) - torch.lgamma(S_beta)
    dg0 = torch.digamma(torch.sum(alpha, dim=-1, keepdim=True))
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=-1, keepdim=False) + lnB + lnB_uni
    return kl

def compute_edl_loss(pred, target, mask, etrans_func='softplus', with_kl=True, 
                   annealing_coef=1.0, num_classes=None):
    """
    evidential loss
    """
    device = pred.device
    
    # 获取词汇表大小
    if num_classes is None:
        num_classes = pred.size(1)  # [batch_size, vocab_size, seq_len]
    batch_size, seq_len = target.size()
    flat_target = target.view(-1)
    flat_mask = mask.view(-1)
    valid_indices = (flat_target != -100) & (flat_mask > 0)
    valid_targets = flat_target[valid_indices]
    
    if valid_targets.numel() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    flat_pred = pred.permute(0, 2, 1).contiguous().view(-1, num_classes)
    valid_pred = flat_pred[valid_indices]
    evidence = evidence_trans(valid_pred, etrans_func)
    alpha = evidence + 1
    
    S = torch.sum(alpha, dim=1, keepdim=True)

    target_indices = valid_targets.unsqueeze(1)
    target_alpha = torch.gather(alpha, 1, target_indices)
    digamma_sum = torch.digamma(S)
    digamma_target = torch.digamma(target_alpha)
    A = digamma_sum - digamma_target
    
    loss = A.mean()
    
    if with_kl:
        non_target_mask = torch.ones_like(alpha).scatter_(1, target_indices, 0)
        S_non_target = S - target_alpha
        target_term = torch.lgamma(target_alpha).sum(1)
        non_target_term = (torch.lgamma(alpha) * non_target_mask).sum(1)
        
        B = (torch.lgamma(S) - target_term - non_target_term) * annealing_coef
        loss = loss + B.mean()
    
    return loss

def get_loss(tgt_tokens, tgt_seq_len, pred, region_pred, region_label, use_kl=True, 
             use_edl=False, etrans_func='softplus', annealing_coef=1.0, with_kl=True, triplet_loss=None, triplet_weight=1.0):
    tgt_seq_len = tgt_seq_len - 1  ### 不算开始符号
    mask = seq_len_to_mask(tgt_seq_len, max_len=tgt_tokens.size(1) - 1).eq(0)
    tgt_tokens = tgt_tokens[:, 1:].masked_fill(mask, -100)   ### 处理之后没有开始0，[  57,   58,   59,   60,    2,    1, -100, -100]
    if use_edl:
        valid_mask = (tgt_tokens != -100).float()
        loss = compute_edl_loss(pred.transpose(1, 2), tgt_tokens, valid_mask, 
                               etrans_func=etrans_func, with_kl=with_kl,
                               annealing_coef=annealing_coef)
    else:
        loss = F.cross_entropy(target=tgt_tokens, input=pred.transpose(1, 2))
    region_mask = region_label[:,:,:-1].sum(dim=-1).gt(0)   ## only for related 
    if region_pred is not None and region_mask.sum()!=0:   
        if use_kl:
            bbox_num = region_pred.size(-1)
            target = region_label[region_mask][:, :-1]
            input = F.log_softmax(region_pred)
            region_loss = F.kl_div(input=input, target=target, reduction='batchmean')
        else:
            region_label = region_label[region_mask][:,:-1]
            pos_tag = region_label.new_full(region_label.size(),fill_value = 1.)
            neg_tag = region_label.new_full(region_label.size(),fill_value = 0.)
            BCE_target = torch.where(region_label > 0,pos_tag,neg_tag)
            bbox_num = region_pred.size(-1)
            sample_weight = region_pred.new_full((bbox_num,),fill_value=1.)
            region_loss = F.binary_cross_entropy_with_logits(region_pred, target=BCE_target, weight=sample_weight)
    else:
        region_loss = torch.tensor(0.,requires_grad=True).to(loss.device)
    if triplet_loss is not None:
        total_loss = loss + region_loss + triplet_weight * triplet_loss
    else:
        total_loss = loss + region_loss
    return total_loss, region_loss

def compute_uncertainty(logits, method='exp', src_len=None, label_start_id=None, src_start_index=None):
    """
    计算预测的不确定性
    
    Args:
        logits: [batch_size, seq_len, vocab_size] 预测logits
        method: evidence计算方法
        src_len: 源文本长度，用于计算指针空间大小
        label_start_id: 标签起始ID，通常为2
        src_start_index: 源文本指针起始索引
    
    Returns:
        uncertainty: [batch_size, seq_len] 每个位置的不确定性值
    """
    batch_size, seq_len, vocab_size = logits.shape
    
    if method == 'softmax':
        probs = F.softmax(logits, dim=-1)
        uncertainty = 1.0 - torch.max(probs, dim=-1)[0]
        id = probs.argmax(dim=-1, keepdim=True)
        return uncertainty.cpu().numpy().tolist(), id.cpu().numpy().tolist()

    uncertainty = torch.zeros((batch_size, seq_len), dtype=torch.float32, device=logits.device)

    pred_classes = torch.argmax(logits, dim=-1)  # [batch_size, seq_len]
    if src_start_index is None:
        evidence = evidence_trans(logits, mode=method)
        alpha = evidence + 1
        S = torch.sum(alpha, dim=-1)  # [batch_size, seq_len]
        return vocab_size / S

    eos_mask = pred_classes.eq(1)
    tag_mask = (pred_classes >= 2) & (pred_classes < src_start_index)
    pointer_mask = pred_classes >= src_start_index
    
    if eos_mask.any():
        eos_indices = torch.nonzero(eos_mask, as_tuple=True)
        for i, j in zip(*eos_indices):
            eos_evidence = evidence_trans(logits[i, j, 1:2].unsqueeze(0), mode=method)
            eos_alpha = eos_evidence + 1
            eos_S = torch.sum(eos_alpha)
            uncertainty[i, j] = 1.0 / eos_S
    
    #  标签空间 (索引2到src_start_index-1)
    if tag_mask.any():
        num_tags = src_start_index - 2  
        tag_indices = torch.nonzero(tag_mask, as_tuple=True)
        for i, j in zip(*tag_indices):
            tag_evidence = evidence_trans(logits[i, j, 2:src_start_index].unsqueeze(0), mode=method)
            tag_alpha = tag_evidence + 1
            tag_S = torch.sum(tag_alpha)
            uncertainty[i, j] = num_tags / tag_S
    
    #  指针空间 (索引src_start_index及以后)
    if pointer_mask.any():
        pointer_indices = torch.nonzero(pointer_mask, as_tuple=True)
        for i, j in zip(*pointer_indices):
            sample_src_len = 0
            if src_len is None:
                sample_src_len = vocab_size - src_start_index
            else:
                sample_src_len = src_len[i].item() if isinstance(src_len, torch.Tensor) else src_len[i]
            pointer_evidence = evidence_trans(logits[i, j, src_start_index:].unsqueeze(0), mode=method)
            pointer_alpha = pointer_evidence + 1
            pointer_S = torch.sum(pointer_alpha)
            uncertainty[i, j] = sample_src_len / pointer_S
    
    return uncertainty

