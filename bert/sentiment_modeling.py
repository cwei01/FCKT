from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

import math
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from bert.modeling import BertModel, BERTLayerNorm
import torch
from bert.dynamic_rnn import DynamicLSTM
import torch.nn.functional as F
cuda=torch.device('cuda')

def flatten(x):
    if len(x.size()) == 2:
        batch_size = x.size()[0]
        seq_length = x.size()[1]
        return x.view([batch_size * seq_length])
    elif len(x.size()) == 3:
        batch_size = x.size()[0]
        seq_length = x.size()[1]
        hidden_size = x.size()[2]
        return x.view([batch_size * seq_length, hidden_size])
    else:
        raise Exception()

def reconstruct(x, ref):
    if len(x.size()) == 1:
        batch_size = ref.size()[0]
        turn_num = ref.size()[1]
        return x.view([batch_size, turn_num])
    elif len(x.size()) == 2:
        batch_size = ref.size()[0]
        turn_num = ref.size()[1]
        sequence_length = x.size()[1]
        return x.view([batch_size, turn_num, sequence_length])
    else:
        raise Exception()

def flatten_emb_by_sentence(emb, emb_mask):
    batch_size = emb.size()[0]
    seq_length = emb.size()[1]
    flat_emb = flatten(emb)
    flat_emb_mask = emb_mask.view([batch_size * seq_length])
    return flat_emb[flat_emb_mask.nonzero().squeeze(), :]

def get_span_representation(span_starts, span_ends, input, input_mask):
    '''
    :param span_starts: [N, M]
    :param span_ends: [N, M]
    :param input: [N, L, D]
    :param input_mask: [N, L]
    :return: [N*M, JR, D], [N*M, JR]
    '''
    #print(input.size())
    input_mask = input_mask.to(dtype=span_starts.dtype)  # fp16 compatibility
    input_len = torch.sum(input_mask, dim=-1) # [N]
    word_offset = torch.cumsum(input_len, dim=0) # [N]
    word_offset -= input_len

    span_starts_offset = span_starts + word_offset.unsqueeze(1)
    span_ends_offset = span_ends + word_offset.unsqueeze(1)

    span_starts_offset = span_starts_offset.view([-1])  # [N*M]
    span_ends_offset = span_ends_offset.view([-1])

    span_width = span_ends_offset - span_starts_offset + 1
    JR = torch.max(span_width)

    context_outputs = flatten_emb_by_sentence(input, input_mask)  # [<N*L, D]
    text_length = context_outputs.size()[0]

    span_indices = torch.arange(JR).unsqueeze(0).to(span_starts_offset.device) + span_starts_offset.unsqueeze(1)  # [N*M, JR]
    span_indices = torch.min(span_indices, (text_length - 1)*torch.ones_like(span_indices))
    span_text_emb = context_outputs[span_indices, :]    # [N*M, JR, D]

    row_vector = torch.arange(JR).to(span_width.device)
    span_mask = row_vector < span_width.unsqueeze(-1)   # [N*M, JR]
    return span_text_emb, span_mask

def get_self_att_representation(input, input_score, input_mask):
    '''
    :param input: [N, L, D]
    :param input_score: [N, L]
    :param input_mask: [N, L]
    :return: [N, D]
    '''
    input_mask = input_mask.to(dtype=input_score.dtype)  # fp16 compatibility
    input_mask = (1.0 - input_mask) * -10000.0
    input_score = input_score + input_mask
    input_prob = nn.Softmax(dim=-1)(input_score)
    input_prob = input_prob.unsqueeze(-1)
    output = torch.sum(input_prob * input, dim=1)
    return output

def distant_cross_entropy(logits, positions, mask=None):
    '''
    :param logits: [N, L]
    :param positions: [N, L]
    :param mask: [N]
    '''
    log_softmax = nn.LogSoftmax(dim=-1)
    log_probs = log_softmax(logits)
    if mask is not None:
        loss = -1 * torch.mean(torch.sum(positions.to(dtype=log_probs.dtype) * log_probs, dim=-1) /
                               (torch.sum(positions.to(dtype=log_probs.dtype), dim=-1) + mask.to(dtype=log_probs.dtype)))
    else:
        loss = -1 * torch.mean(torch.sum(positions.to(dtype=log_probs.dtype) * log_probs, dim=-1) /
                               torch.sum(positions.to(dtype=log_probs.dtype), dim=-1))
    return loss

def pad_sequence(sequence, length):
    while len(sequence) < length:
        sequence.append(0)
    return sequence

def convert_crf_output(outputs, sequence_length, device):
    predictions = []
    for output in outputs:
        pred = pad_sequence(output[0], sequence_length)
        predictions.append(torch.tensor(pred, dtype=torch.long))
    predictions = torch.stack(predictions, dim=0)
    if device is not None:
        predictions = predictions.to(device)
    return predictions

class MultiNonLinearClassifier(nn.Module):
    def __init__(self, hidden_size, num_label, dropout_rate):
        super(MultiNonLinearClassifier, self).__init__()
        self.num_label = num_label
        self.classifier1 = nn.Linear(hidden_size, int(hidden_size / 2))
        self.classifier2 = nn.Linear(int(hidden_size/2), num_label)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features):
        features_output1 = self.classifier1(input_features)
        features_output1 = nn.ReLU()(features_output1)
        features_output1 = self.dropout(features_output1)
        features_output2 = self.classifier2(features_output1)
        return features_output2

def contrastive_loss(sequence_output, start_positions, end_positions, temperature=0.07):
    """
    Compute the contrastive loss to maximize the similarity of positive pairs and minimize the similarity of negative pairs.

    :param sequence_output: [batch_size, seq_len, hidden_dim], embedding representation of each token.
    :param start_positions: [batch_size, seq_len], annotated start positions (positive samples).
    :param end_positions: [batch_size, seq_len], annotated end positions (positive samples).
    :param temperature: Temperature coefficient \( \tau \) for contrastive learning.
    :return: Computed InfoNCE contrastive loss.
    """
    batch_size, seq_len, hidden_dim = sequence_output.size()

    # Step 1: Compute pairwise similarity of token embeddings
    start_extend = sequence_output.unsqueeze(2).expand(-1, -1, seq_len, -1)  # [batch_size, seq_len, seq_len, hidden_dim]
    end_extend = sequence_output.unsqueeze(1).expand(-1, seq_len, -1, -1)    # [batch_size, seq_len, seq_len, hidden_dim]

    # Compute cosine similarity between token pairs
    span_similarity = F.cosine_similarity(start_extend, end_extend, dim=-1)  # [batch_size, seq_len, seq_len]

    # Step 2: Construct positive sample pairs
    start_positions_extend = start_positions.unsqueeze(2).expand(-1, -1, seq_len)
    end_positions_extend = end_positions.unsqueeze(1).expand(-1, seq_len, -1)
    positive_mask = torch.mul(start_positions_extend, end_positions_extend)  # [batch_size, seq_len, seq_len], mask for positive pairs

    # Step 3: Construct negative sample pairs
    # Type 1 negative samples: positive start positions + other end positions
    neg_mask_1 = torch.mul(start_positions_extend, 1 - end_positions_extend)  # [batch_size, seq_len, seq_len]
    # Type 2 negative samples: positive end positions + other start positions
    neg_mask_2 = torch.mul(1 - start_positions_extend, end_positions_extend)  # [batch_size, seq_len, seq_len]

    # Step 4: Randomly select one negative sample from each type
    def select_random_negative(mask):
        """
        Randomly select one negative sample from the given mask.
        :param mask: [batch_size, seq_len, seq_len], negative sample mask.
        :return: Mask with a single randomly selected negative sample.
        """
        negative_indices = mask.nonzero(as_tuple=False)  # Get all positions of negative samples
        if negative_indices.size(0) > 0:  # If negative samples exist
            rand_idx = torch.randint(0, negative_indices.size(0), (1,)).item()  # Randomly select one index
            selected_neg_mask = torch.zeros_like(mask)
            selected_neg_mask[negative_indices[rand_idx][0], negative_indices[rand_idx][1], negative_indices[rand_idx][2]] = 1.0
            return selected_neg_mask
        else:
            return torch.zeros_like(mask)  # Return empty mask if no negative samples

    # Select one negative sample from each type
    selected_neg_mask_1 = select_random_negative(neg_mask_1)
    selected_neg_mask_2 = select_random_negative(neg_mask_2)

    # Combine the two negative sample masks
    negative_mask = torch.clamp(selected_neg_mask_1 + selected_neg_mask_2, max=1)  # [batch_size, seq_len, seq_len]

    # Step 5: Compute similarity for positive and negative samples
    # Similarity for positive samples
    positive_similarity = torch.exp(span_similarity / temperature) * positive_mask  # [batch_size, seq_len, seq_len]
    
    # Similarity for negative samples
    negative_similarity = torch.exp(span_similarity / temperature) * negative_mask  # [batch_size, seq_len, seq_len]

    # Step 6: Compute InfoNCE loss
    numerator = positive_similarity.sum(dim=(1, 2))  # Sum of similarities for positive samples [batch_size]
    denominator = (positive_similarity + negative_similarity).sum(dim=(1, 2))  # Sum of similarities for all samples [batch_size]
    contrastive_loss = -torch.log(numerator / denominator).mean()  # Final loss computation

    return contrastive_loss
    
class BCEFocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.6, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):
        pt = torch.sigmoid(_input)
        # pt = _input
        alpha = self.alpha
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss
class BertForSpanAspectExtraction(nn.Module):

    def __init__(self, config):
        super(BertForSpanAspectExtraction, self).__init__()
        self.bert = BertModel(config)

        self.start_outputs = nn.Linear(config.hidden_size, 1)
        self.end_outputs = nn.Linear(config.hidden_size, 1)
        self.span_outputs=nn.Linear(2*config.hidden_size,1)
        self.span_embedding = MultiNonLinearClassifier(config.hidden_size * 2, 1, 0.1)  # 0.1 dropout
        self.activation_sigmoid = nn.Sigmoid()
        self.activation_softmax = nn.Softmax()

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()
        self.apply(init_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, start_positions=None, end_positions=None, weight_start=None,
                weight_end=None, weight_span=None,weight_bias=None):
        all_encoder_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
        sequence_output = all_encoder_layers[-1]
        batch_size, seq_len, hid_size = sequence_output.size()
        start_logits = self.start_outputs(sequence_output) # [batch_size, seq_len, 1]
        end_logits = self.end_outputs(sequence_output) # [batch_size, seq_len, 1]
        start_logits = torch.squeeze(start_logits)
        end_logits = torch.squeeze(end_logits)

        start_extend = sequence_output.unsqueeze(2).expand(-1, -1, seq_len, -1) # [batch_size , seq_len ,seq_len, hidden_dim]
        end_extend = sequence_output.unsqueeze(1).expand(-1, seq_len, -1, -1) #  [batch_size , seq_len ,seq_len, hidden_dim]
        span_matrix = torch.cat([start_extend, end_extend], 3) # batch x seq_len x seq_len x 2*hidden_dim
        span_logits = self.span_embedding(span_matrix)  # batch x seq_len x seq_len x 1
        span_logits = torch.squeeze(span_logits)  # [batch , seq_len , seq_len]

        if start_positions is not None and end_positions is not None:
            start_positions_extend=start_positions.unsqueeze(2).expand(-1, -1, seq_len)
            end_positions_extend=end_positions.unsqueeze(1).expand(-1, seq_len, -1)
            span_positions = torch.mul(start_positions_extend,end_positions_extend)

            loss_fct = nn.BCELoss()
            start_logits = self.activation_softmax(start_logits)
            end_logits = self.activation_softmax(end_logits)  # [batch_size, seq_len]
            #loss_fct = BCEFocalLoss(gamma=2, alpha=0.1, reduction='elementwise_mean')
            start_loss = loss_fct(start_logits, start_positions.float())
            end_loss = loss_fct(end_logits, end_positions.float())

            #span_loss=BCEFocalLoss(gamma=0, alpha=weight_end, reduction='elementwise_mean')
            span_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight_bias])).to(cuda)
            span_loss = span_loss(span_logits.view(batch_size,-1), span_positions.view(batch_size, -1).float())
            

            #################################################################
            total_loss=weight_start*start_loss+weight_end*end_loss+weight_span*span_loss
            ################################################################

            return total_loss
        else:
            span_logits = torch.sigmoid(span_logits) # [batch , seq_len , seq_len]
            return start_logits, end_logits, span_logits

class BertForSpanAspectClassification(nn.Module):
    def __init__(self, config):
        super(BertForSpanAspectClassification, self).__init__()
        self.bert = BertModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.affine = nn.Linear(config.hidden_size, 1)
        self.classifier = nn.Linear(config.hidden_size, 5)

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()
        self.apply(init_weights)

    def forward(self, mode, attention_mask, input_ids=None, token_type_ids=None, span_starts=None, span_ends=None,
                labels=None, label_masks=None):
        '''
        :param input_ids: [N, L]
        :param token_type_ids: [N, L]
        :param attention_mask: [N, L]
        :param span_starts: [N, M]
        :param span_ends: [N, M]
        :param labels: [N, M]
        '''
        if mode == 'train':
            assert input_ids is not None and token_type_ids is not None
            all_encoder_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
            sequence_output = all_encoder_layers[-1]

            assert span_starts is not None and span_ends is not None and labels is not None
            span_output, span_mask = get_span_representation(span_starts, span_ends, sequence_output,
                                                             attention_mask)  # [N*M, JR, D], [N*M, JR]
            span_score = self.affine(span_output)
            span_score = span_score.squeeze(-1)  # [N*M, JR]
            span_pooled_output = get_self_att_representation(span_output, span_score, span_mask)  # [N*M, D]

            span_pooled_output = self.dense(span_pooled_output)
            span_pooled_output = self.activation(span_pooled_output)
            span_pooled_output = self.dropout(span_pooled_output)
            cls_logits = self.classifier(span_pooled_output)  # [N*M, 4]

            cls_loss_fct = CrossEntropyLoss(reduction='none')
            flat_cls_labels = flatten(labels)
            flat_label_masks = flatten(label_masks)
            loss = cls_loss_fct(cls_logits, flat_cls_labels)
            mean_loss = torch.sum(loss * flat_label_masks.to(dtype=loss.dtype)) / torch.sum(flat_label_masks.to(dtype=loss.dtype))
            return mean_loss

        elif mode == 'inference':
            assert input_ids is not None and token_type_ids is not None
            all_encoder_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
            sequence_output = all_encoder_layers[-1]

            assert span_starts is not None and span_ends is not None
            span_output, span_mask = get_span_representation(span_starts, span_ends, sequence_output,
                                                             attention_mask)  # [N*M, JR, D], [N*M, JR]
            span_score = self.affine(span_output)
            span_score = span_score.squeeze(-1)  # [N*M, JR]
            span_pooled_output = get_self_att_representation(span_output, span_score, span_mask)  # [N*M, D]

            span_pooled_output = self.dense(span_pooled_output)
            span_pooled_output = self.activation(span_pooled_output)
            span_pooled_output = self.dropout(span_pooled_output)
            cls_logits = self.classifier(span_pooled_output)  # [N*M, 4]
            return reconstruct(cls_logits, span_starts)

        else:
            raise Exception

class BertForJointSpanExtractAndClassification(nn.Module):
    def __init__(self, config, args):
        super(BertForJointSpanExtractAndClassification, self).__init__()
        self.bert = BertModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.start_outputs = nn.Linear(config.hidden_size, 1)
        self.end_outputs = nn.Linear(config.hidden_size, 1)
        self.unary_affine = nn.Linear(config.hidden_size, 1)
        self.binary_affine = nn.Linear(config.hidden_size, 2)
        self.dense = nn.Linear(config.hidden_size,config.hidden_size,config.hidden_size)
        self.span_embedding = MultiNonLinearClassifier(config.hidden_size* 2, 1, 0.1)
        self.activation_relu = nn.ReLU()
        self.activation_sigmoid=nn.Sigmoid()
        self.activation_softmax=nn.Softmax()
        self.classifier = nn.Linear(config.hidden_size,5)

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()
        self.apply(init_weights)

    def forward(self, mode, attention_mask, input_ids=None, token_type_ids=None, start_positions=None, end_positions=None,
                span_starts=None, span_ends=None, polarity_labels=None, label_masks=None, sequence_input=None,weight_start=None,weight_end=None,
                                                                           weight_span=None,weight_ac=None,use_expectation=None,):
        if mode == 'train':
            assert input_ids is not None and token_type_ids is not None
            all_encoder_layers,_= self.bert(input_ids, token_type_ids, attention_mask)
            sequence_output = all_encoder_layers[-1]  #[batch_size,seq_len,hidden_dim]

            assert start_positions is not None and end_positions is not None or \
            span_starts is not None and span_ends is not None and polarity_labels is not None \
                   and label_masks is not None and weight_start is not None and weight_end is not None and weight_span is not None  and use_expectation is not None
            # temp=torch.ones(sequence_output_commom.size(0),dtype=torch.int64,device=cuda)
            # seq_len=sequence_output_commom.size(1)*temp  #
            #
            # sequence_output_nerinit, (_, _) = self.ner_bigru(sequence_output_commom, seq_len)  # [batch_size,seq_len,D]
            # sequence_output_absainit, (_, _) = self.absa_bigru(sequence_output_commom,seq_len)  # [batch_size,seq_len,D]
            # layters_GRU  update
            # for i in range(layer_GRU-1):
            #     sequence_output_ner=self.activation_sigmoid(self.same_ner)*sequence_output_nerinit+self.activation_sigmoid(self.absa2ner)*sequence_output_absainit
            #     sequence_output_absa=self.activation_sigmoid(self.same_absa)*sequence_output_absainit+self.activation_sigmoid(self.ner2absa)*sequence_output_nerinit
            #     sequence_output_nerinit, (_, _) = self.ner_bi_GRU(sequence_output_ner, seq_len)
            #     sequence_output_absainit, (_, _) = self.absa_bi_GRU(sequence_output_absa, seq_len)
            batch_size, seq_len, hid_size = sequence_output.size()
            start_logits = self.start_outputs(sequence_output)  # [batch_size , seq_len , 1]
            end_logits = self.end_outputs(sequence_output)  # [batch_size , seq_len , 1]
            start_logits = torch.squeeze(start_logits)
            end_logits = torch.squeeze(end_logits)
            start_logits = self.activation_softmax(start_logits)
            end_logits = self.activation_softmax(end_logits)  #[batch_size , seq_len]
            
            info_loss = contrastive_loss(sequence_output,start_positions,end_positions,temperature=0.07)
            

            # start_extend = sequence_output.unsqueeze(2).expand(-1, -1, seq_len, -1)  # [batch_size ,seq_len ,seq_len,D]
            # end_extend = sequence_output.unsqueeze(1).expand(-1, seq_len, -1,-1)  # [batch_size ,seq_len ,seq_len,D]
            # span_matrix = torch.cat([start_extend, end_extend], 3)  # [batch_size , seq_len , seq_len , 2*D]
            # span_logits = self.span_embedding(span_matrix)  # [batch_size , seq_len , seq_len , 1]
            # span_logits = torch.squeeze(span_logits)  # [batch_size , seq_len , seq_len ]
            # #span_logits=torch.triu(span_logits)   #[batch_size , seq_len , seq_len ]

            # start_positions_extend=start_positions.unsqueeze(2).expand(-1, -1, seq_len)
            # end_positions_extend=end_positions.unsqueeze(1).expand(-1, seq_len, -1)
            # span_positions = torch.mul(start_positions_extend,end_positions_extend)
            # #span_positions=torch.triu(span_positions)

            loss_fct = nn.BCELoss()
            start_loss = loss_fct(start_logits, start_positions.float())
            end_loss = loss_fct(end_logits, end_positions.float())
            # span_loss_fct = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([100.])).to(cuda)
            # span_loss = span_loss_fct(span_logits.view(batch_size, -1), span_positions.view(batch_size, -1).float())
            ############Adjustable parameter################################5
            ae_loss=weight_start*start_loss+weight_end*end_loss+weight_span*info_loss

            if use_expectation:
                mask_distribution=[[0]*seq_len for _ in range(seq_len)]
                for i in range(seq_len):
                    for j in range(seq_len):
                        if i<=j and j-i<3:
                            mask_distribution[i][j]=1
                mask_distribution = torch.tensor(mask_distribution,dtype=torch.float,device=cuda) #[seq_len,seq_len]
                mask_distribution_temp = mask_distribution.unsqueeze(0).expand(batch_size, -1, -1).contiguous().view(-1) # [batch_size*seq_len*seq_len]
                nonzero_index = torch.nonzero(mask_distribution_temp)
                start_logits_extend=start_logits.unsqueeze(2).expand(-1, -1, seq_len)
                end_logits_extend=end_logits.unsqueeze(1).expand(-1, seq_len,-1)
                start_end_distribution_temp = start_logits_extend.mul(end_logits_extend).view(-1) # [batch_size*seq_len*seq_len]
                start_end_distribution = start_end_distribution_temp[nonzero_index].view(batch_size,-1) #
                start_end_distribution = torch.softmax(start_end_distribution,dim=-1)  # [batch_size,l*n]

                seq_index = [i for i in range(seq_len)] #[1,2,....,seq_len]
                seq_index_starts=torch.from_numpy(np.array(seq_index)).to(cuda)
                seq_index_end=torch.from_numpy(np.array(seq_index)).to(cuda) #[seq_len]
                s_i_starts_extend = seq_index_starts.unsqueeze(1).expand(-1,seq_len).contiguous().view(-1)
                s_i_ends_extend = seq_index_end.unsqueeze(0).expand(seq_len,-1).contiguous().view(-1) #[seq_len*seq_len]
                non0_index = torch.nonzero(mask_distribution.view(-1)) #[l*n,1]
                s_i_starts = s_i_starts_extend[non0_index]  #[l*n,1]
                s_i_ends = s_i_ends_extend[non0_index]
                copy_span_starts = s_i_starts.unsqueeze(0).expand(batch_size,-1,-1).contiguous().view(-1,1) # [batch_size*l*n,1]
                copy_span_ends = s_i_ends.unsqueeze(0).expand(batch_size,-1,-1).contiguous().view(-1,1)  # [batch_size*l*n,1]

                seq_out_init=sequence_output.unsqueeze(1).expand(-1, non0_index.size(0),-1, -1) #[batch_size,l*n,seq_len,D]
                seq_out=seq_out_init.contiguous().view(-1,seq_len,hid_size) #[batch_size*n*l,seq_len,D]
                att_mask_init=attention_mask.unsqueeze(1).expand(-1, non0_index.size(0), -1)
                att_mask=att_mask_init.contiguous().view(-1,seq_len)   #[batch_size*n*l,seq_len]
                span_output, span_mask = get_span_representation(copy_span_starts, copy_span_ends, seq_out,
                                                                 att_mask)
                span_score = self.unary_affine(span_output)
                span_score = span_score.squeeze(-1)  # [N*l*n, JR]
                span_pooled_output = get_self_att_representation(span_output, span_score, span_mask)  # [N*l*n, D]
                span_pooled_output = self.dense(span_pooled_output)
                span_pooled_output = self.activation_relu(span_pooled_output)
                span_pooled_output = self.dropout(span_pooled_output)
                ac_logits = self.classifier(span_pooled_output)  # [batch_size*l*n, 5]

                start_end_distribution=start_end_distribution.view(-1) #[batch_size*l*n]
                start_end_distribution=start_end_distribution.unsqueeze(1).expand(-1, 5)   #[batch_size*l*n, 5]
                class_matrix = ac_logits.mul(start_end_distribution)  #[batch_size*l*n, 5]
                class_matrix=class_matrix.view(batch_size,-1,5)  #[batch_size,l*n, 5]
                class_matrix_1= torch.sum(class_matrix,dim=1)  #[batch_size, 5]
                ac_loss_fct = CrossEntropyLoss(reduction='none')
                polarity_labels_ture = torch.index_select(polarity_labels, dim=1, index=torch.tensor([0]).to(cuda))
                label_masks_ture = torch.index_select(label_masks, dim=1, index=torch.tensor([0]).to(cuda))
                flat_polarity_labels = flatten(polarity_labels_ture)
                flat_label_masks = flatten(label_masks_ture).to(dtype=class_matrix_1.dtype)
                ac_loss = ac_loss_fct(class_matrix_1, flat_polarity_labels)
                ac_loss = torch.sum(flat_label_masks * ac_loss) / flat_label_masks.sum()
            else:
                span_starts_ture = torch.index_select(span_starts, dim=1, index=torch.tensor([0]).to(cuda)) #[batch_size,1]
                span_ends_ture = torch.index_select(span_ends, dim=1, index=torch.tensor([0]).to(cuda))  #[batch_size,1]
                span_output, span_mask = get_span_representation(span_starts_ture, span_ends_ture, sequence_output,
                                                                 attention_mask)  # [N*1, JR, D], [N*1, JR]
                span_score = self.unary_affine(span_output)
                span_score = span_score.squeeze(-1)  # [N*1, JR]
                span_pooled_output = get_self_att_representation(span_output, span_score, span_mask)  # [N*1, D]
                span_pooled_output = self.dense(span_pooled_output)
                span_pooled_output = self.activation_relu(span_pooled_output)
                span_pooled_output = self.dropout(span_pooled_output)
                ac_logits = self.classifier(span_pooled_output)  # [batch_size, 5]

                ac_loss_fct = CrossEntropyLoss(reduction='none')
                polarity_labels_ture = torch.index_select(polarity_labels, dim=1, index=torch.tensor([0]).to(cuda))
                label_masks_ture = torch.index_select(label_masks, dim=1, index=torch.tensor([0]).to(cuda))
                flat_polarity_labels = flatten(polarity_labels_ture)
                flat_label_masks = flatten(label_masks_ture).to(dtype=ac_logits.dtype)
                ac_loss = ac_loss_fct(ac_logits, flat_polarity_labels)
                ac_loss = torch.sum(flat_label_masks * ac_loss) / flat_label_masks.sum()

            return ae_loss + weight_ac*ac_loss

        elif mode == 'extract_inference':
            assert input_ids is not None and token_type_ids is not None
            all_encoder_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
            sequence_output = all_encoder_layers[-1]  # [batch_size , seq_len , hidden_dim ]

            # temp = torch.ones(sequence_output_common.size(0), dtype=torch.int64, device=cuda)
            # seq_len = sequence_output_common.size(1) * temp
            # sequence_output_nerinit, (_, _) = self.ner_bigru(sequence_output_common, seq_len)  # [batch_size , seq_len , D ]
            # sequence_output_absainit, (_, _) = self.absa_bigru(sequence_output_common, seq_len)
            # for i in range(layer_GRU-1):
            #     sequence_output_ner = self.activation_sigmoid(
            #         self.same_ner) * sequence_output_nerinit + self.activation_sigmoid(
            #         self.absa2ner) * sequence_output_absainit
            #     sequence_output_absa = self.activation_sigmoid(
            #         self.same_absa) * sequence_output_absainit + self.activation_sigmoid(
            #         self.ner2absa) * sequence_output_nerinit
            #     # sequence_output_ner=sequence_output_nerinit
            #     # sequence_output_absa=sequence_output_absainit
            #     sequence_output_nerinit, (_, _)=self.ner_bi_GRU(sequence_output_ner, seq_len)
            #     sequence_output_absainit, (_, _)=self.absa_bi_GRU(sequence_output_absa, seq_len)

            batch_size, seq_len, hid_size = sequence_output.size()
            start_logits = self.start_outputs(sequence_output)  # [batch_size , seq_len , 1]
            end_logits = self.end_outputs(sequence_output)  # [batch_size , seq_len , 1]
            start_logits = torch.squeeze(start_logits)
            end_logits = torch.squeeze(end_logits)
            #start_logits = self.activation_softmax(start_logits)
            #end_logits = self.activation_softmax(end_logits)

            # start_extend = sequence_output.unsqueeze(2).expand(-1, -1, seq_len,-1)  # [batchsize , seq_len ,seq_len,hidden_dim]
            # end_extend = sequence_output.unsqueeze(1).expand(-1, seq_len, -1,-1)  # [batchsize , seq_len ,seq_len,hidden_dim]
            # span_matrix = torch.cat([start_extend, end_extend], 3)  # [batch_size , seq_len , seq_len , 2*D]

            # span_logits = self.span_embedding(span_matrix)  # [batch_size , seq_len , seq_len , 1]
            # span_logits = torch.squeeze(span_logits)  # [batch_size , seq_len , seq_len]
            # span_logits = torch.sigmoid(span_logits)
            #span_logits = torch.triu(span_logits) # [batch_size , seq_len , seq_len]
            span_logits = torch.zeros((batch_size, seq_len, seq_len), device=start_logits.device)

            return start_logits, end_logits, span_logits, sequence_output

        elif mode == 'classify_inference':
            assert span_starts is not None and span_ends is not None and sequence_input is not None
            #span_starts_ture=torch.index_select(span_starts,dim=1,index=torch.tensor([0]).to(cuda))
            #span_ends_ture=torch.index_select(span_ends,dim=1,index=torch.tensor([0]).to(cuda))
            span_output, span_mask = get_span_representation(span_starts, span_ends, sequence_input,
                                                             attention_mask)  # [N*M, JR, D], [N*1, JR]
            span_score = self.unary_affine(span_output)
            span_score = span_score.squeeze(-1)  # [N*M, JR]
            span_pooled_output = get_self_att_representation(span_output, span_score, span_mask)  # [N*M, D]

            span_pooled_output = self.dense(span_pooled_output)
            span_pooled_output = self.activation_relu(span_pooled_output)
            span_pooled_output = self.dropout(span_pooled_output)
            ac_logits = self.classifier(span_pooled_output)  # [N*M, 5]

            return reconstruct(ac_logits, span_starts)

def distant_loss(start_logits, end_logits, start_positions=None, end_positions=None, mask=None):
    start_loss = distant_cross_entropy(start_logits, start_positions, mask)
    end_loss = distant_cross_entropy(end_logits, end_positions, mask)
    total_loss = (start_loss + end_loss) / 2
    return total_loss

