import torch
import torch.nn as nn
import torch.nn.functional as F

class Atten_Layers(nn.Module):

    def __init__(self, max_length = 512, num_heads = 8, d_model = 768):
        super().__init__()
        self.max_length = max_length #### this is the maximum length
        self.num_heads = num_heads
        self.d_model = d_model
        self.w_query = nn.Linear(self.d_model, int(self.d_model/self.num_heads))
        self.w_key = nn.Linear(self.d_model, int(self.d_model / self.num_heads))
        self.w_value = nn.Linear(self.d_model, int(self.d_model / self.num_heads))

    def forward(self, input_querys, input_key, input_values, masked = False):
        Q = self.w_query(input_querys) #### max_length, int(self.d_model/self.num_heads)
        K = self.w_key(input_key)
        V = self.w_value(input_values)

        attention_matrix = torch.bmm(Q, K.transpose(-1,-2))/torch.sqrt(int(self.d_model/self.num_heads)) #### max_length, max_length

        if masked:
            #### need to set the upper part of the attention_matrix to -inf, so softmax will not apply weights
            mask = torch.triu(torch.ones(attention_matrix.shape[-2], attention_matrix.shape[-1])).bool()
            attention_matrix = attention_matrix.masked_fill(mask, float('-inf'))

        attention_matrix = torch.bmm(F.softmax(attention_matrix, dim = -1), V)  ### attention is all you need formula (1)
        #### post the process, attention_matrix is (batch_size, max_length, int(self.d_model / self.num_heads))

        return attention_matrix

class Multi_Head_Layer(nn.Module):

    def __init__(self, max_length = 512, num_heads = 8, d_model = 768, d_output = None):
        super().__init__()
        if d_output is None:
            d_output = d_model
        self.linear = nn.Linear(int(d_model/num_heads)*num_heads, d_output) #### address the rare case that d_model cannot be fully divided by num_heads
        self.num_heads = num_heads
        self.d_model = d_model
        self.max_length = max_length
        self.multiheads = nn.ModuleList([])
        for _ in range(num_heads):

            scaled_doc_product_attention = Atten_Layers(max_length=512, num_heads=8, d_model=768)
            self.multiheads.append(scaled_doc_product_attention)

    def forward(self, input_querys, input_key, input_values, masked = False):
        multiheads_output = []

        for scaled_doc_product_attention in self.multiheads:
            multiheads_output.append(scaled_doc_product_attention(input_querys, input_key, input_values, masked ))

        #### concat the output
        new_output = torch.cat(multiheads_output, dim = -1) #### batch, max_length, int(self.d_model/self.num_heads)*num_heads
        new_output = self.linear(new_output) ### batch, max_length, d_output

        return new_output

class Bert_Base_Layer(nn.Module):
    #### single structure of the bert layer, there are N of it in the actual bert model

    def __init__(self, max_length = 512, num_heads = 8, d_model = 768, d_output = None, dropout = 0.1):
        super().__init__()
        self.multihead_attention = Multi_Head_Layer(max_length, num_heads, d_model, d_output)
        self.dropout = nn.Dropout(dropout)

        if d_output is None:
            d_output = d_model
        self.normalization = nn.LayerNorm(d_output)
        #### 2 linear feedforward layers with ReLU in between
        self.linear1 = nn.Linear(d_output, d_output*4)
        #### 4 is not explicitly mentioned in the paper, but it is in section 3.3 the last sentence
        self.linear2 = nn.Linear(d_output*4, d_output)

    def forward(self, input, masked = False):
        input_querys = input
        input_key = input
        input_values = input
        output1 = self.multihead_attention(input_querys, input_key, input_values, masked)

        #### there is a dropout directly adfter the multhhead attention output
        input_step2 = self.normalization(input+self.dropout(output1))
        output_step2 = self.linear1(input_step2)
        output_step2 = self.linear2(nn.ReLU(output_step2))

        output = self.normalization(input_step2+output_step2)

        return output

def generate_positional_embeddings(d_model, max_length, batch_size):
    '''
    based on d_model, generate the positional embedding for max_length
    :param d_model: embeddings dimension
    :param max_length: pos from 0 to max_length-1
    :param batch_size: batch size of the input
    :return: positional embeddings
    '''

    res = torch.empty(batch_size, max_length, d_model)

    for d in range(d_model):
        if d%2 == 0:
            res[:,:,d] = torch.sin(res.shape[1]/torch.power(10_000,d/d_model))
        else:
            #### even number
            res[:, :, d] = torch.cos(res.shape[1] / torch.power(10_000, (d-1) / d_model))

    return res

class Bert_Encoder(nn.Module):

    def __init__(self, num_vocab, max_length = 512, num_heads = 8, d_model = 768,
                 d_output = None, dropout = 0.1, batch_size = 1, num_attention_layers = 12):

        self.embedding = nn.Embedding(num_vocab, d_model)
        self.positional_embeddings = generate_positional_embeddings(d_model, max_length, batch_size)
        self.num_attention_layers = num_attention_layers
        self.atten_layers = nn.ModuleList([Bert_Base_Layer(max_length, num_heads, d_model, d_output, dropout)
                                           for _ in range(num_attention_layers)])

    def forward(self, input):
        #### input is token ID of size 512
        #### TODO: relax this requirement to allow flexible input size <= 512

        embedding_output = self.embedding(input)
        output = embedding_output + self.positional_embeddings #### this is fixed

        for atten_layer in self.atten_layers:
            output = atten_layer(output)

        return output














