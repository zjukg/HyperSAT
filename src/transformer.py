import torch
import torch.nn as nn
import math

class PrepareForMultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, bias: bool, use_node: bool, one_st_len: int) -> None:
        super().__init__()
        self.heads = heads
        self.use_node = use_node       
        self.one_st_len = one_st_len
        if self.use_node is True:
            self.layer_s=nn.Linear(hidden_dim,hidden_dim)
            self.layer_r=nn.Linear(hidden_dim,hidden_dim)
            self.layer_o=nn.Linear(hidden_dim,hidden_dim)
            self.layer_a=nn.Linear(hidden_dim,hidden_dim)
            self.layer_v=nn.Linear(hidden_dim,hidden_dim)
        else:
            self.linear = nn.Linear(hidden_dim, hidden_dim, bias=bias)


    def forward(self, x : torch.Tensor):
        shape = x.shape[:-1]

        if self.use_node is False:
            x = self.linear(x)
        else:
            device=x.device
            max_seq_len=x.size(1)
            st_num = max_seq_len//self.one_st_len
            mask_s = torch.tensor([1]+[0]*(self.one_st_len-1)).to(device).repeat(st_num)
            mask_r = torch.tensor([0,1]+[0]*(self.one_st_len-2)).to(device).repeat(st_num)
            mask_o = torch.tensor([0,0,1]+[0]*(self.one_st_len-3)).to(device).repeat(st_num)
            mask_a = torch.tensor([0,0,0]+[1,0]*int(((self.one_st_len-3)/2))).to(device).repeat(st_num)
            mask_v = torch.tensor([0,0,0]+[0,1]*int(((self.one_st_len-3)/2))).to(device).repeat(st_num)

            x_s=self.layer_s(torch.mul(x,mask_s[:,None].expand(-1,x.size(-1))))
            x_r=self.layer_r(torch.mul(x,mask_r[:,None].expand(-1,x.size(-1))))
            x_o=self.layer_o(torch.mul(x,mask_o[:,None].expand(-1,x.size(-1))))
            x_a=self.layer_a(torch.mul(x,mask_a[:,None].expand(-1,x.size(-1))))
            x_v=self.layer_v(torch.mul(x,mask_v[:,None].expand(-1,x.size(-1))))
                            
            x=(x_s+x_r+x_o+x_a+x_v) 
      
        return x.reshape(*shape, self.heads, -1)

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout_prob: float, use_edge: bool, remove_mask: bool, bias: bool, use_node: bool, one_stat_len:int) -> None:
        super().__init__()
        assert hidden_dim % heads == 0
        self.dim = hidden_dim // heads
        self.heads = heads
        self.query = PrepareForMultiHeadAttention(hidden_dim, heads, bias, use_node, one_stat_len)
        self.key = PrepareForMultiHeadAttention(hidden_dim, heads, bias, use_node, one_stat_len)
        self.value = PrepareForMultiHeadAttention(hidden_dim, heads, True, use_node, one_stat_len)
        self.pos = PrepareForMultiHeadAttention(hidden_dim, heads, True, use_node, one_stat_len)
        self.softmax = nn.Softmax(dim=-1)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.use_edge = use_edge
        self.remove_mask = remove_mask
        self.scale = 1 / math.sqrt(self.dim)
        # trasformer-xl
        self.r_w_bias = nn.Parameter(torch.Tensor(heads, self.dim)) # u
        self.r_r_bias = nn.Parameter(torch.Tensor(heads, self.dim)) # v

        self.kv_pair = False
        self.one_stat_len = one_stat_len
        self.pairlinear = nn.Linear(2*hidden_dim, hidden_dim)

    def get_mask(self, graph: torch.Tensor):
        return graph.unsqueeze(1).repeat(1, self.heads, 1, 1)

    def forward(self, *, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                        graph: torch.Tensor, edge_key: torch.Tensor, edge_value: torch.Tensor, edge_query: torch.Tensor):
        # query/key/value: (batch, seq_len, hidden_dim)
        # graph: (batch, kinds, query, key)
        shape = query.shape[:-1]
        query = self.query(query)   # (batch, seq_len, head, hidden)
        key = self.key(key)         # (batch, seq_len, head, hidden)
        value = self.value(value)   # (batch, seq_len, head, hidden)
        seq_len = query.size(1)

        if self.kv_pair:
            new_key = key.clone().reshape(shape,-1)
            statement_num = seq_len//self.one_stat_len
            main_indicator = torch.tensor([0,0,0],device=query.device)
            qual_indicator = torch.tensor([1,2],device=query.device).repeat(self.one_stat_len//2-1)
            seq_indicator = torch.cat((main_indicator,qual_indicator)).repeat(statement_num)
            seq_indicator = seq_indicator.repeat(shape[0],1)
            k_k = new_key[seq_indicator==1]
            k_v = new_key[seq_indicator==2]
            k_rt_pair = self.pairlinear(torch.cat((k_k,k_v),dim=-1))
            new_key[seq_indicator==1] = k_k+k_rt_pair
            new_key[seq_indicator==2] = k_v+k_rt_pair
            key = new_key.reshape(*shape, self.heads, -1)

        if self.use_edge is True:
            scores = torch.einsum("bqhd,bkhd->bhqk", query, key) + torch.einsum("bqhd,bqkd->bhqk", query, edge_key) + torch.einsum("bkqd,bkhd->bhqk", edge_query, key) + torch.einsum("bkqd,bqkd->bqk", edge_query, edge_key).unsqueeze(1)
            scores = scores * self.scale
            mask = self.get_mask(graph)
            if self.remove_mask is True:
                for i in range(3,seq_len,2):
                    if i==3:
                        mask[:,:,i:(i+2),(i+2):]=False
                    elif i==(seq_len-2):
                        mask[:,:,i:(i+2),3:i]=False
                    else:
                        mask[:,:,i:(i+2),(i+2):]=False
                        mask[:,:,i:(i+2),3:i]=False     
            scores = scores.masked_fill(mask == 0, -100000)
            attn = self.softmax(scores)
            attn = self.dropout(attn)
            x = torch.einsum("bhqk,bkhd->bqhd", attn, value) + torch.einsum("bhqk,bqkd->bqhd", attn, edge_value)
            x = x.reshape(*shape, -1)
        else:
            scores = torch.einsum("bqhd,bkhd->bhqk", query, key)
            scores *= self.scale
            mask = self.get_mask(graph)
            if self.remove_mask is True:
                for i in range(3,seq_len,2):
                    if i==3:
                        mask[:,:,i:(i+2),(i+2):]=False
                    elif i==(seq_len-2):
                        mask[:,:,i:(i+2),3:i]=False
                    else:
                        mask[:,:,i:(i+2),(i+2):]=False
                        mask[:,:,i:(i+2),3:i]=False  
            scores = scores.masked_fill(mask == 0, -100000)
            attn = self.softmax(scores)
            attn = self.dropout(attn)
            x = torch.einsum("bhqk,bkhd->bqhd", attn, value)
            x = x.reshape(*shape, -1)

        return self.output(x)  # (batch, query, hidden_dim)

class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation) -> None:
        super().__init__()
        act = None
        if activation == "gelu":
            act = nn.GELU()
        elif activation == "relu":
            act = nn.ReLU()
        elif activation == 'elu':
            act = nn.ELU()
        elif activation == 'tanh':
            act = nn.Tanh()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layer(x)
        
class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout_prob: float, activation: str, use_edge: bool, remove_mask: bool, use_node: bool, one_stat_len: int, bias=True, times=2) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(hidden_dim)
        self.attention = MultiHeadAttention(hidden_dim, heads, dropout_prob, use_edge, remove_mask, bias, use_node, one_stat_len)
        self.dropout = nn.Dropout(dropout_prob)
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = FeedForward(hidden_dim, hidden_dim * times, hidden_dim, activation)

    def forward(self, x: torch.Tensor, graph: torch.Tensor, edge_key: torch.Tensor, edge_value: torch.Tensor, edge_query: torch.Tensor):
        attn = self.attention(query=x, key=x, value=x, graph=graph, edge_key=edge_key, edge_value=edge_value, edge_query=edge_query)
        x = self.norm_attention(x + self.dropout(attn))
        ff = self.ffn(x)
        x = self.norm_ffn(x + self.dropout(ff))
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size: int, one_stat_len: int, neighbor_num: int, res_w:float, positional:bool, fact_index:bool, comp:bool, local_layers: int, hidden_dim: int,
            local_heads: int, local_dropout: float, decoder_activation: str, use_edge: bool, remove_mask: bool, use_node: bool, times=2, bias=True) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.one_st_len = one_stat_len
        self.neighbor_num = neighbor_num
        self.residual_w = res_w
        self.positional = positional
        self.apply_fact_index = fact_index
        self.comp = comp
        self.times = times
        self.layers = nn.ModuleList()
        for _ in range(local_layers):
            self.layers.append(TransformerLayer(hidden_dim, local_heads, local_dropout, decoder_activation, use_edge, remove_mask, use_node, one_stat_len, bias, times=self.times))
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.input_dropout = nn.Dropout(p=local_dropout)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_act = nn.GELU()
        self.output_linear = nn.Linear(hidden_dim, hidden_dim)
        self.output_bias = torch.nn.parameter.Parameter(torch.zeros(vocab_size))
        self.edge_query_embedding = nn.Embedding(14, hidden_dim // local_heads, padding_idx=0)
        self.edge_key_embedding = nn.Embedding(14, hidden_dim // local_heads, padding_idx=0)
        self.edge_value_embedding = nn.Embedding(14, hidden_dim // local_heads, padding_idx=0)
        # self.globl = Global(*graph, vocab_size, hidden_dim, use_global, global_layers, global_heads, global_dropout, global_activation)
        self.node_embedding = nn.parameter.Parameter(torch.Tensor(vocab_size, hidden_dim))

        self.h_fc = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.t_fc = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v_fc = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.hrt2kv_fc = nn.Linear(self.hidden_dim*3, self.hidden_dim)
        self.kv2h_fc = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.kv2t_fc = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.hr_dropout_layer = nn.Dropout(p=0.3)
        self.tr_dropout_layer = nn.Dropout(p=0.3)
        self.hk_dropout_layer = nn.Dropout(p=0.3)
        self.vk_dropout_layer = nn.Dropout(p=0.3)

        self.position_embedding = nn.Embedding(one_stat_len, self.hidden_dim)
        self.fact_index_embedding = nn.Embedding(neighbor_num+1, self.hidden_dim)
        self.init_params()
    def init_params(self):
        for name, param in self.named_parameters():
            if "norm" in name:
                continue
            elif "bias" in name:
                nn.init.zeros_(param)
            elif "weight" in name or "att" in name or "embedding" in name:
                nn.init.normal_(param, mean=0, std=0.02)
            else:
                raise TypeError("Invalid Parameters")
    def forward(self, input_ids, input_mask, mask_position, mask_output, edge_labels):
        # embedding = self.globl().to(input_ids.device)
        embedding = self.node_embedding.to(input_ids.device)
        x = torch.nn.functional.embedding(input_ids, embedding)
        x = self.input_dropout(self.input_norm(x))

        edge_query = self.edge_query_embedding(edge_labels)
        edge_key = self.edge_key_embedding(edge_labels)
        edge_value = self.edge_value_embedding(edge_labels)

        if self.positional:
            if self.one_st_len>3:
                positions_main = torch.tensor([0,1,2], dtype=torch.long, device=input_ids.device) #[0,1,2]
                positions_qual = torch.tensor([3,4], dtype=torch.long, device=input_ids.device).repeat((self.one_st_len-3)//2) #[3,4,3,4,3,4,3,4,3,4]
                positions = torch.cat([positions_main, positions_qual]).repeat(self.neighbor_num+1)
            else :
                positions = torch.tensor([0,1,2], dtype=torch.long, device=input_ids.device).repeat(self.neighbor_num+1)
            positions = positions.repeat(x.shape[0], 1)
            pos_embeddings = self.position_embedding(positions)
            x = x + pos_embeddings

        if self.apply_fact_index:
            fact_index = torch.repeat_interleave(torch.arange(self.neighbor_num+1,dtype=torch.long,device=input_ids.device),self.one_st_len)
            fact_index = fact_index.repeat(x.shape[0],1)
            fact_index_embedding = self.fact_index_embedding(fact_index)
            x = x + fact_index_embedding
        
        input_x = x
        for layer in self.layers:
            if self.comp:
                h_fc = self.h_fc
                t_fc = self.t_fc
                v_fc = self.v_fc
                hrt2kv_fc = self.hrt2kv_fc
                kv2h_fc = self.kv2h_fc
                kv2t_fc = self.kv2t_fc
                x_clone = x.clone()
                new_h = hrt2kv_fc(torch.cat((input_x[:,0,:],input_x[:,1,:],input_x[:,2,:]), dim=1))
                if self.one_st_len>3:
                    qual = list()
                    for qid in range(3,self.one_st_len,2):
                        qual.append(input_x[:,qid+1,:]-input_x[:,qid,:])
                    qual = torch.stack(qual, dim=1)
                    qual = torch.mean(qual, dim=1, keepdim=True).squeeze(1)
                    kv2h = kv2h_fc(qual)
                    kv2t = kv2t_fc(qual)

                    for col in range(0,self.one_st_len,2):
                        if col == 0:
                            tmp = self.hr_dropout_layer(h_fc(input_x[:,2,:] - input_x[:,1,:]))
                            x[:,col,:] = x_clone[:,col,:] + self.residual_w * tmp + self.residual_w * kv2h
                        elif col == 2:
                            tmp = self.tr_dropout_layer(t_fc(input_x[:,0,:] + input_x[:,1,:]))
                            x[:,col,:] = x_clone[:,col,:] + self.residual_w * tmp + self.residual_w * kv2t
                        else:
                            tmp = self.hk_dropout_layer(v_fc(new_h + input_x[:,col-1,:]))
                            x[:,col,:] = x_clone[:,col,:] + self.residual_w * tmp
                
                elif self.one_st_len==3:
                    for col in range(0,self.one_st_len,2):
                        if col == 0:
                            tmp = self.hr_dropout_layer(h_fc(input_x[:,2,:] - input_x[:,1,:]))
                            x[:,col,:] = x_clone[:,col,:] + self.residual_w * tmp
                        elif col == 2:
                            tmp = self.tr_dropout_layer(t_fc(input_x[:,0,:] + input_x[:,1,:]))
                            x[:,col,:] = x_clone[:,col,:] + self.residual_w * tmp
                input_x = x
            x = layer(x, input_mask, edge_key, edge_value, edge_query)
        x = x[torch.arange(x.shape[0]), mask_position]
        x = self.output_linear(x)  # x(batch_size, hiddem_dim)
        x = self.output_act(x)
        x = self.output_norm(x)
        y = torch.mm(x, embedding.transpose(0, 1)) + self.output_bias
        y = y.masked_fill(mask_output == 0, -100000)
        return y
