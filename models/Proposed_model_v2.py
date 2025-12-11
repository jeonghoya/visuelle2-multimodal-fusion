# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import pytorch_lightning as pl
# from fairseq.optim.adafactor import Adafactor
# from torchvision import models

# # ==============================================================================
# # [Part 1] Helpers & Encoders (GTM 호환성 유지)
# # ==============================================================================

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.1, max_len=52):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         x = x + self.pe[:x.size(0), :]
#         return self.dropout(x)

# class TimeDistributed(nn.Module):
#     def __init__(self, module, batch_first=True):
#         super(TimeDistributed, self).__init__()
#         self.module = module 
#         self.batch_first = batch_first

#     def forward(self, x):
#         if len(x.size()) <= 2: return self.module(x)
#         x_reshape = x.contiguous().view(-1, x.size(-1))  
#         y = self.module(x_reshape)
#         if self.batch_first:
#             y = y.contiguous().view(x.size(0), -1, y.size(-1))
#         else:
#             y = y.view(-1, x.size(1), y.size(-1))
#         return y

# class AttributeEncoder(nn.Module):
#     def __init__(self, num_cat, num_col, num_fab, num_store, embedding_dim):
#         super().__init__()
#         self.cat_emb = nn.Embedding(num_cat, embedding_dim)
#         self.col_emb = nn.Embedding(num_col, embedding_dim)
#         self.fab_emb = nn.Embedding(num_fab, embedding_dim)
#         self.store_emb = nn.Embedding(num_store, embedding_dim)
#         self.dropout = nn.Dropout(0.1)

#     def forward(self, cat, col, fab, store):
#         e1 = self.cat_emb(cat)
#         e2 = self.col_emb(col)
#         e3 = self.fab_emb(fab)
#         e4 = self.store_emb(store)
#         return self.dropout(torch.stack([e1, e2, e3, e4], dim=1))

# class SalesEncoder(nn.Module):
#     def __init__(self, input_dim, embedding_dim):
#         super().__init__()
#         self.gru = nn.GRU(input_dim, embedding_dim, batch_first=True)
#         self.dropout = nn.Dropout(0.1)

#     def forward(self, x):
#         output, _ = self.gru(x)
#         return self.dropout(output)

# class ImageEncoder(nn.Module):
#     def __init__(self, embedding_dim=512):
#         super().__init__()
#         resnet = models.resnet101(pretrained=True)
#         modules = list(resnet.children())[:-2]
#         self.cnn = nn.Sequential(*modules)
#         for p in self.cnn.parameters(): p.requires_grad = False
#         for c in list(self.cnn.children())[6:]:
#             for p in c.parameters(): p.requires_grad = True
#         self.projection = nn.Conv2d(2048, embedding_dim, kernel_size=1)
#         self.pool = nn.AdaptiveAvgPool2d((1,1))

#     def forward(self, x):
#         x = self.cnn(x)
#         x = self.projection(x)
#         x = self.pool(x).flatten(1) 
#         return x

# class DummyEmbedder(nn.Module):
#     def __init__(self, embedding_dim):
#         super().__init__()
#         self.day_emb = nn.Linear(1, embedding_dim)
#         self.week_emb = nn.Linear(1, embedding_dim)
#         self.month_emb = nn.Linear(1, embedding_dim)
#         self.year_emb = nn.Linear(1, embedding_dim)
#         self.dummy_fusion = nn.Linear(embedding_dim * 4, embedding_dim)
#         self.dropout = nn.Dropout(0.2)

#     def forward(self, temporal_features):
#         d = self.day_emb(temporal_features[:, 0].unsqueeze(1))
#         w = self.week_emb(temporal_features[:, 1].unsqueeze(1))
#         m = self.month_emb(temporal_features[:, 2].unsqueeze(1))
#         y = self.year_emb(temporal_features[:, 3].unsqueeze(1))
#         concat_feat = torch.cat([d, w, m, y], dim=1) 
#         return self.dropout(self.dummy_fusion(concat_feat)) 

# # ==============================================================================
# # [Part 2] Novelty Implementation: Post-Concat Pure Gating Modules
# # ==============================================================================

# class PureGatedMultiheadAttention(nn.Module):
#     """
#     Custom Multi-Head Attention with Post-Concat Pure Gating (G1).
#     Flow: Q,K,V -> SDPA -> Concat -> Gating(Query-Dependent) -> Linear(Wo)
#     """
#     def __init__(self, embed_dim, num_heads, dropout=0.1):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads
#         assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

#         # Linear Projections
#         self.q_proj = nn.Linear(embed_dim, embed_dim)
#         self.k_proj = nn.Linear(embed_dim, embed_dim)
#         self.v_proj = nn.Linear(embed_dim, embed_dim)
#         self.out_proj = nn.Linear(embed_dim, embed_dim)
        
#         # [Novelty] Post-Concat Gate Projection
#         # Input: Query (Context) -> Output: Gate for Concatenated Vector (embed_dim)
#         self.gate_proj = nn.Linear(embed_dim, embed_dim)

#         self.dropout = nn.Dropout(dropout)
#         self.scale = self.head_dim ** -0.5

#     def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
#         # Input: [Seq_Len, Batch, Dim]
#         tgt_len, bsz, _ = query.size()
#         src_len = key.size(0)

#         # 1. Project & Reshape: [Seq_Len, Batch, Num_Heads, Head_Dim]
#         q = self.q_proj(query).view(tgt_len, bsz, self.num_heads, self.head_dim)
#         k = self.k_proj(key).view(src_len, bsz, self.num_heads, self.head_dim)
#         v = self.v_proj(value).view(src_len, bsz, self.num_heads, self.head_dim)

#         # 2. Transpose: [Batch * Num_Heads, Seq_Len, Head_Dim]
#         q_t = q.permute(1, 2, 0, 3).reshape(bsz * self.num_heads, tgt_len, self.head_dim)
#         k_t = k.permute(1, 2, 0, 3).reshape(bsz * self.num_heads, src_len, self.head_dim)
#         v_t = v.permute(1, 2, 0, 3).reshape(bsz * self.num_heads, src_len, self.head_dim)

#         # 3. Scaled Dot-Product Attention
#         attn_scores = torch.bmm(q_t, k_t.transpose(1, 2)) * self.scale
#         if attn_mask is not None:
#             attn_scores += attn_mask 
        
#         attn_probs = F.softmax(attn_scores, dim=-1)
#         attn_probs = self.dropout(attn_probs)
#         attn_output = torch.bmm(attn_probs, v_t) # [Batch * Heads, Tgt_Len, Head_Dim]

#         # 4. Concatenation (Restore Shape)
#         # [Batch * Heads, Tgt_Len, Head_Dim] -> [Tgt_Len, Batch, Embed_Dim]
#         attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
#         attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(tgt_len, bsz, self.embed_dim)

#         # 5. [Novelty] Post-Concat Pure Gating (G1)
#         # Gate depends on Query (Context)
#         gate_score = torch.sigmoid(self.gate_proj(query)) # [Tgt_Len, Batch, Embed_Dim]
        
#         # Pure Gating: No Residual here (handled by Transformer Block add&norm)
#         gated_output = attn_output * gate_score

#         # 6. Final Linear Projection
#         return self.out_proj(gated_output)

# class PureGatedFusionNetwork(nn.Module):
#     """
#     Fusion Module with Post-Concat Pure Gating.
#     Concat -> Gate -> Linear
#     """
#     def __init__(self, embedding_dim, hidden_dim, dropout=0.2):
#         super().__init__()
#         self.img_dim = embedding_dim
#         self.text_dim = embedding_dim * 4
#         self.dummy_dim = embedding_dim
#         total_dim = self.img_dim + self.text_dim + self.dummy_dim
        
#         # Gate Projection: Takes Concatenated Features -> Gate Score
#         self.gate_fc = nn.Linear(total_dim, total_dim)
        
#         self.fusion_fc = nn.Sequential(
#             nn.Linear(total_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )

#     def forward(self, img_encoding, text_encoding, dummy_encoding):
#         text_flat = text_encoding.view(text_encoding.size(0), -1)
        
#         # 1. Concat
#         concat_features = torch.cat([img_encoding, text_flat, dummy_encoding], dim=1)
        
#         # 2. Pure Gating (Element-wise)
#         # Gate depends on the concatenated features themselves
#         gate_score = torch.sigmoid(self.gate_fc(concat_features))
#         gated_features = concat_features * gate_score
        
#         # 3. Projection
#         return self.fusion_fc(gated_features)

# # ==============================================================================
# # [Part 3] Gated Transformer Layers
# # ==============================================================================

# class GatedTransformerEncoderLayer(nn.Module):
#     def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
#         super().__init__()
#         # Use Custom Gated Attention
#         self.self_attn = PureGatedMultiheadAttention(d_model, nhead, dropout=dropout)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.dropout1 = nn.Dropout(dropout)

#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout2 = nn.Dropout(dropout)

#     def forward(self, src, src_mask=None, src_key_padding_mask=None):
#         # Attention -> Gate -> Dropout -> Add -> Norm
#         src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
#         src = self.norm1(src + self.dropout1(src2))
        
#         src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
#         src = self.norm2(src + self.dropout2(src2))
#         return src

# class GatedTransformerDecoderLayer(nn.Module):
#     def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
#         super().__init__()
#         # Self-Attn (Standard or Gated? Using Standard for AR part usually safer, but consistency suggests Gated. 
#         # Using Standard nn.MHA for Self-Attn to focus Gating on Cross-Attn novelty)
#         self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.dropout1 = nn.Dropout(dropout)

#         # [Novelty] Cross-Attention with Post-Concat Gating
#         self.cross_attn = PureGatedMultiheadAttention(d_model, nhead, dropout=dropout)
#         self.norm2 = nn.LayerNorm(d_model)

#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)
#         self.norm3 = nn.LayerNorm(d_model)
#         self.dropout2 = nn.Dropout(dropout)
#         self.dropout3 = nn.Dropout(dropout)

#     def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
#         # 1. Self Attention
#         tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
#         tgt = self.norm1(tgt + self.dropout1(tgt2))

#         # 2. Gated Cross Attention
#         # Query=tgt, Key/Value=memory
#         tgt2 = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
#         tgt = self.norm2(tgt + tgt2)

#         # 3. FFN
#         tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
#         tgt = self.norm3(tgt + self.dropout3(tgt2))
#         return tgt

# # ==============================================================================
# # [Part 4] Main Model: GatedMultimodal_Visuelle2
# # ==============================================================================

# class GatedMultimodal_Visuelle2(pl.LightningModule):
#     def __init__(self, embedding_dim, hidden_dim, output_dim, num_heads, num_layers, use_text, use_img, 
#                 cat_dict, col_dict, fab_dict, store_num, trend_len, num_trends, gpu_num, use_encoder_mask=1, autoregressive=False):
#         super().__init__()
#         self.hidden_dim = hidden_dim
#         self.embedding_dim = embedding_dim
#         self.output_len = output_dim
#         self.gpu_num = gpu_num
#         self.autoregressive = autoregressive
#         self.save_hyperparameters()

#         # Encoders
#         self.sales_encoder = SalesEncoder(input_dim=1, embedding_dim=hidden_dim)
#         self.text_encoder = AttributeEncoder(len(cat_dict)+1, len(col_dict)+1, len(fab_dict)+1, store_num+1, embedding_dim)
#         self.image_encoder = ImageEncoder(embedding_dim)
#         self.dummy_encoder = DummyEmbedder(embedding_dim)
        
#         # [Modified] Gated Trend Encoder (Self-Attention Gating)
#         self.gtrend_input_linear = TimeDistributed(nn.Linear(num_trends, hidden_dim))
#         self.gtrend_pos_embedding = PositionalEncoding(hidden_dim, max_len=trend_len)
        
#         gated_encoder_layer = GatedTransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=0.2)
#         self.gtrend_encoder = nn.TransformerEncoder(gated_encoder_layer, num_layers=2)
        
#         self.use_encoder_mask = use_encoder_mask
#         self.trend_len = trend_len
        
#         # [Novelty 1] Pure Gated Fusion
#         self.fusion_network = PureGatedFusionNetwork(embedding_dim, hidden_dim)

#         # [Novelty 2] Decoder with Gated Cross-Attention
#         self.decoder_linear = TimeDistributed(nn.Linear(1, hidden_dim))
#         gated_decoder_layer = GatedTransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*4, dropout=0.1)
        
#         if autoregressive: self.pos_encoder = PositionalEncoding(hidden_dim, max_len=12)
#         self.decoder = nn.TransformerDecoder(gated_decoder_layer, num_layers)
        
#         self.decoder_fc = nn.Sequential(
#             nn.Linear(hidden_dim, self.output_len if not autoregressive else 1),
#             nn.Dropout(0.2)
#         )

#     def _generate_encoder_mask(self, size, forecast_horizon):
#         mask = torch.zeros((size, size))
#         split = math.gcd(size, forecast_horizon)
#         for i in range(0, size, split):
#             mask[i:i+split, i:i+split] = 1
#         device = 'cuda:' + str(self.gpu_num) if torch.cuda.is_available() else 'cpu'
#         mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to(device)
#         return mask

#     def _generate_square_subsequent_mask(self, size):
#         mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
#         device = 'cuda:' + str(self.gpu_num) if torch.cuda.is_available() else 'cpu'
#         mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to(device)
#         return mask

#     def forward(self, item_sales, category, color, fabric, store, temporal_features, gtrends, images):
        
#         # 1. Flatten & Repeat
#         if item_sales.dim() == 3: bs, num_splits, window = item_sales.shape
#         else: bs, window = item_sales.shape; num_splits = 1; item_sales = item_sales.unsqueeze(1)
        
#         h_text = self.text_encoder(category, color, fabric, store)
#         h_img = self.image_encoder(images)
#         h_dummy = self.dummy_encoder(temporal_features)
        
#         # Encode Trends (Gated)
#         gtrend_emb = self.gtrend_input_linear(gtrends.permute(0,2,1))
#         gtrend_emb = self.gtrend_pos_embedding(gtrend_emb.permute(1,0,2))
        
#         input_mask = self._generate_encoder_mask(gtrend_emb.shape[0], self.output_len)
#         if self.use_encoder_mask == 1:
#             gtrend_encoding = self.gtrend_encoder(gtrend_emb, input_mask)
#         else:
#             gtrend_encoding = self.gtrend_encoder(gtrend_emb)

#         if num_splits > 1:
#             gtrend_encoding = gtrend_encoding.repeat_interleave(num_splits, dim=1)
#             h_text = h_text.repeat_interleave(num_splits, dim=0)
#             h_img = h_img.repeat_interleave(num_splits, dim=0)
#             h_dummy = h_dummy.repeat_interleave(num_splits, dim=0)
        
#         sales_input = item_sales.reshape(bs * num_splits, window, 1)
#         h_sales = self.sales_encoder(sales_input)

#         # 2. Pure Gated Fusion
#         static_context = self.fusion_network(h_img, h_text, h_dummy)

#         # 3. Combine with Sales
#         sales_base = h_sales[:, -1, :] 
#         decoder_input = sales_base + static_context 

#         # 4. Decode with Gated Cross-Attention
#         if self.autoregressive == 1:
#             tgt = torch.zeros(self.output_len, gtrend_encoding.shape[1], gtrend_encoding.shape[-1]).to(item_sales.device)
#             tgt[0] = decoder_input
#             tgt = self.pos_encoder(tgt)
#             tgt_mask = self._generate_square_subsequent_mask(self.output_len)
            
#             decoder_out = self.decoder(tgt, gtrend_encoding, tgt_mask)
#             forecast = self.decoder_fc(decoder_out)
#         else:
#             tgt = decoder_input.unsqueeze(0)
#             decoder_out = self.decoder(tgt, gtrend_encoding)
#             forecast = self.decoder_fc(decoder_out)

#         return forecast.transpose(0, 1).reshape(bs * num_splits, self.output_len), None

#     def configure_optimizers(self):
#         optimizer = Adafactor(self.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
#         return [optimizer]

#     def training_step(self, train_batch, batch_idx):
#         data_tuple, images = train_batch
#         if len(data_tuple) == 8: 
#             item_sales, y, category, color, fabric, store, temporal_features, gtrends = data_tuple
#         else:
#             y, category, color, fabric, store, temporal_features, gtrends = data_tuple
#             bs = y.shape[0]
#             item_sales = torch.zeros(bs, 1, 2, device=self.device)

#         forecast, _ = self.forward(item_sales, category, color, fabric, store, temporal_features, gtrends, images)
#         loss = F.mse_loss(y.reshape(-1), forecast.reshape(-1))
#         self.log('train_loss', loss)
#         return loss

#     def validation_step(self, test_batch, batch_idx):
#         data_tuple, images = test_batch
#         if len(data_tuple) == 8:
#             item_sales, y, category, color, fabric, store, temporal_features, gtrends = data_tuple
#         else:
#             y, category, color, fabric, store, temporal_features, gtrends = data_tuple
#             bs = y.shape[0]
#             item_sales = torch.zeros(bs, 1, 2, device=self.device)
        
#         forecast, _ = self.forward(item_sales, category, color, fabric, store, temporal_features, gtrends, images)
#         return y.reshape(-1), forecast.reshape(-1)

#     def validation_epoch_end(self, val_step_outputs):
#         item_sales = torch.cat([x[0] for x in val_step_outputs]).view(-1)
#         forecasted_sales = torch.cat([x[1] for x in val_step_outputs]).view(-1)
        
#         rescaled_item_sales = item_sales * 53
#         rescaled_forecasted_sales = forecasted_sales * 53
        
#         loss = F.mse_loss(item_sales, forecasted_sales)
#         mae = F.l1_loss(rescaled_item_sales, rescaled_forecasted_sales)
#         wape = 100 * torch.sum(torch.abs(rescaled_item_sales - rescaled_forecasted_sales)) / torch.sum(rescaled_item_sales)
        
#         self.log('val_mae', mae)
#         self.log('val_wWAPE', wape)
#         self.log('val_loss', loss)

#         lr = self.optimizers().param_groups[0].get('lr')
#         if lr is None: lr_val = 0.0
#         elif torch.is_tensor(lr): lr_val = lr.item()
#         else: lr_val = lr

#         print(f'Validation MAE: {mae.detach().cpu().numpy():.4f}, Validation WAPE: {wape.detach().cpu().numpy():.4f}, LR: {lr_val:.8f}')
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from fairseq.optim.adafactor import Adafactor
from torchvision import models

# ==============================================================================
# [Part 1] Helpers & Encoders (GTM 호환성 유지)
# ==============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=52):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module 
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2: return self.module(x)
        x_reshape = x.contiguous().view(-1, x.size(-1))  
        y = self.module(x_reshape)
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))
        else:
            y = y.view(-1, x.size(1), y.size(-1))
        return y

class AttributeEncoder(nn.Module):
    def __init__(self, num_cat, num_col, num_fab, num_store, embedding_dim):
        super().__init__()
        self.cat_emb = nn.Embedding(num_cat, embedding_dim)
        self.col_emb = nn.Embedding(num_col, embedding_dim)
        self.fab_emb = nn.Embedding(num_fab, embedding_dim)
        self.store_emb = nn.Embedding(num_store, embedding_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, cat, col, fab, store):
        e1 = self.cat_emb(cat)
        e2 = self.col_emb(col)
        e3 = self.fab_emb(fab)
        e4 = self.store_emb(store)
        return self.dropout(torch.stack([e1, e2, e3, e4], dim=1))

class SalesEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, embedding_dim, batch_first=True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        output, _ = self.gru(x)
        return self.dropout(output)

class ImageEncoder(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        resnet = models.resnet101(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.cnn = nn.Sequential(*modules)
        for p in self.cnn.parameters(): p.requires_grad = False
        for c in list(self.cnn.children())[6:]:
            for p in c.parameters(): p.requires_grad = True
        self.projection = nn.Conv2d(2048, embedding_dim, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        x = self.cnn(x)
        x = self.projection(x)
        x = self.pool(x).flatten(1) 
        return x

class DummyEmbedder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.day_emb = nn.Linear(1, embedding_dim)
        self.week_emb = nn.Linear(1, embedding_dim)
        self.month_emb = nn.Linear(1, embedding_dim)
        self.year_emb = nn.Linear(1, embedding_dim)
        self.dummy_fusion = nn.Linear(embedding_dim * 4, embedding_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, temporal_features):
        d = self.day_emb(temporal_features[:, 0].unsqueeze(1))
        w = self.week_emb(temporal_features[:, 1].unsqueeze(1))
        m = self.month_emb(temporal_features[:, 2].unsqueeze(1))
        y = self.year_emb(temporal_features[:, 3].unsqueeze(1))
        concat_feat = torch.cat([d, w, m, y], dim=1) 
        return self.dropout(self.dummy_fusion(concat_feat)) 

# ==============================================================================
# [Part 2] Novelty Implementation: Bias Initialization Added!
# ==============================================================================

class PureGatedMultiheadAttention(nn.Module):
    """
    [처방 1] Bias Init 적용: 초기에는 Gate를 열어둠 (GTM과 유사하게 시작)
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # [Novelty] Post-Concat Gate Projection
        self.gate_proj = nn.Linear(embed_dim, embed_dim)
        
        # [★핵심 수정] Bias Initialization (+2.0)
        # Sigmoid(2.0) ~= 0.88 (정보 88% 통과)
        # 초기 학습 난이도를 낮춰줌
        nn.init.constant_(self.gate_proj.bias, 2.0)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        tgt_len, bsz, _ = query.size()
        src_len = key.size(0)

        q = self.q_proj(query).view(tgt_len, bsz, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(src_len, bsz, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(src_len, bsz, self.num_heads, self.head_dim)

        q_t = q.permute(1, 2, 0, 3).reshape(bsz * self.num_heads, tgt_len, self.head_dim)
        k_t = k.permute(1, 2, 0, 3).reshape(bsz * self.num_heads, src_len, self.head_dim)
        v_t = v.permute(1, 2, 0, 3).reshape(bsz * self.num_heads, src_len, self.head_dim)

        attn_scores = torch.bmm(q_t, k_t.transpose(1, 2)) * self.scale
        if attn_mask is not None:
            attn_scores += attn_mask 
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        attn_output = torch.bmm(attn_probs, v_t) 

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(tgt_len, bsz, self.embed_dim)

        # [Gating]
        gate_score = torch.sigmoid(self.gate_proj(query)) 
        
        # Transformer Block 내부에는 이미 Residual(Add&Norm)이 있으므로 여기선 Pure Gating 유지
        gated_output = attn_output * gate_score

        return self.out_proj(gated_output)

class PureGatedFusionNetwork(nn.Module):
    """
    [처방 2] Fusion Layer는 Soft Gating (Residual 유사) 방식으로 변경
    """
    def __init__(self, embedding_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.img_dim = embedding_dim
        self.text_dim = embedding_dim * 4
        self.dummy_dim = embedding_dim
        total_dim = self.img_dim + self.text_dim + self.dummy_dim
        
        self.gate_fc = nn.Linear(total_dim, total_dim)
        
        # [★핵심 수정] Bias Initialization (+2.0)
        nn.init.constant_(self.gate_fc.bias, 2.0)
        
        self.fusion_fc = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, img_encoding, text_encoding, dummy_encoding):
        text_flat = text_encoding.view(text_encoding.size(0), -1)
        concat_features = torch.cat([img_encoding, text_flat, dummy_encoding], dim=1)
        
        gate_score = torch.sigmoid(self.gate_fc(concat_features))
        
        # [★핵심 수정] Soft Gating: Input + (Input * Gate)
        # 완전히 닫아버리는(0) 것보다, 원본을 보존하되 중요하면 더 강조하는 방식이
        # 데이터가 적은 Visuelle 2.0 Fusion 단계에서 훨씬 안정적임
        gated_features = concat_features + (concat_features * gate_score)
        
        return self.fusion_fc(gated_features)

# ==============================================================================
# [Part 3] Gated Transformer Layers
# ==============================================================================

class HeadSpecificGatedAttention(nn.Module):
    """ Encoder Self-Attention용 (Bias Init 추가) """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.gate_proj = nn.Linear(self.head_dim, self.head_dim)
        # [★핵심 수정] Bias Init
        nn.init.constant_(self.gate_proj.bias, 2.0)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        tgt_len, bsz, _ = query.size()
        src_len = key.size(0)

        q = self.q_proj(query).view(tgt_len, bsz, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(src_len, bsz, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(src_len, bsz, self.num_heads, self.head_dim)

        q_t = q.permute(1, 2, 0, 3).reshape(bsz * self.num_heads, tgt_len, self.head_dim)
        k_t = k.permute(1, 2, 0, 3).reshape(bsz * self.num_heads, src_len, self.head_dim)
        v_t = v.permute(1, 2, 0, 3).reshape(bsz * self.num_heads, src_len, self.head_dim)

        attn_scores = torch.bmm(q_t, k_t.transpose(1, 2)) * self.scale
        if attn_mask is not None:
            attn_scores += attn_mask 
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        attn_output = torch.bmm(attn_probs, v_t) 

        gate_score = torch.sigmoid(self.gate_proj(q_t))
        gated_output = attn_output * gate_score

        gated_output = gated_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        gated_output = gated_output.permute(2, 0, 1, 3).contiguous().view(tgt_len, bsz, self.embed_dim)
        
        return self.out_proj(gated_output)

class GatedTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = HeadSpecificGatedAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = self.norm1(src + self.dropout1(src2))
        
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = self.norm2(src + self.dropout2(src2))
        return src

class GatedTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        # Self-Attn: Standard (일관성 위해 표준 사용)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Cross-Attn: Gated
        self.cross_attn = PureGatedMultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = self.norm1(tgt + self.dropout1(tgt2))

        tgt2 = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
        tgt = self.norm2(tgt + tgt2)

        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout3(tgt2))
        return tgt

# ==============================================================================
# [Part 4] Main Model Class: GatedMultimodal_Visuelle2
# ==============================================================================

class GatedMultimodal_Visuelle2(pl.LightningModule):
    def __init__(self, embedding_dim, hidden_dim, output_dim, num_heads, num_layers, use_text, use_img, 
                cat_dict, col_dict, fab_dict, store_num, trend_len, num_trends, gpu_num, use_encoder_mask=1, autoregressive=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.output_len = output_dim
        self.gpu_num = gpu_num
        self.autoregressive = autoregressive
        self.save_hyperparameters()

        # Encoders
        self.sales_encoder = SalesEncoder(input_dim=1, embedding_dim=hidden_dim)
        self.text_encoder = AttributeEncoder(len(cat_dict)+1, len(col_dict)+1, len(fab_dict)+1, store_num+1, embedding_dim)
        self.image_encoder = ImageEncoder(embedding_dim)
        self.dummy_encoder = DummyEmbedder(embedding_dim)
        
        self.gtrend_input_linear = TimeDistributed(nn.Linear(num_trends, hidden_dim))
        self.gtrend_pos_embedding = PositionalEncoding(hidden_dim, max_len=trend_len)
        
        gated_encoder_layer = GatedTransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=0.2)
        self.gtrend_encoder = nn.TransformerEncoder(gated_encoder_layer, num_layers=2)
        
        self.use_encoder_mask = use_encoder_mask
        self.trend_len = trend_len
        
        # [Modified] Fusion (Bias Init + Residual)
        self.fusion_network = PureGatedFusionNetwork(embedding_dim, hidden_dim)

        self.decoder_linear = TimeDistributed(nn.Linear(1, hidden_dim))
        gated_decoder_layer = GatedTransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*4, dropout=0.1)
        
        if autoregressive: self.pos_encoder = PositionalEncoding(hidden_dim, max_len=12)
        self.decoder = nn.TransformerDecoder(gated_decoder_layer, num_layers)
        
        self.decoder_fc = nn.Sequential(
            nn.Linear(hidden_dim, self.output_len if not autoregressive else 1),
            nn.Dropout(0.2)
        )

    def _generate_encoder_mask(self, size, forecast_horizon):
        mask = torch.zeros((size, size))
        split = math.gcd(size, forecast_horizon)
        for i in range(0, size, split):
            mask[i:i+split, i:i+split] = 1
        device = 'cuda:' + str(self.gpu_num) if torch.cuda.is_available() else 'cpu'
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to(device)
        return mask

    def _generate_square_subsequent_mask(self, size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        device = 'cuda:' + str(self.gpu_num) if torch.cuda.is_available() else 'cpu'
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to(device)
        return mask

    def forward(self, item_sales, category, color, fabric, store, temporal_features, gtrends, images):
        
        if item_sales.dim() == 3: bs, num_splits, window = item_sales.shape
        else: bs, window = item_sales.shape; num_splits = 1; item_sales = item_sales.unsqueeze(1)
        
        h_text = self.text_encoder(category, color, fabric, store)
        h_img = self.image_encoder(images)
        h_dummy = self.dummy_encoder(temporal_features)
        
        gtrend_emb = self.gtrend_input_linear(gtrends.permute(0,2,1))
        gtrend_emb = self.gtrend_pos_embedding(gtrend_emb.permute(1,0,2))
        
        input_mask = self._generate_encoder_mask(gtrend_emb.shape[0], self.output_len)
        if self.use_encoder_mask == 1:
            gtrend_encoding = self.gtrend_encoder(gtrend_emb, input_mask)
        else:
            gtrend_encoding = self.gtrend_encoder(gtrend_emb)

        if num_splits > 1:
            gtrend_encoding = gtrend_encoding.repeat_interleave(num_splits, dim=1)
            h_text = h_text.repeat_interleave(num_splits, dim=0)
            h_img = h_img.repeat_interleave(num_splits, dim=0)
            h_dummy = h_dummy.repeat_interleave(num_splits, dim=0)
        
        sales_input = item_sales.reshape(bs * num_splits, window, 1)
        h_sales = self.sales_encoder(sales_input)

        static_context = self.fusion_network(h_img, h_text, h_dummy)

        sales_base = h_sales[:, -1, :] 
        decoder_input = sales_base + static_context 

        if self.autoregressive == 1:
            tgt = torch.zeros(self.output_len, gtrend_encoding.shape[1], gtrend_encoding.shape[-1]).to(item_sales.device)
            tgt[0] = decoder_input
            tgt = self.pos_encoder(tgt)
            tgt_mask = self._generate_square_subsequent_mask(self.output_len)
            
            decoder_out = self.decoder(tgt, gtrend_encoding, tgt_mask)
            forecast = self.decoder_fc(decoder_out)
        else:
            tgt = decoder_input.unsqueeze(0)
            decoder_out = self.decoder(tgt, gtrend_encoding)
            forecast = self.decoder_fc(decoder_out)

        return forecast.transpose(0, 1).reshape(bs * num_splits, self.output_len), None

    def configure_optimizers(self):
        optimizer = Adafactor(self.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
        return [optimizer]

    def training_step(self, train_batch, batch_idx):
        data_tuple, images = train_batch
        if len(data_tuple) == 8: 
            item_sales, y, category, color, fabric, store, temporal_features, gtrends = data_tuple
        else:
            y, category, color, fabric, store, temporal_features, gtrends = data_tuple
            bs = y.shape[0]
            item_sales = torch.zeros(bs, 1, 2, device=self.device)

        forecast, _ = self.forward(item_sales, category, color, fabric, store, temporal_features, gtrends, images)
        loss = F.mse_loss(y.reshape(-1), forecast.reshape(-1))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, test_batch, batch_idx):
        data_tuple, images = test_batch
        if len(data_tuple) == 8:
            item_sales, y, category, color, fabric, store, temporal_features, gtrends = data_tuple
        else:
            y, category, color, fabric, store, temporal_features, gtrends = data_tuple
            bs = y.shape[0]
            item_sales = torch.zeros(bs, 1, 2, device=self.device)
        
        forecast, _ = self.forward(item_sales, category, color, fabric, store, temporal_features, gtrends, images)
        return y.reshape(-1), forecast.reshape(-1)

    def validation_epoch_end(self, val_step_outputs):
        item_sales = torch.cat([x[0] for x in val_step_outputs]).view(-1)
        forecasted_sales = torch.cat([x[1] for x in val_step_outputs]).view(-1)
        
        rescaled_item_sales = item_sales * 53
        rescaled_forecasted_sales = forecasted_sales * 53
        
        loss = F.mse_loss(item_sales, forecasted_sales)
        mae = F.l1_loss(rescaled_item_sales, rescaled_forecasted_sales)
        wape = 100 * torch.sum(torch.abs(rescaled_item_sales - rescaled_forecasted_sales)) / torch.sum(rescaled_item_sales)
        
        self.log('val_mae', mae)
        self.log('val_wWAPE', wape)
        self.log('val_loss', loss)

        lr = self.optimizers().param_groups[0].get('lr')
        if lr is None: lr_val = 0.0
        elif torch.is_tensor(lr): lr_val = lr.item()
        else: lr_val = lr

        print(f'Validation MAE: {mae.detach().cpu().numpy():.4f}, Validation WAPE: {wape.detach().cpu().numpy():.4f}, LR: {lr_val:.8f}')