import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from fairseq.optim.adafactor import Adafactor
from torchvision import models

# ==============================================================================
# [Part 1] Helpers & Encoders (GTM 원본과 동일 - 호환성 유지)
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

class GTrendEmbedder(nn.Module):
    def __init__(self, forecast_horizon, embedding_dim, use_mask, trend_len, num_trends, gpu_num):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.input_linear = TimeDistributed(nn.Linear(num_trends, embedding_dim))
        self.pos_embedding = PositionalEncoding(embedding_dim, max_len=trend_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4, dropout=0.2)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.use_mask = use_mask
        self.gpu_num = gpu_num

    def _generate_encoder_mask(self, size, forecast_horizon):
        mask = torch.zeros((size, size))
        split = math.gcd(size, forecast_horizon)
        for i in range(0, size, split):
            mask[i:i+split, i:i+split] = 1
        device = 'cuda:' + str(self.gpu_num) if torch.cuda.is_available() else 'cpu'
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to(device)
        return mask

    def forward(self, gtrends):
        gtrend_emb = self.input_linear(gtrends.permute(0,2,1))
        gtrend_emb = self.pos_embedding(gtrend_emb.permute(1,0,2))
        input_mask = self._generate_encoder_mask(gtrend_emb.shape[0], self.forecast_horizon)
        if self.use_mask == 1:
            gtrend_emb = self.encoder(gtrend_emb, input_mask)
        else:
            gtrend_emb = self.encoder(gtrend_emb)
        return gtrend_emb

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
# [Part 3] Novelty 1: Residual Gated Fusion Network (R-GFN)
# ==============================================================================

class GatedResidualBlock(nn.Module):
    """
    Applies Input-Dependent Gating with Residual Connection.
    Output = x + (x * Sigmoid(Linear(x)))
    """
    def __init__(self, input_dim):
        super().__init__()
        self.gate_fc = nn.Linear(input_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim) # Stability

    def forward(self, x):
        gate_score = torch.sigmoid(self.gate_fc(x))
        gated_x = x * gate_score
        return self.norm(x + gated_x) # Residual Connection

class ResidualGatedFusionNetwork(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout=0.2):
        super().__init__()
        # Input Dimensions per modality
        self.img_dim = embedding_dim
        self.text_dim = embedding_dim * 4 # Flattened text embeddings
        self.dummy_dim = embedding_dim
        
        # 1. Modality-Specific Gating Blocks
        self.img_gate = GatedResidualBlock(self.img_dim)
        self.text_gate = GatedResidualBlock(self.text_dim)
        self.dummy_gate = GatedResidualBlock(self.dummy_dim)
        
        # 2. Final Fusion
        total_dim = self.img_dim + self.text_dim + self.dummy_dim
        self.fusion_fc = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, img_encoding, text_encoding, dummy_encoding):
        # Flatten Text: [B, 4, Emb] -> [B, 4*Emb]
        text_flat = text_encoding.view(text_encoding.size(0), -1)
        
        # Apply Residual Gating to each modality (Sparsity + Preservation)
        g_img = self.img_gate(img_encoding)
        g_text = self.text_gate(text_flat)
        g_dummy = self.dummy_gate(dummy_encoding)
        
        # Concatenate and Project
        concat_features = torch.cat([g_img, g_text, g_dummy], dim=1)
        return self.fusion_fc(concat_features)

# ==============================================================================
# [Part 4] Novelty 2: Gated Cross-Attention Decoder
# ==============================================================================

class GatedCrossAttention(nn.Module):
    """
    Query-Dependent Gated Attention (G1 from paper).
    Filters noise in Value (Trends) based on relevance to Query (Static Context).
    """
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Gate Projection: Takes Query (Static Context) -> Produces Gate Score
        self.gate_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        # Standard Cross-Attention
        # query: [Tgt_Len, Batch, Emb], key/value: [Src_Len, Batch, Emb]
        attn_out, _ = self.mha(query, key, value, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        
        # --- Gating Mechanism ---
        # "Query-Dependent": Gate depends on the Query (Static Context)
        # Gate Score: [Tgt_Len, Batch, Emb]
        gate_score = torch.sigmoid(self.gate_proj(query))
        
        # Apply Gate (Element-wise multiplication)
        # Filters out irrelevant trend information
        gated_out = attn_out * gate_score
        
        # Residual Connection (Add & Norm) handled in the Decoder Layer wrapper
        return self.dropout(gated_out)

class GatedTransformerDecoderLayer(nn.Module):
    """
    Custom Decoder Layer replacing standard Cross-Attention with Gated Cross-Attention.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        # 1. Self-Attention (Masked, for autoregressive)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # 2. Gated Cross-Attention (Novelty)
        self.cross_attn = GatedCrossAttention(d_model, nhead, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model) # Residual handled here

        # 3. Feed Forward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # 1. Self Attention (Standard)
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = self.norm1(tgt + self.dropout1(tgt2))

        # 2. Gated Cross Attention (Query=tgt, Key/Value=memory)
        # Note: GatedCrossAttention class handles the gating internally
        tgt2 = self.cross_attn(tgt, memory, memory, key_padding_mask=memory_key_padding_mask)
        tgt = self.norm2(tgt + tgt2) # Residual Add & Norm

        # 3. FFN (Standard)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout3(tgt2))
        return tgt

# ==============================================================================
# [Part 5] Main Model Class: GatedMultimodal
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

        # Encoders (Standard)
        self.gtrend_encoder = GTrendEmbedder(output_dim, hidden_dim, use_encoder_mask, trend_len, num_trends, gpu_num)
        self.sales_encoder = SalesEncoder(input_dim=1, embedding_dim=hidden_dim)
        self.text_encoder = AttributeEncoder(len(cat_dict)+1, len(col_dict)+1, len(fab_dict)+1, store_num+1, embedding_dim)
        self.image_encoder = ImageEncoder(embedding_dim)
        self.dummy_encoder = DummyEmbedder(embedding_dim)
        
        # [Novelty 1] Residual Gated Fusion Network
        self.fusion_network = ResidualGatedFusionNetwork(embedding_dim, hidden_dim)

        # [Novelty 2] Decoder with Gated Cross-Attention
        self.decoder_linear = TimeDistributed(nn.Linear(1, hidden_dim))
        
        # Use custom GatedDecoderLayer instead of standard TransformerDecoderLayer
        decoder_layer = GatedTransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*4, dropout=0.1)
        
        if autoregressive: self.pos_encoder = PositionalEncoding(hidden_dim, max_len=12)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        self.decoder_fc = nn.Sequential(
            nn.Linear(hidden_dim, self.output_len if not autoregressive else 1),
            nn.Dropout(0.2)
        )

    def _generate_square_subsequent_mask(self, size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        device = 'cuda:' + str(self.gpu_num) if torch.cuda.is_available() else 'cpu'
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to(device)
        return mask

    def forward(self, item_sales, category, color, fabric, store, temporal_features, gtrends, images):
        
        # 1. Data Flatten & Repeat
        if item_sales.dim() == 3: bs, num_splits, window = item_sales.shape
        else: bs, window = item_sales.shape; num_splits = 1; item_sales = item_sales.unsqueeze(1)
        
        # Encode Static Features
        h_text = self.text_encoder(category, color, fabric, store)
        h_img = self.image_encoder(images)
        h_dummy = self.dummy_encoder(temporal_features)
        
        # Encode Trends
        gtrend_encoding = self.gtrend_encoder(gtrends) 
        if isinstance(gtrend_encoding, tuple): gtrend_encoding = gtrend_encoding[0]

        # Repeat for window splits
        if num_splits > 1:
            gtrend_encoding = gtrend_encoding.repeat_interleave(num_splits, dim=1)
            h_text = h_text.repeat_interleave(num_splits, dim=0)
            h_img = h_img.repeat_interleave(num_splits, dim=0)
            h_dummy = h_dummy.repeat_interleave(num_splits, dim=0)
        
        # Encode Sales (Zero for Demand, Actual for SO-fore)
        sales_input = item_sales.reshape(bs * num_splits, window, 1)
        h_sales = self.sales_encoder(sales_input)

        # 2. [Novelty] Residual Gated Fusion
        # Fuses static modalities with gating to filter noise and highlight salient features
        static_context = self.fusion_network(h_img, h_text, h_dummy)

        # 3. Combine with Sales (Residual)
        sales_base = h_sales[:, -1, :] 
        decoder_input = sales_base + static_context # Query for Decoder

        # 4. [Novelty] Decode with Gated Cross-Attention
        if self.autoregressive == 1:
            tgt = torch.zeros(self.output_len, gtrend_encoding.shape[1], gtrend_encoding.shape[-1]).to(item_sales.device)
            tgt[0] = decoder_input
            tgt = self.pos_encoder(tgt)
            tgt_mask = self._generate_square_subsequent_mask(self.output_len)
            
            # Here, the custom GatedTransformerDecoderLayer is used.
            # Query (tgt) Gates the Value (gtrend_encoding)
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
        # Task Adaptation Logic (Zero-Padding for Demand)
        if len(data_tuple) == 8: 
            item_sales, y, category, color, fabric, store, temporal_features, gtrends = data_tuple
        else:
            y, category, color, fabric, store, temporal_features, gtrends = data_tuple
            bs = y.shape[0]
            # Demand Task: Zero padding for sales input
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

        # [수정] LR 값을 안전하게 가져오는 로직 추가
        lr = self.optimizers().param_groups[0].get('lr') # .get() 사용 권장
        
        if lr is None:
            lr_val = 0.0
        elif torch.is_tensor(lr):
            lr_val = lr.item()
        else:
            lr_val = lr

        print(f'Validation MAE: {mae.detach().cpu().numpy():.4f}, Validation WAPE: {wape.detach().cpu().numpy():.4f}, LR: {lr_val:.8f}')