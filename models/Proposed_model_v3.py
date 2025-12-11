import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from fairseq.optim.adafactor import Adafactor
from torchvision import models

# ==============================================================================
# [Part 1] Helpers & GTrend Encoder (Google Trends용 - Decoder의 Memory 역할)
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
    """
    구글 트렌드 시계열(52주)을 처리하는 Transformer Encoder.
    Fusion Network에 들어가는 'Temporal'이 아니라, Decoder가 조회할 Context(Memory)입니다.
    """
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

# ==============================================================================
# [Part 2] Static Feature Encoders (Fusion Network의 입력이 됨)
# ==============================================================================

class AttributeEncoder(nn.Module):
    """ Text Embedding 역할 """
    def __init__(self, num_cat, num_col, num_fab, num_store, embedding_dim, hidden_dim):
        super().__init__()
        self.cat_emb = nn.Embedding(num_cat, embedding_dim)
        self.col_emb = nn.Embedding(num_col, embedding_dim)
        self.fab_emb = nn.Embedding(num_fab, embedding_dim)
        self.store_emb = nn.Embedding(num_store, embedding_dim)
        
        # 4개의 속성을 펼친 후 hidden_dim으로 투영 (Summation을 위해 차원 통일)
        self.proj = nn.Linear(embedding_dim * 4, hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, cat, col, fab, store):
        e1 = self.cat_emb(cat)
        e2 = self.col_emb(col)
        e3 = self.fab_emb(fab)
        e4 = self.store_emb(store)
        concat = torch.cat([e1, e2, e3, e4], dim=1) # [B, 4*Emb]
        return self.dropout(self.proj(concat))      # [B, Hidden]

class ImageEncoder(nn.Module):
    """ Vision Embedding 역할 """
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        resnet = models.resnet101(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.cnn = nn.Sequential(*modules)
        for p in self.cnn.parameters(): p.requires_grad = False
        for c in list(self.cnn.children())[6:]:
            for p in c.parameters(): p.requires_grad = True
        
        self.projection = nn.Conv2d(2048, embedding_dim, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        # 차원 맞추기용 Linear
        self.final_proj = nn.Linear(embedding_dim, hidden_dim)

    def forward(self, x):
        x = self.cnn(x)
        x = self.projection(x)
        x = self.pool(x).flatten(1) 
        return self.final_proj(x) # [B, Hidden]

class TemporalEmbedder(nn.Module):
    """ Temporal Embedding 역할 (출시일 연/월/일) """
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.day_emb = nn.Linear(1, embedding_dim)
        self.week_emb = nn.Linear(1, embedding_dim)
        self.month_emb = nn.Linear(1, embedding_dim)
        self.year_emb = nn.Linear(1, embedding_dim)
        
        # 4개의 시간 정보를 합쳐서 hidden_dim으로 변환
        self.proj = nn.Linear(embedding_dim * 4, hidden_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, temporal_features):
        d = self.day_emb(temporal_features[:, 0].unsqueeze(1))
        w = self.week_emb(temporal_features[:, 1].unsqueeze(1))
        m = self.month_emb(temporal_features[:, 2].unsqueeze(1))
        y = self.year_emb(temporal_features[:, 3].unsqueeze(1))
        concat_feat = torch.cat([d, w, m, y], dim=1) 
        return self.dropout(self.proj(concat_feat)) # [B, Hidden]

class SalesEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, embedding_dim, batch_first=True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        output, _ = self.gru(x)
        return self.dropout(output)

# ==============================================================================
# [Part 3] TARG Fusion Network (Text-Anchored Residual Gating) - PROPOSED
# ==============================================================================

class FusionBlock(nn.Module):
    """ M4FT의 MLP Block 재사용 (Fair Comparison) """
    def __init__(self, hidden_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
    def forward(self, x):
        return self.net(x)

class TARGFusionNetwork(nn.Module):
    """
    [제안 방법론] TARG: Target-Anchored Residual Gating
    설정된 Query(Anchor)를 기준으로 나머지 Context 정보들을 Gating하여 더함.
    """
    def __init__(self, hidden_dim, query_modality='text', dropout=0.2):
        super().__init__()
        self.query_modality = query_modality
        self.hidden_dim = hidden_dim
        
        # Gating Networks (2 contexts for 1 query)
        # Input: Concat[Query, Context] -> Output: Gate Score (hidden_dim)
        self.gate_fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate_fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Bias Initialization (0.0 -> Sigmoid(0.0) = 0.5)
        # 너무 열어두면(2.0) 노이즈가 들어오고, 닫아두면(-inf) 학습이 안 되므로 중립 유지
        nn.init.constant_(self.gate_fc1.bias, 0.0)
        nn.init.constant_(self.gate_fc2.bias, 0.0)
        
        # Final Processing Block (Same as M4FT)
        self.fusion_final = FusionBlock(hidden_dim, dropout)

    def forward(self, e_temp, e_text, e_vis):
        """
        Input: 3 Original Embeddings [B, Hidden]
        Logic: Query + (Context1 * Gate1) + (Context2 * Gate2)
        """
        
        # 1. 앵커(Query)와 컨텍스트(Context) 설정
        if self.query_modality == 'text':
            # Text가 중심, 이미지와 시간은 보조
            Q = e_text
            C1, C2 = e_vis, e_temp 
        elif self.query_modality == 'image':
            # Image가 중심, 텍스트와 시간은 보조
            Q = e_vis
            C1, C2 = e_text, e_temp
        elif self.query_modality == 'temporal':
            # Temporal이 중심, 텍스트와 이미지는 보조
            Q = e_temp
            C1, C2 = e_text, e_vis
        else:
            raise ValueError(f"Unknown query modality: {self.query_modality}")

        # 2. Cross-Gating Calculation
        # Context 1 Gating
        gate1_input = torch.cat([Q, C1], dim=1)
        gate1 = torch.sigmoid(self.gate_fc1(gate1_input))
        feat1 = C1 * gate1  # Refined Context 1
        
        # Context 2 Gating
        gate2_input = torch.cat([Q, C2], dim=1)
        gate2 = torch.sigmoid(self.gate_fc2(gate2_input))
        feat2 = C2 * gate2  # Refined Context 2
        
        # 3. Residual Aggregation (TARG Core Logic)
        # 앵커(Q)는 원본 보존(Identity), 나머지는 정제해서 더함(Residual)
        fused_features = Q + feat1 + feat2
        
        # 4. Final MLP
        return self.fusion_final(fused_features)

# ==============================================================================
# [Part 4] Main Model Class: TARG_M4FT_Visuelle2
# ==============================================================================

class TARG_M4FT_Visuelle2(pl.LightningModule):
    def __init__(self, embedding_dim, hidden_dim, output_dim, num_heads, num_layers, use_text, use_img, 
                cat_dict, col_dict, fab_dict, store_num, trend_len, num_trends, gpu_num, 
                query_modality='text', # [New] 쿼리 모달리티 선택 인자 추가
                use_encoder_mask=1, autoregressive=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.output_len = output_dim
        self.gpu_num = gpu_num
        self.autoregressive = autoregressive
        self.save_hyperparameters()

        # 1. Encoders (M4FT와 동일)
        self.gtrend_encoder = GTrendEmbedder(output_dim, hidden_dim, use_encoder_mask, trend_len, num_trends, gpu_num)
        self.sales_encoder = SalesEncoder(input_dim=1, embedding_dim=hidden_dim)
        self.text_encoder = AttributeEncoder(len(cat_dict)+1, len(col_dict)+1, len(fab_dict)+1, store_num+1, embedding_dim, hidden_dim)
        self.image_encoder = ImageEncoder(embedding_dim, hidden_dim)
        self.temporal_encoder = TemporalEmbedder(embedding_dim, hidden_dim)
        
        # 2. Fusion Module (TARG 적용)
        # query_modality='text' (기본), 'image', 'temporal' 중 선택 가능
        self.fusion_network = TARGFusionNetwork(hidden_dim, query_modality=query_modality)

        # 3. Decoder (M4FT/GTM과 동일)
        self.decoder_linear = TimeDistributed(nn.Linear(1, hidden_dim))
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*4, dropout=0.1)
        
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
        
        if item_sales.dim() == 3: bs, num_splits, window = item_sales.shape
        else: bs, window = item_sales.shape; num_splits = 1; item_sales = item_sales.unsqueeze(1)
        
        # 1. Encode Static Features
        e_text = self.text_encoder(category, color, fabric, store)
        e_vis = self.image_encoder(images)
        e_temp = self.temporal_encoder(temporal_features)
        
        if num_splits > 1:
            e_text = e_text.repeat_interleave(num_splits, dim=0)
            e_vis = e_vis.repeat_interleave(num_splits, dim=0)
            e_temp = e_temp.repeat_interleave(num_splits, dim=0)

        # 2. TARG Fusion (Selected Query Driven)
        static_context = self.fusion_network(e_temp, e_text, e_vis)

        # 3. Encode Dynamic Features
        gtrend_encoding = self.gtrend_encoder(gtrends) 
        if isinstance(gtrend_encoding, tuple): gtrend_encoding = gtrend_encoding[0]
        if num_splits > 1:
            gtrend_encoding = gtrend_encoding.repeat_interleave(num_splits, dim=1)

        sales_input = item_sales.reshape(bs * num_splits, window, 1)
        h_sales = self.sales_encoder(sales_input)
        sales_base = h_sales[:, -1, :] 

        # 4. Combine & Decode
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

    # (configure_optimizers, training_step, validation_step 등은 기존과 동일)
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

        lr = self.optimizers().param_groups[0]['lr']
        if torch.is_tensor(lr): lr = lr.item()
        print(f'Validation MAE: {mae.detach().cpu().numpy():.4f}, Validation WAPE: {wape.detach().cpu().numpy():.4f}, LR: {lr}')