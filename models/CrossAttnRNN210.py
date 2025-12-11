import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.models as models
from fairseq.optim.adafactor import Adafactor

# ==============================================================================
# [Part 1] Sub-Modules
# ==============================================================================

class TSEmbedder(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(TSEmbedder, self).__init__()
        self.ts_embedder = nn.GRU(
            input_size=input_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        return self.dropout(self.ts_embedder(x)[0])

class AttributeEncoder(nn.Module):
    def __init__(self, num_cat, num_col, num_fab, num_store, embedding_dim):
        super(AttributeEncoder, self).__init__()
        self.cat_embedder = nn.Embedding(num_cat, embedding_dim)
        self.col_embedder = nn.Embedding(num_col, embedding_dim)
        self.fab_embedder = nn.Embedding(num_fab, embedding_dim)
        self.store_embedder = nn.Embedding(num_store, embedding_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, cat, col, fab, store):
        cat_emb = self.dropout(self.cat_embedder(cat))
        col_emb = self.dropout(self.col_embedder(col))
        fab_emb = self.dropout(self.fab_embedder(fab))
        store_emb = self.dropout(self.store_embedder(store))
        return cat_emb + col_emb + fab_emb + store_emb

class TemporalFeatureEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.day_embedding = nn.Linear(1, embedding_dim)
        self.week_embedding = nn.Linear(1, embedding_dim)
        self.month_embedding = nn.Linear(1, embedding_dim)
        self.year_embedding = nn.Linear(1, embedding_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, temporal_features):
        d = self.dropout(self.day_embedding(temporal_features[:, 0].unsqueeze(1)))
        w = self.dropout(self.week_embedding(temporal_features[:, 1].unsqueeze(1)))
        m = self.dropout(self.month_embedding(temporal_features[:, 2].unsqueeze(1)))
        y = self.dropout(self.year_embedding(temporal_features[:, 3].unsqueeze(1)))
        return d + w + m + y

class ImageEncoder(nn.Module):
    def __init__(self, embedding_dim=300):
        super(ImageEncoder, self).__init__()
        ft_ex_modules = list(models.resnet101(pretrained=True).children())[:-2]
        self.cnn = nn.Sequential(*ft_ex_modules)
        for p in self.cnn.parameters(): p.requires_grad = False
        for c in list(self.cnn.children())[6:]:
            for p in c.parameters(): p.requires_grad = True
        self.fc = nn.Linear(2048, embedding_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.cnn(x) 
        x = x.flatten(2).permute(0, 2, 1) 
        return self.dropout(self.fc(x))

class AdditiveAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(AdditiveAttention, self).__init__()
        self.encoder_linear = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.decoder_linear = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.attn_linear = nn.Linear(attention_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        h_j = self.encoder_linear(encoder_out)
        s_i = self.decoder_linear(decoder_hidden).squeeze(0)
        energy = self.attn_linear(self.tanh(h_j + s_i.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(energy)
        attention_weighted_encoding = alpha.unsqueeze(2) * encoder_out
        return attention_weighted_encoding, alpha

# ==============================================================================
# [Part 2] Main Model (2-10 Task with Full Features)
# ==============================================================================

class CrossAttnRNN(pl.LightningModule):
    def __init__(
        self,
        attention_dim,
        embedding_dim,
        hidden_dim,
        cat_dict, col_dict, fab_dict, store_num, num_trends, 
        use_img=True,
        out_len=10, 
        use_teacher_forcing=True,
        teacher_forcing_ratio=0.5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.use_teacher_forcing = use_teacher_forcing
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.out_len = out_len
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.use_img = use_img
        
        # Encoders
        self.image_encoder = ImageEncoder(embedding_dim)
        self.trend_encoder = TSEmbedder(num_trends, embedding_dim)
        self.temp_encoder = TemporalFeatureEncoder(embedding_dim)
        self.attribute_encoder = AttributeEncoder(
            len(cat_dict)+1, len(col_dict)+1, len(fab_dict)+1, store_num+1, embedding_dim
        )
        self.sales_encoder_gru = nn.GRU(input_size=1, hidden_size=hidden_dim, batch_first=True)

        # Attention
        self.ts_self_attention = nn.MultiheadAttention(embedding_dim, num_heads=4, dropout=0.1)
        self.ts_attention = AdditiveAttention(embedding_dim, hidden_dim, attention_dim)
        self.trend_linear = nn.Linear(52*attention_dim, embedding_dim)
        
        self.img_attention = AdditiveAttention(embedding_dim, hidden_dim, attention_dim)
        self.multimodal_attention = AdditiveAttention(embedding_dim, hidden_dim, attention_dim)
        self.multimodal_embedder = nn.Linear(embedding_dim, embedding_dim)

        # Decoder
        self.decoder_gru = nn.GRU(
            input_size=embedding_dim + 1, 
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.decoder_fc = nn.Linear(hidden_dim, 1)

    def forward(self, X, y, categories, colors, fabrics, stores, temporal_features, gtrends, images):
        # 1. Dynamic Data (X, y) Flatten
        num_windows = 1
        if X.dim() == 3:
            bs, num_windows, hist_len = X.shape
            X = X.reshape(bs * num_windows, hist_len)
            
            # y도 2차원 [Total_Batch, Out_Len]으로 펼침
            if y is not None:
                y = y.reshape(bs * num_windows, -1)
        else:
            bs = X.shape[0]

        # GRU Input Fix
        if X.dim() == 2:
            X = X.unsqueeze(-1)

        # ======================================================================
        # [Static Data Encoding]
        # ======================================================================
        img_encoding = self.image_encoder(images) 
        gtrend_encoding = self.trend_encoder(gtrends.permute(0,2,1)) 
        dummy_encoding = self.temp_encoder(temporal_features)
        attribute_encoding = self.attribute_encoder(categories, colors, fabrics, stores)

        # 복제 (Window 개수만큼)
        if num_windows > 1:
            img_encoding = img_encoding.repeat_interleave(num_windows, dim=0)
            gtrend_encoding = gtrend_encoding.repeat_interleave(num_windows, dim=0) 
            dummy_encoding = dummy_encoding.repeat_interleave(num_windows, dim=0)
            attribute_encoding = attribute_encoding.repeat_interleave(num_windows, dim=0)

        # Trend Self-Attention
        gtrend_encoding = gtrend_encoding.permute(1, 0, 2) 
        gtrend_encoding, _ = self.ts_self_attention(
            gtrend_encoding, gtrend_encoding, gtrend_encoding
        ) 

        # --- Encode Past Sales ---
        _, sales_hidden = self.sales_encoder_gru(X) 
        
        decoder_hidden = sales_hidden 
        decoder_input = X[:, -1, :].unsqueeze(1) 

        outputs = []
        total_bs = X.shape[0] 

        # --- Decoding Loop ---
        for t in range(self.out_len):
            attended_img, _ = self.img_attention(img_encoding, decoder_hidden)
            attended_img = attended_img.sum(1) 

            attended_trend, _ = self.ts_attention(gtrend_encoding.permute(1,0,2), decoder_hidden)
            attended_trend = self.trend_linear(attended_trend.view(total_bs, -1))

            mm_in = torch.stack([
                dummy_encoding,         # Temporal
                attended_img,           # Image
                attribute_encoding,     # Attributes
                attended_trend          # Trends
            ], dim=1)

            attended_mm, _ = self.multimodal_attention(mm_in, decoder_hidden)
            
            final_context = mm_in + attended_mm
            final_context = self.multimodal_embedder(final_context.sum(1)).unsqueeze(1)

            gru_input = torch.cat([final_context, decoder_input], dim=2)
            decoder_out, decoder_hidden = self.decoder_gru(gru_input, decoder_hidden)
            pred = self.decoder_fc(decoder_out)
            outputs.append(pred)

            # [수정된 Teacher Forcing 부분]
            if self.use_teacher_forcing and y is not None:
                is_teacher = (torch.rand(1) < self.teacher_forcing_ratio)
                if is_teacher:
                    # y는 2차원 [Batch, Out_Len] 이므로 y[:, t]로 접근해야 함
                    # 그 후 unsqueeze 2번 -> [Batch, 1, 1]
                    decoder_input = y[:, t].unsqueeze(1).unsqueeze(1)
                else:
                    decoder_input = pred
            else:
                decoder_input = pred

        return torch.cat(outputs, dim=1).squeeze(-1), None

    def configure_optimizers(self):
        return [Adafactor(self.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)]

    def on_train_epoch_start(self):
        self.use_teacher_forcing = True

    def on_validation_epoch_start(self):
        self.use_teacher_forcing = False

    def training_step(self, batch, batch_idx):
        (X, y, cat, col, fab, store, temp, gtrend), images = batch
        
        forecasts, _ = self.forward(X, y, cat, col, fab, store, temp, gtrend, images)
        
        if y.dim() == 3:
            bs, num_windows, out_len = y.shape
            y = y.reshape(bs * num_windows, out_len)

        loss = F.mse_loss(y, forecasts) 
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        (X, y, cat, col, fab, store, temp, gtrend), images = batch
        
        forecasts, _ = self.forward(X, y, cat, col, fab, store, temp, gtrend, images)
        
        if y.dim() == 3:
            bs, num_windows, out_len = y.shape
            y = y.reshape(bs * num_windows, out_len)
            
        return y, forecasts

    def validation_epoch_end(self, outputs):
        gt = torch.cat([x[0] for x in outputs])
        pred = torch.cat([x[1] for x in outputs])
        
        if gt.shape != pred.shape:
             # 예측값이 [N, 1]로 잘못 나오거나 하면 여기서 체크
             # 현재 코드는 view나 squeeze 처리가 되어 있어 맞을 것임
             pass

        norm_scalar = 53.0
        mae = F.l1_loss(gt * norm_scalar, pred * norm_scalar)
        wape = 100 * torch.sum(torch.abs((gt - pred) * norm_scalar)) / torch.sum(torch.abs(gt * norm_scalar))
        
        self.log("val_mae", mae)
        self.log("val_wWAPE", wape)
        
        lr = self.optimizers().param_groups[0]['lr']
        if lr is None:
            lr_val = 0.0
        elif torch.is_tensor(lr):
            lr_val = lr.item()
        else:
            lr_val = lr

        print(f"Validation MAE: {mae:.4f}, WAPE: {wape:.4f}, LR: {lr_val:.8f}")