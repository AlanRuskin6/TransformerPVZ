import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PVZAttentionExtractor(BaseFeaturesExtractor):
    """
    升级版多模态注意力特征提取器 + 记忆增强
    
    创新点:
    1. 修正威胁通道索引 (通道11)
    2. 可学习门控威胁注入
    3. 出怪预告单独编码
    4. 跨模态注意力 (Grid ↔ Global+Card)
    5. 多尺度空间感知 (前/中/后排)
    6. **短期记忆 (LSTM)**: 记住最近序列
    7. **长期记忆 (Memory Bank)**: 可检索的关键历史状态
    8. **循环注意力**: 关注历史重要时刻
    
    输入 obs:
        grid: (B, 5, 9, 13)
        global_features: (B, 71)
        card_attributes: (B, 10, 7)
    输出 features: (B, ff_dim)
    """

    def __init__(
        self,
        observation_space,
        hidden_size: int = 128,
        attn_heads: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.1,
        num_layers: int = 2,
        memory_size: int = 32,  # 长期记忆容量
        lstm_layers: int = 1,   # LSTM层数
    ):
        grid_shape = observation_space["grid"].shape  # (5, 9, 13)
        self.rows, self.cols, self.channels = grid_shape
        self.global_dim = observation_space["global_features"].shape[0]  # 71
        
        # 卡片属性特征
        self.has_card_attrs = "card_attributes" in observation_space.spaces
        self.card_attr_dim = 0
        if self.has_card_attrs:
            card_shape = observation_space["card_attributes"].shape  # (10, 7)
            self.card_attr_dim = card_shape[0] * card_shape[1]  # 70

        super().__init__(observation_space, features_dim=ff_dim)

        # === Grid编码器 ===
        self.grid_embed = nn.Linear(self.channels, hidden_size)
        self.row_embed = nn.Embedding(self.rows, hidden_size)
        self.col_embed = nn.Embedding(self.cols, hidden_size)

        # [CLS] Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        nn.init.normal_(self.cls_token, std=0.02)

        # === 威胁感知模块 (可学习门控) ===
        self.threat_gate = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Sigmoid()  # 门控，输出 0-1
        )
        self.threat_proj = nn.Linear(1, hidden_size)  # 威胁值投影
        
        # === 多尺度空间感知 Token (前/中/后排) ===
        self.zone_tokens = nn.Parameter(torch.zeros(3, 1, hidden_size))  # 前中后3个区域
        nn.init.normal_(self.zone_tokens, std=0.02)

        # === Grid Transformer (Self-Attention) ===
        self.grid_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=attn_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True
            )
            for _ in range(num_layers)
        ])
        
        # === 全局特征编码器 (含出怪预告单独处理) ===
        # 动态计算分割点：最后10维是出怪预告，其余是基础特征
        self.spawn_preview_dim = 10
        self.global_base_dim = self.global_dim - self.spawn_preview_dim
        
        self.global_encoder = nn.Sequential(
            nn.Linear(self.global_base_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 出怪预告专用编码器 (强化战略信息)
        self.spawn_encoder = nn.Sequential(
            nn.Linear(self.spawn_preview_dim, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        
        # === 卡片属性编码器 ===
        if self.has_card_attrs:
            self.card_encoder = nn.Sequential(
                nn.Linear(self.card_attr_dim, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        
        # === 跨模态注意力 (Grid ↔ Global+Card) ===
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=attn_heads,
            dropout=dropout,
            batch_first=True
        )
        self.cross_norm = nn.LayerNorm(hidden_size)
        
        # === 记忆机制 ===
        self.memory_size = memory_size
        self.hidden_size = hidden_size
        
        # 1. 短期记忆 (LSTM) - 记住最近时序
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        self.lstm_layers = lstm_layers
        
        # LSTM隐藏状态 (需要在每个episode开始时重置)
        # 使用 persistent=False 防止保存到模型文件中，避免 batch_size 不匹配问题
        self.register_buffer('lstm_h', torch.zeros(lstm_layers, 1, hidden_size), persistent=False)
        self.register_buffer('lstm_c', torch.zeros(lstm_layers, 1, hidden_size), persistent=False)
        
        # 2. 长期记忆库 (Memory Bank) - 可检索的关键状态
        self.register_buffer('memory_bank', torch.zeros(1, memory_size, hidden_size), persistent=False)
        self.register_buffer('memory_ptr', torch.zeros(1, dtype=torch.long), persistent=False)  # 写入指针
        
        # 记忆注意力 (查询当前状态 ↔ 历史记忆)
        self.memory_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=attn_heads,
            dropout=dropout,
            batch_first=True
        )
        self.memory_norm = nn.LayerNorm(hidden_size)
        
        # 记忆门控 (决定何时写入重要状态)
        self.memory_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # === 最终融合层 ===
        # 新维度: CLS(128) + Zones(384) + Global(128) + Spawn(128) + Card(128) 
        #         + LSTM(128) + Memory(128) = 1152
        # 计算: 128 + 384 + 128 + 128 + 128 + 128 + 128 = 1152
        fusion_dim = hidden_size * 9  # CLS(1) + Zones(3) + Global(1) + Spawn(1) + Card(1) + LSTM(1) + Memory(1)
        self.final = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, ff_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim * 2, ff_dim),
        )
        
        # 可视化
        self.last_attn_weights = None
        self.last_memory_weights = None

    def forward(self, observations: dict) -> torch.Tensor:
        grid = observations["grid"]  # (B, 5, 9, 13)
        bsz = grid.shape[0]
        
        # === 1. 提取威胁热力图 (通道11) ===
        threat_channel_idx = 11
        if grid.shape[-1] > threat_channel_idx:
            threat_map = grid[..., threat_channel_idx:threat_channel_idx+1]  # (B, 5, 9, 1)
        else:
            threat_map = torch.zeros(bsz, self.rows, self.cols, 1, device=grid.device)
        
        threat_flat = threat_map.view(bsz, self.rows * self.cols, 1)  # (B, 45, 1)
        
        # === 2. Grid Embedding + 可学习门控威胁注入 ===
        seq = grid.view(bsz, self.rows * self.cols, self.channels)  # (B, 45, 13)
        base = self.grid_embed(seq)  # (B, 45, 128)
        
        # 威胁门控注入 (比固定系数更智能)
        threat_gate = self.threat_gate(threat_flat)  # (B, 45, 128), sigmoid门控
        threat_proj = self.threat_proj(threat_flat)  # (B, 45, 128), 威胁投影
        base = base + threat_gate * threat_proj  # 门控加权
        
        # === 3. 位置编码 ===
        row_ids = torch.arange(self.rows, device=grid.device).repeat_interleave(self.cols)
        col_ids = torch.arange(self.cols, device=grid.device).repeat(self.rows)
        pos = self.row_embed(row_ids) + self.col_embed(col_ids)
        pos = pos.unsqueeze(0).expand(bsz, -1, -1)
        
        x = base + pos  # (B, 45, 128)
        
        # === 4. 拼接 [CLS] + 多尺度Zone Tokens ===
        cls_tokens = self.cls_token.expand(bsz, -1, -1)  # (B, 1, 128)
        zone_tokens = self.zone_tokens.expand(-1, bsz, -1).transpose(0, 1)  # (B, 3, 128)
        x = torch.cat([cls_tokens, zone_tokens, x], dim=1)  # (B, 4+45=49, 128)
        
        # === 5. Grid Self-Attention ===
        for i, layer in enumerate(self.grid_layers):
            x = layer(x)
            
            # 可视化 (评估模式)
            if i == len(self.grid_layers) - 1 and not self.training:
                with torch.no_grad():
                    _, weights = layer.self_attn(x, x, x, need_weights=True, average_attn_weights=True)
                    self.last_attn_weights = weights[:, 0, 4:]  # CLS对45个格子的注意力
        
        # 提取特征
        cls_feat = x[:, 0]  # (B, 128) - CLS Token
        zone_feats = x[:, 1:4]  # (B, 3, 128) - 前中后区域
        zone_feats_flat = zone_feats.reshape(bsz, -1)  # (B, 384)
        
        # === 6. 全局特征编码 (分离出怪预告) ===
        global_raw = observations["global_features"]  # (B, 71)
        
        # 前61维: 基础全局特征
        global_base = global_raw[:, :self.global_base_dim]  # (B, 61)
        global_feat = self.global_encoder(global_base)  # (B, 128)
        
        # 后10维: 出怪预告 (战略信息强化)
        spawn_preview = global_raw[:, self.global_base_dim:]  # (B, 10)
        spawn_feat = self.spawn_encoder(spawn_preview)  # (B, 128)
        
        # === 7. 卡片属性编码 ===
        if self.has_card_attrs:
            card_attrs = observations["card_attributes"]  # (B, 10, 7)
            card_flat = card_attrs.view(bsz, -1)  # (B, 70)
            card_feat = self.card_encoder(card_flat)  # (B, 128)
        else:
            card_feat = torch.zeros(bsz, self.hidden_size, device=grid.device)
        
        # === 8. 跨模态注意力 (Grid CLS ↔ Global+Spawn+Card) ===
        # Query: Grid CLS, Key/Value: Global+Spawn+Card
        context = torch.stack([global_feat, spawn_feat, card_feat], dim=1)  # (B, 3, 128)
        cls_query = cls_feat.unsqueeze(1)  # (B, 1, 128)
        
        cross_out, _ = self.cross_attn(
            query=cls_query,
            key=context,
            value=context
        )  # (B, 1, 128)
        
        cls_feat_enhanced = self.cross_norm(cls_feat + cross_out.squeeze(1))  # 残差连接
        
        # === 9. 短期记忆 (LSTM) ===
        # 扩展 batch 维度的隐藏状态
        if self.lstm_h.shape[1] != bsz:
            self.lstm_h = self.lstm_h.expand(-1, bsz, -1).contiguous()
            self.lstm_c = self.lstm_c.expand(-1, bsz, -1).contiguous()
        
        lstm_input = cls_feat_enhanced.unsqueeze(1)  # (B, 1, 128)
        lstm_out, (h_new, c_new) = self.lstm(lstm_input, (self.lstm_h, self.lstm_c))
        lstm_feat = lstm_out.squeeze(1)  # (B, 128)
        
        # 更新隐藏状态 (必须 detach 防止梯度累积)
        self.lstm_h = h_new.detach()
        self.lstm_c = c_new.detach()
        
        # === 10. 长期记忆 (Memory Bank) ===
        # 扩展 batch 维度的记忆库
        if self.memory_bank.shape[0] != bsz:
            self.memory_bank = self.memory_bank.expand(bsz, -1, -1).contiguous()
            self.memory_ptr = self.memory_ptr.expand(bsz).contiguous()
        
        # 查询记忆库 (当前状态 ↔ 历史记忆)
        query = cls_feat_enhanced.unsqueeze(1)  # (B, 1, 128)
        memory_out, memory_weights = self.memory_attn(
            query=query,
            key=self.memory_bank,
            value=self.memory_bank,
            need_weights=True
        )  # (B, 1, 128), (B, 1, memory_size)
        memory_feat = self.memory_norm(cls_feat_enhanced + memory_out.squeeze(1))  # (B, 128)
        
        # 可视化记忆注意力
        if not self.training:
            self.last_memory_weights = memory_weights.detach()
        
        # 写入记忆库 (门控机制:只保存重要状态)
        importance = self.memory_gate(cls_feat_enhanced)  # (B, 1)
        
        for b in range(bsz):
            if importance[b] > 0.5:  # 重要性阈值
                ptr = self.memory_ptr[b].item()
                self.memory_bank[b, ptr] = cls_feat_enhanced[b].detach()
                self.memory_ptr[b] = (ptr + 1) % self.memory_size
        
        # === 11. 最终融合 ===
        # CLS(128) + Zones(384) + Global(128) + Spawn(128) + Card(128) 
        # + LSTM(128) + Memory(128) = 1152
        combined = torch.cat([
            cls_feat_enhanced,
            zone_feats_flat,
            global_feat,
            spawn_feat,
            card_feat,
            lstm_feat,
            memory_feat
        ], dim=1)  # (B, 1152)
        
        return self.final(combined)  # (B, 256)
    
    def reset_memory(self, batch_size: int = 1, device=None):
        """
        重置记忆状态 (新Episode开始时调用)
        
        Args:
            batch_size: 批次大小
            device: 设备 (None则自动推断)
        """
        if device is None:
            device = self.lstm_h.device
        
        # 重置LSTM隐藏状态
        self.lstm_h = torch.zeros(
            self.lstm_layers, batch_size, self.hidden_size, 
            device=device
        )
        self.lstm_c = torch.zeros(
            self.lstm_layers, batch_size, self.hidden_size,
            device=device
        )
        
        # 重置记忆库
        self.memory_bank = torch.zeros(
            batch_size, self.memory_size, self.hidden_size,
            device=device
        )
        self.memory_ptr = torch.zeros(batch_size, dtype=torch.long, device=device)
