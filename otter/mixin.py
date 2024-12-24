import random
from pathlib import Path
from typing import Any, Dict, List

import cv2
import lightning as L
from chameleon import build_optimizer
from torch.nn import Identity

__all__ = [
    'BaseMixin', 'BorderValueMixin', 'FillValueMixin',
]


class BaseMixin(L.LightningModule):
    """
    BaseMixin 主要負責：
    1. 接收 YAML 中的參數配置 (cfg)。
    2. 核心功能：建立並配置優化器、學習率策略 (scheduler)。
    3. 根據使用者設定，進行推理 (forward)。

    YAML 配置說明：
    ==============================================================================
    一般結構 (關鍵節點)：
    ------------------------------------------------------------------------------
    common:
      batch_size: int                     # 每個批次的大小
      is_restore: bool                    # 是否從某個 checkpoint 中恢復
      restore_ind: str                    # 恢復 checkpoint 的索引
      restore_ckpt: str                   # 恢復的 ckpt 路徑
      preview_batch: int                  # 進行預覽或可視化時，想處理多少資料

    global_settings:
      image_size: [int, int]             # 圖像大小 (寬, 高)

    trainer:                              # Lightning Trainer 相關參數
      max_epochs: int
      precision: str                      # bf16、fp16 或其他混合精度
      val_check_interval: float
      gradient_clip_val: int
      accumulate_grad_batches: int
      accelerator: str                    # 使用 GPU、CPU 或其他裝置
      devices: [int 或 str]

    model:                                # 模型相關參數
      name: str                           # 模型名稱
      backbone:
        name: str                         # Backbone 模組名稱
        options:                          # 與 backbone 相關的額外配置
          name: str
          pretrained: bool
          features_only: bool
      neck:
        name: str                         # Neck 模組名稱
        options:                          # 與 neck 相關的額外配置
          d_model: int
          image_size: [int, int]
          patch_size: int
          nhead: int
          num_layers: int
          num_classes: int
          using_feature_stage: List
      head:
        name: str                         # Head 模組名稱
        options:                          # 與 head 相關的額外配置
          d_model: int
          image_size: [int, int]
          patch_size: int
          nhead: int
          num_layers: int
          num_classes: int
          using_feature_stage: List[int]

    onnx:                                 # ONNX 導出時相關的設定
      input_shape:
        input: [N, C, H, W]              # 輸入張量維度
      input_names: [str]
      output_names: [str]
      options:
        opset_version: int
        verbose: bool
        do_constant_folding: bool

    dataset:                              # 資料集相關配置
      train_options:
        name: str                         # 訓練集 Dataset 名稱
        options:
          image_size: int
          aug_ratio: float
          return_tensor: bool
          length_of_dataset: int
      valid_options:
        name: str                         # 驗證集 Dataset 名稱
        options:
          image_size: int
          return_tensor: bool

    dataloader:                           # 資料載入器的參數
      train_options:
        batch_size: int
        num_workers: int
        shuffle: bool
        drop_last: bool
      valid_options:
        batch_size: int
        num_workers: int
        shuffle: bool
        drop_last: bool

    optimizer:                            # 優化器設定
      name: str                           # 如 AdamW
      options:
        lr: float
        betas: [float, float]
        weight_decay: float

    lr_scheduler:                         # 學習率策略
      name: str                           # 如 PolynomialLRWarmup
      options:
        warmup_iters: int
        total_iters: int
      pl_options:                         # 與 Lightning 結合時的一些設定
        monitor: str
        interval: str

    callbacks:                            # 需掛載的回呼函數列表
      - name: ModelCheckpoint
        options:
          monitor: str
          mode: str
          verbose: bool
          save_last: bool
          save_top_k: int
      - name: LearningRateMonitor
        options:
          logging_interval: str
      - name: RichModelSummary
        options:
          max_depth: int
      - name: CustomTQDMProgressBar
        options:
          unit_scale: int

    logger:                               # 紀錄器 (Logger) 相關參數
      name: str
      options:
        save_dir: str

    ==============================================================================
    注意：以上僅列出 YAML 中的主要結構與建議參數名稱。若要在程式中取用其它自訂的參數，
    可在此 class 初始化時自行擴充檢查機制 (例如 self.cfg['some_key']...)。
    ==============================================================================
    """

    def __init__(self, cfg: Dict[str, Any]):
        """
        Args:
            cfg (Dict[str, Any]): 來自 YAML 或其他方式讀取的整體配置參數。
        """
        super().__init__()

        # === 基本檢查與提示 ===
        if not isinstance(cfg, dict):
            raise ValueError("cfg 參數必須是一個 dict。請檢查傳入的配置。")

        # 檢查 common 是否存在
        if 'common' not in cfg:
            raise ValueError("cfg 缺少 'common' 配置，請在 cfg 中定義 'common'。")

        # 檢查 preview_batch 是否存在
        if 'preview_batch' not in cfg['common']:
            raise ValueError(
                "cfg['common'] 缺少 'preview_batch' 參數，請在 cfg['common'] 中定義。")

        # （若需要 max_text_length，請在此加入檢查）
        # if 'max_text_length' not in cfg['common']:
        #     raise ValueError("cfg['common'] 缺少 'max_text_length' 參數，請在 cfg['common'] 中定義。")

        # 檢查 optimizer 與 lr_scheduler 是否存在
        if 'optimizer' not in cfg:
            raise ValueError("cfg 缺少 'optimizer' 配置，請在 cfg 中定義 'optimizer'。")
        if 'lr_scheduler' not in cfg:
            raise ValueError(
                "cfg 缺少 'lr_scheduler' 配置，請在 cfg 中定義 'lr_scheduler'。")

        self.cfg = cfg
        self.preview_batch = cfg['common']['preview_batch']
        self.backbone = Identity()
        self.neck = Identity()
        self.head = Identity()

        # 進行優化器與學習率 Scheduler 相關的配置
        self.apply_solver_config(cfg['optimizer'], cfg['lr_scheduler'])

        # for validation
        self.validation_step_outputs = []

    def forward(self, x):
        """
        預設 forward 流程，可根據實際需求做擴充。
        """
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x

    def apply_solver_config(
        self,
        optimizer: Dict[str, Any],
        lr_scheduler: Dict[str, Any]
    ) -> None:
        """
        從 cfg['optimizer'] 與 cfg['lr_scheduler'] 取出必要參數進行設定。
        """
        # 在此可進行進一步的檢查
        if not optimizer:
            raise ValueError("optimizer 配置不可為空。請檢查 cfg['optimizer']。")
        if not lr_scheduler:
            raise ValueError("lr_scheduler 配置不可為空。請檢查 cfg['lr_scheduler']。")

        # 最少要包含兩個 key
        if len(optimizer.values()) < 2:
            raise ValueError("optimizer 配置內容不正確，需包含 name 和對應參數。")
        # lr_scheduler 通常包含三個 key: name, options, pl_options
        if len(lr_scheduler.values()) < 3:
            raise ValueError(
                "lr_scheduler 配置內容不正確，需包含 name、對應參數及 PL scheduler 設定。")

        # 這裡假設 optimizer 結構類似:
        # {
        #   "name": "AdamW",
        #   "options": {...}
        # }
        # 這裡的 name、options 對應到 optimizer.values()
        self.optimizer_name, self.optimizer_opts = optimizer.values()

        # lr_scheduler 結構類似:
        # {
        #   "name": "PolynomialLRWarmup",
        #   "options": {...},
        #   "pl_options": {...}
        # }
        # 這裡的 name、options、pl_options 對應到 lr_scheduler.values()
        self.sche_name, self.sche_opts, self.sche_pl_opts = lr_scheduler.values()

    def get_optimizer_params(self) -> List[Dict[str, Any]]:
        """
        將模型參數分組，以進行不同的 weight_decay 配置。
        """
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)],
                "weight_decay": self.optimizer_opts.get('weight_decay', 0.0),
            },
            {
                "params": [
                    p for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return optimizer_grouped_parameters

    def configure_optimizers(self):
        """
        使用 self.apply_solver_config(...) 後的配置，建立優化器與學習率調度器。
        """

        optimizer = build_optimizer(
            name=self.optimizer_name,
            params=self.get_optimizer_params(),
            **self.optimizer_opts
        )
        scheduler = build_optimizer(
            name=self.sche_name,
            optimizer=optimizer,
            **self.sche_opts
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                **self.sche_pl_opts
            }
        }

    def get_lr(self):
        """
        取得當前優化器的學習率（適用於只有一個優化器時）。
        """
        return self.trainer.optimizers[0].param_groups[0]['lr']

    @property
    def preview_dir(self) -> Path:
        """
        根據 current_epoch 進行區分，產生對應的預覽目錄。
        """
        img_path = Path(self.cfg.root_dir) / "preview" / \
            f'epoch_{self.current_epoch}'
        if not img_path.exists():
            img_path.mkdir(parents=True)
        return img_path


class BorderValueMixin:

    @property
    def pad_mode(self):
        return random.choice([
            cv2.BORDER_CONSTANT,
            cv2.BORDER_REPLICATE,
        ])

    @property
    def border_mode(self):
        return random.choice([
            cv2.BORDER_CONSTANT,
            cv2.BORDER_REPLICATE,
        ])

    @property
    def value(self):
        return [random.randint(0, 255) for _ in range(3)]

    @pad_mode.setter
    def pad_mode(self, x):
        return None

    @border_mode.setter
    def border_mode(self, x):
        return None

    @value.setter
    def value(self, x):
        return None


class FillValueMixin:

    @property
    def fill_value(self):
        return [random.randint(0, 255) for _ in range(3)]

    @fill_value.setter
    def fill_value(self, x):
        return None
