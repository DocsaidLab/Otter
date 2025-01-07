import random
from pathlib import Path
from typing import Any, Dict, List

import cv2
import lightning as L
from chameleon import build_optimizer

__all__ = [
    'BaseMixin', 'BorderValueMixin', 'FillValueMixin',
]

___doc__ =     """
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


class BaseMixin(L.LightningModule):

    def apply_solver_config(
        self,
        optimizer: Dict[str, Any],
        lr_scheduler: Dict[str, Any]
    ) -> None:
        self.optimizer_name, self.optimizer_opts = optimizer.values()
        self.sche_name, self.sche_opts, self.sche_pl_opts = lr_scheduler.values()

    def get_optimizer_params(self) -> List[Dict[str, Any]]:
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
        return self.trainer.optimizers[0].param_groups[0]['lr']

    @property
    def preview_dir(self) -> Path:
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
