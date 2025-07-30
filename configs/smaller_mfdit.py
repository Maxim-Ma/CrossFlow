import ml_collections
from dataclasses import dataclass

@dataclass
class Args:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

# --------------------- 模型大小相关 ------------------
model = Args(
    latent_size = 32,          # ↓ 32 → 24   （latent H=W=32 时显存最大；24 能省 ≈44% token 数）
    learn_sigma = False,
    channels = 4,
    block_grad_to_lowres = False,  # 打断低分支梯度，省一次 backward
    norm_type = "TDRMSN",
    use_t2i = True,
    clip_dim = 768,
    num_clip_token = 77,
    gradient_checking = True,    # 关闭 hooks，每步少一次检查
    image_as_VAEgt = 5,           # 用 VAE latent 直接当 image latent，不再训练 VAE‑decoder
    only_training_diff = True,
    cfg_indicator = 0.10,
    textVAE = Args(
        version = 1.1,
        num_blocks = 11,        # ↓ 11 → 6    Transformer block 数减半
        hidden_dim = 1024,      # ↓ 1024 → 512
        hidden_token_length = 256, # ↓ 256 → 128
        num_attention_heads = 8,   # ↓ 8 → 4
        dropout_prob = 0.1,
    ),
)
# -----------------------------------------------------

def d(**kwargs): return ml_collections.ConfigDict(initial_dictionary=kwargs)

def get_config():
    config = ml_collections.ConfigDict()
    config.seed = 1234

    # ↓ 对应 latent_size 调整
    config.z_shape = (4, 32, 32)

    config.autoencoder = d(
        pretrained_path='assets/stable-diffusion/autoencoder_kl.pth',
        scale_factor=0.23010
    )

    # --- 训练超参 ---
    config.train = d(
        n_steps = 100,    # 先跑 20 万步验证能否收敛，后面再加
        batch_size = 4,       # ↓ 4 → 2，再配合 grad‑accum 可维持有效 batch
        mode = 'cond',
        log_interval = 1000,
        eval_interval = 500,
        save_interval = 2000,
        n_samples_eval=5,
        grad_accum_steps = 2,       # 2 步累积一次梯度
        # mixed_precision = 'fp16',  # 让 accelerate 自动开启 FP16
    )

    config.optimizer = d(
        name = 'adamw',
        lr = 1e-4,            # batch 变小可稍提 LR，经验值
        weight_decay = 0.03,
        betas = (0.9, 0.9),
    )

    config.lr_scheduler = d(
        name = 'customized',
        warmup_steps = 2_000, # 同步缩短 warm‑up
    )

    global model
    config.nnet = d(name='mfdit', model_args=model)
    config.loss_coeffs = []

    config.dataset = d(
        name='JDB_demo_features',
        resolution=256,
        llm='clip',
        train_path='/data/suizhi/data/training_JourneyDB/',
        val_path='/data/suizhi/data/validation_coco2014/',
        cfg=False,
    )

    config.sample = d(
        sample_steps = 50,
        n_samples = 10_000,
        mini_batch_size = 10,
        cfg = False,
        scale = 7,
        path = ''
    )

    return config
