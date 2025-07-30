import ml_collections
from dataclasses import dataclass

@dataclass
class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

model = Args(
    channels = 4,
    block_grad_to_lowres = False,
    norm_type = "TDRMSN",
    use_t2i = True,
    clip_dim=768,
    num_clip_token=77,
    gradient_checking=True, # for larger model
    image_as_VAEgt=5, # 使用真实的图片作为VAE的监督信号！(recon image instead of text); 1 = VAE latent作为image latent，不训练VAE的decoder了; 2 = VAE output 作为image latent，需要训练VAE的decoder; 3 = VAE latent作为image latent，但是使用了SD VAE的decoder; 4/5 = follow Dall-e,使用CLIP 训练 (5: 只针对SS，不需要从VAE的decoder得到图了)
    # image_as_VAEgt_blur=True, # 使用blur的image作为VAEgt
    # use_SD_decoder=True, # 使用了 pretrain 的 SD decoder 作为 VAE 的decoder: 仅适用于 image_as_VAEgt=3/4 的情况 (4 是因为我们现在是只load了image latent，所以训练的时候要decode回去； 但是5直接load了SS的原图，就不需要了)
    # not_training_diff=True, # 只训练text 部分
    only_training_diff=True, # 只训练 diffusion 部分
    cfg_indicator=0.1, # 一种只针对 dir mapping 的 cfg 方式？对于有 indicator 的 item，打乱 target，同时在 sequence 上进行一个 t 的 concat
    textVAE = Args(
        version = 1.1, #LQH:目前只有DirConXatt支持这个 - 1.1 指的是1那个结构，但是只使用一个encoder （double了最后一层layer，分别生成mu和var）
        num_blocks = 11,
        # num_down_sample_block = 3,
        hidden_dim = 1024, # 可能要跟clip_dim差不多？？
        hidden_token_length = 256,
        num_attention_heads = 8,
        dropout_prob = 0.1,
    ),
    # full_ckpt = '../exp1203_dFM_SSclip_arc3-11m-vaeV1.1B11pt500K-6-2-KLv5mp3v1.1en2_imgGT5.2_lr1en4_cfgLeInd_16Node_onlyCKPT/130000.ckpt/nnet_ema.pth',
    stage_configs = [
            Args(
                block_type = "TransformerBlock", 
                dim = 960,
                hidden_dim = 1920,
                num_attention_heads = 16,
                num_blocks = 39,
                max_height = 16,
                max_width = 16,
                image_input_ratio = 1,
                input_feature_ratio = 2,
                final_kernel_size = 3,
                dropout_prob = 0,
            ),
            Args(
                block_type = "ConvNeXtBlock", 
                dim = 480, 
                hidden_dim = 960, 
                kernel_size = 7, 
                num_blocks = 20,
                max_height = 32,
                max_width = 32,
                image_input_ratio = 1,
                input_feature_ratio = 1,
                final_kernel_size = 3,
                dropout_prob = 0,
            ),
    ],
)

def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234
    # config.pred = 'noise_pred'
    config.z_shape = (4, 32, 32)

    config.autoencoder = d(
        # pretrained_path='assets/stable-diffusion/autoencoder_kl_ema.pth', # TODO, why??
        pretrained_path='assets/stable-diffusion/autoencoder_kl.pth',
        scale_factor=0.23010
    )

    config.train = d(
        n_steps=1000000,
        batch_size=4,
        mode='cond',
        log_interval=1000,
        eval_interval=1000,
        save_interval=10000,
        n_samples_eval=5,
    )

    config.optimizer = d(
        name='adamw',
        lr=0.0001,
        weight_decay=0.03,
        betas=(0.9, 0.9),
    )

    config.lr_scheduler = d(
        name='customized',
        warmup_steps=5000
    )

    global model
    config.nnet = d(
        name='dimr',
        model_args=model,
    )
    config.loss_coeffs = [1/4, 1]
    
    config.dataset = d(
        name='JDB_demo_features',
        resolution=256,
        llm='clip',
        train_path='/data/suizhi/data/training_JourneyDB/',
        val_path='/data/suizhi/data/validation_coco2014/',
        cfg=False,
        p_uncond=0.1
    )

    config.sample = d(
        sample_steps=50,
        n_samples=30000,
        mini_batch_size=10,  # the decoder is large
        algorithm='dpm_solver',
        cfg=False,
        scale=2,  # LQH: changed to DiT verision
        path=''
    )

    return config
