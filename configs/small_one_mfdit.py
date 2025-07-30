import ml_collections
from dataclasses import dataclass

@dataclass
class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

model = Args(
    latent_size = 32, # should match z_shape
    learn_sigma = False, # different from DiT, we direct predict noise here
    channels = 4,
    block_grad_to_lowres = False,
    norm_type = "TDRMSN",
    use_t2i = True,
    clip_dim=768,
    num_clip_token=77,
    gradient_checking=True, # for larger model
    image_as_VAEgt=5, # 使用真实的图片作为VAE的监督信号！(recon image instead of text); 1 = VAE latent作为image latent，不训练VAE的decoder了; 2 = VAE output 作为image latent，需要训练VAE的decoder; 3 = VAE latent作为image latent，但是使用了SD VAE的decoder; 4/5 = follow Dall-e,使用CLIP 训练 (5: 只针对SS，不需要从VAE的decoder得到图了),
    only_training_diff=True,
    cfg_indicator=0.10,
    textVAE = Args(
        version = 1.1,
        num_blocks = 11,
        hidden_dim = 1024,
        hidden_token_length = 256,
        num_attention_heads = 8,
        dropout_prob = 0.1,
    ),
)

def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234
    config.z_shape = (4, 32, 32)

    config.autoencoder = d(
        pretrained_path='assets/stable-diffusion/autoencoder_kl.pth',
        scale_factor=0.23010
    )

    config.train = d(
        n_steps=1000000,
        batch_size=4,
        mode='cond',
        log_interval=10,
        eval_interval=100,
        save_interval=1000,
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
        name='mfdit',
        model_args=model,
    )
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
        sample_steps=50,
        n_samples=30000,
        mini_batch_size=10,
        cfg=False,
        scale=7,
        path=''
    )

    return config
