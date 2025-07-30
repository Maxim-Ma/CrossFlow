import ml_collections
import torch
from torch import multiprocessing as mp
from datasets import get_dataset
from torchvision.utils import make_grid, save_image
import utils
import einops
from torch.utils._pytree import tree_map
import accelerate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import tempfile
from absl import logging
import builtins
import os
import wandb
import numpy as np
import time
import random

import libs.autoencoder
from libs.t5 import T5Embedder
from libs.clip import FrozenCLIPEmbedder
from diffusion.flow_matching import FlowMatching, ODEFlowMatchingSolver, ODEEulerFlowMatchingSolver
from tools.fid_score import calculate_fid_given_paths
from tools.clip_score import ClipSocre

import time, contextlib

@contextlib.contextmanager
def tic(tag, device="cuda"):
    start_cpu = time.perf_counter()
    start_gpu = torch.cuda.Event(enable_timing=True)
    end_gpu   = torch.cuda.Event(enable_timing=True)
    if torch.cuda.is_available():
        start_gpu.record()
    yield
    if torch.cuda.is_available():
        end_gpu.record()
        torch.cuda.synchronize()          # 等 GPU 事件完成
        gpu_ms = start_gpu.elapsed_time(end_gpu)
    else:
        gpu_ms = -1
    cpu_ms = (time.perf_counter() - start_cpu) * 1000
    logging.info(f"[TIMER] {tag:15s}  CPU: {cpu_ms:7.1f} ms  GPU: {gpu_ms:7.1f} ms")


# 全局禁用 Flash 和 Mem‑Efficient SDP
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)   # 显式启用 math kernel

import torch
from contextlib import suppress

def sizeof_fmt(num, suffix="B"):
    """把字节数转成易读单位"""
    for unit in ["", "K", "M", "G", "T"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}P{suffix}"

def summary_mem(model, optimizer=None):
    """
    打印模型（和可选优化器）的参数量与显存占用
    model:  nn.Module
    optimizer: torch.optim.Optimizer | None
    """
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())

    total_param = sum(p.numel() for p in model.parameters())
    trainable_param = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 60)
    print(f"可训练参数  : {trainable_param:,}")
    print(f"全部参数   : {total_param:,}")
    print(f"参数显存   : {sizeof_fmt(param_bytes)}")
    print(f"Buffer显存: {sizeof_fmt(buffer_bytes)}")

    if optimizer is not None:
        opt_bytes = 0
        for state in optimizer.state.values():
            for v in state.values():
                if torch.is_tensor(v):
                    opt_bytes += v.numel() * v.element_size()
        print(f"优化器状态 : {sizeof_fmt(opt_bytes)}")
        print(f"—— 参数 + Buffer + Optim : {sizeof_fmt(param_bytes + buffer_bytes + opt_bytes)}")
    else:
        print(f"—— 参数 + Buffer : {sizeof_fmt(param_bytes + buffer_bytes)}")
    print("=" * 60)

# 用法示例
# model = ...
# optimizer = ...
# summary_mem(model, optimizer)


def train(config):
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # grad_acc_steps = config.train.get("grad_accum_steps", 1)

    accelerator = accelerate.Accelerator(
        #  gradient_accumulation_steps=grad_acc_steps,
    )
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f'Process {accelerator.process_index} using device: {device}')

    config.mixed_precision = accelerator.mixed_precision
    config = ml_collections.FrozenConfigDict(config)

    logging.info(f"rank={accelerator.process_index}  reached checkpoint-A")


    assert config.train.batch_size % accelerator.num_processes == 0
    mini_batch_size = config.train.batch_size // accelerator.num_processes

    logging.info(f"rank={accelerator.process_index}  reached checkpoint-B")

    if accelerator.is_main_process:
        os.makedirs(config.ckpt_root, exist_ok=True)
        os.makedirs(config.sample_dir, exist_ok=True)

    logging.info(f"rank={accelerator.process_index}  reached checkpoint-C")
    accelerator.wait_for_everyone()
    # logging.info(f"rank={accelerator.process_index}  reached checkpoint-D")
    if accelerator.is_main_process:
        wandb.init(dir=os.path.abspath(config.workdir), project=f'uvit_{config.dataset.name}', config=config.to_dict(),
                   name=config.hparams, job_type='train', mode='offline')
        utils.set_logger(log_level='info', fname=os.path.join(config.workdir, 'output.log'))
        logging.info(config)
    else:
        utils.set_logger(log_level='error')
        builtins.print = lambda *args: None
    logging.info(f'Run on {accelerator.num_processes} devices')

    logging.info("before get_dataset")
    dataset = get_dataset(**config.dataset)
    logging.info("dataset ready")

    assert os.path.exists(dataset.fid_stat)

    gpu_model = torch.cuda.get_device_name(torch.cuda.current_device())

    num_workers = 8
    persistent_workers = False if accelerator.num_processes > 1 else True

    train_dataset = dataset.get_split(split='train', labeled=True)
    logging.info("before dataloader")
    train_dataset_loader = DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=True, drop_last=True,
                                    num_workers=num_workers, pin_memory=True, persistent_workers=persistent_workers)
    logging.info("dataloader ready")

    test_dataset = dataset.get_split(split='test', labeled=True)  # for sampling
    test_dataset_loader = DataLoader(test_dataset, batch_size=config.sample.mini_batch_size, shuffle=True, drop_last=True,
                                     num_workers=num_workers, pin_memory=True, persistent_workers=persistent_workers)

    train_state = utils.initialize_train_state(config, device)
    summary_mem(train_state.nnet, train_state.optimizer)
    summary_mem(train_state.nnet.context_encoder)
    nnet, nnet_ema, optimizer, train_dataset_loader, test_dataset_loader = accelerator.prepare(
        train_state.nnet, train_state.nnet_ema, train_state.optimizer, train_dataset_loader, test_dataset_loader)
    lr_scheduler = train_state.lr_scheduler
    train_state.resume(config.ckpt_root)

    autoencoder = libs.autoencoder.get_model(**config.autoencoder)
    autoencoder.to(device)

    if config.nnet.model_args.clip_dim == 4096:
        llm = "t5"
        t5 = T5Embedder(device=device)
    elif config.nnet.model_args.clip_dim == 768:
        llm = "clip"
        clip = FrozenCLIPEmbedder()
        clip.eval()
        clip.to(device)
    else:
        raise NotImplementedError

    ss_empty_context = None

    ClipSocre_model = ClipSocre(device=device)

    @ torch.cuda.amp.autocast()
    def encode(_batch):
        return autoencoder.encode(_batch)

    @ torch.cuda.amp.autocast()
    def decode(_batch):
        return autoencoder.decode(_batch)

    def get_data_generator():
        while True:
            for data in tqdm(train_dataset_loader, disable=not accelerator.is_main_process, desc='epoch'):
                with tic("dataloader"):
                    yield data

    data_generator = get_data_generator()

    def get_context_generator(autoencoder):
        while True:
            for data in test_dataset_loader:
                if len(data) == 5:
                    _img, _context, _token_mask, _token, _caption = data
                else:
                    _img, _context = data
                    _token_mask = None
                    _token = None
                    _caption = None

                if len(_img.shape)==5:
                    _testbatch_img_blurred = autoencoder.sample(_img[:,1,:])
                    yield _context, _token_mask, _token, _caption, _testbatch_img_blurred
                else:
                    assert len(_img.shape)==4
                    yield _context, _token_mask, _token, _caption, None

    context_generator = get_context_generator(autoencoder)

    _flow_mathcing_model = FlowMatching()

    def train_step(_batch, _ss_empty_context):
        with tic("data→cuda"):
            _metrics = dict()
            optimizer.zero_grad()

            assert len(_batch)==6
            assert not config.dataset.cfg
            _batch_img = _batch[0]
            _batch_con = _batch[1]
            _batch_mask = _batch[2]
            _batch_token = _batch[3]
            _batch_caption = _batch[4]
            _batch_img_ori = _batch[5]

            _z = autoencoder.sample(_batch_img)

        with tic("forward"):
            loss, loss_dict = _flow_mathcing_model(_z, nnet, loss_coeffs=config.loss_coeffs, cond=_batch_con, con_mask=_batch_mask, batch_img_clip=_batch_img_ori, \
                nnet_style=config.nnet.name, text_token=_batch_token, model_config=config.nnet.model_args, all_config=config, training_step=train_state.step)

            _metrics['loss'] = accelerator.gather(loss.detach()).mean()
            for key in loss_dict.keys():
                _metrics[key] = accelerator.gather(loss_dict[key].detach()).mean()
        
        with tic("backward"):
            accelerator.backward(loss.mean())
        with tic("optimizer"):
            optimizer.step()
            lr_scheduler.step()
        with tic("ema_update"):
            train_state.ema_update(config.get('ema_rate', 0.9999))
            train_state.step += 1
        return dict(lr=train_state.optimizer.param_groups[0]['lr'], **_metrics)

    @torch.no_grad()
    def sample(nnet_ema, n_samples, _sample_steps, context=None, caption=None, testbatch_img_blurred=None, two_stage_generation=-1, token_mask=None, return_clipScore=False, ClipSocre_model=None):
        shape = (n_samples, *config.z_shape) 

        _z_x0, _mu, _log_var = nnet_ema(context, text_encoder = True, shape = shape, mask=token_mask)
        z1 = _z_x0.reshape(shape)

        assert config.sample.scale > 1
        scale = config.sample.scale

        has_null_indicator = hasattr(config.nnet.model_args, "cfg_indicator")

        dt = 1.0 / _sample_steps
        ts = torch.linspace(1.0, 0.0, _sample_steps+1, device=device)[1:] 
        z = z1

        for t in ts:
            if has_null_indicator:
                v_cond = nnet_ema(z, t=t, r=torch.zeros(n_samples, device=device), null_indicator=torch.tensor([False] * n_samples).to(device))
                v_uncond = nnet_ema(z, t=t, r=torch.zeros(n_samples, device=device), null_indicator=torch.tensor([True] * n_samples).to(device))
                v = v_uncond + scale * (v_cond - v_uncond)
            else:
                raise NotImplementedError("Only support has_null_indicator=True for now")
            z = z - dt * v

        image_unprocessed = decode(z)

        if return_clipScore:
            clip_score = ClipSocre_model.calculate_clip_score(caption, image_unprocessed)
            return image_unprocessed, clip_score
        else:
            return image_unprocessed

    def eval_step(n_samples, sample_steps):
        logging.info(f'eval_step: n_samples={n_samples}, sample_steps={sample_steps}, algorithm=ODE_Euler_Flow_Matching_Solver, '
                     f'mini_batch_size={config.sample.mini_batch_size}')

        def sample_fn(_n_samples, return_caption=False, return_clipScore=False, ClipSocre_model=None, config=None):
            _context, _token_mask, _token, _caption, _testbatch_img_blurred = next(context_generator)
            assert _context.size(0) == _n_samples
            assert not return_caption # during training we should not use this
            if return_caption:
                return ode_fm_solver_sample(nnet_ema, _n_samples, sample_steps, context=_context, token_mask=_token_mask), _caption
            elif return_clipScore:
                return ode_fm_solver_sample(nnet_ema, _n_samples, sample_steps, context=_context, token_mask=_token_mask, return_clipScore=return_clipScore, ClipSocre_model=ClipSocre_model, caption=_caption)
            else:
                return ode_fm_solver_sample(nnet_ema, _n_samples, sample_steps, context=_context, token_mask=_token_mask)

        with tempfile.TemporaryDirectory() as temp_path:
            path = config.sample.path or temp_path
            if accelerator.is_main_process:
                os.makedirs(path, exist_ok=True)
            clip_score_list = utils.sample2dir(accelerator, path, n_samples, config.sample.mini_batch_size, sample_fn, dataset.unpreprocess, return_clipScore=True, ClipSocre_model=ClipSocre_model, config=config)
            _fid = 0
            if accelerator.is_main_process:
                _fid = calculate_fid_given_paths((dataset.fid_stat, path))
                _clip_score_list = torch.cat(clip_score_list)
                logging.info(f'step={train_state.step} fid{n_samples}={_fid} clip_score{len(_clip_score_list)} = {_clip_score_list.mean().item()}')
                with open(os.path.join(config.workdir, 'eval.log'), 'a') as f:
                    print(f'step={train_state.step} fid{n_samples}={_fid} clip_score{len(_clip_score_list)} = {_clip_score_list.mean().item()}', file=f)
                wandb.log({f'fid{n_samples}': _fid}, step=train_state.step)
            _fid = torch.tensor(_fid, device=device)
            _fid = accelerator.reduce(_fid, reduction='sum')

        return _fid.item()

    logging.info(f'Start fitting, step={train_state.step}, mixed_precision={config.mixed_precision}')

    step_fid = []
    while train_state.step < config.train.n_steps:
        with tic("whole_step"):
            nnet.train()
            batch = tree_map(lambda x: x, next(data_generator))
            metrics = train_step(batch, ss_empty_context)

        nnet.eval()
        if accelerator.is_main_process and train_state.step % config.train.log_interval == 0:
            logging.info(utils.dct2str(dict(step=train_state.step, **metrics)))
            logging.info(config.workdir)
            wandb.log(metrics, step=train_state.step)

        ############# save rigid image
        if train_state.step % config.train.eval_interval == 0:
            torch.cuda.empty_cache()
            logging.info('Save a grid of images...')
            if hasattr(dataset, "token_embedding"):
                contexts = torch.tensor(dataset.token_embedding, device=device)[ : config.train.n_samples_eval]
                token_mask = torch.tensor(dataset.token_mask, device=device)[ : config.train.n_samples_eval]
            elif hasattr(dataset, "contexts"):
                contexts = torch.tensor(dataset.contexts, device=device)[ : config.train.n_samples_eval]
                token_mask = None
            else:
                raise NotImplementedError
            samples = sample(nnet_ema, n_samples=config.train.n_samples_eval, _sample_steps=8, context=contexts, token_mask=token_mask)
            samples = make_grid(dataset.unpreprocess(samples), 5)
            if accelerator.is_main_process:
                save_image(samples, os.path.join(config.sample_dir, f'{train_state.step}.png'))
                wandb.log({'samples': wandb.Image(samples)}, step=train_state.step)
            accelerator.wait_for_everyone()
            torch.cuda.empty_cache()

        ############ save checkpoint and evaluate results
        if train_state.step % config.train.save_interval == 0 or train_state.step == config.train.n_steps:
            torch.cuda.empty_cache()
            logging.info(f'Save and eval checkpoint {train_state.step}...')

            if accelerator.local_process_index == 0:
                train_state.save(os.path.join(config.ckpt_root, f'{train_state.step}.ckpt'))
            accelerator.wait_for_everyone()

            # fid = eval_step(n_samples=10000, sample_steps=50)  # calculate fid of the saved checkpoint
            # step_fid.append((train_state.step, fid))

            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

    logging.info(f'Finish fitting, step={train_state.step}')
    logging.info(f'step_fid: {step_fid}')
    step_best = sorted(step_fid, key=lambda x: x[1])[0][0]
    logging.info(f'step_best: {step_best}')
    train_state.load(os.path.join(config.ckpt_root, f'{step_best}.ckpt'))
    del metrics
    accelerator.wait_for_everyone()
    # eval_step(n_samples=config.sample.n_samples, sample_steps=config.sample.sample_steps)



from absl import flags
from absl import app
from ml_collections import config_flags
import sys
from pathlib import Path


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("workdir", None, "Work unit directory.")


def get_config_name():
    argv = sys.argv
    for i in range(1, len(argv)):
        if argv[i].startswith('--config='):
            return Path(argv[i].split('=')[-1]).stem


def get_hparams():
    argv = sys.argv
    lst = []
    for i in range(1, len(argv)):
        assert '=' in argv[i]
        if argv[i].startswith('--config.') and not argv[i].startswith('--config.dataset.path'):
            hparam, val = argv[i].split('=')
            hparam = hparam.split('.')[-1]
            if hparam.endswith('path'):
                val = Path(val).stem
            lst.append(f'{hparam}={val}')
    hparams = '-'.join(lst)
    if hparams == '':
        hparams = 'default'
    return hparams


def main(argv):
    config = FLAGS.config
    config.config_name = get_config_name()
    config.hparams = get_hparams()
    config.workdir = FLAGS.workdir or os.path.join('workdir', config.config_name, config.hparams)
    config.ckpt_root = os.path.join(config.workdir, 'ckpts')
    config.sample_dir = os.path.join(config.workdir, 'samples')
    train(config)


if __name__ == "__main__":
    # mp.set_start_method('spawn', force=True)
    app.run(main)
