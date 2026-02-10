"""
run_main_enhanced_v2.py - 极简增强版训练入口

核心改进:
1. 数据划分: 支持丢弃原验证集，使用6:2:2重新划分
2. 测试时机: 支持指定特定epoch进行测试 (final/all/5,10,15)
3. 模块集成: 使用v2版本的极简模块 (DM+C2F, PVDR+AKG)
4. 辅助损失: 权重极低，不影响主任务

数据划分说明:
- 原始ETT: train(12月) -> val(4月) -> test(4月)
- 策略1 (swap_val_test): 交换验证集和测试集
- 策略2 (discard_original_val): 丢弃原验证集，重新按6:2:2划分

修改日志 (2026-02-10):
- 新增 test_epochs 参数支持灵活测试时机
- 新增 discard_original_val 参数
- 新增 data_split 参数
- 集成v2版本模块
"""

import argparse
import torch
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm

# 使用v2增强版模型
from models import TimeLLM_Enhanced_v2 as TimeLLM

from data_provider.data_factory import data_provider
import time
import random
import numpy as np
import os
import shutil

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali, load_content

parser = argparse.ArgumentParser(description='Time-LLM Enhanced v2')


def set_global_seed(seed, deterministic=False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def capture_rng_state():
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    if not state:
        return
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    cuda_state = state.get('cuda')
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def get_loader_sampler(loader):
    if hasattr(loader, "sampler"):
        return loader.sampler
    if hasattr(loader, "data_loader") and hasattr(loader.data_loader, "sampler"):
        return loader.data_loader.sampler
    return None


def get_sampler_state(loader):
    sampler = get_loader_sampler(loader)
    if sampler is not None and hasattr(sampler, "state_dict"):
        return sampler.state_dict()
    return None


def load_sampler_state(loader, state):
    sampler = get_loader_sampler(loader)
    if sampler is None or state is None:
        return False
    if hasattr(sampler, "load_state_dict"):
        sampler.load_state_dict(state)
        return True
    return False


def set_sampler_epoch(loader, epoch):
    if hasattr(loader, "set_epoch"):
        loader.set_epoch(epoch)
    sampler = get_loader_sampler(loader)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def parse_test_epochs(test_epochs_str, total_epochs):
    """解析 test_epochs 参数，返回需要测试的epoch集合"""
    if test_epochs_str == 'final':
        return {total_epochs}
    elif test_epochs_str == 'all':
        return set(range(1, total_epochs + 1))
    else:
        try:
            epochs = {int(e.strip()) for e in test_epochs_str.split(',')}
            epochs.add(total_epochs)
            return epochs
        except ValueError:
            return {total_epochs}


# ========== 基础配置参数 ==========
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast')
parser.add_argument('--is_training', type=int, required=True, default=1)
parser.add_argument('--model_id', type=str, required=True, default='test')
parser.add_argument('--model_comment', type=str, required=True, default='none')
parser.add_argument('--model', type=str, required=True, default='TimeLLM')
parser.add_argument('--seed', type=int, default=2021)
parser.add_argument('--deterministic', action='store_true')

# ========== 数据加载参数 ==========
parser.add_argument('--data', type=str, required=True, default='ETTm1')
parser.add_argument('--root_path', type=str, default='./dataset')
parser.add_argument('--data_path', type=str, default='ETTh1.csv')
parser.add_argument('--features', type=str, default='M')
parser.add_argument('--target', type=str, default='OT')
parser.add_argument('--loader', type=str, default='modal')
parser.add_argument('--freq', type=str, default='h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

# ========== 预测任务参数 ==========
parser.add_argument('--seq_len', type=int, default=96)
parser.add_argument('--label_len', type=int, default=48)
parser.add_argument('--pred_len', type=int, default=96)
parser.add_argument('--seasonal_patterns', type=str, default='Monthly')

# ========== 模型结构参数 ==========
parser.add_argument('--enc_in', type=int, default=7)
parser.add_argument('--dec_in', type=int, default=7)
parser.add_argument('--c_out', type=int, default=7)
parser.add_argument('--d_model', type=int, default=16)
parser.add_argument('--n_heads', type=int, default=8)
parser.add_argument('--e_layers', type=int, default=2)
parser.add_argument('--d_layers', type=int, default=1)
parser.add_argument('--d_ff', type=int, default=32)
parser.add_argument('--moving_avg', type=int, default=25)
parser.add_argument('--factor', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--embed', type=str, default='timeF')
parser.add_argument('--activation', type=str, default='gelu')
parser.add_argument('--output_attention', action='store_true')
parser.add_argument('--patch_len', type=int, default=16)
parser.add_argument('--stride', type=int, default=8)
parser.add_argument('--prompt_domain', type=int, default=0)
parser.add_argument('--llm_model', type=str, default='LLAMA')
parser.add_argument('--llm_dim', type=int, default=4096)
parser.add_argument('--llm_model_path', type=str, default='')
parser.add_argument('--load_in_4bit', action='store_true')

# ========== TAPR_v2模块参数 (DM + C2F) ==========
parser.add_argument('--use_tapr', action='store_true', default=False,
                    help='Enable TAPR_v2 module (DM: Multi-scale Observer + C2F: Trend Classifier)')
parser.add_argument('--n_scales', type=int, default=3,
                    help='Number of scales for DM (Decomposable Multi-scale)')
parser.add_argument('--lambda_trend', type=float, default=0.01,
                    help='C2F auxiliary loss weight (very low, default 0.01)')
parser.add_argument('--apply_trend_fusion', action='store_true', default=False,
                    help='Apply trend-aware output fusion (default: disabled)')

# ========== GRAM_v2模块参数 (PVDR + AKG) ==========
parser.add_argument('--use_gram', action='store_true', default=False,
                    help='Enable GRAM_v2 module (PVDR: Strict Retriever + AKG: Adaptive Gating)')
parser.add_argument('--top_k', type=int, default=3,
                    help='Top-K for PVDR (Pattern-Value Dual Retriever)')
parser.add_argument('--lambda_retrieval', type=float, default=0.001,
                    help='AKG auxiliary loss weight (very low, default 0.001)')
parser.add_argument('--build_memory', action='store_true', default=False,
                    help='Build PVDR memory bank from training data')
parser.add_argument('--d_repr', type=int, default=64,
                    help='PVDR pattern representation dimension')
parser.add_argument('--similarity_threshold', type=float, default=0.95,
                    help='PVDR retrieval threshold (very high, default 0.95)')

# ========== 数据划分参数 ==========
parser.add_argument('--swap_val_test', action='store_true', default=False,
                    help='Swap validation and test sets')
parser.add_argument('--discard_original_val', action='store_true', default=False,
                    help='Discard original validation set (extreme data) and use data_split ratio')
parser.add_argument('--data_split', type=str, default='6:2:2',
                    help='Data split ratio for train:val:test when discard_original_val=True')
parser.add_argument('--test_epochs', type=str, default='final',
                    help='When to run test: "final"=only last epoch, "all"=every epoch, or comma-separated like "5,10,15"')

# ========== 优化参数 ==========
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--itr', type=int, default=1)
parser.add_argument('--train_epochs', type=int, default=10)
parser.add_argument('--align_epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--eval_batch_size', type=int, default=8)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--des', type=str, default='test')
parser.add_argument('--loss', type=str, default='MSE')
parser.add_argument('--lradj', type=str, default='type1')
parser.add_argument('--pct_start', type=float, default=0.2)
parser.add_argument('--use_amp', action='store_true', default=False)
parser.add_argument('--llm_layers', type=int, default=6)
parser.add_argument('--percent', type=int, default=100)

# ========== Checkpoint参数 ==========
parser.add_argument('--save_steps', type=int, default=0)
parser.add_argument('--resume_from_checkpoint', type=str, default='')
parser.add_argument('--save_total_limit', type=int, default=0)
parser.add_argument('--resume_counter', type=int, default=-1)

args = parser.parse_args()
set_global_seed(args.seed, args.deterministic)
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

# 解析测试epoch
test_epoch_set = parse_test_epochs(args.test_epochs, args.train_epochs)

# 打印配置
if accelerator.is_local_main_process:
    print("=" * 60)
    print("Time-LLM Enhanced v2 Configuration")
    print("=" * 60)
    print(f"  TAPR_v2 (DM + C2F): {'ENABLED' if args.use_tapr else 'DISABLED'}")
    if args.use_tapr:
        print(f"    - DM n_scales: {args.n_scales}")
        print(f"    - C2F lambda_trend: {args.lambda_trend} (very low)")
        print(f"    - apply_trend_fusion: {args.apply_trend_fusion}")
    print(f"  GRAM_v2 (PVDR + AKG): {'ENABLED' if args.use_gram else 'DISABLED'}")
    if args.use_gram:
        print(f"    - PVDR top_k: {args.top_k}")
        print(f"    - PVDR similarity_threshold: {args.similarity_threshold} (very high)")
        print(f"    - AKG lambda_retrieval: {args.lambda_retrieval} (very low)")
        print(f"    - build_memory: {args.build_memory}")
    print(f"  Data Split Strategy:")
    print(f"    - discard_original_val: {args.discard_original_val}")
    if args.discard_original_val:
        print(f"    - data_split: {args.data_split}")
    print(f"    - swap_val_test: {args.swap_val_test}")
    print(f"    - test_epochs: {args.test_epochs} -> will test at epochs: {sorted(test_epoch_set)}")
    print("=" * 60)

for ii in range(args.itr):
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
        args.task_name, args.model_id, args.model, args.data, args.features,
        args.seq_len, args.label_len, args.pred_len, args.d_model, args.n_heads,
        args.e_layers, args.d_layers, args.d_ff, args.factor, args.embed, args.des, ii)

    # ========== 数据加载 ==========
    train_data, train_loader = data_provider(args, 'train')

    if args.swap_val_test:
        vali_data, vali_loader = data_provider(args, 'test')
        test_data, test_loader = data_provider(args, 'val')
        accelerator.print("Note: Validation and test sets have been swapped")
    else:
        vali_data, vali_loader = data_provider(args, 'val')
        test_data, test_loader = data_provider(args, 'test')

    args.content = load_content(args)
    model = TimeLLM.Model(args).float()

    path = os.path.join(args.checkpoints, setting + '-' + args.model_comment)
    if not os.path.exists(path) and accelerator.is_local_main_process:
        os.makedirs(path)

    time_now = time.time()
    train_steps = len(train_loader)
    early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)

    trained_parameters = [p for p in model.parameters() if p.requires_grad]
    model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

    if args.lradj == 'COS':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
    else:
        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim, steps_per_epoch=train_steps, pct_start=args.pct_start,
            epochs=args.train_epochs, max_lr=args.learning_rate)

    criterion = nn.MSELoss()
    mae_metric = nn.L1Loss()

    train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
        train_loader, vali_loader, test_loader, model, model_optim, scheduler)

    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

    # ========== 构建PVDR记忆库 ==========
    if args.use_gram and args.build_memory:
        accelerator.print("Building GRAM_v2 PVDR memory bank...")
        unwrapped_model = accelerator.unwrap_model(model)
        if hasattr(unwrapped_model, 'build_retrieval_memory'):
            unwrapped_model.build_retrieval_memory(train_loader, accelerator.device)
        accelerator.print("PVDR memory bank construction complete")

    # Checkpoint恢复
    start_epoch = 0
    global_step = 0
    resume_step_in_epoch = 0

    if args.resume_from_checkpoint:
        ckpt_path = args.resume_from_checkpoint
        if os.path.isdir(ckpt_path):
            ckpt_file = os.path.join(ckpt_path, 'checkpoint.pt')
            if not os.path.exists(ckpt_file):
                ckpt_file = os.path.join(ckpt_path, 'checkpoint')
        else:
            ckpt_file = ckpt_path

        if os.path.exists(ckpt_file):
            accelerator.print(f"Loading checkpoint: {ckpt_file}")
            ckpt = torch.load(ckpt_file, map_location='cpu', weights_only=False)
            model_to_load = accelerator.unwrap_model(model)

            if isinstance(ckpt, dict) and 'model' in ckpt:
                state_dict = ckpt['model']
                memory_buffers = {}
                keys_to_remove = []
                for key in state_dict:
                    if 'memory_keys' in key or 'memory_values' in key or 'memory_size' in key:
                        memory_buffers[key] = state_dict[key]
                        keys_to_remove.append(key)
                for key in keys_to_remove:
                    del state_dict[key]
                model_to_load.load_state_dict(state_dict, strict=False)
                if memory_buffers:
                    for key, value in memory_buffers.items():
                        parts = key.split('.')
                        obj = model_to_load
                        for part in parts[:-1]:
                            obj = getattr(obj, part)
                        setattr(obj, parts[-1], value.to(accelerator.device))
                if 'optimizer' in ckpt and ckpt['optimizer'] is not None:
                    model_optim.load_state_dict(ckpt['optimizer'])
                if 'scheduler' in ckpt and ckpt['scheduler'] is not None:
                    scheduler.load_state_dict(ckpt['scheduler'])
                if scaler is not None and ckpt.get('scaler') is not None:
                    scaler.load_state_dict(ckpt['scaler'])
                if ckpt.get('rng_state') is not None:
                    restore_rng_state(ckpt['rng_state'])
                start_epoch = ckpt.get('epoch', 0)
                global_step = ckpt.get('global_step', 0)
            else:
                model_to_load.load_state_dict(ckpt)

            accelerator.print(f"Resumed at epoch {start_epoch}, global_step {global_step}")
            if 'best_score' in ckpt and ckpt['best_score'] is not None:
                early_stopping.best_score = ckpt['best_score']
                early_stopping.val_loss_min = ckpt.get('val_loss_min', np.Inf)
                early_stopping.counter = ckpt.get('counter', 0)
            if args.resume_counter >= 0:
                early_stopping.counter = args.resume_counter
            if train_steps > 0 and global_step > 0:
                expected_epoch = global_step // train_steps
                start_epoch = max(start_epoch, expected_epoch)
                resume_step_in_epoch = global_step % train_steps

    # ========== 训练循环 ==========
    for epoch in range(start_epoch, args.train_epochs):
        iter_count = 0
        train_loss = []
        aux_loss_total = []
        model.train()
        set_sampler_epoch(train_loader, epoch)
        epoch_time = time.time()
        resume_skip = resume_step_in_epoch if (epoch == start_epoch and resume_step_in_epoch > 0) else 0

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):
            if resume_skip > 0 and i < resume_skip:
                continue
            iter_count += 1
            model_optim.zero_grad()

            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(accelerator.device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(accelerator.device)

            if args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    if args.output_attention:
                        outputs = outputs[0]
                    f_dim = -1 if args.features == 'MS' else 0
                    outputs = outputs[:, -args.pred_len:, f_dim:]
                    batch_y_target = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)
                    loss = criterion(outputs, batch_y_target)
                    if args.use_tapr or args.use_gram:
                        unwrapped = accelerator.unwrap_model(model)
                        if hasattr(unwrapped, 'compute_auxiliary_loss'):
                            aux_loss, _ = unwrapped.compute_auxiliary_loss(batch_y_target, current_step=global_step)
                            loss = loss + aux_loss
                            aux_loss_total.append(aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss)
                    train_loss.append(loss.item())
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                if args.output_attention:
                    outputs = outputs[0]
                f_dim = -1 if args.features == 'MS' else 0
                outputs = outputs[:, -args.pred_len:, f_dim:]
                batch_y_target = batch_y[:, -args.pred_len:, f_dim:]
                loss = criterion(outputs, batch_y_target)
                if args.use_tapr or args.use_gram:
                    unwrapped = accelerator.unwrap_model(model)
                    if hasattr(unwrapped, 'compute_auxiliary_loss'):
                        aux_loss, _ = unwrapped.compute_auxiliary_loss(batch_y_target, current_step=global_step)
                        loss = loss + aux_loss
                        aux_loss_total.append(aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss)
                train_loss.append(loss.item())

            if (i + 1) % 100 == 0:
                avg_aux = np.mean(aux_loss_total[-100:]) if aux_loss_total else 0
                accelerator.print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f} | aux_loss: {avg_aux:.7f}")
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                accelerator.print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                iter_count = 0
                time_now = time.time()

            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.step(model_optim)
                scaler.update()
            else:
                accelerator.backward(loss)
                model_optim.step()
            global_step += 1

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                if accelerator.is_local_main_process:
                    step_dir = os.path.join(path, f'checkpoint_step_{global_step}')
                    os.makedirs(step_dir, exist_ok=True)
                    ckpt_payload = {
                        'model': accelerator.unwrap_model(model).state_dict(),
                        'optimizer': model_optim.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch, 'global_step': global_step,
                        'best_score': early_stopping.best_score,
                        'val_loss_min': early_stopping.val_loss_min,
                        'counter': early_stopping.counter,
                        'rng_state': capture_rng_state(),
                    }
                    if scaler is not None:
                        ckpt_payload['scaler'] = scaler.state_dict()
                    torch.save(ckpt_payload, os.path.join(step_dir, 'checkpoint.pt'))

            if args.lradj == 'TST':
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
                scheduler.step()

        accelerator.print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time:.2f}s")
        train_loss = np.average(train_loss)
        avg_aux_loss = np.average(aux_loss_total) if aux_loss_total else 0

        vali_loss, vali_mae_loss = vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric)

        # ========== 测试时机控制 ==========
        current_epoch = epoch + 1
        should_test = (current_epoch in test_epoch_set) or early_stopping.early_stop

        if should_test:
            test_loss, test_mae_loss = vali(args, accelerator, model, test_data, test_loader, criterion, mae_metric)
            accelerator.print(
                f"Epoch: {current_epoch} | Train Loss: {train_loss:.7f} | Aux Loss: {avg_aux_loss:.7f} | "
                f"Vali Loss: {vali_loss:.7f} | Test Loss: {test_loss:.7f} | MAE: {test_mae_loss:.7f}")
        else:
            accelerator.print(
                f"Epoch: {current_epoch} | Train Loss: {train_loss:.7f} | Aux Loss: {avg_aux_loss:.7f} | "
                f"Vali Loss: {vali_loss:.7f} | (Test at epochs: {sorted(test_epoch_set)})")

        rng_state = capture_rng_state()
        scaler_state = scaler.state_dict() if scaler is not None else None
        sampler_state = get_sampler_state(train_loader)
        early_stopping(vali_loss, model, path, model_optim, scheduler, epoch + 1, global_step,
                       rng_state=rng_state, scaler_state=scaler_state, sampler_state=sampler_state)

        if early_stopping.early_stop:
            accelerator.print("Early stopping")
            if current_epoch not in test_epoch_set:
                test_loss, test_mae_loss = vali(args, accelerator, model, test_data, test_loader, criterion, mae_metric)
                accelerator.print(f"Final Test Loss: {test_loss:.7f} | MAE: {test_mae_loss:.7f}")
            break

        if args.lradj != 'TST':
            if args.lradj == 'COS':
                scheduler.step()
                accelerator.print(f"lr = {model_optim.param_groups[0]['lr']:.10f}")
            else:
                if epoch == 0:
                    args.learning_rate = model_optim.param_groups[0]['lr']
                    accelerator.print(f"lr = {model_optim.param_groups[0]['lr']:.10f}")
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=True)

accelerator.wait_for_everyone()
if accelerator.is_local_main_process:
    accelerator.print(f'Checkpoints saved at: ./checkpoints')
