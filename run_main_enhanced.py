"""
run_main_enhanced.py - 增强版训练入口

支持新增模块:
1. TAPR (Trend-Aware Patch Router) - 趋势感知尺度路由
2. GRA-M (Global Retrieval-Augmented Memory) - 全局检索增强记忆

新增命令行参数:
    --use_tapr: 启用TAPR模块 (默认启用)
    --use_gram: 启用GRA-M模块 (默认启用)
    --n_scales: 多尺度分解的尺度数量 (默认3)
    --top_k: 检索的相似模式数量 (默认5)
    --lambda_trend: 趋势辅助损失权重 (默认0.1)
    --lambda_retrieval: 检索辅助损失权重 (默认0.1)
    --build_memory: 是否构建检索记忆库 (默认启用)

兼容原有参数和checkpoint机制
"""

import argparse
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from models import Autoformer, DLinear
# 导入增强版模型
from models import TimeLLM_Enhanced as TimeLLM

from data_provider.data_factory import data_provider
import time
import random
import numpy as np
import os
import shutil

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali, load_content

parser = argparse.ArgumentParser(description='Time-LLM Enhanced')


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
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                torch.use_deterministic_algorithms(True)


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
    if hasattr(loader, "batch_sampler") and hasattr(loader.batch_sampler, "sampler"):
        return loader.batch_sampler.sampler
    if hasattr(loader, "data_loader") and hasattr(loader.data_loader, "batch_sampler"):
        batch_sampler = loader.data_loader.batch_sampler
        if hasattr(batch_sampler, "sampler"):
            return batch_sampler.sampler
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


# ========== 基础配置参数 ==========
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model_comment', type=str, required=True, default='none', help='prefix when saving test results')
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, DLinear, TimeLLM]')
parser.add_argument('--seed', type=int, default=2021, help='random seed')
parser.add_argument('--deterministic', action='store_true', help='enable deterministic behavior (slower)')

# ========== 数据加载参数 ==========
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; '
                         'M:multivariate predict multivariate, S: univariate predict univariate, '
                         'MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--loader', type=str, default='modal', help='dataset type')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# ========== 预测任务参数 ==========
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

# ========== 模型结构参数 ==========
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--prompt_domain', type=int, default=0, help='')
parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model')
parser.add_argument('--llm_dim', type=int, default=4096, help='LLM model dimension')
parser.add_argument('--llm_model_path', type=str, default='', help='LLM model path (local or HuggingFace ID)')
parser.add_argument('--load_in_4bit', action='store_true', help='Load model in 4-bit quantization to save VRAM')

# ========== 新增: TAPR模块参数 ==========
parser.add_argument('--use_tapr', action='store_true', default=False,
                    help='Enable TAPR (Trend-Aware Patch Router) module')
parser.add_argument('--n_scales', type=int, default=3,
                    help='Number of scales for multi-scale decomposition in TAPR')
parser.add_argument('--lambda_trend', type=float, default=0.1,
                    help='Weight for trend auxiliary loss in TAPR')

# ========== 新增: GRA-M模块参数 ==========
parser.add_argument('--use_gram', action='store_true', default=False,
                    help='Enable GRA-M (Global Retrieval-Augmented Memory) module')
parser.add_argument('--top_k', type=int, default=5,
                    help='Number of similar patterns to retrieve in GRA-M')
parser.add_argument('--lambda_retrieval', type=float, default=0.1,
                    help='Weight for retrieval auxiliary loss in GRA-M')
parser.add_argument('--build_memory', action='store_true', default=False,
                    help='Build retrieval memory bank from training data')
parser.add_argument('--d_repr', type=int, default=128,
                    help='Dimension of pattern representation for retrieval')

# ========== 优化参数 ==========
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--llm_layers', type=int, default=6)
parser.add_argument('--percent', type=int, default=100)

# ========== Checkpoint参数 ==========
parser.add_argument('--save_steps', type=int, default=0,
                    help='save checkpoint every N steps (0=disable)')
parser.add_argument('--resume_from_checkpoint', type=str, default='',
                    help='path to checkpoint directory or checkpoint.pt to resume')
parser.add_argument('--save_total_limit', type=int, default=0,
                    help='keep only the most recent N step checkpoints (0=disable)')
parser.add_argument('--resume_counter', type=int, default=-1,
                    help='override early stopping counter on resume (-1=use checkpoint value)')

args = parser.parse_args()
set_global_seed(args.seed, args.deterministic)
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

# 打印模块配置
if accelerator.is_local_main_process:
    print("=" * 60)
    print("Time-LLM Enhanced Configuration")
    print("=" * 60)
    print(f"  TAPR (Trend-Aware Patch Router): {'ENABLED' if args.use_tapr else 'DISABLED'}")
    if args.use_tapr:
        print(f"    - n_scales: {args.n_scales}")
        print(f"    - lambda_trend: {args.lambda_trend}")
    print(f"  GRA-M (Global Retrieval-Augmented Memory): {'ENABLED' if args.use_gram else 'DISABLED'}")
    if args.use_gram:
        print(f"    - top_k: {args.top_k}")
        print(f"    - lambda_retrieval: {args.lambda_retrieval}")
        print(f"    - build_memory: {args.build_memory}")
    print("=" * 60)

for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.des, ii)

    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')

    # 加载 prompt 内容
    args.content = load_content(args)

    # 模型选择
    if args.model == 'Autoformer':
        model = Autoformer.Model(args).float()
    elif args.model == 'DLinear':
        model = DLinear.Model(args).float()
    else:
        # 使用增强版TimeLLM
        model = TimeLLM.Model(args).float()

    path = os.path.join(args.checkpoints,
                        setting + '-' + args.model_comment)
    if not os.path.exists(path) and accelerator.is_local_main_process:
        os.makedirs(path)

    time_now = time.time()

    train_steps = len(train_loader)
    early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)

    trained_parameters = []
    for p in model.parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)

    model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

    if args.lradj == 'COS':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
    else:
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=args.pct_start,
                                            epochs=args.train_epochs,
                                            max_lr=args.learning_rate)

    criterion = nn.MSELoss()
    mae_metric = nn.L1Loss()

    train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
        train_loader, vali_loader, test_loader, model, model_optim, scheduler)

    scaler = None
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    # ========== 新增: 构建GRA-M检索记忆库 ==========
    if args.use_gram and args.build_memory:
        accelerator.print("Building GRA-M retrieval memory bank...")
        unwrapped_model = accelerator.unwrap_model(model)
        if hasattr(unwrapped_model, 'build_retrieval_memory'):
            unwrapped_model.build_retrieval_memory(train_loader, accelerator.device)
        accelerator.print("Memory bank construction complete")

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
                # 修复: 处理 GRA-M 记忆库的动态形状
                # 记忆库 buffer 形状可能不同 (checkpoint 中有数据，模型初始化为空)
                # 使用 strict=False 加载，然后手动恢复记忆库
                state_dict = ckpt['model']

                # 提取记忆库相关的 buffer (如果存在)
                memory_buffers = {}
                memory_keys_to_remove = []
                for key in state_dict:
                    if 'memory_keys' in key or 'memory_values' in key or 'memory_size' in key:
                        memory_buffers[key] = state_dict[key]
                        memory_keys_to_remove.append(key)

                # 从 state_dict 中移除记忆库 buffer，避免形状不匹配错误
                for key in memory_keys_to_remove:
                    del state_dict[key]

                # 加载其他权重
                model_to_load.load_state_dict(state_dict, strict=False)

                # 手动恢复记忆库 buffer
                if memory_buffers:
                    accelerator.print(f"Restoring GRA-M memory buffers: {list(memory_buffers.keys())}")
                    for key, value in memory_buffers.items():
                        # 解析嵌套属性路径，如 'gram.retriever.memory_keys'
                        parts = key.split('.')
                        obj = model_to_load
                        for part in parts[:-1]:
                            obj = getattr(obj, part)
                        # 直接设置 buffer
                        setattr(obj, parts[-1], value.to(accelerator.device))
                    accelerator.print("GRA-M memory buffers restored successfully")

                if 'optimizer' in ckpt and ckpt['optimizer'] is not None:
                    model_optim.load_state_dict(ckpt['optimizer'])
                if 'scheduler' in ckpt and ckpt['scheduler'] is not None:
                    scheduler.load_state_dict(ckpt['scheduler'])
                if scaler is not None and ckpt.get('scaler') is not None:
                    scaler.load_state_dict(ckpt['scaler'])
                    accelerator.print("Restored AMP scaler state")
                if ckpt.get('rng_state') is not None:
                    restore_rng_state(ckpt['rng_state'])
                    accelerator.print("Restored RNG state")
                if ckpt.get('sampler_state') is not None:
                    if load_sampler_state(train_loader, ckpt['sampler_state']):
                        accelerator.print("Restored sampler state")
                start_epoch = ckpt.get('epoch', 0)
                global_step = ckpt.get('global_step', 0)
            else:
                model_to_load.load_state_dict(ckpt)
            accelerator.print(f"Resumed at epoch {start_epoch}, global_step {global_step}")
            if 'best_score' in ckpt and ckpt['best_score'] is not None:
                early_stopping.best_score = ckpt['best_score']
                early_stopping.val_loss_min = ckpt.get('val_loss_min', np.Inf)
                early_stopping.counter = ckpt.get('counter', 0)
                accelerator.print(
                    f"Restored EarlyStopping: best_score={early_stopping.best_score}, val_loss_min={early_stopping.val_loss_min}, counter={early_stopping.counter}"
                )
            if args.resume_counter is not None and args.resume_counter >= 0:
                early_stopping.counter = args.resume_counter
                accelerator.print(f"Override EarlyStopping counter to {early_stopping.counter} via --resume_counter")
            if train_steps > 0 and global_step > 0:
                expected_epoch = global_step // train_steps
                start_epoch = max(start_epoch, expected_epoch)
                resume_step_in_epoch = global_step % train_steps
                if start_epoch > expected_epoch:
                    resume_step_in_epoch = 0
                if resume_step_in_epoch > 0:
                    accelerator.print(
                        f"Will skip first {resume_step_in_epoch} batches in epoch {start_epoch} to resume."
                    )
        else:
            accelerator.print(f"Checkpoint not found: {ckpt_file}")

    for epoch in range(start_epoch, args.train_epochs):
        iter_count = 0
        train_loss = []
        aux_loss_total = []  # 新增: 辅助损失记录

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

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(accelerator.device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(accelerator.device)

            # encoder - decoder
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if args.features == 'MS' else 0
                    outputs = outputs[:, -args.pred_len:, f_dim:]
                    batch_y_target = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)
                    loss = criterion(outputs, batch_y_target)

                    # ========== 新增: 计算辅助损失 ==========
                    # 修复: 传递 global_step 以启用 warmup 机制
                    # warmup 机制可以防止训练初期辅助损失干扰主任务学习
                    if args.use_tapr or args.use_gram:
                        unwrapped = accelerator.unwrap_model(model)
                        if hasattr(unwrapped, 'compute_auxiliary_loss'):
                            # 传递 current_step 参数，让辅助损失可以根据训练进度调整权重
                            aux_loss, aux_loss_dict = unwrapped.compute_auxiliary_loss(
                                batch_y_target,
                                current_step=global_step
                            )
                            loss = loss + aux_loss
                            aux_loss_total.append(aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss)

                    train_loss.append(loss.item())
            else:
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if args.features == 'MS' else 0
                outputs = outputs[:, -args.pred_len:, f_dim:]
                batch_y_target = batch_y[:, -args.pred_len:, f_dim:]
                loss = criterion(outputs, batch_y_target)

                # ========== 新增: 计算辅助损失 ==========
                # 修复: 传递 global_step 以启用 warmup 机制
                # warmup 机制可以防止训练初期辅助损失干扰主任务学习
                if args.use_tapr or args.use_gram:
                    unwrapped = accelerator.unwrap_model(model)
                    if hasattr(unwrapped, 'compute_auxiliary_loss'):
                        # 传递 current_step 参数，让辅助损失可以根据训练进度调整权重
                        aux_loss, aux_loss_dict = unwrapped.compute_auxiliary_loss(
                            batch_y_target,
                            current_step=global_step
                        )
                        loss = loss + aux_loss
                        aux_loss_total.append(aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss)

                train_loss.append(loss.item())

            if (i + 1) % 100 == 0:
                avg_aux = np.mean(aux_loss_total[-100:]) if aux_loss_total else 0
                accelerator.print(
                    "\titers: {0}, epoch: {1} | loss: {2:.7f} | aux_loss: {3:.7f}".format(
                        i + 1, epoch + 1, loss.item(), avg_aux))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
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
                    sampler_state = get_sampler_state(train_loader)
                    ckpt_payload = {
                        'model': accelerator.unwrap_model(model).state_dict(),
                        'optimizer': model_optim.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch,
                        'global_step': global_step,
                        'best_score': early_stopping.best_score,
                        'val_loss_min': early_stopping.val_loss_min,
                        'counter': early_stopping.counter,
                        'rng_state': capture_rng_state(),
                        'sampler_state': sampler_state,
                    }
                    if scaler is not None:
                        ckpt_payload['scaler'] = scaler.state_dict()
                    torch.save(ckpt_payload, os.path.join(step_dir, 'checkpoint.pt'))
                    accelerator.print(f"Saved step checkpoint: {step_dir}")
                    if args.save_total_limit and args.save_total_limit > 0:
                        step_dirs = []
                        for name in os.listdir(path):
                            if name.startswith('checkpoint_step_'):
                                full = os.path.join(path, name)
                                if os.path.isdir(full):
                                    try:
                                        step = int(name.split('_')[-1])
                                        step_dirs.append((step, full))
                                    except ValueError:
                                        continue
                        step_dirs.sort()
                        if len(step_dirs) > args.save_total_limit:
                            for _, old_dir in step_dirs[:-args.save_total_limit]:
                                shutil.rmtree(old_dir)

            if args.lradj == 'TST':
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
                scheduler.step()

        accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)
        avg_aux_loss = np.average(aux_loss_total) if aux_loss_total else 0
        vali_loss, vali_mae_loss = vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric)
        test_loss, test_mae_loss = vali(args, accelerator, model, test_data, test_loader, criterion, mae_metric)
        accelerator.print(
            "Epoch: {0} | Train Loss: {1:.7f} | Aux Loss: {2:.7f} | Vali Loss: {3:.7f} | Test Loss: {4:.7f} | MAE Loss: {5:.7f}".format(
                epoch + 1, train_loss, avg_aux_loss, vali_loss, test_loss, test_mae_loss))

        rng_state = capture_rng_state()
        scaler_state = scaler.state_dict() if scaler is not None else None
        sampler_state = get_sampler_state(train_loader)
        early_stopping(vali_loss, model, path, model_optim, scheduler, epoch + 1, global_step,
                       rng_state=rng_state, scaler_state=scaler_state, sampler_state=sampler_state)
        if early_stopping.early_stop:
            accelerator.print("Early stopping")
            break

        if args.lradj != 'TST':
            if args.lradj == 'COS':
                scheduler.step()
                accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                if epoch == 0:
                    args.learning_rate = model_optim.param_groups[0]['lr']
                    accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=True)

        else:
            accelerator.print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

accelerator.wait_for_everyone()
if accelerator.is_local_main_process:
    path = './checkpoints'
    accelerator.print('Checkpoints saved at: {}'.format(path))
