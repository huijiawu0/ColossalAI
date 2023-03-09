import argparse
import json

import loralib as lora
import torch
from chatgpt.dataset import RewardDataset
from chatgpt.nn import BLOOMRM, GPTRM, OPTRM, GLMRM
from chatgpt.trainer import RewardModelTrainer
from chatgpt.trainer.strategies import ColossalAIStrategy, DDPStrategy, NaiveStrategy
from datasets import load_dataset, Dataset
from torch.optim import Adam
from transformers import AutoTokenizer, BloomTokenizerFast
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer

from colossalai.nn import CPUAdam
from colossalai.nn.optimizer import HybridAdam
from colossalai.nn.parallel import zero_model_wrapper, zero_optim_wrapper
from colossalai.tensor import ShardSpec, ProcessGroup
from colossalai.utils import get_current_device
from colossalai.utils.model.colo_init_context import ColoInitContext


def train(args):
    # configure strategy
    if args.strategy == 'naive':
        strategy = NaiveStrategy()
    elif args.strategy == 'ddp':
        strategy = DDPStrategy()
    elif args.strategy == 'colossalai_gemini':
        strategy = ColossalAIStrategy(stage=3, placement_policy='cuda')
    elif args.strategy == 'colossalai_zero2':
        strategy = ColossalAIStrategy(stage=2, placement_policy='cuda')
    else:
        raise ValueError(f'Unsupported strategy "{args.strategy}"')

    # configure model
    # with strategy.model_init_context():
    # if args.model == 'bloom':
    #     model = BLOOMRM(pretrained=args.pretrain, lora_rank=args.lora_rank).cuda()
    # elif args.model == 'opt':
    #     model = OPTRM(pretrained=args.pretrain, lora_rank=args.lora_rank).cuda()
    # elif args.model == 'gpt2':
    #     model = GPTRM(pretrained=args.pretrain, lora_rank=args.lora_rank).cuda()
    # elif args.model == 'glm':
    #     model = GLMRM(pretrained=args.pretrain, lora_rank=args.lora_rank).cuda()
    # else:
    #     raise ValueError(f'Unsupported model "{args.model}"')
    world_size = torch.distributed.get_world_size()
    shard_pg = ProcessGroup(tp_degree=world_size) if args.shardinit else None
    default_dist_spec = ShardSpec([-1], [world_size]) if args.shardinit else None
    with ColoInitContext(device=get_current_device(),
                         dtype=torch.bfloat16,
                         default_dist_spec=default_dist_spec,
                         default_pg=shard_pg):
        model = GLMRM(pretrained=args.pretrain, lora_rank=args.lora_rank).cuda()
    gemini_config = dict(strict_ddp_mode=True,
                         device=get_current_device(),
                         placement_policy='cpu',
                         pin_memory=True,
                         hidden_dim=model.config.n_embd,
                         search_range_mb=128)
    optim_config = dict(gpu_margin_mem_ratio=0.)
    zero_stage = 3
    model = zero_model_wrapper(model, zero_stage, gemini_config)
    
    # configure tokenizer
    if args.model == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
    elif args.model == 'bloom':
        tokenizer = BloomTokenizerFast.from_pretrained(args.pretrain)
        tokenizer.pad_token = tokenizer.eos_token
    elif args.model == 'opt':
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
    elif args.model == 'glm':
        tokenizer = AutoTokenizer.from_pretrained(pretrained=args.pretrain, trust_remote_code=True)
    else:
        raise ValueError(f'Unsupported model "{args.model}"')
    tokenizer.pad_token = tokenizer.eos_token

    max_len = 512

    # # configure optimizer
    # if args.strategy.startswith('colossalai'):
    #     optim = HybridAdam(model.parameters(), lr=5e-5)
    # else:
    #     optim = Adam(model.parameters(), lr=5e-5)
    optim = CPUAdam(model.parameters(), lr=1e-5)
    optim = zero_optim_wrapper(model, optim, optim_config=optim_config)
    
    # prepare for data and dataset
    def data_gen():
        with open(args.dataset, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                yield data
    data = Dataset.from_generator(data_gen)
    train_data = data.select(range(800))
    eval_data = data.select(range(800, 887))
    train_dataset = RewardDataset(train_data, tokenizer, max_len)
    eval_dataset = RewardDataset(eval_data, tokenizer, max_len)

    trainer = RewardModelTrainer(model=model,
                                 strategy=strategy,
                                 optim=optim,
                                 train_dataset=train_dataset,
                                 eval_dataset=eval_dataset,
                                 batch_size=args.batch_size,
                                 max_epochs=args.max_epochs)

    trainer.fit(use_lora=args.lora_rank)

    # save model checkpoint after fitting on only rank0
    strategy.save_model(model, 'rm_checkpoint.pt', only_rank0=True)
    # save optimizer checkpoint on all ranks
    strategy.save_optimizer(optim, 'rm_optim_checkpoint_%d.pt' % (torch.cuda.current_device()), only_rank0=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy',
                        choices=['naive', 'ddp', 'colossalai_gemini', 'colossalai_zero2'],
                        default='naive')
    parser.add_argument('--model', choices=['gpt2', 'bloom', 'opt', 'glm'], default='bloom')
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='Dahoas/rm-static')
    parser.add_argument('--save_path', type=str, default='rm_ckpt.pth')
    parser.add_argument('--max_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lora_rank', type=int, default=0, help="low-rank adaptation matrices rank")
    parser.add_argument(
        "--shardinit",
        action='store_true',
        help=
        "Shard the tensors when init the model to shrink peak memory size on the assigned device. Valid when using colossalai as dist plan.",
    )

    args = parser.parse_args()
    train(args)
