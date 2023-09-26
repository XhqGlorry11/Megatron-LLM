import os
import torch
import numpy as np

from megatron import get_args, get_tokenizer
from megatron.training import _setup_model_and_optimizer, build_train_valid_test_data_iterators
from megatron.model import ModelType
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.initialize import initialize_megatron
from megatron.schedules import backward_step


from finetune import model_provider, train_valid_test_datasets_provider

def extra_args(parser):
    """Text generation arguments."""
    group = parser.add_argument_group(title='validation set')
    group.add_argument("--model_name",
                       choices={"gpt", "llama", "falcon", "llama2", "codellama"},
                       default="gpt")
    group.add_argument("--model_type", choices={"encoder_or_decoder", "encoder_and_decoder"},
                       default="encoder_or_decoder")
    group.add_argument("--log_learning_rate_to_tensorboard", type=bool, default=True)
    group.add_argument("--log_loss_scale_to_tensorboard", type=bool, default=True)
    return parser

def build_train_valid_test_data_iterators(build_train_valid_test_datasets_provider,
                                          args):
    if args.train_samples:
        train_samples = args.train_samples
    else:
        train_samples = args.train_iters * args.global_batch_size
    eval_iters = (args.train_iters // args.eval_interval + 1) * \
                    args.eval_iters
    test_iters = args.eval_iters
    train_val_test_num_samples = [train_samples,
                                    eval_iters * args.global_batch_size,
                                    test_iters * args.global_batch_size]

    # Build the datasets.
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets_provider(
        train_val_test_num_samples)
    return train_ds, valid_ds, test_ds

def make_batch(data, device):
    args = get_args()
    tokenizer = get_tokenizer()
    tokens_ = np.expand_dims(data['text'], axis=0)
    tokens_ = torch.from_numpy(tokens_).to(device=device)
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)
    return tokens, labels, loss_mask, attention_mask, position_ids

def make_batch_list(data_list, device):
    args = get_args()
    tokenizer = get_tokenizer()
    data_list = [np.expand_dims(data['text'], axis=0) for data in data_list]
    tokens_ = np.concatenate(data_list, axis=0)
    tokens_ = torch.from_numpy(tokens_).to(device=device)
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)
    return tokens, labels, loss_mask, attention_mask, position_ids

def loss_func(loss_mask, output_tensor):
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    return loss


def check(args):
    local_rank = os.environ['LOCAL_RANK']
    device = torch.device('cuda:%s' % local_rank)
    model, optimizer, _ = _setup_model_and_optimizer(
        model_provider, ModelType.encoder_or_decoder, args=args
    )
    for model_module in model:
        model_module.train()
    model = model[0]

    train_dataset, _, _ = build_train_valid_test_data_iterators(
        train_valid_test_datasets_provider,
        args
    )
    # assert args.bad_iter_file, "bad_iter_file must be provided."
    # with open(args.bad_iter_file, 'r') as f:
    #     bad_iters = f.readlines()
    bad_iters = ["20000", "20001"]
    bad_iters = [int(bad_iter.strip()) for bad_iter in bad_iters]
    for bad_iter in bad_iters:
        # batch size = 1
        for global_batch_index in range(args.global_batch_size):
            model.zero_grad_buffer()
            cur_index = bad_iter * args.global_batch_size + global_batch_index
            cur_data = train_dataset[cur_index]
            cur_batch = make_batch(cur_data, device=device)
            tokens, labels, loss_mask, attention_mask, position_ids = cur_batch
            # forward
            loss = model(tokens, position_ids, attention_mask, labels=labels) # batch_size, seq_length
            loss = loss_func(loss_mask, loss)
            # backward
            optimizer.zero_grad()
            input_tensor, output_tensor_grad = None, None
            backward_step(optimizer, input_tensor, loss, output_tensor_grad, None)
            optimizer._copy_model_grads_to_main_grads()
            grad_norm = optimizer.clip_grad_norm(optimizer.clip_grad)
            grad_norm /= optimizer.grad_scaler.scale
            print (grad_norm)

if __name__ == '__main__':
    initialize_megatron(extra_args_provider=extra_args, args_defaults={})
    args = get_args()
    check(args)
