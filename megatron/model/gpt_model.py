# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""GPT-2 model."""

import torch

from megatron import get_args
from megatron.core import tensor_parallel
from .module import MegatronModule

from .enums import AttnMaskType
from .language_model import parallel_lm_logits
import megatron.model.language_model
from .utils import init_method_normal
from .utils import scaled_init_method_normal
from .utils import small_init_method
from .utils import wang_init_method


def post_language_model_processing(lm_output, labels, logit_weights,
                                   parallel_output,
                                   fp16_lm_cross_entropy):

    # Output. Format [s b h]
    output = parallel_lm_logits(
        lm_output,
        logit_weights,
        parallel_output)
    if labels is None:
        # [s b h] => [b s h]
        return output.transpose(0,1).contiguous()
    else:
        # [b s] => [s b]
        # import numpy as np
        # np.save('/home/xinghq/npy/tp_3.npy', output.transpose(0,1).contiguous().detach().cpu().numpy())
        labels = labels.transpose(0,1).contiguous()
        if fp16_lm_cross_entropy:
            assert output.dtype == torch.half
            loss = tensor_parallel.vocab_parallel_cross_entropy(output, labels)
        else:
            loss = tensor_parallel.vocab_parallel_cross_entropy(output.float(), labels)
            
        # [s b] => [b, s]
        loss = loss.transpose(0,1).contiguous()
        return loss


class GPTModel(MegatronModule):
    """GPT-2 Language model."""

    def __init__(self,
                 num_tokentypes=0,
                 parallel_output=True,
                 pre_process=True,
                 post_process=True,
                 model_type=None,
                 not_calculate_loss=False):

        args = get_args()
        super(GPTModel, self).__init__(share_word_embeddings=args.tie_embed_logits)
        self.tie_embed_logits = args.tie_embed_logits

        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy

        # xhq modification
        # determine which init function to use
        if not args.use_gpt_neox_init_method:
            init_method = init_method_normal(args.init_method_std)
        else:
            init_method = small_init_method(args.hidden_size)
        if not args.use_gpt_neox_output_layer_init_method:
            scaled_init_method = scaled_init_method_normal(args.init_method_std, args.num_layers)
        else:
            scaled_init_method = wang_init_method(args.num_layers, args.hidden_size)

        self.language_model, self._language_model_key = megatron.model.language_model.get_language_model(
            num_tokentypes=num_tokentypes,
            add_pooler=False,
            encoder_attn_mask_type=AttnMaskType.causal,
            # xhq modification
            init_method=init_method,
            scaled_init_method=scaled_init_method,
            pre_process=self.pre_process,
            post_process=self.post_process,
            args=args,
            model_type=model_type)

        if self.tie_embed_logits:
            self.initialize_word_embeddings(init_method_normal, args)
        self.not_calculate_loss = not_calculate_loss

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.language_model.set_input_tensor(input_tensor)

    def forward(self, input_ids, position_ids, attention_mask, labels=None,
                tokentype_ids=None, inference_params=None):

        lm_output = self.language_model(
            input_ids,
            position_ids,
            attention_mask,
            inference_params=inference_params)

        if self.post_process:
            if not self.not_calculate_loss:
                return post_language_model_processing(
                    lm_output, labels,
                    self.word_embeddings_weight(),
                    self.parallel_output,
                    self.fp16_lm_cross_entropy)
            else:
                return post_language_model_processing(
                    lm_output, None,
                    self.word_embeddings_weight(),
                    self.parallel_output,
                    self.fp16_lm_cross_entropy)
        else:
            return lm_output

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        state_dict_ = {}
        state_dict_[self._language_model_key] \
            = self.language_model.state_dict_for_save_checkpoint(
                prefix=prefix, keep_vars=keep_vars)
        # Save word_embeddings.
        if self.post_process and not self.pre_process and self.tie_embed_logits:
            state_dict_[self._word_embeddings_for_head_key] \
                = self.word_embeddings.state_dict(prefix=prefix,
                                                  keep_vars=keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Load word_embeddings.
        if self.post_process and not self.pre_process and self.tie_embed_logits:
            self.word_embeddings.load_state_dict(
                state_dict[self._word_embeddings_for_head_key], strict=strict)
        if self._language_model_key in state_dict:
            state_dict = state_dict[self._language_model_key]
        self.language_model.load_state_dict(state_dict, strict=strict)
