# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""GPT style dataset."""

import os
import time
from typing import Optional, List

import numpy as np
import torch

from megatron import print_rank_0
from megatron.core import mpu
from megatron.data.blendable_dataset import BlendableDataset
from megatron.data.dataset_utils import get_datasets_weights_and_num_samples
from megatron.data.dataset_utils import get_train_valid_test_split_
import megatron.data.indexed_dataset


def build_train_valid_test_datasets(data_prefix: Optional[str],
                                    data_impl: str,
                                    splits_string: str,
                                    train_valid_test_num_samples: List[int],
                                    seq_length: int,
                                    seed: int,
                                    skip_warmup: bool,
                                    train_data_prefix=None,
                                    valid_data_prefix=None,
                                    test_data_prefix=None,
                                    global_batch_size=0,
                                    iteration=0,
                                    force_one_epoch=True):
    """Build train, valid, and test datasets."""
    if data_prefix:
        print_rank_0("Single data path provided for train, valid & test")
        # Single dataset.
        if len(data_prefix) == 1:
            return _build_train_valid_test_datasets(data_prefix[0],
                                                    data_impl,
                                                    splits_string,
                                                    train_valid_test_num_samples,
                                                    seq_length,
                                                    seed,
                                                    skip_warmup,
                                                    global_batch_size,
                                                    iteration,
                                                    force_one_epoch)
        print ('logic modified by xhq, multiple data_pathes not supported currently.')
        raise ValueError
        # Blending dataset.
        # Parse the values.
        output = get_datasets_weights_and_num_samples(data_prefix,
                                                      train_valid_test_num_samples)
        prefixes, weights, datasets_train_valid_test_num_samples = output

        # Build individual datasets.
        train_datasets = []
        valid_datasets = []
        test_datasets = []
        for i in range(len(prefixes)):
            train_ds, valid_ds, test_ds = _build_train_valid_test_datasets(
                prefixes[i], data_impl, splits_string,
                datasets_train_valid_test_num_samples[i],
                seq_length, seed, skip_warmup)
            if train_ds:
                train_datasets.append(train_ds)
            if valid_ds:
                valid_datasets.append(valid_ds)
            if test_ds:
                test_datasets.append(test_ds)

        # Blend.
        blending_train_dataset = None
        if train_datasets:
            blending_train_dataset = BlendableDataset(train_datasets, weights)
        blending_valid_dataset = None
        if valid_datasets:
            blending_valid_dataset = BlendableDataset(valid_datasets, weights)
        blending_test_dataset = None
        if test_datasets:
            blending_test_dataset = BlendableDataset(test_datasets, weights)

        return (blending_train_dataset, blending_valid_dataset,
                blending_test_dataset)
    else:
        print_rank_0("Separate data paths provided for train, valid & test. Split string will be ignored.")
        train_dataset, valid_dataset, test_dataset = None, None, None
        # Single dataset.
        if train_data_prefix is not None:
            train_dataset = _build_dataset("train", train_data_prefix, data_impl,
                                        train_valid_test_num_samples[0], seq_length, seed,
                                        skip_warmup)

        if valid_data_prefix is not None:
            valid_dataset = _build_dataset("valid", valid_data_prefix, data_impl,
                                    train_valid_test_num_samples[1], seq_length, seed,
                                    False)

        if test_data_prefix is not None:
            test_dataset = _build_dataset("test", test_data_prefix, data_impl,
                                    train_valid_test_num_samples[2], seq_length, seed,
                                    False)
        return train_dataset, valid_dataset, test_dataset


def _build_dataset(dataset_name,
                   data_prefix,
                   data_impl,
                   num_samples,
                   seq_length,
                   seed,
                   skip_warmup):
    dataset = None
    if len(data_prefix) == 1:
        dataset = _build_dataset_kernel(dataset_name,
                        data_prefix[0], data_impl,
                        num_samples, seq_length,
                        seed, skip_warmup)
    else:
        # Blending dataset.
        # Parse the values.
        output = get_datasets_weights_and_num_samples(data_prefix, num_samples)
        prefixes, weights, dataset_num_samples = output

        # Build individual datasets.
        datasets = []
        for i in range(len(prefixes)):
            ds = _build_dataset_kernel(dataset_name, prefixes[i],
                            data_impl, dataset_num_samples[i],
                            seq_length, seed, skip_warmup)
            if ds:
                datasets.append(ds)

        if datasets:
            dataset = BlendableDataset(datasets, weights)
    return dataset


def _build_dataset_kernel(dataset_name, data_prefix, data_impl,
                num_samples, seq_length, seed, skip_warmup):
    """
    Build dataset. This method is called when individual
    train, valid, test datasets are provided
    """

    # Indexed dataset.
    indexed_dataset = get_indexed_dataset_(data_prefix,
                                           data_impl,
                                           skip_warmup)

    total_num_of_documents = indexed_dataset.sizes.shape[0]

    print_rank_0('    {}:'.format(dataset_name))
    print_rank_0('     document indices in [0, {}) total of {} '
                 'documents'.format(total_num_of_documents, total_num_of_documents))

    documents = np.arange(start=0, stop=total_num_of_documents,
                        step=1, dtype=np.int32)

    dataset = GPTDataset(dataset_name, data_prefix,
                        documents, indexed_dataset,
                        num_samples, seq_length, seed)

    return dataset


def _build_train_valid_test_datasets(data_prefix,
                                     data_impl,
                                     splits_string: str,
                                     train_valid_test_num_samples,
                                     seq_length,
                                     seed,
                                     skip_warmup,
                                     global_batch_size,
                                     iteration,
                                     force_one_epoch):
    """Build train, valid, and test datasets."""

    # Indexed dataset.
    indexed_dataset = get_indexed_dataset_(data_prefix,
                                           data_impl,
                                           skip_warmup)

    total_num_of_documents = indexed_dataset.sizes.shape[0]
    splits = get_train_valid_test_split_(splits_string, total_num_of_documents)

    # Print stats about the splits.
    print_rank_0(' > dataset split:')

    def print_split_stats(name, index):
        print_rank_0('    {}:'.format(name))
        print_rank_0('     document indices in [{}, {}) total of {} '
                     'documents'.format(splits[index], splits[index + 1],
                                        splits[index + 1] - splits[index]))
    print_split_stats('train', 0)
    print_split_stats('validation', 1)
    print_split_stats('test', 2)

    def _f(index, name, documents_total, global_batch_size, iteration, force_one_epoch):
        dataset = None
        if splits[index + 1] > splits[index]:
            documents = documents_total[splits[index]: splits[index + 1]]
            dataset = GPTDataset(name, data_prefix,
                                  documents, indexed_dataset,
                                  train_valid_test_num_samples[index],
                                  seq_length, seed, global_batch_size, iteration, force_one_epoch)
        return dataset

    # shuffle documents index in advance to avoid continuos train/valid/test dataset within a whole bin data file.
    documents_total = np.arange(start=0, stop=splits[-1], step=1, dtype=np.int64)
    np.random.shuffle(documents_total)

    train_dataset = _f(0, 'train', documents_total, global_batch_size, iteration, force_one_epoch)
    valid_dataset = _f(1, 'valid', documents_total, global_batch_size, iteration, force_one_epoch)
    test_dataset = _f(2, 'test', documents_total, global_batch_size, iteration, force_one_epoch)

    return train_dataset, valid_dataset, test_dataset


def get_indexed_dataset_(data_prefix, data_impl, skip_warmup):
    print_rank_0(' > building dataset index ...')

    start_time = time.time()
    indexed_dataset = megatron.data.indexed_dataset.make_dataset(data_prefix,
                                           data_impl,
                                           skip_warmup)
    assert indexed_dataset is not None
    print_rank_0(' > finished creating indexed dataset in {:4f} seconds'.format(time.time() - start_time))
    print_rank_0('    number of documents: {}'.format(indexed_dataset.sizes.shape[0]))
    n_tokens = _num_tokens(np.arange(start=0, stop=indexed_dataset.sizes.shape[0], step=1, dtype=np.int32), indexed_dataset.sizes)
    print_rank_0('    number of tokens: {}'.format(n_tokens))
    return indexed_dataset


class GPTDataset(torch.utils.data.Dataset):

    def __init__(self, name, data_prefix, documents, indexed_dataset,
                 num_samples, seq_length, seed, global_batch_size, iteration, force_one_epoch):

        self.name = name
        self.indexed_dataset = indexed_dataset

        # Checks
        assert np.min(documents) >= 0
        assert np.max(documents) < indexed_dataset.sizes.shape[0]

        # Build index mappings.
        self.doc_idx, self.sample_idx, self.shuffle_idx = _build_index_mappings(
            self.name, data_prefix, documents, self.indexed_dataset.sizes,
            num_samples, seq_length, seed, global_batch_size, iteration, force_one_epoch)

    def __len__(self):
        # -1 is due to data structure used to retieve the index:
        #    sample i --> [sample_idx[i], sample_idx[i+1])
        return self.sample_idx.shape[0] - 1

    def __getitem__(self, idx):
        # Get the shuffled index.
        idx = self.shuffle_idx[idx]
        # Start and end documents and offsets.
        doc_index_f = self.sample_idx[idx][0]
        doc_index_l = self.sample_idx[idx + 1][0]
        offset_f = self.sample_idx[idx][1]
        offset_l = self.sample_idx[idx + 1][1]
        # If we are within the same document, just extract the chunk.
        if doc_index_f == doc_index_l:
            sample = self.indexed_dataset.get(self.doc_idx[doc_index_f],
                                              offset=offset_f,
                                              length=offset_l - offset_f + 1)
        else:
            # Otherwise, get the rest of the initial document.
            sample_list = [self.indexed_dataset.get(self.doc_idx[doc_index_f],
                                                    offset=offset_f)]
            # Loop over all in between documents and add the entire document.
            for i in range(doc_index_f + 1, doc_index_l):
                sample_list.append(self.indexed_dataset.get(self.doc_idx[i]))
            # And finally add the relevant portion of last document.
            sample_list.append(self.indexed_dataset.get(
                self.doc_idx[doc_index_l],
                length=offset_l + 1))
            sample = np.concatenate(sample_list)

        return {'text': np.array(sample, dtype=np.int64)}


def _build_index_mappings(name, data_prefix, documents, sizes,
                          num_samples, seq_length, seed, global_batch_size, iteration, force_one_epoch=True):
    """Build doc-idx, sample-idx, and shuffle-idx.
    doc-idx: is an array (ordered) of documents to be used in training.
    sample-idx: is the start document index and document offset for each
       training sample.
    shuffle-idx: maps the sample index into a random index into sample-idx.
    """
    # Number of tokens in each epoch and number of required epochs.
    tokens_per_epoch = _num_tokens(documents, sizes)
    num_epochs = _num_epochs(tokens_per_epoch, seq_length, num_samples)
    # rng state
    np_rng = np.random.RandomState(seed=seed)

    # Filename of the index mappings.
    _filename = data_prefix
    _filename += '_{}_indexmap'.format(name)
    _filename += '_{}ns'.format(num_samples)
    _filename += '_{}sl'.format(seq_length)
    _filename += '_{}s'.format(seed)
    doc_idx_filename = _filename + '_doc_idx.npy'
    sample_idx_filename = _filename + '_sample_idx.npy'
    shuffle_idx_filename = _filename + '_shuffle_idx.npy'

    # Build the indexed mapping if not exist.
    if torch.distributed.get_rank() == 0 or os.environ.get('LOCAL_RANK', None) == '0':
        if (not os.path.isfile(doc_idx_filename)) or \
           (not os.path.isfile(sample_idx_filename)) or \
           (not os.path.isfile(shuffle_idx_filename)):

            print(' > WARNING: could not find index map files, building '
                         'the indices on rank 0 ...')

            # For the last epoch, decide whether include the entire epoch
            # in the global shuffle or not.

            # If we need only one epoch, then separating last epoch  does
            # not mean anything.
            if num_epochs == 1:
                separate_last_epoch = False
                print(' > only one epoch required, setting '
                      'separate_last_epoch to False', flush=True)

            else:
                # Get the number of samples for the last epoch
                num_samples_from_epochs_minus_one = (
                    (num_epochs - 1) * tokens_per_epoch - 1) // seq_length
                last_epoch_num_samples = num_samples - \
                                         num_samples_from_epochs_minus_one
                assert last_epoch_num_samples >= 0, \
                    'last epoch number of samples should be non-negative.'
                num_samples_per_epoch = (tokens_per_epoch - 1) // seq_length
                assert last_epoch_num_samples < (num_samples_per_epoch + 1), \
                    'last epoch number of samples exceeded max value.'
                # If we have less than 80% of the samples for the last epoch,
                # seperate out the epoch and treat it differently.
                # Note: the 80% number is just based on common sense and can
                # be adjusted if needed.
                separate_last_epoch = (last_epoch_num_samples <
                                       int(0.80 * num_samples_per_epoch))
                if separate_last_epoch:
                    string = ' > last epoch number of samples ({}) is smaller '\
                             'than 80% of number of samples per epoch ({}), '\
                             'setting separate_last_epoch to True'
                else:
                    string = ' > last epoch number of samples ({}) is larger '\
                             'than 80% of number of samples per epoch ({}), '\
                             'setting separate_last_epoch to False'
                print(string.format(last_epoch_num_samples,
                                    num_samples_per_epoch), flush=True)

            # doc-idx.
            start_time = time.time()
            if 'train' in os.path.basename(doc_idx_filename):
                if force_one_epoch:
                    doc_idx = _build_doc_idx_train(documents, num_epochs, np_rng,
                                                separate_last_epoch)
                else:
                    doc_idx = _build_doc_idx_val_test(documents, num_epochs, np_rng, separate_last_epoch)
            else:
                assert 'valid' in os.path.basename(doc_idx_filename) or 'test' in os.path.basename(doc_idx_filename)
                doc_idx = _build_doc_idx_val_test(documents, num_epochs, np_rng,
                                                  separate_last_epoch)
            np.save(doc_idx_filename, doc_idx, allow_pickle=True)
            print(' > elasped time to build and save doc-idx mapping '
                         '(seconds): {:4f}'.format(time.time() - start_time))
            # sample-idx.
            start_time = time.time()
            # Use C++ implementation for speed.
            # First compile and then import.
            from megatron.data import helpers
            assert doc_idx.dtype == np.int32
            assert sizes.dtype == np.int32
            sample_idx = helpers.build_sample_idx(sizes, doc_idx, seq_length,
                                                  num_epochs, tokens_per_epoch)
            # sample_idx = _build_sample_idx(sizes, doc_idx, seq_length,
            #                               num_epochs, tokens_per_epoch)
            np.save(sample_idx_filename, sample_idx, allow_pickle=True)
            print(' > elasped time to build and save sample-idx mapping '
                         '(seconds): {:4f}'.format(time.time() - start_time))
            # shuffle-idx.
            start_time = time.time()
            # -1 is due to data structure used to retieve the index:
            #    sample i --> [sample_idx[i], sample_idx[i+1])
            if 'train' in os.path.basename(shuffle_idx_filename):
                if force_one_epoch:
                    num_samples_per_epoch = (tokens_per_epoch - 1) // seq_length
                    shuffle_idx = _build_shuffle_idx_train(num_samples_per_epoch, sample_idx.shape[0] - 1, np_rng, global_batch_size, iteration)
                else:
                    if separate_last_epoch:
                        num_samples_ = num_samples_from_epochs_minus_one
                    else:
                        num_samples_ = sample_idx.shape[0] - 1
                    shuffle_idx = _build_shuffle_idx_val_test(num_samples_, sample_idx.shape[0] - 1, np_rng, global_batch_size, iteration)
            else:
                assert 'valid' in os.path.basename(shuffle_idx_filename) or 'test' in os.path.basename(shuffle_idx_filename)
                if separate_last_epoch:
                    num_samples_ = num_samples_from_epochs_minus_one
                else:
                    num_samples_ = sample_idx.shape[0] - 1
                shuffle_idx = _build_shuffle_idx_val_test(num_samples_,
                                                          sample_idx.shape[0] - 1, np_rng, global_batch_size, iteration)
            
            np.save(shuffle_idx_filename, shuffle_idx, allow_pickle=True)
            print(' > elasped time to build and save shuffle-idx mapping'
                         ' (seconds): {:4f}'.format(time.time() - start_time))

    # This should be a barrier but nccl barrier assumes
    # device_index=rank which is not the case for model
    # parallel case
    counts = torch.cuda.LongTensor([1])
    torch.distributed.all_reduce(counts, group=mpu.get_data_parallel_group())
    torch.distributed.all_reduce(counts, group=mpu.get_pipeline_model_parallel_group())
    assert counts[0].item() == (
        torch.distributed.get_world_size() //
        torch.distributed.get_world_size(group=mpu.get_tensor_model_parallel_group()))

    # Load mappings.
    start_time = time.time()
    print_rank_0(' > loading doc-idx mapping from {}'.format(
        doc_idx_filename))
    doc_idx = np.load(doc_idx_filename, allow_pickle=True, mmap_mode='r')
    print_rank_0(' > loading sample-idx mapping from {}'.format(
        sample_idx_filename))
    sample_idx = np.load(sample_idx_filename, allow_pickle=True, mmap_mode='r')
    print_rank_0(' > loading shuffle-idx mapping from {}'.format(
        shuffle_idx_filename))
    shuffle_idx = np.load(shuffle_idx_filename, allow_pickle=True, mmap_mode='r')
    print_rank_0('    loaded indexed file in {:3.3f} seconds'.format(
        time.time() - start_time))
    print_rank_0('    total number of tokens: {}'.format(_num_tokens(documents, sizes)))
    print_rank_0('    total number of samples: {}'.format(
        sample_idx.shape[0]))
    print_rank_0('    total number of epochs: {}'.format(num_epochs))

    return doc_idx, sample_idx, shuffle_idx


def _num_tokens(documents, sizes):
    """Total number of tokens in the dataset."""
    return np.sum(sizes[documents])


def _num_epochs(tokens_per_epoch, seq_length, num_samples):
    """Based on number of samples and sequence lenght, calculate how many
    epochs will be needed."""
    num_epochs = 0
    total_tokens = 0
    while True:
        num_epochs += 1
        total_tokens += tokens_per_epoch
        # -1 is because we need to retrieve seq_length + 1 token each time
        # but the last token will overlap with the first token of the next
        # sample except for the last sample.
        if ((total_tokens - 1) // seq_length) >= num_samples:
            return num_epochs


def _build_doc_idx_val_test(documents, num_epochs, np_rng, separate_last_epoch):
    """Build an array with length = number-of-epochs * number-of-dcuments.
    Each index is mapped to a corresponding document."""
    if not separate_last_epoch or num_epochs == 1:
        doc_idx = np.mgrid[0:num_epochs, 0:len(documents)][1]
        doc_idx[:] = documents
        doc_idx = doc_idx.reshape(-1)
        doc_idx = doc_idx.astype(np.int32)
        np_rng.shuffle(doc_idx)
        return doc_idx

    doc_idx_first = _build_doc_idx_val_test(documents, num_epochs-1, np_rng, False)
    doc_idx_last = _build_doc_idx_val_test(documents, 1, np_rng, False)
    return np.concatenate((doc_idx_first, doc_idx_last))


def _build_doc_idx_train(documents, num_epochs, np_rng, separate_last_epoch):
    """Build an array with length = number-of-epochs * number-of-dcuments.
    Each index is mapped to a corresponding document."""
    if not separate_last_epoch or num_epochs == 1:
        doc_idx = np.mgrid[0:num_epochs, 0:len(documents)][1]
        doc_idx[:] = documents
        doc_idx = doc_idx.reshape(-1)
        doc_idx = doc_idx.astype(np.int32)
        np_rng.shuffle(doc_idx)
        return doc_idx


    # modify doc index generation logic to disable epoch > 1 during training. Make total doc index：
    # shuffle([index in one epoch] * num_epoch) -> shuffle([index in one epoch]) * num_epoch
    doc_holders = []
    for _ in range(num_epochs):
        cur_doc_idx = _build_doc_idx_train(documents, 1, np_rng, False)
        doc_holders.append(cur_doc_idx)
    doc_idx_total = np.concatenate(doc_holders)
    return doc_idx_total


def _build_sample_idx(sizes, doc_idx, seq_length,
                      num_epochs, tokens_per_epoch):
    """Sample index mapping is a 2D array with sizes
    [number-of-samples + 1, 2] where [..., 0] contains
    the index into `doc_idx` and [..., 1] is the
    starting offset in that document."""

    # Total number of samples. For -1 see comments in `_num_epochs`.
    num_samples = (num_epochs * tokens_per_epoch - 1) // seq_length
    sample_idx = np.zeros([num_samples + 1, 2], dtype=np.int32)

    # Index into sample_idx.
    sample_index = 0
    # Index into doc_idx.
    doc_idx_index = 0
    # Begining offset for each document.
    doc_offset = 0
    # Start with first document and no offset.
    sample_idx[sample_index][0] = doc_idx_index
    sample_idx[sample_index][1] = doc_offset
    sample_index += 1
    while sample_index <= num_samples:
        # Start with a fresh sequence.
        remaining_seq_length = seq_length + 1
        while remaining_seq_length != 0:
            # Get the document length.
            doc_id = doc_idx[doc_idx_index]
            doc_length = sizes[doc_id] - doc_offset
            # And add it to the current sequence.
            remaining_seq_length -= doc_length
            # If we have more than a full sequence, adjust offset and set
            # remaining length to zero so we return from the while loop.
            # Note that -1 here is for the same reason we have -1 in
            # `_num_epochs` calculations.
            if remaining_seq_length <= 0:
                doc_offset += (remaining_seq_length + doc_length - 1)
                remaining_seq_length = 0
            else:
                # Otherwise, start from the begining of the next document.
                doc_idx_index += 1
                doc_offset = 0
        # Record the sequence.
        sample_idx[sample_index][0] = doc_idx_index
        sample_idx[sample_index][1] = doc_offset
        sample_index += 1

    return sample_idx


def _build_shuffle_idx_val_test(num_samples, total_size, np_rng, global_batch_size, iteration):
    """Build the range [0, size) and shuffle."""
    print(' > building shuffle index with split [0, {}) and [{}, {}) '
          '...'.format(num_samples, num_samples, total_size), flush=True)
    
    dtype_ = np.int64

    shuffle_idx_first = np.arange(start=0, stop=num_samples,
                                  step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx_first)
    if num_samples == total_size:
        return shuffle_idx_first

    shuffle_idx_last = np.arange(start=num_samples, stop=total_size,
                                 step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx_last)
    shuffle_idx = np.concatenate((shuffle_idx_first, shuffle_idx_last))

    return shuffle_idx

# modify shuffle index generation logic by xhq11 to disable epoch > 1
# make shuffle index = [shuffle index with 1 epoch] + [-1] * remaining length
def _build_shuffle_idx_train(num_samples_per_epoch, total_size, np_rng, global_batch_size, iteration):

    print(' > building shuffle index with split [0, {}) with normal index and [{}, {}) with invalid index to disable epoch > 1 during training'
          '...'.format(num_samples_per_epoch, num_samples_per_epoch, total_size), flush=True)

    dtype_ = np.int64
    # make sure max_index is larger than shuffle length and will throw an exception when
    # 1 epoch finishes or start with wrong index when finetune with different data package
    max_index = np.iinfo(np.int64).max - 1
    shuffle_idx_in_one_epoch = np.arange(start=0, stop=num_samples_per_epoch,
                                         step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx_in_one_epoch)
    
    shuffle_idx_beyond_one_epoch = np.array([max_index] * (total_size - num_samples_per_epoch), dtype=dtype_)
    shuffle_idx = np.concatenate((shuffle_idx_in_one_epoch, shuffle_idx_beyond_one_epoch))
    # if finetune from previously checkpoint, skip iteration * global_batch_size samples
    # used in training with discrete data packages
    if iteration > 0:
        consumed_samples = global_batch_size * iteration
        shuffle_idx[consumed_samples:] = shuffle_idx[:-consumed_samples]
        shuffle_idx[:consumed_samples] = max_index
    return shuffle_idx