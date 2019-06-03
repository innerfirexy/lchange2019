import argparse
from collections import Counter, deque
import copy
import pdb
import re
import sys
import time
import queue
import itertools
import os
import pickle
# from numba import jit, jitclass

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.multiprocessing as mp
from multiprocessing import set_start_method

import data_producer
from utils import *
from models import *


np.random.seed(1)


parser = argparse.ArgumentParser()
parser.add_argument("--train", type=str, default="", help="training file")
parser.add_argument('--vocab_data', type=str, default='', help='load previously built vocabulary data to save time')
parser.add_argument('--save_vocab', type=str, default='', help='save the vocabulary data to .pkl format')
parser.add_argument("--output", type=str, default="vectors.txt", help="output word embedding file")
parser.add_argument('--output_emb0', type=int, choices=[0,1], default=1, help='0 for "no" and 1 for "yes"')
parser.add_argument('--output_emb0_char', type=int, choices=[0,1], default=1, help='0 for "no" and 1 for "yes"')
parser.add_argument('--output_emb1', type=int, choices=[0,1], default=0, help='0 for "no" and 1 for "yes"')
parser.add_argument('--output_avg', type=int, choices=[0,1], default=1, help='0 for "no" and 1 for "yes"')
parser.add_argument('--output_h', type=int, choices=[0,1], default=1, help='0 for "no" and 1 for "yes"')
parser.add_argument('--output_per_iter', action='store_true', default=False, help='output average vector and theta(k) after each iter, for debug use')
parser.add_argument('--output_loss', type=str, default='')
parser.add_argument('--output_loss_step', type=float, default=1.0, help='the step of outputing loss with unit million words')
parser.add_argument('--output_per_step', action='store_true', default=False, help='output emb0 after each step, for debug use')

parser.add_argument("--size", type=int, default=300, help="word embedding dimension")
parser.add_argument("--cbow", type=int, default=1, help="1 for cbow, 0 for skipgram")
parser.add_argument("--window", type=int, default=5, help="context window size")
parser.add_argument("--sample", type=float, default=1e-4, help="subsample threshold")
parser.add_argument("--negative", type=int, default=10, help="number of negative samples")
parser.add_argument("--min_count", type=int, default=5, help="minimum frequency of a word")
parser.add_argument("--processes", type=int, default=4, help="number of processes")
parser.add_argument("--num_workers", type=int, default=6, help="number of workers for data processsing")
parser.add_argument("--iter", type=int, default=5, help="number of iterations")
parser.add_argument("--lr", type=float, default=-1.0, help="initial learning rate")
parser.add_argument("--batch_size", type=int, default=128, help="(max) batch size")
parser.add_argument("--cuda", action='store_true', default=False, help="enable cuda")
parser.add_argument('--gpu_id', type=int, default=0, help='GPU id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument("--output_ctx", action='store_true', default=False, help="output context embeddings")

parser.add_argument('--lang', type=str, choices=['zh', 'ko', 'en'], default='zh')
parser.add_argument('--model', type=str,
    choices=['CWE', 'w2v', 'DSE',
            'CBOW_dropout', 'SG_dropout', 'fastText'],
    nargs='?', default='CWE')
parser.add_argument('--h_init', type=str, choices=['random', 'fixed'], nargs='?', default='fixed')
parser.add_argument('--pt_emb0', type=str, default=None, help='pre-trained emb0_lookup table')
parser.add_argument('--pt_emb0_char', type=str, default=None, help='pre-trained emb0_char_lookup table')
parser.add_argument('--pt_emb1', type=str, default=None, help='pre-trained emb1_lookup table')
parser.add_argument('--fix_emb0', action='store_true', default=False, help='Make the emb0_lookup table untrainable')
parser.add_argument('--fix_emb1', action='store_true', default=False, help='Make the emb1_lookup table untrainable')
parser.add_argument('--fix_emb0_char', action='store_true', default=False, help='Make the emb0_char_lookup untrainable')
parser.add_argument('--pretrain_word', action='store_true', default=False, help='pretrain emb0 and emb1, with emb0_char disabled')
parser.add_argument('--pretrain_char', action='store_true', default=False, help='pretrain emb0_char, with emb0 disabled')
parser.add_argument('--wordlen_lim', type=int, default=0, help='Upper limit of the number of characters considered in each word')
parser.add_argument('--l2', type=float, default=0.0, help='L2 regularization for DSE models')

parser.add_argument('--drop_prob', type=float, default=0.2)

MAX_SENT_LEN = 1000
# For convenience
OTHER_MODELS = ['CBOW_dropout', 'SG_dropout']


# Initialize model.
def init_model(args):
    if args.lr == -1.0:
        if args.cbow == 1:
            vars(args)['lr'] = 0.05
        elif args.cbow == 0:
            vars(args)['lr'] = 0.025
    if args.model == 'CWE':
        if args.cbow == 1:
            return CWE_cbow(args)
        elif args.cbow == 0:
            return CWE_sg(args)
    elif args.model == 'DSE': # Dynamic Subword-incorporated Embeddings (DSE)
        if args.cbow == 1:
            return DSE_cbow(args)
        elif args.cbow == 0:
            return DSE_sg(args)
    elif args.model == 'w2v':
        if args.cbow == 1:
            return CBOW(args)
        elif args.cbow == 0:
            return SG(args)
    elif args.model == 'CBOW_dropout':
        vars(args)['cbow'] = 1
        return CBOW_dropout(args)
    elif args.model == 'SG_dropout':
        vars(args)['cbow'] = 0
        return SG_dropout(args)
    elif args.model == 'fastText':
        assert args.lang == 'en' # FastText model must be trained on non-Chinese data
        return FastText_sg(args)


# Training
def train_process_sent_producer(p_id, data_queue, word_count_actual, word2idx, word_list, freq, wid2cid_list, wid2cid_count, args):
    if args.negative > 0:
        table_ptr_val = data_producer.init_unigram_table(word_list, freq, args.word_count)

    train_file = open(args.train)
    file_pos = args.file_size * p_id // args.processes
    train_file.seek(file_pos, 0)
    while True:
        try:
            train_file.read(1)
        except UnicodeDecodeError:
            file_pos -= 1
            train_file.seek(file_pos, 0)
        else:
            train_file.seek(file_pos, 0)
            break

    batch_count = 0

    if args.model in ['CWE', 'DSE', 'fastText']:
        if args.cbow == 1:
            batch_ph_word = np.zeros([args.batch_size, 2*args.window+2+2*args.negative], np.int64)
            batch_ph_char = np.zeros([args.batch_size, 2*args.window*args.max_word_len+1], np.int64)
        elif args.cbow == 0:
            batch_ph_word = np.zeros([args.batch_size, 2+2*args.negative], np.int64)
            batch_ph_char = np.zeros([args.batch_size, args.max_word_len+1], np.int64)
    if args.model == 'w2v' or args.model in OTHER_MODELS:
        if args.cbow == 1:
            batch_ph = np.zeros((args.batch_size, 2*args.window+2+2*args.negative), 'int64')
        elif args.cbow == 0:
            batch_ph = np.zeros((args.batch_size, 2+2*args.negative), 'int64')

    for it in range(args.iter):
        train_file.seek(file_pos, 0)

        last_word_cnt = 0
        word_cnt = 0
        sentence = []
        prev = ''
        eof = False
        while True:
            if eof or train_file.tell() > file_pos + args.file_size / args.processes:
                break

            while True:
                s = train_file.read(1)
                if not s:
                    eof = True
                    break
                elif s == ' ' or s == '\t':
                    if prev in word2idx:
                        sentence.append(prev)
                    prev = ''
                    if len(sentence) >= MAX_SENT_LEN:
                        break
                elif s == '\n':
                    if prev in word2idx:
                        sentence.append(prev)
                    prev = ''
                    break
                else:
                    prev += s

            if len(sentence) > 0:
                # subsampling
                word_ids = []
                char_ids = []
                char_count = []
                if args.sample != 0:
                    sent_len = len(sentence)
                    i = 0
                    while i < sent_len:
                        word = sentence[i]
                        if word in freq: # To be compatible with smaller vocab_data
                            f = freq[word] / args.word_count
                            pb = (np.sqrt(f / args.sample) + 1) * args.sample / f;
                            if pb > np.random.random_sample():
                                wid = word2idx[word]
                                word_ids.append(wid)
                                char_ids.append(wid2cid_list[wid])
                                char_count.append(wid2cid_count[wid])
                        i += 1

                if len(word_ids) < 2:
                    word_cnt += len(sentence)
                    sentence.clear()
                    continue

                next_random = (2**24) * np.random.randint(0, 2**24) + np.random.randint(0, 2**24)
                if args.model in ['CWE', 'DSE', 'fastText']: # train CWE, DSE, fastText
                    if args.cbow == 1:
                        chunk = data_producer.cwe_cbow_producer(word_ids, len(word_ids), char_ids, char_count, args.max_word_len, args.char_vocab_size, table_ptr_val, args.window, args.negative, args.vocab_size, args.batch_size, next_random)
                    elif args.cbow == 0:
                        chunk = data_producer.cwe_sg_producer(word_ids, len(word_ids), char_ids, char_count, args.max_word_len, args.char_vocab_size, table_ptr_val, args.window, args.negative, args.vocab_size, args.batch_size, next_random)
                if args.model == 'w2v' or args.model in OTHER_MODELS:
                    if args.cbow == 1: # train CBOW
                        chunk = data_producer.cbow_producer(word_ids, len(word_ids), table_ptr_val,
                                args.window, args.negative, args.vocab_size, args.batch_size, next_random)
                    elif args.cbow == 0: # train skipgram
                        chunk = data_producer.sg_producer(word_ids, len(word_ids), table_ptr_val,
                                args.window, args.negative, args.vocab_size, args.batch_size, next_random)

                if args.model in ['CWE', 'DSE', 'fastText']:
                    chunk_pos = 0
                    chunk_w, chunk_c = chunk
                    while chunk_pos < chunk_w.shape[0]:
                        remain_space = args.batch_size - batch_count
                        remain_chunk = chunk_w.shape[0] - chunk_pos

                        if remain_chunk < remain_space:
                            take_from_chunk = remain_chunk
                        else:
                            take_from_chunk = remain_space

                        batch_ph_word[batch_count:batch_count+take_from_chunk, :] = chunk_w[chunk_pos:chunk_pos+take_from_chunk, :]
                        batch_ph_char[batch_count:batch_count+take_from_chunk, :] = chunk_c[chunk_pos:chunk_pos+take_from_chunk, :]
                        batch_count += take_from_chunk

                        if batch_count == args.batch_size:
                            data_queue.put((batch_ph_word, batch_ph_char))
                            batch_count = 0
                        chunk_pos += take_from_chunk
                if args.model == 'w2v' or args.model in OTHER_MODELS:
                    chunk_pos = 0
                    while chunk_pos < chunk.shape[0]:
                        remain_space = args.batch_size - batch_count
                        remain_chunk = chunk.shape[0] - chunk_pos

                        if remain_chunk < remain_space:
                            take_from_chunk = remain_chunk
                        else:
                            take_from_chunk = remain_space

                        batch_ph[batch_count:batch_count+take_from_chunk, :] = chunk[chunk_pos:chunk_pos+take_from_chunk, :]
                        batch_count += take_from_chunk

                        if batch_count == args.batch_size:
                            data_queue.put(batch_ph)
                            batch_count = 0
                        chunk_pos += take_from_chunk

                word_cnt += len(sentence)
                if word_cnt - last_word_cnt > 10000:
                    with word_count_actual.get_lock():
                        word_count_actual.value += word_cnt - last_word_cnt
                    last_word_cnt = word_cnt
                sentence.clear()

        with word_count_actual.get_lock():
            word_count_actual.value += word_cnt - last_word_cnt

    if args.model in ['CWE', 'DSE', 'fastText']:
        if batch_count > 0:
            data_queue.put((batch_ph_word[:batch_count,:], batch_ph_char[:batch_count,:]))
        data_queue.put((None, None))
    if args.model == 'w2v' or args.model in OTHER_MODELS:
        if batch_count > 0:
            data_queue.put(batch_ph[:batch_count,:])
        data_queue.put(None)


def train_process(p_id, word_count_actual, recent_loss, avg_loss, loss_cnt, word2idx, word_list, freq, wid2cid_list, wid2cid_count, args, model):
    data_queue = mp.SimpleQueue()

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    t = mp.Process(target=train_process_sent_producer, args=(p_id, data_queue, word_count_actual, word2idx, word_list, freq, wid2cid_list, wid2cid_count, args))
    t.start()

    # get from data_queue and feed to model
    prev_word_cnt = 0
    while True:
        if args.model in ['CWE', 'DSE', 'fastText']:
            dw, dc = data_queue.get()
            if dw is None or dc is None:
                break
        if args.model == 'w2v' or args.model in OTHER_MODELS: # For other experiments
            d = data_queue.get()
            if d is None:
                break

        # lr anneal & output, record loss
        record_loss_flag = False
        if word_count_actual.value - prev_word_cnt > 10000:
            record_loss_flag = True
            lr = args.lr * (1 - word_count_actual.value / (args.iter * args.word_count))
            if lr < 0.0001 * args.lr:
                lr = 0.0001 * args.lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            speed = word_count_actual.value / (time.monotonic() - args.t_start)
            remain_time_sec = (args.iter * args.word_count - word_count_actual.value) / speed
            sys.stdout.write("\rAlpha: %0.8f, Progress: %0.2f, Words/sec: %0.1f, Remaining time: %dh %dm %0.1fs" % (lr, word_count_actual.value / (args.iter * args.word_count) * 100, speed, remain_time_sec//3600, remain_time_sec%3600//60, remain_time_sec%3600%60))
            sys.stdout.flush()
            prev_word_cnt = word_count_actual.value

        if args.model in ['CWE', 'DSE', 'fastText']:
            if args.cuda:
                data_w = Variable(torch.LongTensor(dw).cuda(), requires_grad=False)
                data_c = Variable(torch.LongTensor(dc).cuda(), requires_grad=False)
            else:
                data_w = Variable(torch.LongTensor(dw), requires_grad=False)
                data_c = Variable(torch.LongTensor(dc), requires_grad=False)
            data = (data_w, data_c)
            # DEBUG:
            zero_count = (data_c[:,-1]==0).nonzero().sum().item()
            if zero_count > 0:
                # print(data_c.data)
                print(data_c.size(), 'zero_count={}'.format(zero_count))
                torch.save(data_c.data, open('data_c_{}.pkl'.format(p_id), 'wb'))
                torch.save(data_w.data,open('data_w_{}.pkl'.format(p_id), 'wb'))
                sys.exit('oops!')
        if args.model == 'w2v' or args.model in OTHER_MODELS:
            if args.cuda:
                data = Variable(torch.LongTensor(d).cuda(), requires_grad=False)
            else:
                data = Variable(torch.LongTensor(d), requires_grad=False)

        if args.cbow == 1:
            optimizer.zero_grad()
            loss = model(data)
            if record_loss_flag and args.output_loss:
                with recent_loss.get_lock(), loss_cnt.get_lock():
                    recent_loss.value += loss.data[0]
                    loss_cnt.value += 1
            loss.backward()
            optimizer.step()
            model.emb0_lookup.weight.data[args.vocab_size].fill_(0)
            if args.model in ['CWE', 'DSE', 'fastText']:
                model.emb0_char_lookup.weight.data[args.char_vocab_size].fill_(0) # This step is important, because we don't need the embedding for the filler character
        elif args.cbow == 0:
            optimizer.zero_grad()
            loss = model(data)
            if record_loss_flag and args.output_loss:
                with recent_loss.get_lock(), loss_cnt.get_lock():
                    recent_loss.value += loss.data[0]
                    loss_cnt.value += 1
            loss.backward()
            optimizer.step()
            if args.model in ['CWE', 'DSE', 'fastText']:
                model.emb0_char_lookup.weight.data[args.char_vocab_size].fill_(0)

    t.join()


# Monitor process for debug
def monitor_per_iter(word_count_actual, recent_loss, avg_loss, loss_cnt, word_list, wid2cid_np, args, model):
    prev_iter = 0
    prev_step = 0
    output_loss_step = int(args.output_loss_step * 1e6) # 1M words as one step
    while True:
        if word_count_actual.value >= args.word_count*args.iter:
            break

        curr_iter = word_count_actual.value // args.word_count
        if args.output_per_iter:
            if curr_iter > prev_iter:
                print('\nIteration {} is done, output vectors...'.format(prev_iter))

                if args.cuda:
                    emb0 = model.emb0_lookup.weight.data.cpu().numpy()
                    emb1 = model.emb1_lookup.weight.data.cpu().numpy()
                    if args.model in ['CWE', 'DSE', 'fastText']:
                        emb0_char = model.emb0_char_lookup.weight.data.cpu().numpy()
                    if args.model == 'DSE':
                        embs_h = np_softmax(model.h_lookup.weight.data.cpu().numpy(), axis=1)
                else:
                    emb0 = model.emb0_lookup.weight.data.numpy()
                    emb1 = model.emb1_lookup.weight.data.numpy()
                    if args.model in ['CWE', 'DSE', 'fastText']:
                        emb0_char = model.emb0_char_lookup.weight.data.numpy()
                    if args.model == 'DSE':
                        embs_h = np_softmax(model.h_lookup.weight.data.numpy(), axis=1)

                out_file, out_ext = os.path.splitext(args.output)
                if args.model in ['CWE', 'DSE', 'fastText']:
                    if args.output_avg:
                        avgvec_out = out_file + '_avgvec_iter{}'.format(curr_iter) + out_ext
                        data_producer.write_embs_avg(avgvec_out, word_list, wid2cid_np, emb0, emb0_char, args.vocab_size, args.char_vocab_size, args.size, embs_h)
                    if args.output_h:
                        h_out = out_file + '_h_iter{}'.format(curr_iter) + out_ext
                        data_producer.write_embs(h_out, word_list, embs_h, args.vocab_size, embs_h.shape[1])
                if args.output_emb0:
                    emb0_out = out_file + '_emb0_iter{}'.format(curr_iter) + out_ext
                    data_producer.write_embs(emb0_out, word_list, emb0, args.vocab_size, args.size)
                if args.output_emb1:
                    emb1_out = out_file + '_emb1_iter{}'.format(curr_iter) + out_ext
                    data_producer.write_embs(emb1_out, word_list, emb1, args.vocab_size, args.size)
                prev_iter = curr_iter

        curr_step = word_count_actual.value // output_loss_step
        if args.output_loss:
            if curr_step > prev_step:
                with recent_loss.get_lock(), loss_cnt.get_lock(), avg_loss.get_lock():
                    if loss_cnt.value > 0:
                        avg_loss.value = recent_loss.value / loss_cnt.value
                    recent_loss.value = 0.0
                    loss_cnt.value = 0
                with open(args.output_loss, 'a') as f:
                    f.write(str(avg_loss.value) + ',' + str(curr_step) + ',' + str(curr_iter) + '\n')
                prev_step = curr_step

        # update avg_loss every 100 records of loss
        # if loss_cnt.value >= 50 and args.output_loss:
        #     with recent_loss.get_lock(), loss_cnt.get_lock(), avg_loss.get_lock():
        #         if loss_cnt.value > 0:
        #             avg_loss.value = recent_loss.value / loss_cnt.value
        #         recent_loss.value = 0.0
        #         loss_cnt.value = 0


def monitor_per_step(word_count_actual, word_list, args, model):
    step_len = 10*1e6 # 1M words as one step
    prev_step = 0
    out_file, out_ext = os.path.splitext(args.output)

    while True:
        if word_count_actual.value >= args.word_count*args.iter:
            break
        curr_step = int(word_count_actual.value // step_len)
        if curr_step > prev_step:
            if args.cuda:
                emb0 = model.emb0_lookup.weight.data.cpu().numpy()
            else:
                emb0 = model.emb0_lookup.weight.data.numpy()
            emb0_out = out_file + '_emb0_step{}'.format(curr_step) + out_ext
            data_producer.write_embs(emb0_out, word_list, emb0, args.vocab_size, args.size)
            print('emb0 at step {} written to {}'.format(curr_step, emb0_out))
            prev_step = curr_step


if __name__ == '__main__':
    set_start_method('forkserver')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    print("Starting training using file %s" % args.train)
    train_file = open(args.train)
    train_file.seek(0, 2)
    vars(args)['file_size'] = train_file.tell()

    # Add word embs and character embs output files
    out_file, out_ext = os.path.splitext(args.output)
    vars(args)['emb0_outfile'] = out_file + '_emb0' + out_ext
    vars(args)['emb0_char_outfile'] = out_file + '_emb0char' + out_ext
    vars(args)['emb1_outfile'] = out_file + '_emb1' + out_ext
    vars(args)['avg_outfile'] = out_file + '_avgvec' + out_ext
    vars(args)['h_outfile'] = out_file + '_h' + out_ext

    # Build vocabulary
    if args.vocab_data:
        print('Loading previously built vocabulary data...')
        if args.lang in ['zh', 'ko']: # 中文
            word2idx, word_list, freq, char2id, char_list, wid2cid_list, wid2cid_np, wid2cid_count, word_count, max_word_len = pickle.load(open(args.vocab_data, 'rb'))
            print("Number of tokens in training set: %ld" % word_count)
            print("Word vocab size: %ld" % len(word2idx))
            print('Character vocab size: %ld' % len(char2id))
        elif args.lang == 'en': # Other languagues: English etc.
            word2idx, word_list, freq, ngram2id, ngram_list, wid2nid_list, wid2nid_np, wid2nid_count, word_count, ngram_perword_max = pickle.load(open(args.vocab_data, 'rb'))
            print("Number of tokens in training set: %ld" % word_count)
            print("Word vocab size: %ld" % len(word2idx))
            print('Ngram vocab size: %ld' % len(ngram2id))
    else:
        print('Building vocabulary...')
        if args.lang in ['zh', 'ko']:
            word2idx, word_list, freq, char2id, char_list, wid2cid_list, wid2cid_np, wid2cid_count, word_count, max_word_len = build_vocab(args, fs='cython')
        elif args.lang == 'en':
            word2idx, word_list, freq, ngram2id, ngram_list, wid2nid_list, wid2nid_np, wid2nid_count, word_count, ngram_perword_max = build_vocab_en(args)
        if args.save_vocab:
            print('Saving vocabulary data...')
            if args.lang in ['zh', 'ko']:
                pickle.dump((word2idx, word_list, freq, char2id, char_list, wid2cid_list, wid2cid_np, wid2cid_count, word_count, max_word_len), open(args.save_vocab, 'wb'))
            elif args.lang =='en':
                pickle.dump((word2idx, word_list, freq, ngram2id, ngram_list, wid2nid_list, wid2nid_np, wid2nid_count, word_count, ngram_perword_max), open(args.save_vocab, 'wb'))

    vars(args)['vocab_size'] = len(word2idx)
    vars(args)['word_count'] = word_count
    if args.lang in ['zh', 'ko']:
        vars(args)['char2id'] = char2id
        vars(args)['char_list'] = char_list
        vars(args)['char_vocab_size'] = len(char2id)
        vars(args)['max_word_len'] = max_word_len
        vars(args)['wid2cid_list'] = wid2cid_list
        vars(args)['wid2cid_np'] = wid2cid_np
        vars(args)['wid2cid_count'] = wid2cid_count
    elif args.lang == 'en':
        vars(args)['char2id'] = ngram2id
        vars(args)['char_list'] = ngram_list
        vars(args)['char_vocab_size'] = len(ngram2id)
        vars(args)['max_word_len'] = ngram_perword_max
        vars(args)['wid2cid_list'] = wid2nid_list
        vars(args)['wid2cid_np'] = wid2nid_np
        vars(args)['wid2cid_count'] = wid2nid_count

    # Initialize model
    model = init_model(args)
    model.share_memory()
    if args.cuda:
        # torch.cuda.set_device(args.gpu_id)
        model.cuda()

    # Train
    vars(args)['t_start'] = time.monotonic()
    word_count_actual = mp.Value('L', 0)
    recent_loss = mp.Value('d', 0.0) # track loss
    avg_loss = mp.Value('d', 0.0)
    loss_cnt = mp.Value('L', 0)
    processes = []
    for p_id in range(args.processes):
        p = mp.Process(target=train_process, args=(p_id, word_count_actual, recent_loss, avg_loss, loss_cnt, word2idx, word_list, freq, args.wid2cid_list, args.wid2cid_count, args, model))
        p.start()
        processes.append(p)

    # append the monitor process
    if args.output_per_iter:
        p = mp.Process(target=monitor_per_iter, args=(word_count_actual, recent_loss, avg_loss, loss_cnt, word_list, wid2cid_np, args, model))
        p.start()
        processes.append(p)
    if args.output_per_step:
        p = mp.Process(target=monitor_per_step, args=(word_count_actual, word_list, args, model))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Generate vectors for output
    if args.cuda:
        emb0 = model.emb0_lookup.weight.data.cpu().numpy()
        emb1 = model.emb1_lookup.weight.data.cpu().numpy()
        if args.model in ['CWE', 'DSE', 'fastText']:
            emb0_char = model.emb0_char_lookup.weight.data.cpu().numpy()
        if args.model == 'DSE':
            embs_h = np_softmax(model.h_lookup.weight.data.cpu().numpy(), axis=1)
    else:
        emb0 = model.emb0_lookup.weight.data.numpy()
        emb1 = model.emb1_lookup.weight.data.numpy()
        if args.model in ['CWE', 'DSE', 'fastText']:
            emb0_char = model.emb0_char_lookup.weight.data.numpy()
        if args.model == 'DSE':
            embs_h = np_softmax(model.h_lookup.weight.data.numpy(), axis=1)

    print('\nWriting embeddings..')
    # Write emb0, emb0_char, emb1
    if args.output_emb0:
        data_producer.write_embs(args.emb0_outfile, word_list, emb0, args.vocab_size, args.size)
    if args.output_emb1:
        data_producer.write_embs(args.emb1_outfile, word_list, emb1, args.vocab_size, args.size)
    if args.output_emb0_char:
        data_producer.write_embs(args.emb0_char_outfile, args.char_list, emb0_char, args.char_vocab_size, args.size)

    # Write average embs
    # Do not output average embs during pretrain phase
    if args.output_avg and not args.pretrain_word and not args.pretrain_char:
        if args.model in ['DSE', 'fastText']:
            data_producer.write_embs_avg(args.avg_outfile, word_list, args.wid2cid_np, emb0, emb0_char, args.vocab_size, args.char_vocab_size, args.size, embs_h)
        elif args.model == 'CWE':
            data_producer.write_embs_avg(args.avg_outfile, word_list, args.wid2cid_np, emb0, emb0_char, args.vocab_size, args.char_vocab_size, args.size)

    # Write parameter k, or sk
    if args.output_h:
        if args.model == 'DSE':
            data_producer.write_embs(args.h_outfile, word_list, embs_h, args.vocab_size, embs_h.shape[1])
    print("")
