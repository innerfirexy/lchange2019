import numpy as np
cimport numpy as np
import cython
import sys
from libc.stdio cimport FILE, fopen, fwrite, fscanf, fclose, fprintf, fseek, ftell, SEEK_END, rewind, fread, getline
from libc.stdlib cimport malloc, free, rand, RAND_MAX, atof, atoi, strtod
from libc.math cimport pow, sqrt, exp
from libc.stdint cimport uintptr_t
from libc.string cimport strtok, strcpy, memset

try:
    from scipy.linalg.blas import fblas
except ImportError:
    import scipy.linalg.blas as fblas

cdef extern from "voidptr.h":
    void* PyCObject_AsVoidPtr(object obj)

ctypedef float (*sdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil
ctypedef double (*snrm2_ptr) (const int *N, const float *X, const int *incX) nogil
ctypedef void (*saxpy_ptr) (const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY) nogil

cdef sdot_ptr sdot=<sdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)  # float = dot(x, y)
cdef snrm2_ptr snrm2=<snrm2_ptr>PyCObject_AsVoidPtr(fblas.snrm2._cpointer)  # sqrt(x^2)
cdef saxpy_ptr saxpy=<saxpy_ptr>PyCObject_AsVoidPtr(fblas.saxpy._cpointer) # y += alpha * x

def dot(int size, float[:] x, float[:] y):
    cdef int ONE = 1
    v1 = sdot(&size, &x[0], &ONE, &y[0], &ONE)
    return v1

#cdef cos_sim(int size, float[:] x, float[:] y):
#    cdef int ONE = 1
#    return sdot(&size, &x[0], &ONE, &y[0], &ONE) / snrm2(&size, &x[0], &ONE) / snrm2(&size, &y[0], &ONE)

cdef float cosine(const int size, const float* x, const float* y):
    cdef int d
    cdef float sim, n1, n2
    sim = 0.0
    n1 = 0.0
    n2 = 0.0
    for d in range(size):
        sim += x[d] * y[d]
        n1 += x[d] * x[d]
        n2 += y[d] * y[d]
    return sim / sqrt(n1) / sqrt(n2)

cdef float cos_sim(const int size, const float* x, const float* y):
    cdef int ONE = 1
    return <float> sdot(&size, x, &ONE, y, &ONE) / snrm2(&size, x, &ONE) / snrm2(&size, y, &ONE)

cdef my_saxpy(const int size, const float a, const float* x, float* y):
    cdef int ONE = 1
    saxpy(&size, &a, x, &ONE, y, &ONE)

cdef sigmoid(float x):
    return 1 / (1+exp(-x))

cdef int* unigram_table

@cython.cdivision(True)
def init_unigram_table(word_list, freq, int train_words):
    global unigram_table
    cdef int table_size = int(1e8)
    cdef int a, idx, vocab_size
    cdef double power = 0.75
    cdef double d1
    cdef double train_words_pow = 0.0
    unigram_table = <int *>malloc(table_size * sizeof(int));
    idx = 0
    vocab_size = len(word_list)

    for word in freq:
        train_words_pow += pow(freq[word], power)

    d1 = pow(freq[ word_list[idx] ], power) / train_words_pow;
    for a in range(table_size):
        unigram_table[a] = idx
        if (<double>a / table_size) > d1:
            idx += 1
            d1 += pow(freq[ word_list[idx] ], power) / train_words_pow;
        if idx >= vocab_size:
            idx = vocab_size - 1;

    return <uintptr_t>unigram_table

cdef int get_unigram_table_at_idx(int* arr, unsigned long long next_random):
    #return *(arr + ((next_random >> 16) % 100000000))
    return arr [ (next_random >> 16) % 100000000 ]
#def test_ptr(uintptr_t ptr_val):
#    cdef int* unigram_table
#    unigram_table = <int*>ptr_val
#    return [ unigram_table[a] for a in range(int(1e8)) ]

#cdef uniform():
#    return <double> rand() / RAND_MAX


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cwe_cbow_producer(sent_id, int sent_id_len, char_ids, char_count, int max_word_len, int char_vocab_size, uintptr_t ptr_val, int window, int negative, int vocab_size, int batch_size, unsigned long long next_random, bint dynamic_window=True):
    cdef int i,j,tar_id,pos,t,n,q,r
    cdef int ctx_count, neg_count, actual_window
    cdef unsigned long long modulo = 281474976710655ULL

    # columns of data and its length:
    #   ctx_indices:  [2 * window]
    #   ctx_lens:     [1]
    #   word_idx:     [1]
    #   neg_indices:  [negative]
    #   neg_mask:     [negative]
    # columns of char_data
    #   ctx_char_indices:  [2 * window]
    #   actual_char_num:   [1]
    cdef long[:,:] data = np.zeros([sent_id_len,2*window+1+1+2*negative], dtype=np.int64)
    cdef long[:,:] char_data = np.zeros([sent_id_len, 2*window*max_word_len+1], dtype=np.int64)
    cdef int k, actual_char_num

    cdef int* unigram_table
    unigram_table = <int*>ptr_val

    for i in range(sent_id_len):
        ctx_count = 0
        actual_window = rand() % window + 1 if dynamic_window else window
        tar_id = sent_id[i]
        actual_char_num = 0

        for j in range(i-window, i-actual_window):
            pos = j - (i-window)
            data[i, pos] = vocab_size
            char_data[i, pos*max_word_len:(pos+1)*max_word_len] = char_vocab_size
        for j in range(i-actual_window, i):
            pos = j - (i-window)
            if j < 0 or j >= sent_id_len:
                data[i, pos] = vocab_size
                char_data[i, pos*max_word_len:(pos+1)*max_word_len] = char_vocab_size
            else:
                data[i, pos] = sent_id[j]
                for k in range(max_word_len):
                    char_data[i, pos*max_word_len + k] = char_ids[i][k]
                ctx_count += 1
                actual_char_num += char_count[j]

        for j in range(i+1, i+1+actual_window):
            pos = j - (i-window) - 1
            if j < 0 or j >= sent_id_len:
                data[i, pos] = vocab_size
                char_data[i, pos*max_word_len:(pos+1)*max_word_len] = char_vocab_size
            else:
                data[i, pos] = sent_id[j]
                for k in range(max_word_len):
                    char_data[i, pos*max_word_len + k] = char_ids[i][k]
                ctx_count += 1
                actual_char_num += char_count[j]
        for j in range(i+1+actual_window, i+1+window):
            pos = j - (i-window) - 1
            data[i, pos] = vocab_size
            char_data[i, pos*max_word_len:(pos+1)*max_word_len] = char_vocab_size

        data[i, 2*window] = ctx_count
        data[i, 2*window+1] = tar_id
        char_data[i, 2*window*max_word_len] = actual_char_num

        # negative sampling
        neg_count = 0
        for n in range(negative):
            t = get_unigram_table_at_idx(unigram_table, next_random)
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if t == tar_id:
                continue
            data[i, 2*window+2+neg_count] = t
            neg_count += 1

        # neg mask
        for n in range(neg_count):
            data[i, 2*window+2+negative+n] = 1

    return np.asarray(data), np.asarray(char_data)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cwe_sg_producer(sent_id, int sent_id_len, char_ids, char_count, int max_word_len, int char_vocab_size, uintptr_t ptr_val, int window, int negative, int vocab_size, int batch_size, unsigned long long next_random, bint dynamic_window=True):
    cdef int i,j,t,n,k,prev_i
    cdef int batch_count, neg_count, actual_window
    cdef unsigned long long modulo = 281474976710655ULL

    # columns of data and its length:
    #   ctx_word_idx:     [1], i.e., the word to be predicted, in the window
    #   tar_word_idx:      [1], i.e., the word at the center of the window
    #   neg_indices:  [negative]
    #   neg_mask:     [negative]
    # columns of char_data
    #   tar_char_indices:   [max_word_len], i.e., character ids in context word
    #   actual_char_num:   [1]
    cdef long[:,:] data = np.zeros([sent_id_len*2*window, 2+2*negative], dtype=np.int64)
    cdef long[:,:] char_data = np.zeros([sent_id_len*2*window, max_word_len+1], dtype=np.int64)

    cdef int* unigram_table
    unigram_table = <int*>ptr_val

    batch_count = 0
    prev_i = -1
    for i in range(sent_id_len):
        actual_window = rand() % window + 1 if dynamic_window else window
        for j in range(i-actual_window, i+actual_window+1):
            if j < 0 or j >= sent_id_len or j == i:
                continue

            data[batch_count, 0] = sent_id[j]
            data[batch_count, 1] = sent_id[i]

            # prepare char_data
            # for k in range(max_word_len):
            #     char_data[batch_count, k] = char_ids[i][k]
            #     char_data[batch_count, max_word_len] = char_count[i]
            if i == prev_i:
                char_data[batch_count, :] = char_data[batch_count-1, :]
            else:
                for k in range(max_word_len):
                    char_data[batch_count, k] = char_ids[i][k]
                char_data[batch_count, max_word_len] = char_count[i]
                prev_i = i

            # negative sampling
            neg_count = 0
            for n in range(negative):
                t = get_unigram_table_at_idx(unigram_table, next_random)
                next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
                if t == sent_id[i]:
                    continue
                data[batch_count, 2+neg_count] = t
                neg_count += 1

            # neg mask
            for n in range(neg_count):
                data[batch_count, 2+negative+n] = 1

            batch_count += 1

    # remove all-zero rows (resulting from dynamic window)
    data = data[:batch_count, :]
    char_data = char_data[:batch_count, :]
    assert data.shape[0] == char_data.shape[0]

    return np.asarray(data), np.asarray(char_data)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cbow_producer(sent_id, int sent_id_len, uintptr_t ptr_val, int window, int negative, int vocab_size, int batch_size, unsigned long long next_random, bint dynamic_window=True):
    cdef int i,j,tar_id,pos,t,n,q,r
    cdef int ctx_count, neg_count, actual_window
    cdef unsigned long long modulo = 281474976710655ULL

    # columns of data and its length:
    #   ctx_indices:  [2 * window]
    #   ctx_lens:     [1]
    #   word_idx:     [1]
    #   neg_indices:  [negative]
    #   neg_mask:     [negative]
    cdef long[:,:] data = np.zeros([sent_id_len,2*window+1+1+2*negative], dtype=np.int64)

    cdef int* unigram_table
    unigram_table = <int*>ptr_val

    for i in range(sent_id_len):
        ctx_count = 0
        actual_window = rand() % window + 1 if dynamic_window else window
        tar_id = sent_id[i]
        for j in range(i-window, i-actual_window):
            pos = j - (i-window)
            data[i, pos] = vocab_size
        for j in range(i-actual_window, i):
            pos = j - (i-window)
            if j < 0 or j >= sent_id_len:
                data[i, pos] = vocab_size
            else:
                data[i, pos] = sent_id[j]
                ctx_count += 1
        for j in range(i+1, i+1+actual_window):
            pos = j - (i-window) - 1
            if j < 0 or j >= sent_id_len:
                data[i, pos] = vocab_size
            else:
                data[i, pos] = sent_id[j]
                ctx_count += 1
        for j in range(i+1+actual_window, i+1+window):
            pos = j - (i-window) - 1
            data[i, pos] = vocab_size

        data[i, 2*window] = ctx_count
        data[i, 2*window+1] = tar_id

        # negative sampling
        neg_count = 0
        for n in range(negative):
            #t= unigram_table[ (next_random >> 16) % int(1e8) ]
            t = get_unigram_table_at_idx(unigram_table, next_random)
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            #t = rand() % vocab_size
            #t = unigram_table[ <int>(<double>(rand()>>16) / RAND_MAX * 1e8) ]
            #t = unigram_table[ <int>(<double> rand() / RAND_MAX * 1e8) ]
            #t = unigram_table[ rand() % 1e8 ]
            #t = rand() % 1e8 ]
            if t == tar_id:
                continue

            data[i, 2*window+2+neg_count] = t
            neg_count += 1

        # neg mask
        for n in range(neg_count):
            data[i, 2*window+2+negative+n] = 1

    return np.asarray(data)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sg_producer(sent_id, int sent_id_len, uintptr_t ptr_val, int window, int negative, int vocab_size, int batch_size, unsigned long long next_random, bint dynamic_window=True):
    cdef int i,j,t,n,q,r
    cdef int batch_count, neg_count, actual_window
    cdef unsigned long long modulo = 281474976710655ULL

    # columns of data and its length:
    #   word_idx:     [1]
    #   ctx_idx:      [1]
    #   neg_indices:  [negative]
    #   neg_mask:     [negative]
    #cdef long[:,:] data = np.zeros([batch_size, 2+2*negative], dtype=np.int64)
    cdef long[:,:] data = np.zeros([sent_id_len*2*window, 2+2*negative], dtype=np.int64)

    cdef int* unigram_table
    unigram_table = <int*>ptr_val

    batch_count = 0
    for i in range(sent_id_len):
        actual_window = rand() % window + 1 if dynamic_window else window
        for j in range(i-actual_window, i+actual_window+1):
            if j < 0 or j >= sent_id_len or j == i:
                continue

            data[batch_count, 0] = sent_id[j]
            data[batch_count, 1] = sent_id[i]

            # negative sampling
            neg_count = 0
            for n in range(negative):
                t= unigram_table[ (next_random >> 16) % int(1e8) ]
                next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
                if t == sent_id[i]:
                    continue

                data[batch_count, 2+neg_count] = t
                neg_count += 1

            # neg mask
            for n in range(neg_count):
                data[batch_count, 2+negative+n] = 1

            batch_count += 1

    return np.asarray(data)


def write_embs(str fn, word_list, float[:,:] embs, int vocab_size, int dim):
    cdef int i,j
    with open(fn, 'w') as out_f:
        if embs.shape[0] == vocab_size+1:
            out_f.write('%d %d\n' % (vocab_size+1, dim))
        else:
            out_f.write('%d %d\n' % (vocab_size, dim))

        if embs.shape[0] == vocab_size+1:
            out_f.write('</s> ')
            for j in range(dim):
                out_f.write( '%.6f ' % embs[vocab_size, j] )
            out_f.write('\n')

        for i in range(vocab_size):
            out_f.write('%s ' % word_list[i])
            for j in range(dim):
                out_f.write( '%.6f ' % embs[i, j] )
            out_f.write('\n')

# Write the average embedding for CWE, CWE_sk, and CWE_k
def write_embs_avg(str fn, word_list, wid2cid, float[:,:] embs_word, float[:,:] embs_char, int vocab_size, int char_vocab_size, int dim, embs_k=None):
    cdef int i,j
    cdef int max_word_len = wid2cid.shape[1]
    cdef int k
    cdef float count_ch = 0.0
    cdef long ch
    cdef float[:,:] tmp_vec = np.zeros([1, dim], dtype=np.float32)
    cdef float ek, w, kw, kc

    with open(fn, 'w') as out_f:
        if embs_word.shape[0] == vocab_size+1:
            out_f.write('%d %d\n' % (vocab_size+1, dim))
        else:
            out_f.write('%d %d\n' % (vocab_size, dim))

        if embs_word.shape[0] == vocab_size+1:
            out_f.write('</s> ')
            for j in range(dim):
                out_f.write( '%.6f ' % embs_word[vocab_size, j] )
            out_f.write('\n')

        for i in range(vocab_size):
            tmp_vec[:,:] = 0.0
            count_ch = 0.0
            out_f.write('%s ' % word_list[i])
            for k in range(max_word_len):
                ch = wid2cid[i, k]
                if ch != char_vocab_size:
                    count_ch += 1
                    tmp_vec = np.add(tmp_vec, embs_char[ch,:])
            tmp_vec = np.divide(tmp_vec, count_ch)

            if embs_k is not None:
                if isinstance(embs_k, np.ndarray):
                    if embs_k.shape[1] == 2: # 2 softmaxed coefs per row
                        kw = embs_k[i,0]
                        kc = embs_k[i,1]
                        tmp_vec = np.add(np.multiply(embs_word[i,:], kw), np.multiply(tmp_vec, kc))
                    elif embs_k.shape[1] == 1: # 1 sigmoid coefs per row
                        kw = embs_k[i,0]
                        kc = 1.0 - kw
                        tmp_vec = np.add(np.multiply(embs_word[i,:], kw), np.multiply(tmp_vec, kc))
                elif isinstance(embs_k, float):
                    tmp_vec = np.add(np.multiply(sigmoid(embs_k), embs_word[i,:]), np.multiply(1-sigmoid(embs_k), tmp_vec))
                elif isinstance(embs_k, tuple) and len(embs_k)==2:
                    tmp_vec = np.add(np.multiply(embs_k[0], embs_word[i,:]), np.multiply(embs_k[1], tmp_vec))
            else:
                tmp_vec = np.add(embs_word[i,:], tmp_vec)
                tmp_vec = np.multiply(0.5, tmp_vec)

            for j in range(dim):
                out_f.write('%.6f ' % tmp_vec[0, j])
            out_f.write('\n')


# limited max # of sense
#def create_n_update_sense(long[:] type_ids, float[:,:] context_feats, float[:,:] sense_embs, int[:,:] word2sense, int[:] word_sense_cnt, float[:] counter_list, int type_ids_len, int emb_dim, float delta, int current_n_sense):

def create_n_update_sense(long[:] type_ids, float[:,:] context_feats, float[:,:] sense_embs, word2sense, float[:] counter_list, int type_ids_len, int emb_dim, float delta, int current_n_sense):
    cdef int b, d, pos, t_id, s_id
    cdef int max_sense_id, new_sense_id, create_count
    cdef float sim, max_sim

    create_count = 0
    for b in range(type_ids_len):
        t_id = type_ids[b]
        # first encounter
        if counter_list[t_id] == 0.0:
            for d in range(emb_dim):
                sense_embs[t_id, d] = context_feats[b, d]
            counter_list[t_id] += 1.0
            continue

        # not first encounter
        max_sense_id = -1
        max_sim = -10.0
        for s_id in word2sense[t_id]:
            sim = cosine(emb_dim, &context_feats[b,0], &sense_embs[s_id,0])
            if sim > max_sim:
                max_sim = sim
                max_sense_id = s_id

        if len(word2sense[t_id]) < 5:
            if max_sim < delta:
                max_sense_id = current_n_sense + create_count
                word2sense[t_id].append(max_sense_id)
                create_count += 1

        for d in range(emb_dim):
            sense_embs[max_sense_id,d] += context_feats[b,d]
        counter_list[ max_sense_id ] += 1.0

    return create_count

def select_sense(long[:,:] chunk, float[:,:] context_feats, sense2idx, float[:,:] sense_embs, word2sense, int chunk_size, int emb_dim, int window, int negative):
    cdef int b, d, pos, t_id, s_id
    cdef int max_sense_id
    cdef float sim, max_sim

    for b in range(chunk_size):
        pos = 2*window+1
        t_id = chunk[b, pos]
        max_sense_id = -1
        max_sim = -10.0
        for s_id in word2sense[t_id]:
            sim = cos_sim(emb_dim, &context_feats[b,0], &sense_embs[sense2idx[s_id],0])
            if sim > max_sim:
                max_sim = sim
                max_sense_id = s_id
        chunk[b, pos] = max_sense_id

        for pos in range(2*window+2, 2*window+2+negative):
            t_id = chunk[b, pos]
            chunk[b, pos] = word2sense[t_id][ rand() % len(word2sense[t_id]) ]

    return chunk

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def npmssg_producer(sent_id, int sent_id_len, uintptr_t ptr_val, int window, int negative, int vocab_size, int batch_size, unsigned long long next_random):
    cdef int i,j,tar_id,pos,t,n,q,r
    cdef int ctx_count, neg_count, actual_window
    cdef unsigned long long modulo = 281474976710655ULL

    # columns of data and its length:
    #   ctx_indices:  [2 * window]
    #   ctx_mask:     [2 * window]
    #   word_idx:     [1]
    #   neg_indices:  [negative]
    #   neg_mask:     [negative]
    cdef long[:,:] data = np.zeros([sent_id_len,4*window+1+2*negative], dtype=np.int64)

    cdef int* unigram_table
    unigram_table = <int*>ptr_val

    for i in range(sent_id_len):
        ctx_count = 0
        actual_window = rand() % window + 1
        tar_id = sent_id[i]
        for j in range(i-window, i-actual_window):
            pos = j - (i-window)
            data[i, pos] = vocab_size
        for j in range(i-actual_window, i):
            pos = j - (i-window)
            if j < 0 or j >= sent_id_len:
                data[i, pos] = vocab_size
            else:
                data[i, pos] = sent_id[j]
                data[i, pos + 2*window] = 1
                ctx_count += 1
        for j in range(i+1, i+1+actual_window):
            pos = j - (i-window) - 1
            if j < 0 or j >= sent_id_len:
                data[i, pos] = vocab_size
            else:
                data[i, pos] = sent_id[j]
                data[i, pos + 2*window] = 1
                ctx_count += 1
        for j in range(i+1+actual_window, i+1+window):
            pos = j - (i-window) - 1
            data[i, pos] = vocab_size

        data[i, 4*window] = tar_id

        # negative sampling
        neg_count = 0
        for n in range(negative):
            t = get_unigram_table_at_idx(unigram_table, next_random)
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if t == tar_id:
                continue

            data[i, 4*window+1+neg_count] = t
            neg_count += 1

        # neg mask
        for n in range(neg_count):
            data[i, 4*window+1+negative+n] = 1

    return np.asarray(data)

def npmssg_select_sense(long[:] word_ids, float[:,:] context_feats, float[:,:] cluster_embs,
        int[:,:] word2sense, int[:] word_sense_cnts, float[:] counter_list, int word_ids_len,
        int emb_dim, int max_senses, float delta, int current_n_senses):
    cdef int b, d, pos, w_id, s_id
    cdef int max_sense_id, new_sense_id, create_count
    cdef float sim, max_sim
    cdef float[:] cluster_emb = np.zeros([emb_dim], dtype=np.float32)
    cdef long[:] senses = np.zeros([word_ids_len], dtype=np.int64)

    create_count = 0
    for b in range(word_ids_len):
        w_id = word_ids[b]

        # first encounter
        if counter_list[w_id] == 0.0:
            senses[b] = w_id
            for d in range(emb_dim):
                cluster_embs[w_id, d] += context_feats[b, d]
            counter_list[w_id] += 1.0
            continue

        # not first encounter
        max_sense_id = -1
        max_sim = -10.0
        for s_id in range(word_sense_cnts[w_id]):
            s_id = word2sense[w_id][s_id]
            if counter_list[s_id] == 0:
                print("zero:", s_id)
            for d in range(emb_dim):
                cluster_emb[d] = cluster_embs[s_id, d] / counter_list[s_id]
            sim = cosine(emb_dim, &context_feats[b,0], &cluster_emb[0])
            if sim > max_sim:
                max_sim = sim
                max_sense_id = s_id

        # create new sense
        if word_sense_cnts[w_id] < max_senses:
            if max_sim < delta:
                max_sense_id = current_n_senses + create_count
                word2sense[w_id][ word_sense_cnts[w_id] ] = max_sense_id
                word_sense_cnts[w_id] += 1
                create_count += 1

        senses[b] = max_sense_id
        for d in range(emb_dim):
            cluster_embs[max_sense_id, d] += context_feats[b, d]
        counter_list[max_sense_id] += 1.0

    return senses, create_count


# Read embedding file to dict object
def emb2dict(str emb_file, normalize=True, keep_eos=True):
    emb_file_byte_string = emb_file.encode('utf-8')
    cdef char* fname = emb_file_byte_string
    cdef char* token
    cdef char[100] word # should not be a point, but a static array
    cdef int count, size
    cdef double num
    cdef FILE* cfile
    cdef char* line = NULL
    cdef size_t l = 0
    cdef ssize_t read

    vectors = {}
    cfile = fopen(fname, 'r')
    if cfile == NULL:
        sys.exit('embedding file {} not found'.format(emb_file))

    # get file header
    getline(&line, &l, cfile)
    token = strtok(line, ' \t')
    token = strtok(NULL, ' \t\n')
    size = atoi(token)
    cyvec = cython.view.array(shape=(size,), itemsize=sizeof(double), format='d')
    cdef double [:] cyvec_view = cyvec

    while True:
        read = getline(&line, &l, cfile)
        if read == -1:
            break
        token = strtok(line, ' \t')
        memset(word, '\0', sizeof(word))
        strcpy(word, token)
        count = 0
        while token != NULL and count < size:
            token = strtok(NULL, ' \t\n')
            num = atof(token)
            # num = strtod(token, NULL)
            cyvec_view[count] = num
            count += 1

        vec = np.asarray(cyvec_view)
        if normalize:
            norm = np.linalg.norm(vec)
            if norm > 0.0:
                vec /= norm # normalize
        vec = vec.reshape(1,-1)
        vectors[word.decode('utf-8')] = vec.copy() # w/o copy, it will be the last word emb

    fclose(cfile)
    return vectors


# Read embedding file to numpy array
def emb2np(str emb_file, int vocab_size, int size, char2id=None, keep_eos=True):
    """
    This is a faster version of utils.emb2np.
    char2id: when emb_file is a file of character embeddings, number fo rows in emb_file, i.e., n,
        is not necessarily equal to vocab_size. When char2id is provided, retrieve
        only len(char2id), i.e., vocab_size, rows from emb_file, and use char2id[items[0]] as the index in arr.
    """
    emb_file_byte_string = emb_file.encode('utf-8')
    cdef char* fname = emb_file_byte_string
    cdef int head_vs, head_s
    cdef char* token
    cdef char[100] word # should not be a point, but a static array, cuz strcpy token -> word
    cdef int count, idx, char_count
    cdef double num
    cdef FILE* cfile
    cdef char* line = NULL
    cdef size_t l = 0
    cdef ssize_t read

    arr = cython.view.array(shape=(vocab_size+1, size), itemsize=sizeof(double), format='d') # the array to be returned
    cdef double[:,:] arr_view = arr
    vec = cython.view.array(shape=(size,), itemsize=sizeof(double), format='d')
    cdef double[:] vec_view = vec

    cfile = fopen(fname, 'r')
    if cfile == NULL:
        sys.exit('embedding file {} not found'.format(emb_file))

    # Read header line (first line) and check if head_s matches size
    getline(&line, &l, cfile)
    token = strtok(line, ' ')
    head_vs = atoi(token)
    token = strtok(NULL, ' \n')
    head_s = atoi(token)
    if head_s != size:
        sys.exit('size {} does not match the m {} in emb_file {}!'.format(size, head_s, emb_file))

    # Read the second line (first line after header) and check if </s> is in vocab
    getline(&line, &l, cfile)
    token = strtok(line, ' ')
    memset(word, '\0', sizeof(word))
    strcpy(word, token)
    count = 0
    while token != NULL and count < size:
        token = strtok(NULL, ' \n')
        num = atof(token)
        vec_view[count] = num
        count += 1

    if char2id is None:
        if word.decode('utf-8') == '</s>':
            if vocab_size+1 != head_vs:
                sys.exit('vocab_size {} does not match the head_vs {} in emb_file {}!'.format(vocab_size, head_vs, emb_file))
            arr_view[vocab_size,:] = vec_view.copy()
            idx = 0
        else:
            if vocab_size != head_vs:
                sys.exit('vocab_size {} does not match the head_vs {} in emb_file {}!'.format(vocab_size, head_vs, emb_file))
            arr_view[0,:] = vec_view.copy()
            idx = 1
        while True:
            read = getline(&line, &l, cfile)
            if read == -1:
                break
            token = strtok(line, ' ')
            count = 0
            while token != NULL and count < size:
                token = strtok(NULL, ' \n')
                num = atof(token)
                vec_view[count] = num
                count += 1
            arr_view[idx,:] = vec_view.copy()
            idx += 1
    else:
        if word.decode('utf-8') == '</s>':
            if vocab_size+1 > head_vs:
                sys.exit('len(char2id) {} is larger than the head_vs {} in emb_file {}!'.format(len(char2id), head_vs, emb_file))
            arr_view[vocab_size,:] = vec_view.copy()
            char_count = 0
        else:
            if vocab_size > head_vs:
                sys.exit('len(char2id) {} is larger than the head_vs {} in emb_file {}!'.format(len(char2id), head_vs, emb_file))
            idx = char2id[word.decode('utf-8')]
            arr_view[idx,:] = vec_view.copy()
            char_count = 1
        while True:
            read = getline(&line, &l, cfile)
            if read == -1:
                break
            token = strtok(line, ' ')
            memset(word, '\0', sizeof(word))
            strcpy(word, token)
            if word.decode('utf-8') not in char2id:
                continue
            idx = char2id[word.decode('utf-8')]
            count = 0
            while token != NULL and count < size:
                token = strtok(NULL, ' \n')
                num = atof(token)
                vec_view[count] = num
                count += 1
            arr_view[idx,:] = vec_view.copy()
            char_count += 1
        if char_count != vocab_size:
            sys.exit('not enough characters in emb_file {}'.format(emb_file))

    if not keep_eos:
        return np.asarray(arr_view[:-1,:])

    return np.asarray(arr_view)


# Faster file_split?
def file_split(str file_name):
    file_name_byte_string = file_name.encode('utf-8')
    cdef char* fname = file_name_byte_string
    cdef char* token
    cdef char[100] word
    cdef FILE* cfile
    cdef char* line = NULL
    cdef size_t l = 0
    cdef ssize_t read

    cfile = fopen(fname, 'r')
    while True:
        read = getline(&line, &l, cfile)
        if read == -1:
            break
        token = strtok(line, ' \t\n')
        while token != NULL:
            yield token.decode('utf-8')
            token = strtok(NULL, ' \t\n')
