import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import data_producer


class CBOWMean(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lens):
        ctx.save_for_backward(x)
        x = torch.sum(x, 1, keepdim=True)
        x = x.permute(1,2,0) / lens
        return x.permute(2,0,1)
    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_variables
        return g.expand_as(x), None


class CBOW(nn.Module):
    def __init__(self, args):
        super(CBOW, self).__init__()
        self.emb0_lookup = nn.Embedding(args.vocab_size+1, args.size, padding_idx=args.vocab_size, sparse=True)
        self.emb1_lookup = nn.Embedding(args.vocab_size, args.size, sparse=True)
        self.emb0_lookup.weight.data.uniform_(-0.5/args.size, 0.5/args.size)
        self.emb0_lookup.weight.data[args.vocab_size].fill_(0)
        self.emb1_lookup.weight.data.uniform_(-0.5/args.size, 0.5/args.size)
        self.window = args.window
        self.negative = args.negative

    def forward(self, data):
        ctx_indices = data[:, 0:2*self.window]
        ctx_lens = data[:, 2*self.window].float()
        word_idx = data[:, 2*self.window+1]
        neg_indices = data[:, 2*self.window+2:2*self.window+2+self.negative]
        neg_mask = data[:, 2*self.window+2+self.negative:].float()

        c_embs = self.emb0_lookup(ctx_indices)
        w_embs = self.emb1_lookup(word_idx)
        n_embs = self.emb1_lookup(neg_indices)

        c_embs = CBOWMean.apply(c_embs, ctx_lens)

        pos_ips = torch.sum(c_embs[:,0,:] * w_embs, 1)
        neg_ips = torch.bmm(n_embs, c_embs.permute(0,2,1))[:,:,0]

        # Neg Log Likelihood
        pos_loss = torch.sum(-F.logsigmoid(torch.clamp(pos_ips,max=10,min=-10)))
        neg_loss = torch.sum(-F.logsigmoid(torch.clamp(-neg_ips,max=10,min=-10)) * neg_mask)
        loss = pos_loss + neg_loss

        return loss


class CBOW_dropout(nn.Module):
    def __init__(self, args):
        super(CBOW_dropout, self).__init__()
        self.emb0_lookup = nn.Embedding(args.vocab_size+1, args.size, padding_idx=args.vocab_size, sparse=True)
        self.emb1_lookup = nn.Embedding(args.vocab_size, args.size, sparse=True)
        self.emb0_lookup.weight.data.uniform_(-0.5/args.size, 0.5/args.size)
        self.emb0_lookup.weight.data[args.vocab_size].fill_(0)
        self.emb1_lookup.weight.data.uniform_(-0.5/args.size, 0.5/args.size)
        self.window = args.window
        self.negative = args.negative
        self.drop_prob = args.drop_prob

    def forward(self, data):
        ctx_indices = data[:, 0:2*self.window]
        ctx_lens = data[:, 2*self.window].float()
        word_idx = data[:, 2*self.window+1]
        neg_indices = data[:, 2*self.window+2:2*self.window+2+self.negative]
        neg_mask = data[:, 2*self.window+2+self.negative:].float()

        c_embs = self.emb0_lookup(ctx_indices)
        w_embs = self.emb1_lookup(word_idx)
        n_embs = self.emb1_lookup(neg_indices)

        # Apply dropout masks
        c_mask = torch.rand(c_embs.shape[0], c_embs.shape[1], c_embs.shape[2], requires_grad=False) < (1 - self.drop_prob)
        c_embs = c_mask.type(torch.FloatTensor) * c_embs
        # w_mask = torch.rand(w_embs.shape[0], w_embs.shape[1], requires_grad=False) < (1 - self.drop_prob)
        # w_embs = w_mask * w_embs
        # n_mask = torch.rand(n_embs.shape[0], n_embs.shape[1], requires_grad=False) < (1 - self.drop_prob)
        # n_embs = n_mask * n_embs


        c_embs = CBOWMean.apply(c_embs, ctx_lens)

        pos_ips = torch.sum(c_embs[:,0,:] * w_embs, 1)
        neg_ips = torch.bmm(n_embs, c_embs.permute(0,2,1))[:,:,0]

        # Neg Log Likelihood
        pos_loss = torch.sum(-F.logsigmoid(torch.clamp(pos_ips,max=10,min=-10)))
        neg_loss = torch.sum(-F.logsigmoid(torch.clamp(-neg_ips,max=10,min=-10)) * neg_mask)
        loss = pos_loss + neg_loss

        return loss


class SG_dropout(nn.Module):
    def __init__(self, args):
        super(SG_dropout, self).__init__()
        self.emb0_lookup = nn.Embedding(args.vocab_size+1, args.size, padding_idx=args.vocab_size, sparse=True)
        self.emb1_lookup = nn.Embedding(args.vocab_size, args.size, sparse=True)
        self.emb0_lookup.weight.data.uniform_(-0.5/args.size, 0.5/args.size)
        self.emb1_lookup.weight.data.zero_()
        self.window = args.window
        self.negative = args.negative
        self.drop_prob = args.drop_prob

    def forward(self, data):
        word_idx = data[:, 1]
        ctx_idx = data[:, 0]
        neg_indices = data[:, 2:2+self.negative]
        neg_mask = data[:, 2+self.negative:].float()

        w_embs = self.emb0_lookup(word_idx)
        c_embs = self.emb1_lookup(ctx_idx)
        n_embs = self.emb1_lookup(neg_indices)

        # Apply dropout masks
        # c_mask = torch.rand(c_embs.shape[0], c_embs.shape[1], requires_grad=False) < (1 - self.drop_prob)
        # c_embs = c_mask.type(torch.FloatTensor) * c_embs
        w_mask = torch.rand(w_embs.shape[0], w_embs.shape[1], requires_grad=False) < (1 - self.drop_prob)
        w_embs = w_mask.type(torch.FloatTensor) * w_embs
        # n_mask = torch.rand(n_embs.shape[0], n_embs.shape[1], requires_grad=False) < (1 - self.drop_prob)
        # n_embs = n_mask * n_embs

        pos_ips = torch.sum(w_embs * c_embs, 1)
        neg_ips = torch.bmm(n_embs, torch.unsqueeze(w_embs,1).permute(0,2,1))[:,:,0]

        # Neg Log Likelihood
        pos_loss = torch.sum( -F.logsigmoid(torch.clamp(pos_ips,max=10,min=-10)) )
        neg_loss = torch.sum( -F.logsigmoid(torch.clamp(-neg_ips,max=10,min=-10)) * neg_mask )

        return pos_loss + neg_loss


# Class for CWE_cbow
class CWE_cbow(nn.Module):
    def __init__(self, args):
        super(CWE_cbow, self).__init__()
        self.emb0_lookup = nn.Embedding(args.vocab_size+1, args.size, padding_idx=args.vocab_size, sparse=True)
        self.emb1_lookup = nn.Embedding(args.vocab_size, args.size, sparse=True)
        self.emb0_lookup.weight.data.uniform_(-0.5/args.size, 0.5/args.size)
        self.emb0_lookup.weight.data[args.vocab_size].fill_(0)
        self.emb1_lookup.weight.data.uniform_(-0.5/args.size, 0.5/args.size)
        self.emb0_char_lookup = nn.Embedding(args.char_vocab_size+1, args.size, padding_idx=args.char_vocab_size, sparse=True)
        self.emb0_char_lookup.weight.data.uniform_(-0.5/args.size, 0.5/args.size)
        self.emb0_char_lookup.weight.data[args.char_vocab_size].fill_(0)

        if args.pt_emb0:
            print('reading pretrained emb0_lookup table')
            self.emb0_lookup.weight.data.copy_(torch.from_numpy(
                data_producer.emb2np(args.pt_emb0, args.vocab_size, args.size)))
        if args.pt_emb0_char:
            print('reading pretrained emb0_char_lookup table')
            self.emb0_char_lookup.weight.data.copy_(torch.from_numpy(
                data_producer.emb2np(args.pt_emb0_char, args.char_vocab_size, args.size, args.char2id)))
        if args.pt_emb1:
            print('reading pretrained emb1_lookup table')
            self.emb1_lookup.weight.data.copy_(torch.from_numpy(
                data_producer.emb2np(args.pt_emb1, args.vocab_size, args.size, keep_eos=False)))
        if args.fix_emb0:
            self.emb0_lookup.weight.requires_grad = False
        if args.fix_emb1:
            self.emb1_lookup.weight.requires_grad = False
        if args.fix_emb0_char:
            self.emb0_char_lookup.weight.requires_grad = False

        self.pretrain_word = args.pretrain_word
        self.pretrain_char = args.pretrain_char
        self.window = args.window
        self.negative = args.negative
        self.vocab_size = args.vocab_size
        self.char_vocab_size = args.char_vocab_size
        self.use_cuda = args.cuda
        self.max_word_len = args.max_word_len

    def forward(self, data):
        word_data, char_data = data
        ctx_word_indices = word_data[:, 0:2*self.window]
        ctx_lens = word_data[:, 2*self.window].float()
        tar_word_idx = word_data[:, 2*self.window+1]
        neg_indices = word_data[:, 2*self.window+2:2*self.window+2+self.negative]
        neg_mask = word_data[:, 2*self.window+2+self.negative:].float()
        ctx_char_indices = char_data[:, 0:2*self.window*self.max_word_len]
        actual_char_num = char_data[:, 2*self.window*self.max_word_len].float()

        if self.pretrain_word and not self.pretrain_char:
            ctx_word_embs = self.emb0_lookup(ctx_word_indices)
            avg_word_embs = CBOWMean.apply(ctx_word_embs, ctx_lens)
            avg_embs = avg_word_embs
        elif not self.pretrain_word and self.pretrain_char:
            ctx_char_embs = self.emb0_char_lookup(ctx_char_indices)
            avg_char_embs = CBOWMean.apply(ctx_char_embs, actual_char_num)
            avg_embs = avg_char_embs
        else:
            ctx_word_embs = self.emb0_lookup(ctx_word_indices)
            ctx_char_embs = self.emb0_char_lookup(ctx_char_indices) # This step is slow because ctx_char_indices has two many columns
            avg_word_embs = CBOWMean.apply(ctx_word_embs, ctx_lens)
            avg_char_embs = CBOWMean.apply(ctx_char_embs, actual_char_num)
            avg_embs = 0.5*avg_word_embs + 0.5*avg_char_embs

        # SLOW methods
        # ctx_char_embs = torch.cat([self.emb0_char_lookup(ctx_char_indices[:, i*self.max_word_len:(i+1)*self.max_word_len]) for i in range(2*self.window)], 1)
        # SLOW: ctx_char_embs = torch.stack([self.emb0_char_lookup(ind) for ind in torch.unbind(ctx_char_indices, 1)], 1)
        # SLOW: ctx_char_embs = torch.stack([self.emb0_char_lookup(ctx_char_indices[:,i]) for i in range(2*self.window*self.max_word_len)], 1)

        tar_word_embs = self.emb1_lookup(tar_word_idx)
        neg_embs = self.emb1_lookup(neg_indices)

        pos_ips = torch.sum(avg_embs[:,0,:] * tar_word_embs, 1)
        neg_ips = torch.bmm(neg_embs, avg_embs.permute(0,2,1))[:,:,0]
        neg_ips = neg_ips * neg_mask

        # Neg Log Likelihood
        pos_loss = torch.sum( -F.logsigmoid(torch.clamp(pos_ips,max=10,min=-10)) )
        neg_loss = torch.sum( -F.logsigmoid(torch.clamp(-neg_ips,max=10,min=-10)) * neg_mask )

        return pos_loss + neg_loss


class SG(nn.Module):
    def __init__(self, args):
        super(SG, self).__init__()
        self.emb0_lookup = nn.Embedding(args.vocab_size+1, args.size, padding_idx=args.vocab_size, sparse=True)
        self.emb1_lookup = nn.Embedding(args.vocab_size, args.size, sparse=True)
        self.emb0_lookup.weight.data.uniform_(-0.5/args.size, 0.5/args.size)
        self.emb1_lookup.weight.data.zero_()
        self.window = args.window
        self.negative = args.negative

    def forward(self, data):
        word_idx = data[:, 1]
        ctx_idx = data[:, 0]
        neg_indices = data[:, 2:2+self.negative]
        neg_mask = data[:, 2+self.negative:].float()

        w_embs = self.emb0_lookup(word_idx)
        c_embs = self.emb1_lookup(ctx_idx)
        n_embs = self.emb1_lookup(neg_indices)

        pos_ips = torch.sum(w_embs * c_embs, 1)
        neg_ips = torch.bmm(n_embs, torch.unsqueeze(w_embs,1).permute(0,2,1))[:,:,0]

        # Neg Log Likelihood
        pos_loss = torch.sum( -F.logsigmoid(torch.clamp(pos_ips,max=10,min=-10)) )
        neg_loss = torch.sum( -F.logsigmoid(torch.clamp(-neg_ips,max=10,min=-10)) * neg_mask )

        return pos_loss + neg_loss


class CWE_sg(SG):
    def __init__(self, args):
        super(CWE_sg, self).__init__(args)
        self.emb0_char_lookup = nn.Embedding(args.char_vocab_size+1, args.size, sparse=True)
        # self.emb0_char_lookup.weight.data.zero_()
        self.emb0_char_lookup.weight.data.uniform_(-0.5/args.size, 0.5/args.size)
        self.emb0_char_lookup.weight.data[args.char_vocab_size].fill_(0)

        if args.pt_emb0:
            print('reading pretrained emb0_lookup table')
            self.emb0_lookup.weight.data.copy_(torch.from_numpy(
                data_producer.emb2np(args.pt_emb0, args.vocab_size, args.size)))
        if args.pt_emb0_char:
            print('reading pretrained emb0_char_lookup table')
            self.emb0_char_lookup.weight.data.copy_(torch.from_numpy(
                data_producer.emb2np(args.pt_emb0_char, args.char_vocab_size, args.size, args.char2id)))
        if args.pt_emb1:
            print('reading pretrained emb1_lookup table')
            self.emb1_lookup.weight.data.copy_(torch.from_numpy(
                data_producer.emb2np(args.pt_emb1, args.vocab_size, args.size, keep_eos=False)))
        if args.fix_emb0:
            self.emb0_lookup.weight.requires_grad = False
        if args.fix_emb1:
            self.emb1_lookup.weight.requires_grad = False
        if args.fix_emb0_char:
            self.emb0_char_lookup.weight.requires_grad = False

        self.pretrain_word = args.pretrain_word
        self.pretrain_char = args.pretrain_char
        self.vocab_size = args.vocab_size
        self.char_vocab_size = args.char_vocab_size
        self.use_cuda = args.cuda
        self.max_word_len = args.max_word_len

    def forward(self, data):
        word_data, char_data = data
        tar_word_idx = word_data[:, 1]
        ctx_word_idx = word_data[:, 0]
        neg_indices = word_data[:, 2:2+self.negative]
        neg_mask = word_data[:, 2+self.negative:].float()
        tar_char_indices = char_data[:, 0:self.max_word_len]
        actual_char_num = char_data[:, self.max_word_len].float()

        if self.pretrain_word and not self.pretrain_char:
            tar_word_embs = self.emb0_lookup(tar_word_idx)
            avg_embs = tar_word_embs.unsqueeze(1)
        elif self.pretrain_char and not self.pretrain_word:
            tar_char_embs = self.emb0_char_lookup(tar_char_indices)
            avg_embs = CBOWMean.apply(tar_char_embs, actual_char_num)
        else:
            tar_word_embs = self.emb0_lookup(tar_word_idx)
            tar_char_embs = self.emb0_char_lookup(tar_char_indices)
            avg_embs = 0.5*tar_word_embs.unsqueeze(1) + 0.5*CBOWMean.apply(tar_char_embs, actual_char_num)
        
        ctx_embs = self.emb1_lookup(ctx_word_idx)
        neg_embs = self.emb1_lookup(neg_indices)

        pos_ips = torch.sum(avg_embs.squeeze(1) * ctx_embs, 1)
        neg_ips = torch.bmm(neg_embs, avg_embs.permute(0,2,1))[:,:,0]

        pos_loss = torch.sum(-F.logsigmoid(torch.clamp(pos_ips,max=10,min=-10)))
        neg_loss = torch.sum(-F.logsigmoid(torch.clamp(-neg_ips,max=10,min=-10)) * neg_mask)
        # pos_loss = torch.sum(-F.logsigmoid(pos_ips))
        # neg_loss = torch.sum(-F.logsigmoid(-neg_ips) * neg_mask)

        return pos_loss + neg_loss


class DSE_sg(CWE_sg):
    def __init__(self, args):
        super(DSE_sg, self).__init__(args)
        self.h_lookup = nn.Embedding(args.vocab_size+1, 2)
        if args.h_init == 'random':
            self.h_lookup.weight.data.uniform_(0.5, 0.5)
        elif args.h_init == 'fixed':
            self.h_lookup.weight.data.fill_(0)
        self.l2 = args.l2

    def forward(self, data):
        word_data, char_data = data
        tar_word_idx = word_data[:, 1]
        ctx_word_idx = word_data[:, 0]
        neg_indices = word_data[:, 2:2+self.negative]
        neg_mask = word_data[:, 2+self.negative:].float()
        tar_char_indices = char_data[:, 0:self.max_word_len]
        actual_char_num = char_data[:, self.max_word_len].float()

        if self.pretrain_word and not self.pretrain_char:
            tar_word_embs = self.emb0_lookup(tar_word_idx)
            avg_embs = tar_word_embs.unsqueeze(1)
        elif self.pretrain_char and not self.pretrain_word:
            tar_char_embs = self.emb0_char_lookup(tar_char_indices)
            avg_embs = CBOWMean.apply(tar_char_embs, actual_char_num)
        else:
            tar_word_embs = self.emb0_lookup(tar_word_idx) # Shape: [batch_size, embs_size]
            tar_char_embs = self.emb0_char_lookup(tar_char_indices) # Shape: [batch_size, max_word_len, embs_size]

            h = self.h_lookup(tar_word_idx)
            w = F.softmax(h, dim=1) # Shape: [batch_size, 2]
            word_weight = w[:,0].unsqueeze(1) # Shape: [batch_size, 1]
            tar_word_embs = word_weight * tar_word_embs # Weighted word embeddings for target word

            batch_size = char_data.size()[0]
            char_weight = w[:,1].unsqueeze(1).expand(batch_size, self.max_word_len).contiguous().view(batch_size, -1).unsqueeze(2)
            tar_char_embs = char_weight * tar_char_embs # Weighted char embeddings for all chars in target word
            avg_embs = tar_word_embs.unsqueeze(1) + CBOWMean.apply(tar_char_embs, actual_char_num)
        
        ctx_embs = self.emb1_lookup(ctx_word_idx)
        neg_embs = self.emb1_lookup(neg_indices)
        
        pos_ips = torch.sum(avg_embs.squeeze(1) * ctx_embs, 1)
        neg_ips = torch.bmm(neg_embs, avg_embs.permute(0,2,1))[:,:,0]

        pos_loss = torch.sum(-F.logsigmoid(torch.clamp(pos_ips,max=10,min=-10)))
        neg_loss = torch.sum(-F.logsigmoid(torch.clamp(-neg_ips,max=10,min=-10)) * neg_mask)

        return pos_loss + neg_loss


class DSE_sg2(CWE_sg):
    """
    """
    pass


# Class for CWE_cbow with h
class DSE_cbow(CWE_cbow):
    def __init__(self, args):
        super(DSE_cbow, self).__init__(args)
        self.h_lookup = nn.Embedding(args.vocab_size+1, 2)
        if args.h_init == 'random':
            self.h_lookup.weight.data.uniform_(-0.5, 0.5)
        elif args.h_init == 'fixed':
            self.h_lookup.weight.data.fill_(0)
        self.l2 = args.l2

    def forward(self, data):
        word_data, char_data = data
        ctx_word_indices = word_data[:, 0:2*self.window]
        ctx_lens = word_data[:, 2*self.window].float()
        tar_word_idx = word_data[:, 2*self.window+1]
        neg_indices = word_data[:, 2*self.window+2:2*self.window+2+self.negative]
        neg_mask = word_data[:, 2*self.window+2+self.negative:].float()

        ctx_char_indices = char_data[:, 0:2*self.window*self.max_word_len]
        actual_char_num = char_data[:, 2*self.window*self.max_word_len].float()

        ctx_word_embs = self.emb0_lookup(ctx_word_indices)
        ctx_char_embs = self.emb0_char_lookup(ctx_char_indices)
        tar_word_embs = self.emb1_lookup(tar_word_idx)
        neg_embs = self.emb1_lookup(neg_indices)

        k = self.h_lookup(ctx_word_indices)
        w = F.softmax(k, 2)
        word_weight = w[:,:,0].unsqueeze(2) # [batch_size, 2*window] -> [batch_size, 2*window, 1]
        ctx_word_embs = word_weight*ctx_word_embs # put weight on word embedding
        bs = char_data.size()[0]
        char_weight = w[:,:,1].unsqueeze(2).expand(bs, 2*self.window, self.max_word_len).contiguous().view(bs, -1).unsqueeze(2) # [batch_size, 2*window] -> [batch_size, 2*window, max_word_len] -> [batch_size, 2*window*max_word_len, 1]
        ctx_char_embs = char_weight*ctx_char_embs

        avg_word_embs = CBOWMean.apply(ctx_word_embs, ctx_lens)
        avg_char_embs = CBOWMean.apply(ctx_char_embs, actual_char_num)
        avg_embs = avg_word_embs + avg_char_embs

        ##
        # Old way: getting k, w from tar_word_idx
        # k = self.h_lookup(tar_word_idx)
        # w = F.softmax(k, 1) # pytorch>=0.3
        # w = F.softmax(k) # for pytorch<=0.2x
        # w = F.tanh(k) # experiment with not using softmax
        # avg_embs = w[:,0].unsqueeze(1).expand_as(avg_word_embs[:,0,:]).unsqueeze(1)*avg_word_embs + w[:,1].unsqueeze(1).expand_as(avg_char_embs[:,0,:]).unsqueeze(1)*avg_char_embs

        pos_ips = torch.sum(avg_embs[:,0,:] * tar_word_embs, 1)
        neg_ips = torch.bmm(neg_embs, avg_embs.permute(0,2,1))[:,:,0]
        neg_ips = neg_ips * neg_mask

        # Neg Log Likelihood
        pos_loss = torch.sum( -F.logsigmoid(torch.clamp(pos_ips,max=10,min=-10)) )
        neg_loss = torch.sum( -F.logsigmoid(torch.clamp(-neg_ips,max=10,min=-10)) * neg_mask )
        l2_reg = self.l2 * torch.sum(torch.clamp(k ** 2, max=10, min=-10))
        loss = pos_loss + neg_loss + l2_reg

        return loss


# CWE_cbow with one k, which is a trainable scalar var, instead of a lookup table
class CWE_cbow_sk(CWE_cbow):
    def __init__(self, args):
        super(CWE_cbow_sk, self).__init__(args)
        self.sk = nn.Parameter(torch.FloatTensor(1,2))
        self.l2 = args.l2
        self.sk.data.fill_(0.0)

    def forward(self, data):
        word_data, char_data = data
        ctx_word_indices = word_data[:, 0:2*self.window]
        ctx_lens = word_data[:, 2*self.window].float()
        tar_word_idx = word_data[:, 2*self.window+1]
        neg_indices = word_data[:, 2*self.window+2:2*self.window+2+self.negative]
        neg_mask = word_data[:, 2*self.window+2+self.negative:].float()

        ctx_char_indices = char_data[:, 0:2*self.window*self.max_word_len]
        actual_char_num = char_data[:, 2*self.window*self.max_word_len].float()

        ctx_word_embs = self.emb0_lookup(ctx_word_indices)
        ctx_char_embs = self.emb0_char_lookup(ctx_char_indices)
        tar_word_embs = self.emb1_lookup(tar_word_idx)
        neg_embs = self.emb1_lookup(neg_indices)

        avg_word_embs = CBOWMean.apply(ctx_word_embs, ctx_lens)
        avg_char_embs = CBOWMean.apply(ctx_char_embs, actual_char_num)
        w = F.softmax(self.sk)
        # avg_embs = w[0,0].expand_as(avg_word_embs)*avg_word_embs + w[0,1].expand_as(avg_char_embs)*avg_char_embs
        avg_embs = w[0,0]*avg_word_embs + w[0,1]*avg_char_embs
        # avg_embs = avg_word_embs
        # avg_embs = avg_char_embs

        pos_ips = torch.sum(avg_embs[:,0,:] * tar_word_embs, 1)
        neg_ips = torch.bmm(neg_embs, avg_embs.permute(0,2,1))[:,:,0]
        neg_ips = neg_ips * neg_mask

        # Neg Log Likelihood
        pos_loss = torch.sum( -F.logsigmoid(torch.clamp(pos_ips,max=10,min=-10)) )
        neg_loss = torch.sum( -F.logsigmoid(torch.clamp(-neg_ips,max=10,min=-10)) * neg_mask )
        # sk_sq = torch.clamp(self.sk[0,0],max=10,min=-10) * torch.clamp(self.sk[0,0],max=10,min=-10)
        # reg_loss = self.l2 * sk_sq * word_data.data.shape[0]
        loss = pos_loss + neg_loss

        return loss


# fastText
class FastText_sg(SG):

    pass


# Element-wise weights
class CWE_cbow_elm(CWE_cbow):
    def __init__(self, args):
        super(CWE_cbow_elm, self).__init__(args)
        self.emb_weight = nn.Embedding(args.vocab_size+1, args.size)
        self.emb_weight.weight.data.uniform_(-0.5, 0.5)
        # self.att_model = nn.Sequential(
        #     nn.Linear(args.size, 600),
        #     nn.Tanh(),
        #     nn.Linear(600, args.size),
        #     nn.Sigmoid()
        # )
        self.l2 = args.l2

    def forward(self, data):
        word_data, char_data = data
        ctx_word_indices = word_data[:, 0:2*self.window]
        ctx_lens = word_data[:, 2*self.window].float()
        tar_word_idx = word_data[:, 2*self.window+1]
        neg_indices = word_data[:, 2*self.window+2:2*self.window+2+self.negative]
        neg_mask = word_data[:, 2*self.window+2+self.negative:].float()

        ctx_char_indices = char_data[:, 0:2*self.window*self.max_word_len]
        actual_char_num = char_data[:, 2*self.window*self.max_word_len].float()

        ctx_word_embs = self.emb0_lookup(ctx_word_indices)
        ctx_char_embs = self.emb0_char_lookup(ctx_char_indices)
        tar_word_embs = self.emb1_lookup(tar_word_idx)
        neg_embs = self.emb1_lookup(neg_indices)

        avg_word_embs = CBOWMean.apply(ctx_word_embs, ctx_lens)
        avg_char_embs = CBOWMean.apply(ctx_char_embs, actual_char_num)
        # weight = self.att_model(avg_word_embs[:,0,:])
        # avg_embs = F.sigmoid(weight).unsqueeze(1)*avg_word_embs + (1 - F.sigmoid(weight)).unsqueeze(1)*avg_char_embs
        weight = self.emb_weight(ctx_word_indices)
        avg_embs = F.sigmoid(weight)*avg_word_embs + (1 - F.sigmoid(weight))*avg_char_embs
        # weight = self.emb_weight(tar_word_idx)
        # avg_embs = F.sigmoid(weight).unsqueeze(1)*avg_word_embs + (1 - F.sigmoid(weight)).unsqueeze(1)*avg_char_embs

        pos_ips = torch.sum(avg_embs[:,0,:] * tar_word_embs, 1)
        neg_ips = torch.bmm(neg_embs, avg_embs.permute(0,2,1))[:,:,0]
        neg_ips = neg_ips * neg_mask

        # Neg Log Likelihood
        pos_loss = torch.sum(-F.logsigmoid(torch.clamp(pos_ips,max=10,min=-10)))
        neg_loss = torch.sum(-F.logsigmoid(torch.clamp(-neg_ips,max=10,min=-10)) * neg_mask)
        l2_reg = self.l2 * torch.sum(torch.clamp(weight ** 2, max=10, min=-10))
        loss = pos_loss + neg_loss + l2_reg

        return loss


# Attention
class CWE_cbow_att(CWE_cbow):
    def __init__(self, args):
        super(CWE_cbow_att, self).__init__(args)
        self.att_model = nn.Sequential(
            nn.Linear(args.size, 100),
            nn.Tanh(),
            # nn.Linear(100, 50),
            # nn.Tanh(),
            nn.Linear(100, 2),
            nn.Sigmoid()
        )

    def forward(self, data):
        word_data, char_data = data
        ctx_word_indices = word_data[:, 0:2*self.window]
        ctx_lens = word_data[:, 2*self.window].float()
        tar_word_idx = word_data[:, 2*self.window+1]
        neg_indices = word_data[:, 2*self.window+2:2*self.window+2+self.negative]
        neg_mask = word_data[:, 2*self.window+2+self.negative:].float()

        ctx_char_indices = char_data[:, 0:2*self.window*self.max_word_len]
        actual_char_num = char_data[:, 2*self.window*self.max_word_len].float()

        ctx_word_embs = self.emb0_lookup(ctx_word_indices)
        ctx_char_embs = self.emb0_char_lookup(ctx_char_indices)
        tar_word_embs = self.emb1_lookup(tar_word_idx)
        neg_embs = self.emb1_lookup(neg_indices)

        avg_word_embs = CBOWMean.apply(ctx_word_embs, ctx_lens)
        out = self.att_model(avg_word_embs[:,0,:])
        w = F.softmax(out)
        avg_char_embs = CBOWMean.apply(ctx_char_embs, actual_char_num)
        avg_embs = w[:,0].unsqueeze(1).expand_as(avg_word_embs[:,0,:]).unsqueeze(1)*avg_word_embs + w[:,1].unsqueeze(1).expand_as(avg_char_embs[:,0,:]).unsqueeze(1)*avg_char_embs

        pos_ips = torch.sum(avg_embs[:,0,:] * tar_word_embs, 1)
        neg_ips = torch.bmm(neg_embs, avg_embs.permute(0,2,1))[:,:,0]
        neg_ips = neg_ips * neg_mask

        # Neg Log Likelihood
        pos_loss = torch.sum(-F.logsigmoid(torch.clamp(pos_ips,max=10,min=-10)))
        neg_loss = torch.sum(-F.logsigmoid(torch.clamp(-neg_ips,max=10,min=-10)) * neg_mask)
        # l2_reg = self.l2 * torch.sum(torch.clamp(l2_out ** 2, max=10, min=-10))
        loss = pos_loss + neg_loss

        return loss

    def output_theta(self):
        out = self.att_model(self.emb0_lookup.weight)
        theta = F.softmax(out) # A.k.a., w
        if self.use_cuda:
            theta = theta.data.cpu().numpy()
        else:
            theta = theta.data.numpy()
        return theta
