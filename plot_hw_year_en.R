library(data.table)
library(ggplot2)
library(stringr)


# Input options
lang1 = 'de'
if (lang1 == "de") {
    lang2 = "german"
} else if (lang1 == "fr") {
    lang2 = "french"
} else if (lang1 == "es") {
    lang2 = "spanish"
} else if (lang1 == "en") {
    lang2 = "english"
}


year_file = paste0("results/1gram_first_year_", lang2, ".txt")
vocab_file = paste0("results/wiki", lang1, "_wll7_mc5_vocab.csv")


# Join 1gram_earliest_year.txt with the resulting *_h.txt file of running main.py
join_hw = function(hw_file) {
    d1 = fread(year_file)
    setnames(d1, c('word', 'year'))
    setkey(d1, word)

    d2 = fread(hw_file, skip=2)
    setnames(d2, c('word', 'hw', 'hc'))
    setkey(d2, word)
    d12j = d1[d2,nomatch=0]
    
    d3 = fread(vocab_file)
    d3$V2 = NULL
    setnames(d3, c("word", "ngramCount"))
    setkey(d3, word)
    d123j = d12j[d3, nomatch=0]
    d123j$wordLen = as.character(d123j$ngramCount)
    d123j
}

## DSE sg
model_str = "DSE_sg"
hw_file = paste0("results/wiki", lang1, "_", model_str, "_wll7_mc5_iter5_t1_h.txt")

d = join_hw(hw_file)

# All years & all words
p = ggplot(d, aes(x=year, y=hw)) +
    stat_summary(fun.y=mean, geom='line') +
    stat_summary(fun.data=mean_cl_boot, geom='errorbar') +
    geom_smooth() + theme_bw()
pdf(paste0("figs/hw_year_allWords_", model_str, "_wiki", lang1, ".pdf"), 5, 5)
plot(p)
dev.off()

# All years & ngramCount > 1 words
p = ggplot(d[ngramCount >= 2], aes(x=year, y=hw)) +
    stat_summary(fun.y=mean, geom='line') +
    stat_summary(fun.data=mean_cl_boot, geom='errorbar') +
    geom_smooth() + theme_bw()
pdf(paste0("figs/hw_year_multiNgram_", model_str, "_wiki", lang1, ".pdf"), 5, 5)
plot(p)
dev.off()

# All years & charNum = 2,3,4
p = ggplot(d[ngramCount >= 2 & ngramCount <= 7], aes(x=year, y=hw, color=wordLen, fill=wordLen)) +
    geom_smooth() + theme_bw()
pdf(paste0("figs/hw_year_ngramCount2to7_", model_str, "_wiki", lang1, ".pdf"), 5, 5)
plot(p)
dev.off()



## Combine CBOW and Skipgram
# d1 = fread('results/wikifullchn_CWE_cbow_k_wll7_mc5_iter5_p4_l20_t1_k.txt', skip=2)
# setnames(d1, c('word', 'hw', 'hc'))
# setkey(d1, word)
# d1$model = 'DSE(CBOW)'

# d2 = fread('results/wikifullchn_CWE_sg_k_wll7_mc5_iter5_p4_l20_t1_k.txt', skip=2)
# setnames(d2, c('word', 'hw', 'hc'))
# setkey(d2, word)
# d2$model = 'DSE(Skipgram)'

# d_year = fread('results/1gram_earliest_year.txt')
# setnames(d_year, c('word', 'year'))
# setkey(d_year, word)
# dc = rbindlist(list(d1, d2))
# dc = d_year[dc, nomatch=0]
# dc$charNum = str_length(dc$word)
# dc$wordLen = as.character(dc$charNum)

# All years & multi-character words
# p = ggplot(dc[charNum >= 2], aes(x=year, y=hw)) +
#     stat_summary(fun.data = mean_cl_boot, geom='errorbar', aes(linetype=model), alpha=.7, width=5) + 
#     geom_smooth(aes(fill=model, color=model)) + theme_bw() + 
#     labs(x='First-appearance-year of the word', y=expression(h^w)) + ylim(c(0.35,1)) + theme(legend.position=c(0.2,0.9)) + 
#     guides(linetype=guide_legend(title='Model'), 
#             fill=guide_legend(title='Model'),
#             color=guide_legend(title='Model'))
# pdf('figs/hw_year_allWords_cbow&sg_wikifull.pdf', 5, 5)
# plot(p)
# dev.off()

# All years & charNum = 2,3,4
# p = ggplot(dc[charNum >= 2 & charNum <= 4], aes(x=year, y=hw, color=wordLen, fill=wordLen, linetype=model)) +
#     geom_smooth() + theme_bw() + 
#     stat_summary(fun.y = mean, geom='point') + 
#     labs(x='First-appearance-year of the word', y=expression(h^w)) + 
#     theme(legend.position='bottom', legend.box='vertical') + 
#     guides(fill=guide_legend(title='Number of characters'), 
#            color=guide_legend(title='Number of characters'), 
#            linetype=guide_legend(title='Model'))
# pdf('figs/hw_year_charNum234_cbow&sg_wikifull.pdf', 5, 6)
# plot(p)
# dev.off()

# library(mgcv)
# new_year = data.table(year=seq(1550, 2000, 50))
# b0_2 = gam(hw ~ s(year, bs='cs'), data=dc[model=='DSE(Skipgram)' & charNum==2])
# hw_b0_2 = predict.gam(b0_2, new_year)

# data_pred = copy(new_year)
# data_pred$hw = hw_b0_2



###############################
# Joint data from all languages
join_all_langs = function() {
    lang_short = c("en", "de", "fr", "es")
    lang_long = c("english", "german", "french", "spanish")
    model_str = "DSE_sg"

    data = list()
    for (i in 1:4) {
        lang1 = lang_short[i]
        lang2 = lang_long[i]
        year_file = paste0("results/1gram_first_year_", lang2, ".txt")
        hw_file = paste0("results/wiki", lang1, "_", model_str, "_wll7_mc5_iter5_t1_h.txt")
        vocab_file = paste0("results/wiki", lang1, "_wll7_mc5_vocab.csv")

        d1 = fread(year_file)
        setnames(d1, c('word', 'year'))
        setkey(d1, word)

        # Include lower case words
        d1_lower = copy(d1)
        d1_lower[, word := str_to_lower(word)]
        d1_ext = unique(rbindlist(list(d1, d1_lower)))
        setkey(d1_ext, word)

        d2 = fread(hw_file, skip=2)
        setnames(d2, c('word', 'hw', 'hc'))
        setkey(d2, word)
        d12j = d1_ext[d2,nomatch=0]

        d3 = fread(vocab_file)
        d3$V2 = NULL
        setnames(d3, c("word", "ngramCount"))
        setkey(d3, word)
        d123j = d12j[d3, nomatch=0]
        d123j$wordLen = as.character(d123j$ngramCount)
        
        d123j$Language = str_to_title(lang2)
        data[[i]] = d123j
    }
    rbindlist(data)
}

d_all_langs = join_all_langs()

p = ggplot(d_all_langs[ngramCount >= 2 & ngramCount <= 7], aes(x=year, y=hw, color=wordLen, fill=wordLen)) +
    geom_smooth() + theme_bw() + facet_wrap(~Language) + 
    guides(fill=guide_legend(title="Ngram count"), color=guide_legend(title="Ngram count")) + 
    labs(x='First-appearance-year of the word', y=expression(h^w))
pdf("figs/hw_year_ngramCount2to7_DSE_sg_wikiAllLangs.pdf", 8, 7)
plot(p)
dev.off()