library(data.table)
library(ggplot2)
library(stringr)


# Input files
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



# Join 1gram_earliest_year.txt with the resulting *_h.txt file of running main.py
join_h_coefs = function(file) {
    d1 = fread('results/1gram_earliest_year.txt')
    setnames(d1, c('word', 'year'))
    setkey(d1, word)
    d2 = fread(file, skip=2)
    setnames(d2, c('word', 'hw', 'hc'))
    setkey(d2, word)
    d1j = d1[d2,nomatch=0]
    d1j
}


## DSE (CBOW)
d = join_h_coefs('results/wikifullchn_CWE_cbow_k_wll7_mc5_iter5_p4_l20_t1_k.txt')
d$charNum = str_length(d$word)
d$wordLen = as.character(d$charNum)

# All years & all words
p = ggplot(d, aes(x=year, y=hw)) +
    stat_summary(fun.y=mean, geom='line') +
    stat_summary(fun.data=mean_cl_boot, geom='errorbar') +
    geom_smooth() + theme_bw()
pdf('figs/hw_year_allWords_cbow_wikifull.pdf', 10, 4)
plot(p)
dev.off()

# All years & multi-character words
p = ggplot(d[charNum >= 2], aes(x=year, y=hw)) +
    stat_summary(fun.y=mean, geom='line') +
    stat_summary(fun.data=mean_cl_boot, geom='errorbar') +
    geom_smooth() + theme_bw()
pdf('figs/hw_year_multiCharWords_cbow_wikifull.pdf', 10, 4)
plot(p)
dev.off()

# All years & charNum = 2,3,4
p = ggplot(d[charNum >= 2 & charNum <= 4], aes(x=year, y=hw, color=wordLen, fill=wordLen)) +
    geom_smooth() + theme_bw()
pdf('figs/hw_year_charNum234_cbow_wikifull.pdf', 5, 5)
plot(p)
dev.off()


## DSE (Skipgram)
d = join_h_coefs('results/wikifullchn_CWE_sg_k_wll7_mc5_iter5_p4_l20_t1_k.txt')
d$charNum = str_length(d$word)
d$wordLen = as.character(d$charNum)

# All years & all words
p = ggplot(d, aes(x=year, y=hw)) +
    stat_summary(fun.y=mean, geom='line') +
    stat_summary(fun.data=mean_cl_boot, geom='errorbar') +
    geom_smooth() + theme_bw()
pdf('figs/hw_year_allWords_sg_wikifull.pdf', 10, 4)
plot(p)
dev.off()

# All years & multi-character words
p = ggplot(d[charNum >= 2], aes(x=year, y=hw)) +
    stat_summary(fun.y=mean, geom='line') +
    stat_summary(fun.data=mean_cl_boot, geom='errorbar') +
    geom_smooth() + theme_bw()
pdf('figs/hw_year_multiCharWords_sg_wikifull.pdf', 10, 4)
plot(p)
dev.off()

# All years & charNum = 2,3,4
p = ggplot(d[charNum >= 2 & charNum <= 4], aes(x=year, y=hw, color=wordLen, fill=wordLen)) +
    geom_smooth() + theme_bw()
pdf('figs/hw_year_charNum234_sg_wikifull.pdf', 5, 5)
plot(p)
dev.off()


## Combine CBOW and Skipgram
d1 = fread('results/wikifullchn_CWE_cbow_k_wll7_mc5_iter5_p4_l20_t1_k.txt', skip=2)
setnames(d1, c('word', 'hw', 'hc'))
setkey(d1, word)
d1$model = 'DSE(CBOW)'

d2 = fread('results/wikifullchn_CWE_sg_k_wll7_mc5_iter5_p4_l20_t1_k.txt', skip=2)
setnames(d2, c('word', 'hw', 'hc'))
setkey(d2, word)
d2$model = 'DSE(Skipgram)'

d_year = fread('results/1gram_earliest_year.txt')
setnames(d_year, c('word', 'year'))
setkey(d_year, word)
dc = rbindlist(list(d1, d2))
dc = d_year[dc, nomatch=0]
dc$charNum = str_length(dc$word)
dc$wordLen = as.character(dc$charNum)

# All years & multi-character words
p = ggplot(dc[charNum >= 2], aes(x=year, y=hw)) +
    stat_summary(fun.data = mean_cl_boot, geom='errorbar', aes(linetype=model), alpha=.7, width=5) + 
    geom_smooth(aes(fill=model, color=model)) + theme_bw() + 
    labs(x='First-appearance-year of the word', y=expression(h^w)) + ylim(c(0.35,1)) + theme(legend.position=c(0.2,0.9)) + 
    guides(linetype=guide_legend(title='Model'), 
            fill=guide_legend(title='Model'),
            color=guide_legend(title='Model'))
pdf('figs/hw_year_allWords_cbow&sg_wikifull.pdf', 5, 5)
plot(p)
dev.off()

# All years & charNum = 2,3,4
p = ggplot(dc[charNum >= 2 & charNum <= 4], aes(x=year, y=hw, color=wordLen, fill=wordLen, linetype=model)) +
    geom_smooth() + theme_bw() + 
    stat_summary(fun.y = mean, geom='point') + 
    labs(x='First-appearance-year of the word', y=expression(h^w)) + 
    theme(legend.position='bottom', legend.box='vertical') + 
    guides(fill=guide_legend(title='Number of characters'), 
           color=guide_legend(title='Number of characters'), 
           linetype=guide_legend(title='Model'))
pdf('figs/hw_year_charNum234_cbow&sg_wikifull.pdf', 5, 6)
plot(p)
dev.off()

library(mgcv)
new_year = data.table(year=seq(1550, 2000, 50))
b0_2 = gam(hw ~ s(year, bs='cs'), data=dc[model=='DSE(Skipgram)' & charNum==2])
hw_b0_2 = predict.gam(b0_2, new_year)

data_pred = copy(new_year)
data_pred$hw = hw_b0_2
