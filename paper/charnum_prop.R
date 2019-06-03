# Examine the proportions of 2-, 3-, and 4-character words in training data and GBN data

library(data.table)
library(stringr)


# Among training data
d_freq = fread('results/wikifullchn_wll7_mc5_vocab.csv')
setnames(d_freq, c('word', 'freq'))
setkey(d_freq, word)

d_freq$charNum = str_length(d_freq$word)
t1 = table(d_freq$charNum)
#      1      2      3      4
#   8728 258632 139851  38623
t1[2] / nrow(d_freq) # 57.5%
t1[3] / nrow(d_freq) # 31.0%
t1[4] / nrow(d_freq) # 8.6%

# Proportions computed using word frequencies
sum(d_freq[charNum==2, freq]) / sum(d_freq[charNum>=2, freq]) # 82.9%
sum(d_freq[charNum==3, freq]) / sum(d_freq[charNum>=2, freq]) # 11.8%
sum(d_freq[charNum==4, freq]) / sum(d_freq[charNum>=2, freq]) # 4.6%


# Among GBN data
d_year = fread('results/1gram_earliest_year.txt')
setnames(d_year, c('word', 'year'))

d_year$charNum = str_length(d_year$word)
t2 = table(d_year$charNum)
#    1     2     3     4     5     6     7 
# 6334 39333  8792  2881    40     1     1 
t2[2] / nrow(d_year) # 68.5%
t2[3] / nrow(d_year) # 15.3%
t2[4] / nrow(d_year) # 5.0%