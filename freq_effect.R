#!/usr/bin/env Rscript
# Analyze if the weight on words, i.e., h^w, is still affected by year, after 
# frequency effect is taken into account.
# Example: $> Rscript --vanilla freq_effect.R *_k.txt

library(data.table)
library(stringr)
# library(reticulate)
# use_python('/usr/local/Cellar/python/3.7.0/Frameworks/Python.framework/Versions/3.7/bin/python3')

node_name = Sys.info()[4]
if (node_name == 'COS-CS-E050971') {
    # setwd('/Users/xy/academic_projects/zhwiki')
    setwd('/Users/xy/academic_projects/cogsci2019repo') # laptop
} else if (node_name == 'dreitter-cent.ist.local') {
    setwd('/home/yangxu/projects/zhwiki')
} else if (node_name == 'COS-CS-E051358') {
    setwd('/home/yxu-admin/projects/cogsci2019repo') # power
} else if(node_name == 'COS-CS-E050970') {
    setwd('/Users/xy/academic_projects/cogsci2019repo') # dream
}


# Take one argument, i.e., the _k file as input
args = commandArgs(trailingOnly=TRUE)
if (length(args)==0) {
    stop("At least one argument must be supplied (input file)", call.=FALSE)
}
hw_file = args[1]


# Read data from a *_k.txt file (2 columns)
d1 = fread(hw_file, skip=2)
setnames(d1, c('word', 'hw', 'hc'))
setkey(d1, word)

# Read word frequency data 
d_freq = fread('results/wikifullchn_wll7_mc5_vocab.csv')
setnames(d_freq, c('word', 'freq'))
setkey(d_freq, word)

# Read year data
d_year = fread('results/1gram_earliest_year.txt')
setnames(d_year, c('word', 'year'))
setkey(d_year, word)

# Join 
d3 = d1[d_freq, nomatch=0]
d3 = d3[d_year, nomatch=0]

# Count number of characters in words
d3$charNum = str_length(d3$word)


##
# Models

# all words
# m1 = lm(hw ~ freq + year, d3)
# summary(m1)

# m2 = lm(hw ~ freq, d3)
# m2_res = lm(residuals(m2) ~ year, d3)
# summary(m2_res)

# multi-character words
m3 = lm(hw ~ freq + year, d3[charNum>1])
summary(m3)

m4 = lm(hw ~ freq, d3[charNum>1])
m4_res = lm(residuals(m4) ~ year, d3[charNum>1])
summary(m4_res)