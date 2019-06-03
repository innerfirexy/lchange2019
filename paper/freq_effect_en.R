#!/usr/bin/env Rscript
# Analyze if the weight on words, i.e., h^w, is still affected by year, after 
# frequency effect is taken into account.
# Example: $> Rscript --vanilla freq_effect.R *_k.txt

library(data.table)
library(stringr)
# library(reticulate)
# use_python('/usr/local/Cellar/python/3.7.0/Frameworks/Python.framework/Versions/3.7/bin/python3')
library(optparse)


node_name = Sys.info()[4]
if (node_name == 'COS-CS-E050971') {
    # setwd('/Users/xy/academic_projects/zhwiki')
    setwd('/Users/xy/academic_projects/acl2019repo') # laptop
} else if (node_name == 'dreitter-cent.ist.local') {
    setwd('/home/yangxu/projects/zhwiki')
} else if (node_name == 'COS-CS-E051358') {
    setwd('/home/yxu-admin/projects/acl2019repo')
} else if(node_name == 'COS-CS-E050970') {
    setwd('/Users/xy/academic_projects/acl2019repo') # dream
}

# Take one argument, i.e., the _h file as input
# args = commandArgs(trailingOnly=TRUE)
# if (length(args)==0) {
#     stop("At least one argument must be supplied (input file)", call.=FALSE)
# }
# hw_file = args[1]

# Setup arguments
option_list = list(
    make_option(c('-y', '--year'), type='character'),
    make_option(c("-v", "--vocab"), type="character"),
    make_option(c("-i", "--input"), type="character"))

opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)


# Read data from a *_h.txt file (2 columns)
d1 = fread(opt$input, skip=2)
setnames(d1, c('word', 'hw', 'hc'))
setkey(d1, word)

# Read word frequency and ngramNum data 
d_freq = fread(opt$vocab)
setnames(d_freq, c('word', 'freq', 'ngramNum'))
setkey(d_freq, word)

# Read year data
d_year = fread(opt$year)
setnames(d_year, c('word', 'year'))
setkey(d_year, word)

# Join 
d3 = d1[d_freq, nomatch=0]
d3 = d3[d_year, nomatch=0]


#########
# Models

# Direct model
m1 = lm(hw ~ freq + year, d3[ngramNum>1])
summary(m1)

# Auxiliary model
m2 = lm(hw ~ freq, d3[ngramNum>1])
m2_res = lm(residuals(m2) ~ year, d3[ngramNum>1])
summary(m2_res)
