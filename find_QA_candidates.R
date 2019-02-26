library(data.table)
library(stringr)

d_year = fread('results/1gram_earliest_year.txt')
setnames(d_year, c('word', 'year'))
setkey(d_year, word)

dh_cbow = fread('results/wikifullchn_CWE_cbow_k_wll7_mc5_iter5_p4_l20_t1_k.txt', skip=1)
setnames(dh_cbow, c('word', 'hw', 'hc'))
setkey(dh_cbow, word)

dh_sg = fread('results/wikifullchn_CWE_sg_k_wll7_mc5_iter5_p4_l20_t1_k.txt', skip=1)
setnames(dh_sg, c('word', 'hw', 'hc'))
setkey(dh_sg, word)

# 示 
d_year[grepl('示', word)][dh_cbow, nomatch=0][order(year)]
d_year[grepl('示', word)][dh_sg, nomatch=0][order(year)]
#  5:   表示 1648 0.685150 0.314850
#  2:   指示 1574 0.680273 0.319727
# 13: 示意图 1898 0.854989 0.145011
# 19: 示波器 1900 0.884740 0.115260
# 30:   示例 1955 0.792946 0.207054

# 安
d_year[grepl('安', word)][dh_cbow, nomatch=0][order(year)]
d_year[grepl('安', word)][dh_sg, nomatch=0][order(year)]
#  2:     安全 1581 0.752314 0.247686
#  3:     安定 1632 0.719027 0.280973
# 45:   安眠药 1945 0.849715 0.150285
# 80:     安打 1959 0.850070 0.149930
#104:     安检 1987 0.872362 0.127638
#106:   安捷伦 1995 0.887502 0.112498

# Other examples
# 把握: 0.685299, 0.314701; 1591
# 投机倒把: 0.875172, 0.124828; 1949
# 把头: 0.849862, 0.150138; 1961
# 拖把: 0.861653, 0.138347; 1985

# 组成: 0.669771, 0.330229; 1568
# 组阁: 0.793639, 0.206361; 1900
# 机组: 0.760919, 0.239081; 1900
# 剧组: 0.761169, 0.238831; 1955
# 课题组: 0.918295, 0.081705; 1988

# 覆盖: 0.688801, 0.311199; 1747
# 盖帽: 0.906955, 0.093045; 1972