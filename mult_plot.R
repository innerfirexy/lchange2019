require("ggplot2")
require("data.table")
require("stringr")


join_all_langs = function(model="DSE_sg") {
    lang_short = c("en", "de", "fr", "it", "es")
    lang_long = c("english", "german", "french", "italian", "spanish")
    model_str = model

    data = list()
    for (i in 1:5) {
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

# Data for 6 Indo-Euro languages
d_6lang = join_all_langs(model="DSE_sg")


# For Chinese data
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
d_zh = join_h_coefs('results/wikifullchn_CWE_cbow_k_wll7_mc5_iter5_p4_l20_t1_k.txt')
d_zh$charNum = str_length(d_zh$word)
d_zh$wordLen = as.character(d_zh$charNum)


# Get the plots for three languages: EN, DE, ES
p1 = ggplot(d_6lang[Language == "English" & ngramCount >= 2 & ngramCount <= 7], aes(x=year, y=hw, color=wordLen, fill=wordLen)) +
    geom_smooth() + theme_bw() + 
    scale_fill_brewer(palette="Blues", direction=-1) + scale_color_brewer(palette="Blues", direction=-1) + 
    # guides(fill=guide_legend(title="Ngram count"), color=guide_legend(title="Ngram count")) + 
    theme(legend.position="none", plot.title = element_text(hjust=0.5, face="bold")) + 
    labs(x='First-appearance-year of the word', y=expression(h^w)) + 
    ggtitle("English")

p2 = ggplot(d_6lang[Language == "German" & ngramCount >= 2 & ngramCount <= 7], aes(x=year, y=hw, color=wordLen, fill=wordLen)) +
    geom_smooth() + theme_bw() + 
    scale_fill_brewer(palette="Blues", direction=-1) + scale_color_brewer(palette="Blues", direction=-1) + 
    guides(fill=guide_legend(title="Ngram\ncount"), color=guide_legend(title="Ngram\ncount")) + 
    theme(legend.position=c(0.2, 0.3), plot.title = element_text(hjust=0.5, face="bold")) + 
    labs(x='First-appearance-year of the word', y=expression(h^w)) + 
    ggtitle("German")

p3 = ggplot(d_6lang[Language == "Spanish" & ngramCount >= 2 & ngramCount <= 7], aes(x=year, y=hw, color=wordLen, fill=wordLen)) +
    geom_smooth() + theme_bw() + 
    scale_fill_brewer(palette="Blues", direction=-1) + scale_color_brewer(palette="Blues", direction=-1) + 
    # guides(fill=guide_legend(title="Ngram\ncount"), color=guide_legend(title="Ngram\ncount")) + 
    theme(legend.position="none", plot.title = element_text(hjust=0.5, face="bold")) + 
    labs(x='First-appearance-year of the word', y=expression(h^w)) + 
    ggtitle("Spanish")

p4 = ggplot(d_zh[charNum >= 2 & charNum <= 4], aes(x=year, y=hw, color=wordLen, fill=wordLen)) +
    geom_smooth() + theme_bw() + 
    scale_fill_brewer(palette="Oranges", direction=-1) + scale_color_brewer(palette="Oranges", direction=-1) + 
    guides(fill=guide_legend(title="Character\ncount"), color=guide_legend(title="Character\ncount")) + 
    theme(legend.position=c(0.55, 0.2), plot.title = element_text(hjust=0.5, face="bold")) + 
    labs(x='First-appearance-year of the word', y=expression(h^w)) + 
    ggtitle("Chinese")


pdf("figs/hw_year_4langs_contrast.pdf", 8.5, 8.5)
multiplot(p1, p2, p3, p4, cols=2)
dev.off()




# Multiple plot function
#
# ggplot objects can be passed in ..., or to plotlist (as a list of ggplot objects)
# - cols:   Number of columns in layout
# - layout: A matrix specifying the layout. If present, 'cols' is ignored.
#
# If the layout is something like matrix(c(1,2,3,3), nrow=2, byrow=TRUE),
# then plot 1 will go in the upper left, 2 will go in the upper right, and
# 3 will go all the way across the bottom.
#
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)

  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)

  numPlots = length(plots)

  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                    ncol = cols, nrow = ceiling(numPlots/cols))
  }

 if (numPlots==1) {
    print(plots[[1]])

  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))

    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))

      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}