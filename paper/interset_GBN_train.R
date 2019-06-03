# Find the intersection of vocabulary between GBN and training data.
library(data.table)

lang_short = c("en", "de", "fr", "it", "es", "zh25p")
lang_long = c("english", "german", "french", "italian", "spanish", "chinese")
model_str = "DSE_sg"

for (i in 6:6) {
    lang1 = lang_short[i]
    lang2 = lang_long[i]
    year_file = paste0("results/1gram_first_year_", lang2, ".txt")
    vocab_file = paste0("results/wiki", lang1, "_wll7_mc5_vocab.csv")

    d_year = fread(year_file)
    setnames(d_year, c('word', 'year'))
    setkey(d_year, word)

    d_vocab = fread(vocab_file)
    if (ncol(d_vocab) == 3) {
        setnames(d_vocab, c("word", "freq", "ngramCount"))
    } else if (ncol(d_vocab) == 2) {
        setnames(d_vocab, c("word", "freq"))
    }
    
    setkey(d_vocab, word)

    dj = d_year[d_vocab, nomatch=0]
    r1 = nrow(dj) / nrow(d_vocab)
    r2 = sum(dj$freq) / sum(d_vocab$freq)

    print(paste(lang2, r1, r2))
}