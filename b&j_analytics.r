####
# Libraries ----
####
library(plyr)
library(dplyr)
library(quanteda)
library(quanteda.textplots)
library(quanteda.textstats)
library(data.table)
library(tidyr)
library(ggplot2)
library(tidytext)
library(textdata)
library("sentimentr")


library("syuzhet")
library("RColorBrewer")
library("tm")
library("topicmodels")
library("dendextend")
library("vosonSML")
library("urltools")
library("formattable")
library("stringr")
library(fmsb)
library(stringi)

####
# Datasets ----
####
product <- read.csv("products.csv", fileEncoding = "UTF-8")

#product$name <- enc2utf8(product$name)
#Encoding(product$name)

reviews <- read.csv("reviews.csv")
reviews <- reviews %>%
  mutate(title = str_replace(title, "Â", "")) %>% 
  mutate(title = str_replace(title, "â€™", "'")) %>%
  mutate(title = str_replace(title, "â€œ", '"')) %>%
  mutate(title = str_replace(title, 'â€“', '-')) %>%
  mutate(title = str_replace(title, 'â€', '"'))  %>% 
  mutate(text = str_replace(text, "Â", "")) %>% 
  mutate(text = str_replace(text, "â€™", "'")) %>%
  mutate(text = str_replace(text, "â€œ", '"')) %>%
  mutate(text = str_replace(text, 'â€“', '-')) %>%
  mutate(text = str_replace(text, 'â€', '"'))



dfm <- (merge(product, reviews, by="key")) #Merge datasets



####
# Word Frequencies & Wordcloud ----
####
  
  dfmat_corp_twitter <- dfm$text %>% corpus() %>% 
  dfm(remove = c(stopwords("english")),
      what = "word", # dans le what on mlet les words for tokenizing
      stem = TRUE, 
      remove_punct = TRUE,
      remove_numbers=TRUE,
      remove_symbols=TRUE,
      remove_separators=TRUE,
      remove_url=TRUE)
  
textplot_wordcloud(dfmat_corp_twitter, min_count = 6, random_order = FALSE,
                     rotation = .25,
                     color = RColorBrewer::brewer.pal(8, "Dark2"))

#Top 25 tokens 
dfFreq <- textstat_frequency(dfmat_corp_twitter) %>% as.data.table
ggplot(dfFreq[1:25,], aes(x=reorder(feature, -rank), y=frequency)) + 
  geom_col() +
  coord_flip() +
  labs(x = "Stemmed word", y = "Count")

####
# Tokens ----
####
  
tok <- dfm$text %>% 
  gsub("#","", . ) %>% 
  corpus %>% 
  tokens(what="word", 
         remove_numbers=TRUE,
         remove_punct=TRUE,
         remove_symbols=TRUE,
         remove_separators=TRUE,
         remove_url=TRUE)
tok <- tokens_remove(tok,stopwords("english"))

####-
# Tokens 2-grams, 3grams ----
####-
  
#2-grams
TokensStemmed <- tokens_remove(tok,c(stopwords("English"),"it�", "i�", "ben", 
                                     "didn�", "core", "ice", "jerry�", "b" ))

dfm2 <- dfm(tokens_ngrams(TokensStemmed,n=2))

dfFreq2 <- textstat_frequency(dfm2)

#Top 15 2-grams tokens
ggplot(dfFreq2[1:15,], aes(x=reorder(feature, frequency), y=frequency)) + 
  geom_col() +
  coord_flip() +
  scale_x_discrete(name = "2 gram") +
  theme(text=element_text(size=12))

#3-grams
dfm3 <- dfm(tokens_ngrams(TokensStemmed,n=3))

dfFreq3 <- textstat_frequency(dfm3)

#Top 15 3-grams token
ggplot(dfFreq3[1:15,], aes(x=reorder(feature, frequency), y=frequency)) + 
  geom_col() +
  coord_flip() +
  scale_x_discrete(name = "3 gram") +
  theme(text=element_text(size=12))
####
# Sentiment Analysis----
####
  
TextDoc <- Corpus(VectorSource(dfm$text))

#Replacing "/", "@" and "|" with space
toSpace <- content_transformer(function (x , pattern ) gsub(pattern, " ", x))
TextDoc <- tm_map(TextDoc, toSpace, "/")
TextDoc <- tm_map(TextDoc, toSpace, "@")
TextDoc <- tm_map(TextDoc, toSpace, "\\|")
# Convert the text to lower case
TextDoc <- tm_map(TextDoc, content_transformer(tolower))
# Remove numbers
TextDoc <- tm_map(TextDoc, removeNumbers)
# Remove english common stopwords
TextDoc <- tm_map(TextDoc, removeWords, stopwords("english"))
# custom stopwords as a character vector
TextDoc <- tm_map(TextDoc, removeWords, c("https","cho","tco","maga","just")) 
# Remove punctuations
TextDoc <- tm_map(TextDoc, removePunctuation)
# Eliminate extra white spaces
TextDoc <- tm_map(TextDoc, stripWhitespace)
# Text stemming - which reduces words to their root form
TextDoc <- tm_map(TextDoc, stemDocument)


# Build a term-document matrix
TextDoc_dtm <- TermDocumentMatrix(TextDoc)
dtm_m <- as.matrix(TextDoc_dtm)
# Sort by descearing value of frequency
dtm_v <- sort(rowSums(dtm_m),decreasing=TRUE)
dtm_d <- data.frame(word = names(dtm_v),freq=dtm_v)
# Display the top 5 most frequent words
head(dtm_d, 5)

# Plot the most frequent words
barplot(dtm_d[1:5,]$freq, las = 2, names.arg = dtm_d[1:5,]$word,
        col ="lightgreen", main ="Top 5 most frequent words",
        ylab = "Word frequencies")

syuzhet_vector <- get_sentiment(dfm$text, method="syuzhet")
# see the first row of the vector
head(syuzhet_vector)
# see summary statistics of the vector
summary(syuzhet_vector)


# Bing lexicon :  It categorizes words in a binary fashion into positive and negative categories
bing_vector <- get_sentiment(dfm$text, method="bing")
head(bing_vector)
summary(bing_vector)

# Affin lexicon : The AFINN lexicon assigns words with a score that runs between -5 and 5
afinn_vector <- get_sentiment(dfm$text, method="afinn")
head(afinn_vector)
summary(afinn_vector)

rbind(
  sign(head(syuzhet_vector)),
  sign(head(bing_vector)),
  sign(head(afinn_vector))
)

d<-get_nrc_sentiment(dfm$text)  # The nrc lexicon categorizes words in a binary fashion (“yes”/“no”) into categories of positive, negative, anger, anticipation, disgust, fear, joy, sadness, surprise, and trust
# head(d,10) - to see top 10 lines of the get_nrc_sentiment dataframe
head (d,10)

#transpose
td<-data.frame(t(d))
#The function rowSums computes column sums across rows for each level of a grouping variable.
td_new <- data.frame(rowSums(td))
#Transformation and cleaning
names(td_new)[1] <- "count"
td_new <- cbind("sentiment" = rownames(td_new), td_new)
rownames(td_new) <- NULL
td_new2<-td_new[1:8,]


# Barplot of the Sentiment Analysis using the NRC lexicon
barplot(
  sort(colSums(prop.table(d[, 1:8]))), 
  horiz = TRUE, 
  cex.names = 0.7, 
  col = brewer.pal(8, "Spectral"),
  las = 1, 
  main = "Emotions in Ice Cream reviews", xlab="Percentage"
)

# Top 10 negative & positive words

library(janeaustenr)
tidy_books2 <- dfm %>%
  group_by(text) %>%
  mutate(
    linenumber = row_number(),
    chapter = cumsum(str_detect(text, 
                                regex("^chapter [\\divxlc]", 
                                      ignore_case = TRUE)))) %>%
  ungroup() %>%
  unnest_tokens(word, text)

bing_word_counts2 <- tidy_books2 %>%
  inner_join(get_sentiments("bing")) %>%
  count(word, sentiment, sort = TRUE) %>%
  ungroup()

bing_word_counts2 %>%
  group_by(sentiment) %>%
  slice_max(n, n = 20) %>% 
  ungroup() %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(n, word, fill = sentiment)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~sentiment, scales = "free_y") +
  labs(x = "Contribution to sentiment",
       y = NULL)

# Sentiment Barplots of Ice Cream Flavors ----
# Function to calculate sentiment by row
batch_sentiment_by <- function(reviews, batch_size = 200, ...) {
  review_batches <- split(reviews, ceiling(seq_along(reviews)/batch_size))
  x <- rbindlist(lapply(review_batches, sentiment_by, ...))
  x[, element_id := .I]
  x[]
}

sentiment_df <- batch_sentiment_by(dfm$text)
setDF(sentiment_df)

ave_sent <- c(sentiment_df$ave_sentiment)

dfm$ave_sentiment <- ave_sent

grouped_by_sent <- setNames(aggregate(dfm$ave_sentiment, list(dfm$name), mean),c("name","average_sentiment"))

# Ice Cream BarPlot by sentiment score 
ggplot(grouped_by_sent[head(order(grouped_by_sent$average_sentiment),60),], aes(x= reorder(name, -average_sentiment), average_sentiment, fill = average_sentiment)) + 
  geom_bar(stat="identity") + 
  labs(y="Sentiment Score", x="Ice Cream Flavors") +
  theme(axis.text.x = element_text(angle=90)) + 
  ggtitle("Ice Cream barplot by Sentiment Score")

# Ice Cream BarPlot by Star Rating
grouped_by_rank <- setNames(aggregate(dfm$rating, list(dfm$name), mean),c("name","ranking"))

ggplot(grouped_by_rank[head(order(grouped_by_rank$ranking),60),], aes(x= reorder(name, -ranking), ranking, fill= ranking)) + 
  geom_bar(stat="identity") + 
  labs(y="Rating Score", x="Ice Cream Flavors") + 
  theme_minimal() +
  theme(axis.text.x = element_text(angle=90)) + 
  ggtitle("Ice Cream Barplot by Score rating")


####
# Worst IceCream flavor----
####
worst_flavor_dfm <- filter(dfm, key == "50_bj")
worst_flavor_dfm$text

worst_flavor_corp <- worst_flavor_dfm$text %>% corpus() %>% 
  dfm(remove = c(stopwords("english")),
                   what = "word",
                   stem = TRUE, remove_punct = TRUE,
                   remove_numbers=TRUE,
                   remove_symbols=TRUE,
                   remove_separators=TRUE,
                   remove_url=TRUE)

dfFreq <- textstat_frequency(worst_flavor_corp) %>% as.data.table
ggplot(dfFreq[1:25,], aes(x=reorder(feature, -rank), y=frequency)) + 
  geom_col() +
  coord_flip() +
  labs(x = "Stemmed word", y = "Count")

# Tokens

tok <- worst_flavor_dfm$text %>% 
  gsub("#","", . ) %>% 
  corpus %>% 
  tokens(what="word",
         remove_numbers=TRUE,
         remove_punct=TRUE,
         remove_symbols=TRUE,
         remove_separators=TRUE,
         remove_url=TRUE)
tok <- tokens_remove(tok,stopwords("english"))


####-
# Worst Flavor : Tokens 2-grams, 3grams ----
####-

#2-grams
TokensStemmed <- tokens_remove(tok, stopwords("english"))

dfm2 <- dfm(tokens_ngrams(TokensStemmed,n=2))

dfFreq2 <- textstat_frequency(dfm2)

#Top 15 2-grams tokens
ggplot(dfFreq2[1:15,], aes(x=reorder(feature, frequency), y=frequency)) + 
  geom_col() +
  coord_flip() +
  scale_x_discrete(name = "2 gram") +
  theme(text=element_text(size=12))

#3-grams
dfm3 <- dfm(tokens_ngrams(TokensStemmed,n=3))

dfFreq3 <- textstat_frequency(dfm3)

#Top 15 3-grams token
ggplot(dfFreq3[1:15,], aes(x=reorder(feature, frequency), y=frequency)) + 
  geom_col() +
  coord_flip() +
  scale_x_discrete(name = "3 gram") +
  theme(text=element_text(size=12))

####-
# Worst Flavor Sentiment Analysis ----
####-

d<-get_nrc_sentiment(worst_flavor_dfm$text)  # The nrc lexicon categorizes words in a binary fashion (“yes”/“no”) into categories of positive, negative, anger, anticipation, disgust, fear, joy, sadness, surprise, and trust
# head(d,10) - to see top 10 lines of the get_nrc_sentiment dataframe
head (d,10)

#transpose
td<-data.frame(t(d))
#The function rowSums computes column sums across rows for each level of a grouping variable.
td_new <- data.frame(rowSums(td))
#Transformation and cleaning
names(td_new)[1] <- "count"
td_new <- cbind("sentiment" = rownames(td_new), td_new)
rownames(td_new) <- NULL
td_new2<-td_new[1:8,]


# Barplot of the Sentiment Analysis using the NRC lexicon
barplot(
  sort(colSums(prop.table(d[, 1:8]))), 
  horiz = TRUE, 
  cex.names = 0.7, 
  col = brewer.pal(8, "Spectral"),
  las = 1, 
  main = "Emotions in Ice Cream reviews", xlab="Percentage"
)
barplot(
  sort(colSums(prop.table(d[, 9:10]))), 
  horiz = TRUE, 
  cex.names = 0.7, 
  col = c("firebrick2","springgreen2"),
  las = 1, 
  main = "Proportion of Positive and Negative Sentiments on the \n Chocolate Cookie Dough Core Flavor", xlab="Percentage"
)


# Top 10 negative & positive words

tidy_books3 <- worst_flavor_dfm %>%
  group_by(text) %>%
  mutate(
    linenumber = row_number(),
    chapter = cumsum(str_detect(text, 
                                regex("^chapter [\\divxlc]", 
                                      ignore_case = TRUE)))) %>%
  ungroup() %>%
  unnest_tokens(word, text)

bing_word_counts3 <- tidy_books3 %>%
  inner_join(get_sentiments("bing")) %>%
  count(word, sentiment, sort = TRUE) %>%
  ungroup()

bing_word_counts3 %>%
  group_by(sentiment) %>%
  slice_max(n, n = 20) %>% 
  ungroup() %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(n, word, fill = sentiment)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~sentiment, scales = "free_y") +
  labs(x = "Contribution to sentiment",
       y = NULL)
# Words organized by Sentiment :
bing_word_counts4 <- tidy_books3 %>%
  inner_join(get_sentiments("nrc")) %>%
  count(word, sentiment, sort = TRUE) %>%
  ungroup()

bing_word_counts4 %>%
  group_by(sentiment) %>%
  slice_max(n, n = 20) %>% 
  ungroup() %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(n, word, fill = sentiment)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~sentiment, scales = "free_y") +
  labs(x = "Contribution to sentiment",
       y = NULL)
