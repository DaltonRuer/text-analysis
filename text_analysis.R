
##---------------------------pathway 1-----------------------------##
##  corpus --> dtm --> summary text --> visualization (wordcloud)  ##
##-----------------------------------------------------------------##

##---------------------------------
## CORPUS GENERATION: tm package
##---------------------------------

# load libraries
library(tm)

# create a corpus
docs <- VCorpus(DirSource("~/TA-data-science-club/docs"))

# look at ingest
summary(docs)

# inspect docs
inspect(docs[2])

# demonstrate lines in each document
writeLines(as.character(docs[2]))

##-------------------------------
## NORMALIZE TEXT: tm package
##-------------------------------

# remove numbers
docs <- tm_map(docs, removeNumbers)

# convert to lower case
docs <- tm_map(docs, content_transformer(tolower))

# create toSpace function
toSpace <- content_transformer(function(x, pattern) {return (gsub(pattern, " ", x))})

# add space around dashes and colons
docs <- tm_map(docs, toSpace, "-")
docs <- tm_map(docs, toSpace, ":")

# remove punctuation & special character
docs <- tm_map(docs, removePunctuation)

# optional vector to remove additional words
rm_add_words <- c("albus", "dumbledore", "harry", "potter", "hermione", "granger", "fenrir", "greyback", "dedalus", "diggle", "luna", "lovegood", 
                  "ron", "weasley", "delacour", "fleur", "macgonagall", "mcgonagall", "arthur", "laurel", "clearwater", "yaxley", "andromeda", 
                  "cedric", "penelope", "tonks", "marvolo", "corban", "mcgonagall", "minerva")

# remove stopwords & additional words
docs <- tm_map(docs, removeWords, stopwords("english"))
docs <- tm_map(docs, removeWords, rm_add_words)

# manually stem 'thanks' to thank for more accurate counts
stemThanks <- content_transformer(function(x, pattern) {return (gsub(pattern, "thank", x))})
docs <- tm_map(docs, stemThanks, "thanks")

# manually stem 'patients' to thank for more accurate counts
stemPatients <- content_transformer(function(x, pattern) {return (gsub(pattern, "patient", x))})
docs <- tm_map(docs, stemPatients, "patients")

##------------------------------------------
## GENERATE DTM: tm package 
##------------------------------------------

# create document term matrix and term document matrix
dtm <- DocumentTermMatrix(docs)
tdm <- TermDocumentMatrix(docs)

# create document term matrix for each document
dtm_PR <- DocumentTermMatrix(docs[1])
dtm_UR <- DocumentTermMatrix(docs[2])
dtm_TT <- DocumentTermMatrix(docs[3])

# convert to sparse matrix
dtms <- removeSparseTerms(dtm, 0.2)

##------------------------------------------
## GENERATE SUMMARY TEXT: tm package 
##------------------------------------------

# determine frequency of words for all documents
freq_ALL <- sort(colSums(as.matrix(dtm)), decreasing=TRUE)
head(freq_ALL, 20)

# determine frequency of presenter words
freq_PR <- sort(colSums(as.matrix(dtm_PR)), decreasing=TRUE)
head(freq_PR, 20)

# determine frequency of user words
freq_UR <- sort(colSums(as.matrix(dtm_UR)), decreasing=TRUE)
head(freq_UR, 20)

# determine frequency of trouble ticket words
freq_TT <- sort(colSums(as.matrix(dtm_TT)), decreasing=TRUE)
head(freq_TT, 20)

## find frequent terms for the complete dtm
findMostFreqTerms(dtm, lowfreq = 20)

##------------------------------------------
## VISUALIZATION: tm package 
##------------------------------------------

library(RColorBrewer)
library(wordcloud)

# set grid to display wordclouds as 1 row and 2 columns
par(mfrow=c(1,2))

# get word cloud of trouble tickets comments
set.seed(1234)
wordcloud(names(freq_TT), freq_TT, 
          scale = c(4, 0.5),
          min.freq = 5,
          colors = brewer.pal(6, "Dark2"))

# get word cloud of user comments
set.seed(1235)
wordcloud(names(freq_UR), freq_UR, 
          scale = c(4, 0.5),
          min.freq = 5,
          colors = brewer.pal(6, "Dark2"))


##---------------------------pathway 2-----------------------------##
##  corpus --> dtm --> tidytext --> summarytext --> visualization  ##
##-----------------------------------------------------------------##

##------------------------------------------------
## CONVERTING CORPUS TO TIDYTEXT: tidytext package
##------------------------------------------------

library(tidytext)
library(dplyr)
library(ggplot2)

# convert sparse document term matrix into tidy text
docs_td <- tidy(dtms)
docs_td

# convert document dtms into tidy text
PR_tidy <- tidy(dtm_PR)
UR_tidy <- tidy(dtm_UR)
TT_tidy <- tidy(dtm_TT)

##------------------------------------------------
## BEGINNING TO END: tidytext package
##------------------------------------------------

library(readtext)
library(tidyr)
data("stop_words")

# load data fresh data
new_docs <- as.data.frame(readtext("~/TA-data-science-club/docs"))
View(new_docs)

# get special word list ready for anti-join
rm_add_words <- as.data.frame(rm_add_words, stringsAsFactors = FALSE)
names(rm_add_words) <- "word"

# create tidytext of users data
tidy_docs <- new_docs %>%
  unnest_tokens(word, text) %>%
  anti_join(stop_words, by = "word") %>%
  anti_join(rm_add_words, by = "word")

# determine most common words in data
tidy_docs %>%
  count(word, sort = TRUE)

# explore sentiments
afinn <- get_sentiments(lexicon = "afinn")
bing <- get_sentiments(lexicon = "bing")
ncr <- get_sentiments(lexicon = "nrc")
loughran <- get_sentiments(lexicon = "loughran")

# explore joy sentiment for user comments
nrcjoy <- get_sentiments("nrc") %>% 
  filter(sentiment == "joy")

# filter docs for MHSPHP user comments & look for joy words
tidy_docs %>%
  filter(doc_id == "TA_MHSPHP_USERS.txt") %>%
  inner_join(nrcjoy, by = "word") %>%
  count(word, sort = TRUE)

# prep text for graph
head(tidy_docs)

# data wrangling for scatterplot
comment_docs <- tidy_docs %>%
  inner_join(get_sentiments("nrc"), by = "word") %>%
  count(word, index = doc_id, sentiment) %>%
  spread(sentiment, n, fill = 0) %>%
  mutate(sentiment = positive - negative)

# graph sentiments of comments
library(ggplot2)
ggplot(comment_docs, aes(index, sentiment, fill = index)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~index, ncol = 3, scales = "free_x")
