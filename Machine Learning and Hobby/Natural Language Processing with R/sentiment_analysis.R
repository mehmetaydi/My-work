
library(dplyr)
library(stringr)
library(gutenbergr)

# Sentiment Analysis among given books below


# 14985 =Valere Aude: Dare to Be Healthy, Or, The Light of Physical Regeneration by Dechmann
# 17682 = The Healthy Life, Vol. V, Nos. 24-28 by Various
## 18487 = Food Remedies: Facts About Foods And Their Medicinal Uses by Florence Daniel

books <- gutenberg_download(c(17682, 18487,14985), meta_fields = "title")


tidy_books <- books %>%
  group_by(title) %>%
  mutate(gutenberg_id = row_number(),
         chapter = cumsum(str_detect(text, 
                                     regex("^chapter [\\divxlc]",
                                           ignore_case = TRUE)))) %>%
  ungroup() %>%
  unnest_tokens(word, text)

sentiments_check <- get_sentiments("nrc")

sentiments_check
nrc_joy <- get_sentiments("nrc") %>%
  filter(sentiment == "joy")

nrc_joy
tidy_books %>%
  #filter(title == "The Comedy of Errors") %>%
  inner_join(nrc_joy) %>%
  count(word, sort = TRUE)
library(tidyr)

# Subtracting the number of negative words from the Positive. Othello appears to have the most
# number of negative words.

plays_sentiment <- tidy_books %>%
  inner_join(get_sentiments("bing")) %>%
  count(title, index = gutenberg_id %% 80, sentiment) %>%
  spread(sentiment, n, fill = 0) %>%
  mutate(sentiment = positive - negative)


library(ggplot2)

ggplot(plays_sentiment, aes(index, sentiment, fill = title)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~title, ncol = 2, scales = "free_x")

### 2.3 Comparing the three sentiment dictionaries

pride_prejudice <- tidy_books %>% 
  filter(title == "Food Remedies: Facts About Foods And Their Medicinal Uses")

pride_prejudice

afinn <- pride_prejudice %>% 
  inner_join(get_sentiments("afinn")) %>% 
  group_by(index = gutenberg_id %/% 80) %>% 
  summarise(sentiment = sum(value)) %>% 
  mutate(method = "AFINN")

bing_and_nrc <- bind_rows(
  pride_prejudice %>% 
    inner_join(get_sentiments("bing")) %>%
    mutate(method = "Bing et al."),
  pride_prejudice %>% 
    inner_join(get_sentiments("nrc") %>% 
                 filter(sentiment %in% c("positive", 
                                         "negative"))
    ) %>%
    mutate(method = "NRC")) %>%
  count(method, index = gutenberg_id %/% 80, sentiment) %>%
  spread(sentiment, n, fill = 0) %>%
  mutate(sentiment = positive - negative)

bind_rows(afinn, 
          bing_and_nrc) %>%
  ggplot(aes(index, sentiment, fill = method)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~method, ncol = 1, scales = "free_y")

## 2.4 Most common positive and negative words

bing_word_counts <- tidy_books %>%
  inner_join(get_sentiments("bing")) %>%
  count(word, sentiment, sort = TRUE) %>%
  ungroup()

bing_word_counts

bing_word_counts %>%
  group_by(sentiment) %>%
  top_n(10) %>%
  ungroup() %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(word, n, fill = sentiment)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~sentiment, scales = "free_y") +
  labs(y = "Contribution to sentiment",
       x = NULL) +
  coord_flip()


## word cloud for positive and negative words
library(wordcloud)

tidy_books %>%
  anti_join(stop_words) %>%
  count(word) %>%
  with(wordcloud(word, n, max.words = 100))

library(reshape2)

tidy_books %>%
  inner_join(get_sentiments("bing")) %>%
  count(word, sentiment, sort = TRUE) %>%
  acast(word ~ sentiment, value.var = "n", fill = 0) %>%
  comparison.cloud(colors = c("gray20", "gray80"),
                   max.words = 100)










