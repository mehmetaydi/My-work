
library(dplyr)
library(stringr)
library(gutenbergr)
library(tidytext)
library(janeaustenr)
library(tidyverse )

# Statistical experiments in NLP such as Zipf's law, term distribution, tf-idf

# 14985 =Valere Aude: Dare to Be Healthy, Or, The Light of Physical Regeneration by Dechmann
# 17682 = The Healthy Life, Vol. V, Nos. 24-28 by Various
## 18487 = Food Remedies: Facts About Foods And Their Medicinal Uses by Florence Daniel

books <- gutenberg_download(c(18487), meta_fields = "title")


tidy_books <- books %>%
  group_by(title) %>%
  mutate(gutenberg_id = row_number(),
         chapter = cumsum(str_detect(text, 
                                     regex("^chapter [\\divxlc]",
                                           ignore_case = TRUE)))) %>%
  ungroup() %>%
  unnest_tokens(word, text)


total_words <- tidy_books %>% 
  group_by(title) %>% 
  summarize(total = sum(n))


tidy_books <- left_join(total_words, total_words)

book_words
library(ggplot2)

ggplot(book_words, aes(n/total, fill = title)) +
  geom_histogram(show.legend = FALSE) +
  xlim(NA, 0.0009) +
  facet_wrap(~title, ncol = 2, scales = "free_y")

freq_by_rank <- book_words %>% 
  group_by(title) %>% 
  mutate(rank = row_number(), 
         `term frequency` = n/total) %>%
  ungroup()

freq_by_rank

freq_by_rank %>% 
  ggplot(aes(rank, `term frequency`, color = title)) + 
  geom_line(size = 1.1, alpha = 0.8, show.legend = FALSE) + 
  scale_x_log10() +
  scale_y_log10()

rank_subset <- freq_by_rank %>% 
  filter(rank < 500,
         rank > 10)

lm(log10(`term frequency`) ~ log10(rank), data = rank_subset)


freq_by_rank %>% 
  ggplot(aes(rank, `term frequency`, color = title)) + 
  geom_abline(intercept = -0.62, slope = -1.1, 
              color = "gray50", linetype = 2) +
  geom_line(size = 1.1, alpha = 0.8, show.legend = FALSE) + 
  scale_x_log10() +
  scale_y_log10()


books_for_tf <- gutenberg_download(c(17682,18487), meta_fields = "author")

books_for_tf <- physics_words %>%
  bind_tf_idf(word, author, n) %>%
  mutate(author = factor(author, levels = c("Various",
                                            "Daniel, Florence")))

books_for_tf %>% 
  group_by(author) %>% 
  slice_max(tf_idf, n = 15) %>% 
  ungroup() %>%
  mutate(word = reorder(word, tf_idf)) %>%
  ggplot(aes(tf_idf, word, fill = author)) +
  geom_col(show.legend = FALSE) +
  labs(x = "tf-idf", y = NULL) +
  facet_wrap(~author, ncol = 2, scales = "free")


