
library(dplyr)
library(stringr)
library(gutenbergr)

# Presenting the Pearson’s correlation 
# analysis results for the comparison of the two texts.


# 57215 = Some Medical Aspects of Old Age by Sir Humphry Davy Rolleston
# 17682 = The Healthy Life, Vol. V, Nos. 24-28 by Various
healty_life <- gutenberg_download(17682)
library(tidytext)
tidy_books <- healty_life %>%
  unnest_tokens(word, text)

tidy_books
data(stop_words)

tidy_books <- tidy_books %>%
  anti_join(stop_words)
tidy_books %>%
  count(word, sort = TRUE) 
library(ggplot2)

tidy_books %>%
  count(word, sort = TRUE) %>%
  filter(n > 107) %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(n, word)) +
  geom_col() +
  labs(y = NULL)

## second book

medical_aspect <- gutenberg_download(57215)
tidy_bronte <- medical_aspect %>%
  unnest_tokens(word, text) %>%
  anti_join(stop_words)
tidy_bronte %>%
  count(word, sort = TRUE)
library(ggplot2)

tidy_bronte %>%
  count(word, sort = TRUE) %>%
  filter(n > 58) %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(n, word)) +
  geom_col() +
  labs(y = NULL)

library(tidyr)

frequency <- bind_rows(mutate(tidy_bronte, author = "Humphry Davy"),
                       
                       mutate(tidy_books, author = "Various")) %>% 
  mutate(word = str_extract(word, "[a-z']+")) %>%
  count(author, word) %>%
  group_by(author) %>%
  mutate(proportion = n / sum(n)) %>% 
  select(-n) %>% 
  spread(author, proportion) %>% 
  gather(author, proportion, `Humphry Davy`)

frequency

library(scales)

# expect a warning about rows with missing values being removed
ggplot(frequency, aes(x = proportion, y = `Various`, 
                      color = abs(`Various` - proportion))) +
  geom_abline(color = "gray40", lty = 2) +
  geom_jitter(alpha = 0.1, size = 2.5, width = 0.3, height = 0.3) +
  geom_text(aes(label = word), check_overlap = TRUE, vjust = 1.5) +
  scale_x_log10(labels = percent_format()) +
  scale_y_log10(labels = percent_format()) +
  scale_color_gradient(limits = c(0, 0.001), 
                       low = "darkslategray4", high = "gray75") +
  facet_wrap(~author, ncol = 2) +
  theme(legend.position="none") +
  labs(y = "Various", x = NULL)

## Pearson correlation
cor.test(data = frequency[frequency$author == "Humphry Davy",],
         ~ proportion + `Various`)

