# Generates a synthetic stand-in for all_posts_w_topic_and_candidate.csv, the
# input to Benchmark 2, until the real file is available. It has only the
# columns 02_random_effects.R uses, with the same names and coding, and
# realistic proportions (platform mix and candidate counts from Dataset 1).
# The values are random: regression results from it mean nothing.
#
# Usage: Rscript generate_synthetic_data.R [output.csv] [number of posts]

library(data.table)

args <- commandArgs(trailingOnly = TRUE)
out_file <- if (length(args) >= 1) args[1] else "synthetic_all_posts_w_topic_and_candidate.csv"
n_posts  <- if (length(args) >= 2) as.integer(args[2]) else 3321323L  # Dataset 1's post count

set.seed(330)

# Candidates: Dataset 1 has 1,758 (1,565 House, 193 Senate; 879 Republican,
# 799 Democratic, 80 third party)
n_cand <- 1758L
candidates <- data.table(
  candidate_UUID = sprintf("synthetic-%04d", seq_len(n_cand)),
  Office.Name = sample(c("U.S. Representative", "U.S. Senator"), n_cand,
                       replace = TRUE, prob = c(1565, 193)),
  Party.Standardized = sample(c("Republican Party", "Democratic Party", "Third Party"), n_cand,
                              replace = TRUE, prob = c(879, 799, 80)),
  # Some candidates post far more than others
  activity = rlnorm(n_cand, 0, 1.5),
  # Candidate-level popularity, the random intercept a random effects model estimates
  popularity = rnorm(n_cand, 0, 1)
)

# Platforms as coded in Dataset 1, with its post counts
platforms <- c(twitter = 1983240, facebook = 758586, instagram = 355970,
               truthsocial = 74399, youtube = 69700, tiktok = 32253, t = 20397,
               gettr = 18754, rumble = 7621, threads = 403)
platform_mean_loglikes <- c(twitter = 3, facebook = 4, instagram = 5, truthsocial = 4,
                            youtube = 3, tiktok = 5, t = 1, gettr = 2, rumble = 2, threads = 3)

cand_idx <- sample.int(n_cand, n_posts, replace = TRUE, prob = candidates$activity)
df <- candidates[cand_idx, .(candidate_UUID, Office.Name, Party.Standardized, popularity)]
df[, id := seq_len(n_posts)]
df[, platform := sample(names(platforms), n_posts, replace = TRUE, prob = platforms)]

# Topic flags: 1 if the post is about the topic, else 0. Most posts have none.
topics <- c("crt_dei", "abortion", "cancel_culture", "court", "environ",
            "guns", "health", "imm", "lgbt", "police", "socialism")
topic_rates <- c(crt_dei = 0.01, abortion = 0.04, cancel_culture = 0.01, court = 0.03,
                 environ = 0.05, guns = 0.03, health = 0.08, imm = 0.05, lgbt = 0.02,
                 police = 0.03, socialism = 0.01)
for (t in topics) set(df, j = t, value = rbinom(n_posts, 1L, topic_rates[[t]]))

# Engagement: log-normal around a platform- and candidate-specific level, with
# a small boost for culture-war topics. About 16% of posts have no like count,
# as in Dataset 1.
culture <- df$crt_dei | df$lgbt | df$cancel_culture | df$socialism
loglikes <- platform_mean_loglikes[df$platform] + df$popularity + 0.2 * culture +
  rnorm(n_posts, 0, 1.5)
df[, likes := as.numeric(pmax(0, round(exp(loglikes) - 1)))]
df[sample.int(n_posts, round(0.16 * n_posts)), likes := NA]
df[, shares := as.numeric(rpois(n_posts, fcoalesce(likes, 0) / 10))]
df[is.na(likes), shares := NA]

df[, popularity := NULL]
setcolorder(df, c("id", "candidate_UUID", "platform", "likes", "shares",
                  "Office.Name", "Party.Standardized", topics))
fwrite(df, out_file)
cat(sprintf("Wrote %s posts by %s synthetic candidates to %s\n",
            format(n_posts, big.mark = ","), format(n_cand, big.mark = ","), out_file))
