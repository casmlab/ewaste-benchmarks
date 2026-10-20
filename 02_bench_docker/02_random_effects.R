# Benchmark libraries
library(broom)
library(dplyr)
library(tidyr)  # nest(); missing from the original
library(ggplot2)
library(scales)
library(data.table)
library(furrr)
library(tictoc)

# Usage: Rscript 02_random_effects.R [input.csv] [max rows]
args <- commandArgs(trailingOnly = TRUE)
# Defaults to the real data file, or the synthetic one if the image was built
# without it (see README)
input_file <- if (length(args) >= 1) {
  args[1]
} else if (file.exists("all_posts_w_topic_and_candidate.csv")) {
  "all_posts_w_topic_and_candidate.csv"
} else {
  "synthetic_all_posts_w_topic_and_candidate.csv"
}
if (grepl("synthetic", input_file)) cat("NOTE: using synthetic data; the results mean nothing\n")
max_rows   <- if (length(args) >= 2) as.numeric(args[2]) else Inf

# ---------------------------------------------------------
# Measurement Hooks
# ---------------------------------------------------------
# Inside a container, every process in /proc belongs to the benchmark: this R
# session (main) and the furrr worker sessions (workers).
clk_tck <- as.numeric(system("getconf CLK_TCK", intern = TRUE))

proc_stat <- function(pid) {
  line <- tryCatch(readLines(sprintf("/proc/%s/stat", pid), warn = FALSE),
                   error = function(e) NA_character_)
  if (is.na(line)) return(NULL)
  # Fields after the ")" closing the process name; utime, stime, cutime and
  # cstime are fields 14-17 of the full line
  as.numeric(strsplit(sub("^.*\\) ", "", line), " ")[[1]][12:15])
}

cpu_seconds <- function() {
  me <- as.character(Sys.getpid())
  main <- 0; workers <- 0
  for (pid in list.files("/proc", pattern = "^[0-9]+$")) {
    st <- proc_stat(pid)
    if (is.null(st)) next
    if (pid == me) {
      main <- sum(st[1:2]) / clk_tck
      workers <- workers + sum(st[3:4]) / clk_tck  # children that have exited
    } else {
      workers <- workers + sum(st) / clk_tck
    }
  }
  c(main = main, workers = workers)
}

memory_mb <- function() {
  total_kb <- 0
  for (pid in list.files("/proc", pattern = "^[0-9]+$")) {
    status <- tryCatch(readLines(sprintf("/proc/%s/status", pid), warn = FALSE),
                       error = function(e) character(0))
    rss <- grep("^VmRSS:", status, value = TRUE)
    if (length(rss)) total_kb <- total_kb + as.numeric(gsub("[^0-9]", "", rss))
  }
  total_kb / 1024
}

measure_execution <- function(expr, name = "Task") {
  cat(sprintf("\n--- [START] %s ---\n", name))
  mem_before <- memory_mb()
  cpu_before <- cpu_seconds()
  start_wall <- Sys.time()

  result <- force(expr)

  wall <- as.numeric(difftime(Sys.time(), start_wall, units = "secs"))
  cpu <- cpu_seconds() - cpu_before
  mem_after <- memory_mb()

  cat(sprintf("--- [END] %s ---\n", name))
  cat(sprintf("Wall-clock time : %.4f seconds\n", wall))
  cat(sprintf("CPU time        : %.4f seconds (main %.4f + workers %.4f)\n",
              sum(cpu), cpu[["main"]], cpu[["workers"]]))
  cat(sprintf("Effective Cores : %.2f / %d\n", sum(cpu) / max(wall, 1e-6), n_cpus))
  cat(sprintf("RAM Used Delta  : %.2f MB\n\n", mem_after - mem_before))
  invisible(result)
}

# Setup multi-core parallel execution across available CPU cores
# (availableCores() respects docker --cpuset-cpus and --cpus)
n_cpus <- availableCores()
cat(sprintf("Running with %d CPUs\n", n_cpus))
measure_execution(plan(multisession, workers = n_cpus), name = "Starting Worker Sessions")

# Rscript prints each plot to Rplots.pdf in the working directory before
# ggsave() saves it; discard that copy
pdf(NULL)

# Ensure output directory exists to prevent ggsave errors
dir.create("output/figures", recursive = TRUE, showWarnings = FALSE)

topics <- c("crt_dei", "abortion", "cancel_culture", "court",  "environ", 
            "guns", "health", "imm", "lgbt", "police", "socialism")
mainstream <- c("Facebook", "Twitter", "Instagram", "TikTok", "YouTube", "Threads")
alttech    <- c("Truth Social", "Telegram", "Rumble", "Gettr")
issue_groups <- list(
  culture_wars = c("crt_dei", "lgbt", "cancel_culture", "socialism"),
  hybrid       = c("abortion", "guns"),
  policy       = c("imm", "health", "court", "environ", "police")
)

tic("Total Execution Pipeline")

# 1. High-Performance Data Ingestion & Cleaning
tic("Data Ingestion & Cleaning")
df <- measure_execution(fread(input_file, nrows = max_rows), name = "Data Ingestion")
cat(sprintf("Loaded %s posts from %s\n", format(nrow(df), big.mark = ","), input_file))

# Light cleaning
df <- df[!is.na(likes) & Office.Name %in% c("U.S. Representative", "U.S. Senator")]

# Renaming platforms
platform_map <- c(
  "facebook" = "Facebook", "twitter" = "Twitter", "instagram" = "Instagram",
  "rumble" = "Rumble", "t" = "Telegram", "threads" = "Threads",
  "tiktok" = "TikTok", "truthsocial" = "Truth Social", "youtube" = "YouTube", 
  "gettr" = "Gettr"
)
df[, platform := recode(platform, !!!platform_map)]

# Fast vectorized row checks
df[, policy := as.integer(rowSums(.SD == 1, na.rm = TRUE) > 0), .SDcols = issue_groups$policy]
df[, hybrid := as.integer(rowSums(.SD == 1, na.rm = TRUE) > 0), .SDcols = issue_groups$hybrid]
df[, culture_wars := as.integer(rowSums(.SD == 1, na.rm = TRUE) > 0), .SDcols = issue_groups$culture_wars]

df[, issue_group := case_when(
  culture_wars == 1 ~ "culture wars",
  hybrid       == 1 ~ "hybrid",
  policy       == 1 ~ "policy issue",
  TRUE              ~ NA_character_
)]

df[, issue_group := factor(issue_group, levels = c("policy issue", "hybrid", "culture wars"))]

# Calculate values
df[, loglikes := log(likes + 2)]
df[, totalengagement := likes + shares]
df[, logtotal := log(totalengagement + 2)]
toc()

# 2. Parallel Model Fitting Across Cores
tic("Parallel Linear Regressions")

# Parallel group-by regression to test multi-core CPU throughput
single_issue_groups <- df[!is.na(issue_group) & !is.na(Party.Standardized),
                          .(posts = .N, issue_groups = uniqueN(issue_group)),
                          by = .(Office.Name, Party.Standardized, platform)][issue_groups < 2]
if (nrow(single_issue_groups) > 0) {
  cat("Skipping groups whose posts are all in one issue group:\n")
  print(single_issue_groups)
}
n_fit_posts <- sum(!is.na(df$issue_group) & !is.na(df$Party.Standardized)) - sum(single_issue_groups$posts)
fit_start <- Sys.time()
all_results <- measure_execution(name = "Parallel Linear Regressions", df %>%
  filter(!is.na(issue_group) & !is.na(Party.Standardized)) %>%
  group_by(Office.Name, Party.Standardized, platform) %>%
  filter(n() > 0) %>%
  # lm can't fit a factor with one level; skip groups with posts from only one issue group
  filter(n_distinct(issue_group) >= 2) %>%
  nest() %>%
  future_pmap_dfr(function(Office.Name, Party.Standardized, platform, data) {
    result <- lm(loglikes ~ issue_group, data = data)
    
    tidy_result <- tidy(result) %>%
      mutate(
        platform = platform,
        party = Party.Standardized, # Fixed: Mapped from Party.Standardized correctly
        office = Office.Name
      )
    return(tidy_result)
  }, .options = furrr_options(packages = "broom")))  # workers need broom's tidy() method for lm
fit_time <- as.numeric(difftime(Sys.time(), fit_start, units = "secs"))

toc()
cat(sprintf("Fitted %d regressions on %s posts\n", nrow(distinct(all_results, office, party, platform)),
            format(n_fit_posts, big.mark = ",")))
cat(sprintf("Throughput (regressions): %.2f posts/sec\n", n_fit_posts / fit_time))

# 3. Post-Processing Coefficients
topic_results <- all_results %>%
  filter(term %in% c("issue_grouphybrid", "issue_groupculture wars"))

topic_results <- topic_results %>%
  mutate(
    estimate_likes   = exp(estimate) - 1,
    conf.low_likes   = exp(estimate - 1.96 * std.error) - 1,
    conf.high_likes  = exp(estimate + 1.96 * std.error) - 1
  )

# Renaming
topic_results$term <- recode(
  topic_results$term,
  "issue_grouphybrid" = "Hybrid",
  "issue_groupculture wars" = "Culture Wars"
)

topic_results$group <- ifelse(topic_results$platform %in% mainstream, "Mainstream", "Alt-tech")
topic_results$platform <- factor(topic_results$platform, levels = c(mainstream, alttech))

# 4. Figure Generation (Original Logic Preserved)
tic("Plot Rendering & File Output")

# Senate Plot
senate <- topic_results[topic_results$office == "U.S. Senator", ]

ggplot(senate, aes(x = platform, y = estimate_likes, color = term)) +
  geom_point(position = position_dodge(width = 0.6), size = 2) +
  geom_errorbar(
    aes(ymin = conf.low_likes, ymax = conf.high_likes),
    width = 0.2,
    position = position_dodge(width = 0.6)
  ) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  coord_flip() +
  theme_minimal() +
  facet_wrap(~ party, scales = "fixed") +
  labs(
    title = "Topic Effects on Likes (Senate)",
    x = "Platform",
    y = "Change in likes",
    color = "Issue Group"
  )

ggsave("output/figures/senate_topic_engagement_preds.pdf")

# House Plot
house <- topic_results[topic_results$office == "U.S. Representative", ]

ggplot(house, aes(x = platform, y = estimate, color = term)) +
  geom_point(position = position_dodge(width = 0.6), size = 2) +
  geom_errorbar(
    aes(ymin = estimate - 1.96 * std.error, ymax = estimate + 1.96 * std.error),
    width = 0.2,
    position = position_dodge(width = 0.6)
  ) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  coord_flip() +
  theme_minimal() +
  facet_wrap(~ party, scales = "fixed") +
  labs(
    title = "Topic Effects on Log Likes (House)",
    x = "Topic",
    y = "Coefficient Estimate (log likes)",
    color = "Platform"
  )

ggsave("output/figures/house_topic_engagement_preds.pdf")

toc() # End plot timing
toc() # End total execution timing