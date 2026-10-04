#!/usr/bin/env bash
# Functionality test for the Dockerized benchmarks: builds each benchmark's
# image and runs it once, serially, using quick-test options for the long
# benchmarks (05-08). A benchmark passes if its container exits cleanly and
# prints its final results line. This checks that everything runs; it doesn't
# produce meaningful timings.
#
# Usage:
#   ./test_functionality.sh            # test all benchmarks
#   ./test_functionality.sh 05 07      # test only the listed benchmarks
#
# Each benchmark's build and run output is saved to a log file; the summary at
# the end says where. Takes about 6-8 minutes for all benchmarks on an 8-CPU
# laptop once the images are built (the first build downloads several GB).

set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$(mktemp -d "${TMPDIR:-/tmp}/benchmark-functionality-XXXXXX")"
TIMEOUT_SECONDS=1800  # per benchmark run, not counting the build

# id | folder | image | expected line in the output | docker run options | arguments for the script
BENCHMARKS=(
  "03|03_bench_docker|ams-create-best-classifiers|Cohen's kappa|"
  "04|04_bench_docker|ams-label-comparisons|Predicted label counts|"
  "05|05_bench_docker|ams-lda-mallet|Throughput .*Gibbs sampling only|-e NUM_DOCS=20000 -e NUM_ITERATIONS=50"
  "06|06_bench_docker|ams-lda-inference|Wrote topic distributions|-e INPUT_FILE=russell_processed_0.9.csv.gz"
  "07|07_bench_docker|ams-bertopic-fit|OVERALL THROUGHPUT||-n 2000"
  "08|08_bench_docker|ams-bertopic-infer|OVERALL INFERENCE SPEED||-n 2000"
)

if ! docker info >/dev/null 2>&1; then
  echo "Docker isn't available (is the daemon running, and can this user use it?)" >&2
  exit 2
fi

selected=("$@")
is_selected() {
  [ ${#selected[@]} -eq 0 ] && return 0
  local s
  for s in "${selected[@]}"; do
    [ "$s" = "$1" ] && return 0
  done
  return 1
}

results=()
failures=0

for entry in "${BENCHMARKS[@]}"; do
  IFS='|' read -r id folder image expected run_opts script_args <<<"$entry"
  is_selected "$id" || continue

  log="$LOG_DIR/${id}.log"
  echo "=== Benchmark $id ($folder) ==="
  start=$(date +%s)

  echo "  building $image..."
  if ! docker build -t "$image" "$REPO_DIR/$folder" >"$log" 2>&1; then
    echo "  FAIL: build failed (see $log)"
    results+=("$id  FAIL  build failed")
    failures=$((failures + 1))
    continue
  fi

  echo "  running ${run_opts:+with $run_opts }${script_args}..."
  name="functest-$id-$$"
  # Run as the current user, as the READMEs do. Word splitting of run_opts and
  # script_args is intended: each holds separate command-line options.
  # shellcheck disable=SC2086
  timeout "$TIMEOUT_SECONDS" docker run --rm --name "$name" --user "$(id -u):$(id -g)" \
    $run_opts "$image" $script_args >>"$log" 2>&1
  status=$?
  if [ $status -eq 124 ]; then
    # timeout stops only the docker client; stop the container too
    docker kill "$name" >/dev/null 2>&1
  fi
  elapsed=$(( $(date +%s) - start ))

  if [ $status -eq 124 ]; then
    outcome="FAIL  timed out after ${TIMEOUT_SECONDS}s"
  elif [ $status -ne 0 ]; then
    outcome="FAIL  exited with status $status"
  elif ! grep -Eq "$expected" "$log"; then
    outcome="FAIL  finished without printing '$expected'"
  else
    outcome="PASS"
  fi
  [ "$outcome" = "PASS" ] || failures=$((failures + 1))
  echo "  $outcome (${elapsed}s)"
  results+=("$id  $outcome  (${elapsed}s)")
done

if [ ${#results[@]} -eq 0 ]; then
  echo "No benchmarks matched: $*" >&2
  exit 2
fi

echo
echo "=== Summary ==="
printf '%s\n' "${results[@]}"
echo "Logs: $LOG_DIR"
[ $failures -eq 0 ]
