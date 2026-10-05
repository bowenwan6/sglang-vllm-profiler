#!/usr/bin/env bash
# What a client receives when SGLANG_REQ_WAITING_TIMEOUT aborts a queued request.
# All requests use temperature 0: the first sampled request on a cold server stalls the scheduler ~70 s.
set -uo pipefail
pkill -9 -f sglang.launch_server 2>/dev/null; pkill -f diag5.sh 2>/dev/null; sleep 3
set +u; source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate sgl-profiler; set -u
M=Qwen/Qwen3-8B; U=http://127.0.0.1:30000; J='Content-Type: application/json'
D="$HOME/sgl/logs/p1/diag6"; rm -rf "$D"; mkdir -p "$D"; cd "$D"
SGLANG_REQ_WAITING_TIMEOUT=2 python -m sglang.launch_server --model-path "$M" --host 127.0.0.1 --port 30000 --max-running-requests 2 > server.log 2>&1 < /dev/null &
PID=$!
for _ in $(seq 1 120); do curl -sf "$U/health" > /dev/null 2>&1 && break; sleep 3; done
sleep 2
chat() { printf '{"model":"%s","messages":[{"role":"user","content":"%s"}],"max_tokens":%s,"temperature":0,"ignore_eos":true,"stream":%s%s}' "$M" "$1" "$2" "$3" "$4"; }
curl -s -N --max-time 90 "$U/v1/chat/completions" -H "$J" -d "$(chat 'Write a long story one.' 3000 true '')" > blocker_1.txt & B1=$!
curl -s -N --max-time 90 "$U/v1/chat/completions" -H "$J" -d "$(chat 'Write a long story two.' 3000 true '')" > blocker_2.txt & B2=$!
sleep 2
T0=$(date +%s.%N); el() { echo "$(date +%s.%N) - $T0" | bc; }
( curl -s -i -N --max-time 30 "$U/v1/chat/completions" -H "$J" -d "$(chat 'Say hi.' 16 true ',"stream_options":{"include_usage":true}')" > q_chat_stream.txt; echo "[curl_exit=$? after=$(el)s]" >> q_chat_stream.txt ) & Q1=$!
( curl -s -i --max-time 30 "$U/v1/chat/completions" -H "$J" -d "$(chat 'Say hi.' 16 false '')" > q_chat_nonstream.txt; echo "[curl_exit=$? after=$(el)s]" >> q_chat_nonstream.txt ) & Q2=$!
( curl -s -i -N --max-time 30 "$U/v1/completions" -H "$J" -d "{\"model\":\"$M\",\"prompt\":\"Say hi.\",\"max_tokens\":16,\"temperature\":0,\"stream\":true}" > q_completions_stream.txt; echo "[curl_exit=$? after=$(el)s]" >> q_completions_stream.txt ) & Q3=$!
( curl -s -i --max-time 30 "$U/generate" -H "$J" -d '{"text":"Say hi.","sampling_params":{"max_new_tokens":16,"temperature":0}}' > q_generate_nonstream.txt; echo "[curl_exit=$? after=$(el)s]" >> q_generate_nonstream.txt ) & Q4=$!
( curl -s -i -N --max-time 30 "$U/generate" -H "$J" -d '{"text":"Say hi.","sampling_params":{"max_new_tokens":16,"temperature":0},"stream":true}' > q_generate_stream.txt; echo "[curl_exit=$? after=$(el)s]" >> q_generate_stream.txt ) & Q5=$!
wait $Q1 $Q2 $Q3 $Q4 $Q5
echo "blocker bytes so far: $(wc -c < blocker_1.txt) $(wc -c < blocker_2.txt)"
kill $B1 $B2 2>/dev/null
kill -TERM $PID 2>/dev/null; sleep 5; pkill -9 -f sglang.launch_server 2>/dev/null
for f in q_chat_stream q_chat_nonstream q_completions_stream q_generate_nonstream q_generate_stream; do echo "=== $f"; head -c 900 "$f.txt" | tr -d '\r' | grep -v -E '^(date|server|content-length|x-request|transfer-encoding|connection|cache-control):' ; echo; done
