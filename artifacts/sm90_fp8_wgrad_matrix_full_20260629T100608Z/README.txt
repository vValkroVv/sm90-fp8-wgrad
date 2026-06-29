SM90 FP8 Wgrad model-shape matrix
created_at_utc: 20260629T100608Z
models: all
ep_list: 1,8,16
route: both
global_tokens_list: 4096,8192,16384,32768
warmup_iters: 500
active_iters: 5000
repeat: 3
deepgemm_local_token_cap: 8192
parallel: 1
devices: 1,2,4,5
job_granularity: sweep
case_isolation: isolated
clean_child_outputs: 1
ep_token_mode: real

EP proxy:
local_experts = global_experts / EP
real mode local_tokens = global_tokens
stress mode local_tokens = global_tokens * EP

DeepGEMM is run separately from Sonic/custom. By default it is skipped for
local token counts above the local cap because the public DeepGEMM Wgrad API
has known failures for larger sequences in this benchmark.
