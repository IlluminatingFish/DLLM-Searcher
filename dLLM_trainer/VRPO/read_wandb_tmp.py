from wandb.proto import wandb_internal_pb2 as pb2
from wandb.sdk.internal.datastore import DataStore
import json

def read_history(path):
    ds = DataStore()
    ds.open_for_scan(path)
    step_data = {}
    cur_step = None
    while True:
        raw = ds.scan_data()
        if raw is None:
            break
        record = pb2.Record()
        record.ParseFromString(raw)
        if record.WhichOneof('record_type') != 'history':
            continue
        for item in record.history.item:
            nk_list = item.nested_key
            vj_raw  = item.value_json
            nk = nk_list[0] if len(nk_list) > 0 else ""
            try:
                vj_f = float(vj_raw)
            except Exception:
                vj_f = vj_raw
            if nk == '_step':
                cur_step = int(vj_f)
                if cur_step not in step_data:
                    step_data[cur_step] = {}
            elif nk and cur_step is not None:
                step_data[cur_step][nk] = vj_f
    ds.close()
    return step_data

runs = [
    ('/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/SFT/dLLM-RL/wandb/run-20260611_131322-a7tivh9s/run-a7tivh9s.wandb', 'SFT', '/tmp/sft_history.json'),
    ('/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/SFT/dLLM-RL/wandb/run-20260616_175351-q848anr9/run-q848anr9.wandb', 'X0pred', '/tmp/x0_history.json'),
]

for path, label, fname in runs:
    sd = read_history(path)
    print(f'{label}: {len(sd)} steps')
    steps = sorted(sd.keys())
    if steps:
        print('  Keys:', list(sd[steps[0]].keys()))
        for s in steps[:3]:
            print('   step', s, sd[s])
        for s in steps[-3:]:
            print('   step', s, sd[s])
    with open(fname, 'w') as f:
        json.dump({str(k): v for k, v in sd.items()}, f)
    print(f'  -> {fname}')
