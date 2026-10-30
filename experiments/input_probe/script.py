import glob, os, subprocess
print('input entries:', flush=True)
total = 0.0
for d in sorted(glob.glob('/kaggle/input/*')):
    out = subprocess.run(['du', '-sb', d], capture_output=True, text=True).stdout.split()
    n = int(out[0]) if out else 0
    total += n
    print(f'{n/1e9:8.2f} GB  {d}', flush=True)
print(f'TOTAL {total/1e9:.2f} GB', flush=True)
hits = glob.glob('/kaggle/input/**/modernbert-large-a/fold0/config.json', recursive=True)
print('modernbert hits:', hits, flush=True)
print('modernbert b fold0:', glob.glob('/kaggle/input/**/modernbert-large-b/fold0/config.json', recursive=True), flush=True)
print('lora24 heads:', glob.glob('/kaggle/input/**/lora24/final/heads.safetensors', recursive=True), flush=True)
print('lora24 adapter:', glob.glob('/kaggle/input/**/lora24/final/adapter/adapter_config.json', recursive=True), flush=True)
