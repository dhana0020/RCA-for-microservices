from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="phamquiluan/RCAEval",
    repo_type="dataset",
    allow_patterns="re2ob_checkoutservice_cpu_1/*",
    local_dir="data/re2_sample"
)

print("Downloaded to:", path)