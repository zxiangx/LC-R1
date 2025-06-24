from huggingface_hub import snapshot_download
import os
dataset_path = '../dataset'
LC_extractor_path = '../extractor_model'
if not os.path.exists(dataset_path):
    print("Dataset not found locally, downloading from Huggingface...")
    snapshot_download(repo_id="zx10086/LCR1_training", 
                     local_dir=dataset_path,
                     repo_type="dataset")

if not os.path.exists(LC_extractor_path):
    print("Extractor model not found locally, downloading from Huggingface...")
    snapshot_download(repo_id="zx10086/extractor_model",
                     local_dir=LC_extractor_path,
                     repo_type="model")