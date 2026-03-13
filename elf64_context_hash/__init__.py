from huggingface_hub import hf_hub_download

from elf64_context_hash.constants import (DEFAULT_VOCAB_PATH,
                                          DEFAULT_CHECKPOINT_PATH,
                                          DEFAULT_DATA_PATH,
                                          DEFAULT_REPO_ID)

# initializing the module storage
if not DEFAULT_DATA_PATH.is_dir():
    DEFAULT_DATA_PATH.mkdir()

# downloading the vocabulary
if not DEFAULT_VOCAB_PATH.is_file():
    print(f"vocabulary not found, downloading the vocabulary from {DEFAULT_REPO_ID}, into {DEFAULT_DATA_PATH}")
    hf_hub_download(repo_id=DEFAULT_REPO_ID,filename="vocab.json",local_dir=DEFAULT_DATA_PATH)

# downloading the checkpoint
if not DEFAULT_CHECKPOINT_PATH.is_file():
    print(f"checkpoint not found, downloading the checkpoint from {DEFAULT_REPO_ID}, into {DEFAULT_DATA_PATH}")
    hf_hub_download(repo_id=DEFAULT_REPO_ID,filename="latest.pt",local_dir=DEFAULT_DATA_PATH)