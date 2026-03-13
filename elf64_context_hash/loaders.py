from pathlib import Path
import json
from elf64_context_hash.constants import (DEFAULT_VOCAB_PATH,
                       DEFAULT_CHECKPOINT_PATH,
                       DEFAULT_VOCAB_SIZE,
                       DEFAULT_PREDICTOR_DIM)
import torch.nn as nn
from torch import load as load_model, Tensor,tensor
from elf64_context_hash.model.encoder import Conv1DEncoder
from elf64_context_hash.model.predictor import Predictor
from pickle import loads as pickle_loads
from base64 import b64decode
import sys



def load_vocabulary(vocab_path: str | Path = DEFAULT_VOCAB_PATH) -> dict:
    vocab_path = Path(vocab_path)
    vocab = None

    if not vocab_path.exists():
        print(f"ERROR: the vocabulary file {vocab_path} doesn't exist")

    with vocab_path.open("r") as f:
        vocab = json.load(f)

    if not vocab:
        print("ERROR: unable to retrieve the vocabulary")

    return vocab

# load encoder from checkpoint
def load_encoder(vocab_size : int = DEFAULT_VOCAB_SIZE, checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH) -> Conv1DEncoder:
    model = Conv1DEncoder(vocab_size)
    checkpoint = load_model(checkpoint_path)
    model.load_state_dict(checkpoint["encoder"])
    return model

# load predictor from checkpoint
def load_predictor(dim : int = DEFAULT_PREDICTOR_DIM, checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH) -> Predictor:
    model = Predictor(dim)
    checkpoint = load_model(checkpoint_path)
    model.load_state_dict(checkpoint["predictor"])
    return model


def load_embeddings(file_path : str | Path,target_function : str = None) -> list:
    try:
        with open(file_path,"r") as f:
            functions = json.load(f)
    except:
        print(f"ERROR: could not load json data from the file {file_path}")
        sys.exit(1)

    output_paths = {}

    if target_function != None:
        if target_function in functions.keys():
            output_paths[target_function] = []
            for b64embedding in functions[target_function]:
                output_paths[target_function].append(tensor(pickle_loads(b64decode(b64embedding))))
        else:
            print(f"ERROR: could not find the target function {target_function}")

    else:
        for function in functions:
            output_paths[function] = []
            for b64embedding in functions[function]:
                output_paths[function].append(tensor(pickle_loads(b64decode(b64embedding))))

    return output_paths