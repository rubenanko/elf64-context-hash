from __future__ import annotations
import argparse
import sys
from pathlib import Path
from elf_processing import BinaryAnalyzer
# import torch
from tqdm import tqdm
from constants import UNKNOWN_TOKEN,MASK_TOKEN
from loaders import load_vocabulary,load_encoder,load_predictor,load_embeddings
from torch import tensor,int64,flatten,norm,dot,device as torch_device,cuda,nn

from pickle import dumps as pickle_dumps, loads as pickle_loads
from base64 import b64encode
import json

def encode_tokens(vocab : dict, tokens: list) -> list:
    encoded_unkown_token = vocab.get(UNKNOWN_TOKEN)
    encoded_tokens = []

    for token in tokens:
        encoded_tokens.append(vocab.get(token, encoded_unkown_token))

    return tensor([encoded_tokens],dtype=int64)

def main() -> None:
    # storing arguments
    argv = sys.argv[1:]
    exit_code = 1

    parser = argparse.ArgumentParser()

    parser.add_argument("-E","--encode",
                        help="compute the embedding of the target elf64 binaries",
                        type=str,
                        nargs="*",
                        default=None)

    parser.add_argument("-P","--use-predictor",
                        help="Append a masked token at the end of each execution path, and output the corresponding predicted embedding",
                        action="store_true",
                        default=False)
    
    parser.add_argument("--stdout",
                        help="output the function addresses and b64 encoded embeddings in stdout",
                        action="store_true",
                        default=False)
    
    parser.add_argument("-C","--compare",
                        help="compare the calculated embeddings of two programs",
                        type=str,
                        nargs=2,
                        default=False)

    parser.add_argument("-F","--functions",
                        help="compare two specific functions (use with --compare)",
                        type=str,
                        nargs=2,
                        default=[None,None])
    
    parser.add_argument("-D","--dot-product",
                    help="compare two specific execution paths using the dot product instead of the distance.",
                    action="store_true",
                    default=False)

    args = parser.parse_args()

    # outputing the encoded bag of path for each input elf64 file
    if args.encode:
        device = torch_device("cuda" if cuda.is_available() else "cpu")
        vocab = load_vocabulary()
        predictor = load_predictor().to(device)
        encoder = load_encoder().to(device)

        for file in args.encode:
            file = Path(file)
            if not file.is_file():
                print(f"ERROR: the file '{file}' does not exist")
                continue

            # analysing the binary
            analyser = BinaryAnalyzer(file)
            bag_of_path = analyser.extract_bag_of_paths()

            b64_embeddings = {}
            # looping through paths
            for path in tqdm(bag_of_path,desc=f"encoding {len(bag_of_path)} execution paths",unit="path"):
                # add a mask token if predictor mode is used
                token_sequence = path[1]

                if args.use_predictor:
                    token_sequence.append(MASK_TOKEN)

                #encode the token sequence
                token_sequence = encode_tokens(vocab,token_sequence).to(device)

                # computing the embedding
                embedding = encoder(token_sequence)

                if args.use_predictor:
                    embedding = predictor(embedding)[0][-1]
                else:
                    embedding = embedding[0]
                
                # flatten
                embedding = flatten(embedding)
                # normalize
                embedding /= norm(embedding)
                # encode to b64
                embedding = embedding.tolist()

                # b64_embedding = b64encode(pickle_dumps(embedding))
                b64_embedding = b64encode(pickle_dumps(embedding)).decode("utf-8")
                if args.stdout:
                    print(f'function address : {hex(path[0])}\n{b64_embedding}',end="\n\n")

                if hex(path[0]) in b64_embeddings.keys():
                    b64_embeddings[hex(path[0])].append(b64_embedding)
                else:
                    b64_embeddings[hex(path[0])] = [b64_embedding]

            # saving_embeddings in a json file

            with open(f'{file}-embeddings{"-predictor" if args.use_predictor else ""}.json',"w") as f:
                json.dump(b64_embeddings,f,indent=4)

            exit_code = 0

    elif args.compare:
        file1,file2 = args.compare
        file1 = Path(file1)
        file2 = Path(file2)

        if not file1.is_file():
            print(f"ERROR: the file '{file1}' does not exist")
            sys.exit(1)
    
        if not file2.is_file():
            print(f"ERROR: the file '{file2}' does not exist")
            sys.exit(1)


        embeddings1 = load_embeddings(file1,args.functions[0])    
        embeddings2 = load_embeddings(file2,args.functions[1])


        for function1 in embeddings1:
            for function2 in embeddings2:
                for embedding1 in embeddings1[function1]:
                    for embedding2 in embeddings2[function2]:
                        max_length = max(embedding1.shape[0],embedding2.shape[0])
                        if embedding1.shape[0] < max_length:
                            embedding1 = nn.ZeroPad1d((0,max_length-embedding1.shape[0]))(embedding1)
                        else:
                            embedding2 = nn.ZeroPad1d((0,max_length-embedding2.shape[0]))(embedding2)
                        
                        if args.dot_product:
                            print(f'{function1} * {function2} : {"%.2f"}' % dot(embedding1,embedding2).item())
                        else:
                            print(f'd({function1},{function2}) : {"%.2f"}' % norm(embedding1 - embedding2).item())

    sys.exit(exit_code)
