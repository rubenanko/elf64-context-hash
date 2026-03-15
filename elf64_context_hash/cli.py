from __future__ import annotations
import argparse
import sys
import os
from pathlib import Path
from elf64_context_hash.elf_processing import BinaryAnalyzer
# import torch
from tqdm import tqdm
from elf64_context_hash.constants import UNKNOWN_TOKEN,MASK_TOKEN, DEFAULT_OUTPUT_PATH

from elf64_context_hash.loaders import (load_vocabulary,
                     load_encoder,
                     load_predictor,
                     load_embeddings)

from torch import (tensor,
                   int64,
                   flatten,
                   norm,dot,
                   device as torch_device,
                   cuda,
                   nn)

import matplotlib.pyplot as plt

from pickle import (dumps as pickle_dumps, 
                    loads as pickle_loads)
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
    current_dir = os.getcwd()

    parser = argparse.ArgumentParser()

    parser.add_argument("-O", "--output",
                        help="Specify the output path of the computation.",
                        type=str,
                        nargs=1,
                        default=""
                        )

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
    
    parser.add_argument("--plot",
                    help="generate a heatmap of the comparision",
                    action="store_true",
                    default=False)

    args = parser.parse_args()

    output_path = DEFAULT_OUTPUT_PATH if args.output is not None else os.path.join(current_dir, args.output)

    if(not os.path.exists(output_path)):
        os.mkdir(output_path)

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
                    print(f'function : {path[0]}\n{b64_embedding}',end="\n\n")

                if path[0] in b64_embeddings.keys():
                    b64_embeddings[path[0]].append(b64_embedding)
                else:
                    b64_embeddings[path[0]] = [b64_embedding]

            # saving_embeddings in a json file

            with open(os.path.join(output_path, f'{file}-embeddings{"-predictor" if args.use_predictor else ""}.json'),"w") as f:
                json.dump(b64_embeddings,f,indent=4)

            exit_code = 0

    elif args.compare:
        exit_code = 0
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
        functions_dist = {}

        for function1 in embeddings1:
            functions_dist[function1] = {}
            for function2 in embeddings2:

                functions_dist[function1][function2] = int(not args.dot_product)
                cmp_function = max if args.dot_product else min

                embeddings_pairs_counter = 0
                for embedding1 in embeddings1[function1]:
                    for embedding2 in embeddings2[function2]:
                        embeddings_pairs_counter += 1
                        max_length = max(embedding1.shape[0],embedding2.shape[0])
                        if embedding1.shape[0] < max_length:
                            embedding1 = nn.ZeroPad1d((0,max_length-embedding1.shape[0]))(embedding1)
                        else:
                            embedding2 = nn.ZeroPad1d((0,max_length-embedding2.shape[0]))(embedding2)

                        if args.dot_product:
                            result = dot(embedding1,embedding2).item()
                            output = f'{function1} * {function2} : {"%.2f"}' % result
                        else:
                            result = norm(embedding1 - embedding2).item()
                            output = f'd({function1},{function2}) : {"%.2f"}' % result

                        if args.plot:
                            functions_dist[function1][function2] = cmp_function(functions_dist[function1][function2],result)
                        else:
                            print(output)

                # functions_dist[function1][function2] /= embeddings_pairs_counter
        

        if args.plot:
            proximity = [
                [
                    functions_dist[function1][function2] for function1 in embeddings1
                ] for function2 in embeddings2
            ]

            fig, ax = plt.subplots(figsize=(8, 5))
            cmap = "plasma" if args.dot_product else "plasma_r"
            label = "Dot product" if args.dot_product else "Distance" 

            heatmap = ax.imshow(proximity, aspect="auto", cmap=cmap, vmin=0, vmax=1)

            ax.set_xticks(range(len(embeddings1.keys())))
            ax.set_xticklabels(embeddings1.keys(), rotation=45, ha="right")

            ax.set_yticks(range(len(embeddings2.keys())))
            ax.set_yticklabels(embeddings2.keys())

            ax.set_xlabel(f"Functions from {file1} (adresses)")
            ax.set_ylabel(f"Functions from {file2} (adresses)")
            ax.set_title("Proximity heatmap between functions")

            plt.colorbar(heatmap, ax=ax, label=label)
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, "heatmap.png"), dpi=150)
            plt.show()


    sys.exit(exit_code)
