"""
elf_processing_core.py

Pipeline d'extraction statique de bag-of-paths tokenises depuis des binaires ELF.

Dependances :
    angr networkx pebble

Usage CLI :
    python elf_processing_core.py dataset/ 8 output/
    # => traite tous les *.elf de dataset/ avec 8 workers, timeout 120s/binaire
    # => produit output/<stem>.jsonl par binaire
"""

import gc
import json
import logging
import random
from concurrent.futures import TimeoutError as FuturesTimeoutError
from concurrent.futures import as_completed
from pathlib import Path
from typing import Any, Generator, Iterator

# disable an annoying useless log
logging.getLogger("angr.state_plugins.unicorn_engine").disabled = True

import angr
import networkx as nx
from pebble import ProcessPool  # pip install pebble
from tqdm import tqdm  # pip install tqdm
# Silence angr/cle — les erreurs de lifting sont gerees via <UNLIFTABLE>
logging.getLogger("angr").setLevel(logging.ERROR)
logging.getLogger("cle").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_MAX_PATH_LENGTH = 50
DEFAULT_MAX_PATHS = 500


# ══════════════════════════════════════════════════════════════════════════════
# COEUR DE TRAITEMENT
# ══════════════════════════════════════════════════════════════════════════════


class BinaryAnalyzer:
    """
    Analyse un unique binaire ELF et produit un bag-of-paths tokenise.

    Utiliser comme context manager pour garantir la liberation memoire :

        with BinaryAnalyzer(elf_path) as analyzer:
            bag = analyzer.extract_bag_of_paths()

    Retourne une liste de (func_addr, token_path) pour l'agregation
    par fonction dans le DataLoader PyTorch (groupby -> Max-Pooling).
    """

    def __init__(
        self,
        binary_path: str | Path,
        max_path_length: int = DEFAULT_MAX_PATH_LENGTH,
        max_paths: int = DEFAULT_MAX_PATHS,
        random_seed: int | None = None,
    ):
        self.binary_path = Path(binary_path)
        self.max_path_length = max_path_length
        self.max_paths = max_paths

        # RNG isole par instance : pas d'effet de bord sur le module global,
        # indispensable quand plusieurs BinaryAnalyzer coexistent.
        self.rng = random.Random(random_seed)

        self._proj: angr.Project | None = None
        self._cfg: Any = None

    # -- Context Manager -----------------------------------------------------

    def __enter__(self) -> "BinaryAnalyzer":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._cfg = None
        self._proj = None
        gc.collect()

    # -- Chargement paresseux ------------------------------------------------

    @property
    def proj(self) -> angr.Project:
        if self._proj is None:
            self._proj = angr.Project(str(self.binary_path), auto_load_libs=False)
        return self._proj

    @property
    def cfg(self) -> Any:
        if self._cfg is None:
            self._cfg = self.proj.analyses.CFGFast(normalize=True)
        return self._cfg

    # -- API publique --------------------------------------------------------

    def extract_bag_of_paths(self) -> list[tuple[int, list[str]]]:
        """
        Retourne la liste de (func_addr, tokens) pour tout le binaire.
        Le cache de blocs garantit que chaque adresse n'est liftee en VEX
        qu'une seule fois, meme si le bloc apparait dans 500 chemins.
        """
        bag: list[tuple[int, list[str]]] = []
        block_cache: dict[int, list[str]] = {}

        for func_addr, func in self.cfg.functions.items():
            if func.is_simprocedure or func.is_syscall or func.is_plt:
                continue

            blocks = list(func.blocks)
            if not blocks:
                continue

            g, addr_to_cfgnode = self._build_function_graph(func, blocks)

            entry = func_addr if func_addr in g else blocks[0].addr
            if entry not in g:
                continue

            for path_addrs in self._enumerate_paths(g, entry):
                token_path: list[str] = []
                for blk_addr in path_addrs:
                    if blk_addr not in block_cache:
                        block_cache[blk_addr] = self._tokenize_block(
                            blk_addr, addr_to_cfgnode.get(blk_addr)
                        )
                    token_path.extend(block_cache[blk_addr])

                if token_path:
                    bag.append((func_addr, token_path))

        return bag

    # -- Construction du graphe intra-fonction --------------------------------

    def _build_function_graph(
        self, func: Any, blocks: list[Any]
    ) -> tuple[nx.DiGraph, dict[int, Any]]:
        g = nx.DiGraph()
        addr_to_cfgnode: dict[int, Any] = {}
        func_addrs = {b.addr for b in blocks}

        for blk in blocks:
            cfg_node = self.cfg.model.get_any_node(blk.addr)
            addr_to_cfgnode[blk.addr] = cfg_node
            g.add_node(blk.addr)

        for blk in blocks:
            cfg_node = addr_to_cfgnode[blk.addr]
            if cfg_node is None:
                continue
            for succ in cfg_node.successors:
                if succ.addr in func_addrs:
                    g.add_edge(blk.addr, succ.addr)

        return g, addr_to_cfgnode

    # -- Enumeration des chemins (DFS aleatoire) ------------------------------

    def _enumerate_paths(
        self, graph: nx.DiGraph, source: int
    ) -> Generator[list[int], None, None]:
        count = 0
        stack = [(source, [source], frozenset([source]))]

        while stack:
            node, path, visited = stack.pop()
            successors = list(graph.successors(node))
            self.rng.shuffle(successors)

            if not successors or len(path) >= self.max_path_length:
                yield path
                count += 1
                if count >= self.max_paths:
                    return
                continue

            pushed_any = False
            for succ in successors:
                if succ not in visited:
                    stack.append((succ, path + [succ], visited | {succ}))
                    pushed_any = True

            if not pushed_any:
                yield path
                count += 1
                if count >= self.max_paths:
                    return

    # -- Tokenisation d'un bloc -----------------------------------------------

    def _tokenize_block(self, block_addr: int, cfg_node: Any) -> list[str]:
        tokens: list[str] = []

        try:
            vex = self.proj.factory.block(block_addr).vex
        except Exception:
            return ["<UNLIFTABLE>"]

        for stmt in vex.statements:
            tag = getattr(stmt, "tag", None)
            if tag is None or tag == "Ist_IMark":
                continue
            if tag == "Ist_WrTmp":
                tokens.append(self._token_wrtmp(getattr(stmt, "data", None)))
            elif tag == "Ist_Store":
                tokens.append("VEX_STORE")
            elif tag == "Ist_Put":
                tokens.append("VEX_REG_WRITE")
            elif tag == "Ist_Exit":
                tokens.append("VEX_EXIT_COND")
            else:
                tokens.append(f"VEX_{tag}")

        api_tok = self._get_terminal_api(cfg_node)
        tokens.append(
            api_tok
            if api_tok
            else f"JK_{getattr(vex, 'jumpkind', 'Ijk_Unknown').replace('Ijk_', '').upper()}"
        )
        return tokens

    # -- Helpers de tokenisation ----------------------------------------------

    @staticmethod
    def _token_wrtmp(data: Any) -> str:
        if data is None:
            return "VEX_WrTmp"
        tag = getattr(data, "tag", "")
        op_val = getattr(data, "op", None)
        if op_val and isinstance(op_val, str) and "_" in op_val:
            return f"VEX_OP_{op_val.split('_')[1][:3].upper()}"
        return {
            "Iex_Load": "VEX_LOAD",
            "Iex_Const": "VEX_CONST",
            "Iex_Get": "VEX_REG_READ",
        }.get(tag, "VEX_WrTmp")

    def _get_terminal_api(self, cfg_node: Any) -> str | None:
        if cfg_node is None:
            return None
        for succ in cfg_node.successors:
            tok = self._api_token(self.cfg.functions.get(succ.addr))
            if tok:
                return tok
        return None

    @staticmethod
    def _api_token(func: Any) -> str | None:
        if func is None:
            return None
        if func.is_plt or func.is_simprocedure:
            return f"<API_{func.name.split('@')[0].upper()}>"
        if func.is_syscall:
            return f"<SYSCALL_{func.name.upper()}>"
        return None


# ══════════════════════════════════════════════════════════════════════════════
# WORKER
# ══════════════════════════════════════════════════════════════════════════════


def _analyze_one(args: tuple[Path, Path, dict[str, Any]]) -> tuple[str, int]:
    """
    Worker execute dans un sous-processus pebble.

    Ecrit directement <stem>.jsonl dans output_dir.
    Seul (nom, nb_chemins) traverse l'IPC — aucun token ne transite,
    ce qui evite la serialisation de centaines de Mo vers le process principal.
    """
    elf_path, output_dir, kwargs = args
    out_file = output_dir / f"{elf_path.stem}.jsonl"
    tmp_file = output_dir / f"{elf_path.stem}.tmp"  # ecriture dans le .tmp
    paths_count = 0

    with tmp_file.open("w") as f:
        with BinaryAnalyzer(elf_path, **kwargs) as analyzer:
            for func_addr, tokens in analyzer.extract_bag_of_paths():
                f.write(
                    json.dumps(
                        {
                            "file": elf_path.name,
                            "func_addr": hex(func_addr),
                            "tokens": tokens,
                        }
                    )
                    + "\n"
                )
                paths_count += 1

    # Renommage atomique : n'a lieu que si tout s'est bien passe.
    # En cas de SIGTERM (timeout), le .tmp reste orphelin mais aucun
    # .jsonl corrompu ne pollue le dataset.
    tmp_file.replace(
        out_file
    )  # replace() est atomique ET cross-platform (rename() crashe sur Windows si la cible existe)
    return elf_path.name, paths_count


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE DATASET
# ══════════════════════════════════════════════════════════════════════════════


def process_dataset(
    elf_paths: Iterator[str | Path],
    output_dir: str | Path = "output",
    max_workers: int = 4,
    timeout_sec: int = 120,
    **analyzer_kwargs: Any,
) -> None:
    """
    Traite un dataset de binaires ELF en parallele via pebble.ProcessPool.

    Pourquoi pebble et pas ProcessPoolExecutor natif ?
    - ProcessPoolExecutor cree des workers "daemoniques" : ils n'ont pas le
      droit de spawner des sous-processus enfants (AssertionError immediate).
    - Les fonctions locales/imbriquees ne sont pas picklables (AttributeError).
    - pebble.ProcessPool resout les deux problemes et ajoute un hard-timeout
      par tache via SIGTERM, indispensable quand angr se bloque en C++ natif
      sur des malwares obfusques (UPX, Themida).

    Chaque binaire produit output/<stem>.jsonl (convention HuggingFace shards).

    Args:
        elf_paths:       iterable de chemins vers les binaires.
        output_dir:      dossier de sortie (cree si absent).
        max_workers:     processus paralleles. angr est RAM-bound, pas CPU-bound.
                         Comptez ~2-4 Go/worker. Sur 16 Go : max_workers=4 max.
                         Un OOM-Kill se manifeste en BrokenProcessPool — baisser
                         max_workers si cela arrive.
        timeout_sec:     timeout strict par binaire (SIGTERM via pebble).
        analyzer_kwargs: max_paths, max_path_length, random_seed.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    elf_list = [Path(e) for e in elf_paths]
    total_paths = 0

    with ProcessPool(max_workers=max_workers) as pool:
        futures = {
            pool.schedule(
                _analyze_one,
                args=((elf, output_dir, analyzer_kwargs),),
                timeout=timeout_sec,
            ): elf
            for elf in elf_list
        }

        for future in tqdm(
            as_completed(futures), total=len(elf_list), desc="Extraction", unit="elf"
        ):
            elf_path = futures[future]
            try:
                name, count = future.result()
                total_paths += count
                logger.info("OK      %-40s -> %d paths", name, count)
            except FuturesTimeoutError:
                # Nettoyage du .tmp laisse par le worker tue par SIGTERM
                (output_dir / f"{elf_path.stem}.tmp").unlink(missing_ok=True)
                logger.warning(
                    "TIMEOUT %-40s (> %ds) — ejete", elf_path.name, timeout_sec
                )
            except Exception as exc:
                (output_dir / f"{elf_path.stem}.tmp").unlink(missing_ok=True)
                logger.error("ERR     %-40s — %s", elf_path.name, exc)

    logger.info("Done. %d total paths dans %s/", total_paths, output_dir)


# ══════════════════════════════════════════════════════════════════════════════
# POINT D'ENTREE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    dataset_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dataset")
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    elf_paths = sorted(dataset_dir.glob("*.elf"))
    output_dir = Path(sys.argv[3])

    process_dataset(
        elf_paths,
        output_dir=output_dir,
        max_workers=workers,
        timeout_sec=120,
        max_paths=500,
        max_path_length=50,
    )
