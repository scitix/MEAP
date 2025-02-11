import glob
import json
import os
import sys
from pathlib import Path
import time
import numpy as np
from tqdm import tqdm
from multiprocessing import Process, cpu_count
# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import lit_gpt.packed_dataset as packed_dataset
from transformers import AutoTokenizer



filename_sets = {
    "c4": "json_c4/c4-train*",
}



def prepare_full(
    filenames: list[Path], checkpoint_dir: Path, destination_path: Path, chunk_size: int, match: str = "",process_id: int = 0
) -> None:
    """Prepare the "Red Pajama" dataset using the original tokenizer."""
    import zstandard as zstd

    destination_path.mkdir(parents=True, exist_ok=True)
    print(checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir,legacy=False)

    builder = packed_dataset.PackedDatasetBuilder(
        outdir=destination_path,
        prefix="c4_" + str(process_id),
        chunk_size=chunk_size,
        sep_token=tokenizer.eos_token_id,
        dtype="auto",
        vocab_size=tokenizer.vocab_size,
    )

    for filepath in filenames:
        #filepath = source_path / name

        print(f"Processing {filepath}")


        with open(filepath, encoding="utf-8") as f:
            for row in tqdm(f):
                try:
                    text = json.loads(row)["text"]
                    text_ids = tokenizer.encode(text)+[tokenizer.eos_token_id]
                    builder.add_array(np.array(text_ids, dtype=builder.dtype))
                except Exception as e:
                    print(f"Error processing row: {e}")
                    continue
        #os.remove(filepath)

    builder.write_reminder()


def prepare(
    source_path: Path = Path("data/RedPajama-Data-1T-Sample"),
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
    destination_path: Path = Path("data/redpajama_sample"),
    sample: bool = True,
    match: str = "",
) -> None:
    """Prepare the "Red Pajama" dataset. We assume tokenizer has been trained."""
    # with open(checkpoint_dir / "lit_config.json") as fp:
    #     config = Config(**json.load(fp))
    sample = False

    #num_processes = cpu_count() 
    num_processes = 80
    pattern = filename_sets["c4"]
    filenames = glob.glob(os.path.join(source_path, pattern), recursive=True)
    print("source_path",source_path)
    print("pattern",pattern)
    print("os.path.join(source_path, pattern)",os.path.join(source_path, pattern))
    print("filenames",filenames)
    chunked_filenames = np.array_split(filenames, num_processes)
    print("chunked_filenames",chunked_filenames)

    processes = []
    start_time = time.time()
    chunk_size = 4097 * 1024
    for i,filename in enumerate(chunked_filenames):
        print("iter filename",filename)
        p = Process(target=prepare_full, args=(filename, checkpoint_dir, destination_path, chunk_size, match,i))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")



if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)