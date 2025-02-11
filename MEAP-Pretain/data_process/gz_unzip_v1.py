import os
import gzip
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

lock = Lock()

def decompress_and_count_json(file_path, folder_path):
    output_file = file_path.rstrip('.gz')  
    try:
        
        with gzip.open(file_path, 'rb') as gz_file:
            with open(output_file, 'wb') as out_file:
                shutil.copyfileobj(gz_file, out_file)
        
        json_count = count_json_files(folder_path)
        
        with lock:  
            print(f"Extraction completed successfully.: {file_path} -> {output_file}")
            print(f" {json_count} .json 。")
    except Exception as e:
        with lock:
            print(f"Extra fail: {file_path}, error: {e}")

def count_json_files(folder_path):
    return sum(1 for f in os.listdir(folder_path) if f.endswith('.json'))

def decompress_all_and_track_json_multithreaded(folder_path, max_workers=4):
    gz_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.gz')]
    
    if not gz_files:
        print("not found .gz")
        return

    print(f"start {len(gz_files)}  ....")
    

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(decompress_and_count_json, file_path, folder_path) for file_path in gz_files]
        for future in as_completed(futures):
            future.result()  
    
    print("all sucessful ！")

if __name__ == "__main__":
    folder_path = input("input file folder: ")
    if os.path.isdir(folder_path):
        decompress_all_and_track_json_multithreaded(folder_path)
    else:
        print("input file path wrong,please input again")

