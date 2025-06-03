import os
import requests

def download_if_missing(local_path, url):
    """
    Download a file from a URL if it does not exist at the local path.
    """
    if os.path.isfile(local_path):
        return
    print(f"File not found: {local_path}\nDownloading from {url} ...")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"Downloaded to {local_path}")
