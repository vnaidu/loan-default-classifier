import os
import shutil
import fnmatch
import requests
import hashlib
import zipfile
import json
import box

def file_hash_sha1(filename):
   """"Function returns the hex representation of the SHA-1 hash
   of a file"""

   # make a hash object
   h = hashlib.sha1()

   # open file for reading in binary mode
   with open(filename,'rb') as file:

       # loop till the end of the file
       chunk = 0
       while chunk != b'':
           # read only 1024 bytes at a time
           chunk = file.read(1024)
           h.update(chunk)

   # return the hex representation of digest
   return h.hexdigest()

def is_valid_file(filename, valid_hash, hash_type='SHA-1'):
    if valid_hash is not None and os.path.isfile(filename):
        if hash_type == 'SHA-1':
            file_hash = file_hash_sha1(filename)
            return file_hash == valid_hash
    else:
        return False

def is_downloadable(url):
    """
    Does the url contain a downloadable resource
    """
    h = requests.head(url, allow_redirects=True)
    header = h.headers
    content_type = header.get('content-type')
    if 'text' in content_type.lower():
        return False
    if 'html' in content_type.lower():
        return False
    return True


def extract_zip(filename, project_dir, zip_dir='data/external', extract_dir='data/raw'):
    zipfile_path = os.path.join(project_dir, zip_dir, filename)
    dest_dir = os.path.join(project_dir, extract_dir)
    if os.path.isfile(zipfile_path):
        with open(zipfile_path, 'rb') as f:
            print(" Unzipping data:", zipfile_path)
            z = zipfile.ZipFile(f)
            for name in z.namelist():
                print("    Extracting file", name)
                z.extract(name, dest_dir)
        return True
    else:
        return False

def download_file(url, filepath, valid_hash=None):
    if is_downloadable(url):
        r = requests.get(url, allow_redirects=True)
        open(filepath, 'wb').write(r.content)
        if valid_hash is not None:
            return is_valid_file(filepath, valid_hash)
        else:
            return os.path.isfile(filepath)
    else:
        return False

def validate_data_dirs(data_hash, project_dir='.'):
    valid_nfo = {dirname: True for dirname in data_hash.keys()}
    for dirname, nfo_dict in data_hash.items():
        for filename, filehash in nfo_dict.items():
            filepath = os.path.join(project_dir, 'data', dirname, filename)
            if not is_valid_file(filepath, filehash):
                valid_nfo[dirname] = False
    return valid_nfo

def load_json_box(filename):
    with open(filename) as json_file:
        data = json.load(json_file)
    return box.SBox(data, default_box=True, default_box_attr=None)
