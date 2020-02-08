#!/usr/bin/env python
# coding: utf-8

from bs4 import BeautifulSoup
import os
import sys
import time
import requests

import threading
from queue import Queue

DIRECTORY = 'moma/images'
TOTAL_PAGES = 1000000 # Unclear what the limit is, getting 404s/no images is cheap enough though to brute-force
if not os.path.exists(DIRECTORY): os.makedirs(DIRECTORY)

lock = threading.Lock()
q = Queue()


def save(url):
    data = requests.get(url).content
    name = url.split('?sha=')[-1] # SHA as name
    file = f'{DIRECTORY}/{name}.jpg'
    with open(file, 'wb') as f:
        f.write(data)

        
def worker():
    with lock:
        print(f'{threading.current_thread().name} started')
    
    while True:
        error = None
        
        page = q.get() 
        url = f'https://www.moma.org/collection/works/{page}'   
        try:
            response = requests.get(url)
        except:
            with lock:
                print(f'E1: No response for page {page}')
            continue
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            imgs = soup.findAll('img', 'picture__img--scale--focusable')
            if len(imgs) == 1:
                src = imgs[0].get('src')
                save(f'https://www.moma.org{src}')
            else:
                with lock:
                    print(f'E2: {len(imgs)} links on page {page}')
        else: 
            with lock:
                print(f'E3: Response {response.status_code} for page {page}')
            
        q.task_done()
    
    with lock:
        print(f'{threading.current_thread().name} finished')
    driver.quit()

    
def main():
    n = 250
    for i in range(n):
        t = threading.Thread(target=worker)
        t.daemon = True  # thread dies when main thread (only non-daemon thread) exits.
        t.start()
        time.sleep(0.2)
    for page in range(200000, TOTAL_PAGES + 1):
        q.put(page)
    q.join()

    
if __name__ == "__main__":
    main()
