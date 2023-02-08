
import json
from clip_retrieval.clip_client import ClipClient, Modality
import os
import pathlib
from pebble import ProcessPool, ThreadPool
from concurrent.futures import TimeoutError
import sys
import colorsys
import traceback
import os
import cv2
import random
import re
import unicodedata
import numpy as np
import requests
import io
import time
import math
from PIL import Image
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import multiprocessing
import subprocess
from duckduckgo_images_api import search as ddg_search_image_api
  
file_path = os.path.abspath(os.path.dirname(__file__))
prompts = []

with open(os.path.join(file_path, "lora_words.txt"), 'r') as f:
    lines = f.readlines()
    for line in lines:
        if len(line) > 0:
            prompts.append(line.strip())
  
def estimate_noise(img_tensor):
  greyscaler = torchvision.transforms.Grayscale()
  img_tensor = greyscaler(img_tensor)
  W, H = img_tensor.squeeze().shape
  K = torch.tensor(
      [
          [ 1, -2,  1],
          [-2,  4, -2],
          [ 1, -2,  1]
      ], 
      device = img_tensor.device,
      dtype = torch.float32
  ).unsqueeze(0).unsqueeze(0)
  torch_conv = F.conv2d(img_tensor, K, bias=None, stride=(1, 1), padding=1) 
  sigma = torch.sum(torch.abs(torch_conv))
  sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W - 2) * (H - 2))
  return sigma

def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()

def image_filter(img: Image, skip = []) -> str:
  if not 'ratio' in skip:
    ratio = min(img.size) / max(img.size)
    if ratio < 0.6:
      return f"Ratio too far from 1.0! (ratio: {ratio})"
  if not 'size' in skip:
    min_size = 128
    if img.size[0] < min_size or img.size[1] < min_size:
      return f"Size too small! ({img.size[0]}x{img.size[1]})"
  sigma = estimate_noise(TF.to_tensor(img))
  if sigma > 0.1:
    return f"Skipped as image is too noisy! (sigma: {sigma})"
  if not 'lapvar' in skip:
    lapvar = variance_of_laplacian(np.array(img)[:, :, ::-1])
    if lapvar < 50:
      return f"Skipped as image is too blurry! (lapvar: {lapvar})"
  return None

def task_done(future):
    try:
        result = future.result()  # blocks until results are ready
    except TimeoutError as error:
        print("Function took longer than %d seconds" % error.args[1])
    except Exception as error:
        print("Function raised %s" % error)
        print(error.traceback)  # traceback of the function

def download_image_from_row_worker(prompt: str, row, count: int, images_folder, headers, max_images: int):
    try:
        amount_of_images_in_folder = len(os.listdir(images_folder))
        if amount_of_images_in_folder >= max_images:
          return
        image_name = f"{prompt}_{count}"
        req = requests.get(row['url'], headers=headers)
        with Image.open(io.BytesIO(req.content)) as img:
          img = img.convert('RGB')
          filter_result = image_filter(img)
          if filter_result != None:
            print(f"Image filtered: {filter_result}. [{row['url']}]")
            return
          min_size = min(img.size[0], img.size[1])
          max_size = (2048, 2048)
          img.thumbnail(max_size, Image.Resampling.LANCZOS)
          image_path = os.path.join(images_folder, f"{image_name}.jpg")
          img.save(image_path)
          print(f"Saved {image_path}")
    except KeyboardInterrupt:
        print('KeyboardInterrupt exception is caught, stopping')
        return

def slugify(value):
    """
    Converts to lowercase, removes non-word characters (alphanumerics and
    underscores) and converts spaces to hyphens. Also strips leading and
    trailing whitespace.
    """
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub('[^\w\s-]', '', value).strip().lower()
    return re.sub('[-\s]+', '-', value)

def clip_search(query):
  client = ClipClient(url="https://knn5.laion.ai/knn-service", indice_name="laion5B", num_images=100, aesthetic_weight=0.2)
  result = client.query(text=query)

def ddg_search(query):
  result = []
  ddg_result = ddg_search_image_api(query, max_results=100)
  for item in ddg_result["results"]:
    result.append({
      "url": item["image"]
    })
  return result

def trim_images(images_folder: str, max_images: int):
  images_list = os.listdir(images_folder)
  if len(images_list) > max_images:  
    images_list.sort()
    for image_name in images_list[:len(images_list) - max_images]:
      image_path = os.path.join(images_folder, image_name)
      os.remove(image_path)
      print(f"Trimmed {image_path}")
      
def download_images(prompt, images_folder):
  max_images = 10
  images_folder = os.path.join(images_folder, slugify(prompt), "images")
  if os.path.exists(images_folder):
    print(f"Skipping: {images_folder} as it already exists (but will trim).")
    trim_images(images_folder, max_images)
    return

  result = None
  while result == None:
    try:
      # result = clip_search(prompt)
      result = ddg_search(prompt)
    except:
      print("CLIP search error (is it offline?) sleeping for 5 seconds then trying again...")
      traceback.print_exc()
      time.sleep(5)
  
  result = list(filter(lambda item: item['url'].endswith(".png") or item['url'].endswith(".jpg") or item['url'].endswith(".webp"), result))
  os.makedirs(images_folder)
  
  print(f"Making training database for {prompt}. {len(result)} candidates")
  headers = {
      'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36',
  }
  count = 0
  with ProcessPool(max_workers=4, max_tasks=len(result)) as pool:
    try:
      for row in result:
        future = pool.schedule(download_image_from_row_worker, args=(prompt, row, count, images_folder, headers, max_images), timeout=60)
        future.add_done_callback(task_done)
        count += 1
    except KeyboardInterrupt:
      print("Keyboard interrupt, closing pool")
      pool.close()
      pool.stop()
  trim_images(images_folder, max_images)

def main():
    images_folder = os.path.join(file_path, "lora_dataset")
    for prompt in prompts:
        download_images(prompt, images_folder)

if __name__ == "__main__":
    main()