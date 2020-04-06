import os
import numpy as np
from io import BytesIO
import PIL.Image 
from IPython import display
from scipy.spatial.distance import cdist
import scipy.ndimage
import lap # pip install lap
import umap
from math import floor, sqrt, ceil
import copy
import torch as t
import torchvision as tv
import torch.nn as nn
from tqdm import tqdm
import threading
from queue import Queue

def load_img(file):
    return PIL.Image.open(file).convert('RGB')

def from_device(tensor):
    return tensor.detach().cpu().numpy()

# Show an image within a Jupyter environment
# Can do PyTorch tensors, NumPy arrays, file paths, and PIL images
def show_img(img, fmt='jpeg', normalize=False):
    if type(img) is np.ndarray:
        img = PIL.Image.fromarray(img)
    elif type(img) is t.Tensor:
        img = _deprocess(img, normalize)
    elif type(img) is str or type(img) is np.str_:
        img = PIL.Image.open(img)
    out = BytesIO()
    img.save(out, fmt)
    display.display(display.Image(data=out.getvalue()))

# Save an image
# Can do PyTorch tensors, NumPy arrays, file paths, and PIL images
def save_img(img, filename, normalize=False):
    if type(img) is np.ndarray:
        img = PIL.Image.fromarray(img)
    elif type(img) is t.Tensor:
        img = _deprocess(img, normalize)
    elif type(img) is str or type(img) is np.str_:
        img = PIL.Image.open(img)
    img.save(filename)
    
# Reverse of preprocess, PyTorch tensor to PIL image
def _deprocess(tensor, normalize):
    # Clone tensor first, otherwise we are NOT making a copy by using .cpu()!
    img = t.clone(tensor)
    img = img.cpu().data.numpy().squeeze() # Get rid of batch dimension
    img = img.transpose((1, 2, 0)) # Channels first to channels last
    
    if normalize:
        mean = np.array([0.485, 0.456, 0.406]) 
        std = np.array([0.229, 0.224, 0.225]) 
        img = std * img + mean

    # 0./1. range to 0./255. range
    img *= 255
    
    img = img.astype(np.uint8)
    img = PIL.Image.fromarray(img)
    return img

def _smart_resize(img, thumb_size):
    max_dim = np.argmax(img.size)
    scale = thumb_size/img.size[max_dim]
    new_size = (int(img.size[0]*scale), int(img.size[1]*scale))
    img = img.resize(new_size, PIL.Image.ANTIALIAS)
    return img

def get_all_files(folder, extension=None):
    all_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if extension and file.endswith(extension) or extension is None:
                try: 
                    img = PIL.Image.open(os.path.join(root, file)) # If PIL can't open it we don't want it
                    all_files.append(f'{root}/{file}')
                except:
                    continue
    return all_files

def new_dir(folder):
    if not os.path.exists(folder): os.makedirs(folder)

def plot_features(files, features, num_datapoints=None, thumb_size=32, thumbs=None, num_workers=32, html=False, grid=False):
    
    if num_datapoints is None:
        assert features.shape[0] == len(files)
        num_datapoints = len(files)
    
    if grid:
        print('Computing grid')
        
        # https://gist.github.com/vmarkovtsev/74e3a973b19113047fdb6b252d741b42
        # https://github.com/gatagat/lap
    
        gs = floor(sqrt(features.shape[0]))
        samples = gs*gs # Determine number of data points to keep
        print(f'Grid size: {gs}x{gs}, samples: {samples}')
    
        # Cut excess data points
        files = files[:samples]
        features = features[:samples]

        # Make grid
        grid = np.dstack(np.meshgrid(np.linspace(0, 1, gs), np.linspace(0, 1, gs))).reshape(-1, 2)

        cost_matrix = cdist(grid, features, "sqeuclidean").astype(np.float32)
        cost, row_asses, col_asses = lap.lapjv(cost_matrix)
    
        features = grid[col_asses]

    if html:
        html_map = []

    # Generate thumbnails
    if thumbs is None:
        print('Generating thumbnails in parallel')
        thumbs = _thumbnails_parallel(files, thumb_size, num_workers)
    
    # Find max. and min. feature values
    value_max = np.max(features)
    value_min = np.min(features)
    
    # Determine max possible grid size
    gs = thumb_size * floor(sqrt(num_datapoints))
    
    # Calculate size of the plot based on these values
    canvas_size = int((abs(value_max) + abs(value_min)) * gs) + thumb_size # Images are anchored at upper left corner

    # Define plot as empty (white) canvas
    canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255

    print('Plotting image')
    for i, file in enumerate(files):
    
        img = thumbs[file]
    
        # Read features and calculate x,y
        y = int((features[i,0] + abs(value_min)) * gs)
        x = int((features[i,1] + abs(value_min)) * gs)   
        
        # Plot image
        canvas[y:y+img.shape[0],x:x+img.shape[1],:] = img
        
        if html:
            # Add to HTML map area list
            mapstring = f'<area shape="rect" coords="{x},{y},{x + img.shape[1]},{y + img.shape[0]}" href="{files[i]}" alt=""/>'
            html_map.append(mapstring)
    
    if html:
        # Return plot and HTML map area list
        print('Writing HTML map')
        return canvas, html_map, thumbs
    else:
        return canvas, thumbs

def _thumbnails_parallel(files, thumb_size, num_workers):
    d = dict()
    q = Queue()
    l = threading.Lock()
   
    def _worker(thumb_size, d):
        while True:
            file = q.get() 
            d[file] = np.array(_smart_resize(load_img(file), thumb_size))
            q.task_done()
            
    for i in range(num_workers):
        t = threading.Thread(target=_worker, args=(thumb_size,d,))
        t.daemon = True  # Thread dies when main thread (only non-daemon thread) exits.
        t.start()
        
    for file in files:
        q.put(file)
        
    q.join()
    return d

class UnsupervisedImageDataset(t.utils.data.Dataset):

    def __init__(self, folder, extension=None, transforms=None):
        self.folder = folder
        self.transforms = transforms
        self.files = get_all_files(folder, extension)
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if t.is_tensor(idx):
            idx = idx.tolist()
        sample = load_img(self.files[idx])
        if self.transforms:
            sample = self.transforms(sample)
        return sample

def train_model(model, dataloaders, criterion, optimizer, device, epochs):

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):
        
        print(f'Epoch {epoch}/{epochs-1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history if only in train
                with t.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = t.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += t.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    print(f'Best val acc: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model