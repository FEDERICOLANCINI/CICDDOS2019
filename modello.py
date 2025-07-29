import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import torch.nn.functional as F
import pdb
from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from sklearn.preprocessing import StandardScaler

def compute_output_size(height, width, conv_params, pool_params):
    """
    Calcola le dimensioni dell'output dopo i layer convolutivi e di pooling alternati.

    Args:
        height (int): Altezza dell'immagine in input.
        width (int): Larghezza dell'immagine in input.
        conv_params (list of tuples): Parametri dei layer convolutivi.
            Ogni tupla deve contenere (kernel_size, stride, padding).
            pool_params (list of tuples): Parametri dei layer di max pooling.
            Ogni tupla deve contenere (kernel_size, stride).

    Returns:
        tuple: Una tupla contenente le dimensioni dell'output (output_height, output_width).
    """
    for i in range(len(conv_params)):
        # Calcolo delle dimensioni dell'output dopo il layer convolutivo
        kernel_size, stride, padding = conv_params[i]
        height = ((height + 2 * padding - kernel_size) // stride) + 1
        width = ((width + 2 * padding - kernel_size) // stride) + 1

        if i < len(pool_params):
            # Se ci sono layer di pooling successivi, calcola le dimensioni dell'output dopo il layer di max pooling
            kernel_size, stride = pool_params[i]
            height = (height - kernel_size) // stride + 1
            width = (width - kernel_size) // stride + 1

    return height, width


def calculate_metrics(predictions, targets):
    predictions = predictions.cpu().detach().numpy()  # Applica la sigmoide per ottenere probabilità
    predictions = (predictions > 0.5).astype(int)  # Applica una soglia di 0.5 e converte in 0 o 1
    targets = targets.cpu().detach().numpy()  # Converte le etichette in un array NumPy
    accuracy = accuracy_score(targets, predictions)  # Calcola l'accuratezza
    return accuracy

class CNN(nn.Module):
    def __init__(self,
                 lunghezza_vettore=46,  # lunghezza del vettore
                 canali_input=1,  # numero di canali di input
                 num_classi=1,  # numero di classi
                 ):
        super(CNN, self).__init__()

        self.lunghezza_vettore = lunghezza_vettore
        self.colonne_encoding = 20
        self.canali_input = canali_input
        self.num_classi = num_classi
        
        self.encoding_row = nn.Parameter(torch.randn(1, self.colonne_encoding))

        
        self.conv1 = nn.Conv2d(canali_input, 6, kernel_size=5, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=1)

        # Calcolo delle dimensioni di output dopo le convoluzioni
        altezza_immagine = lunghezza_vettore
        larghezza_immagine = self.colonne_encoding
        conv_params = [(5, 1, 1), (5, 1, 1)]
        pool_params = []
        self.output_height, self.output_width = compute_output_size(altezza_immagine, larghezza_immagine, conv_params, pool_params)

        # Fully Connected layer
        self.fc = nn.Linear(480, num_classi)

    def forward(self, x):        
        batch_size = x.size(0)  # Prendi la dimensione del batch
        #print("dimensioni x iniziali:", x.shape)
        x = x.unsqueeze(1) 
        #print("dimensioni dopo unsqueeze:", x.shape)
        x = x.transpose(1, 2) # Trasposizione di ogni vettore del batch      
        #print("Dimensioni dell'input x dopo la trasposizione:", x.shape)
        encoding_matrix = self.encoding_row.repeat(batch_size, 1, 1)
        #print("dimensioni del vettore di encoding:", encoding_matrix.shape)
        x = torch.bmm(x, encoding_matrix)
        #print("Dimensioni dell'immagine dopo la moltiplicazione batch-wise:", x.shape)  # (batch_size, lunghezza_vettore, colonne_encoding)
        x = x.unsqueeze(1)  # (batch_size, 1, lunghezza_vettore, colonne_encoding)
        #print("dimensione dopo unsqueeze:", x.shape)

        # Passaggio attraverso i layer convoluzionali
        x = self.pool(F.relu(self.conv1(x)))  # Applica conv1 e poi max pooling
        #print("dimensioni dopo la prima conv:", x.shape)
        x = self.pool(F.relu(self.conv2(x)))  # Applica conv2 e poi max pooling

        #print("dimensioni dopo le convoluzioni:", x.shape)
        # Flatten per il fully connected
        #print("dimensioni prima del flattening:", x.shape)
        x = x.view(x.size(0), -1)

        #print("dimensioni dopo il flattening:", x.shape)
        #print("dimensioni previste:", self.output_height, self.output_width)
        
        # Fully connected
        x = self.fc(x)

        # Applica la sigmoid per la classificazione binaria
        x = torch.sigmoid(x)
        
        return x