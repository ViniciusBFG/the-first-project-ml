# Importa funções para dividir datasets e criar loaders para batches
from torch.utils.data import random_split, DataLoader
# Importa datasets e ferramentas para transformação de imagens
from torchvision import datasets, transforms
# Manipulação de diretórios e arquivos
import os
# Manipulação de imagens
from PIL import Image
# Visualização de imagens
import matplotlib.pyplot as plt
# Biblioteca principal para cálculos com tensores
import torch
# Ferramentas para criar e treinar redes neurais
import torch.nn as nn
# Ferramentas de otimização
import torch.optim as optim

# Define o diretório base do dataset
base_folder = "/content/dataset/PetImages"
# Define as categorias do dataset
categories = ["Dog", "Cat"]

# Loop para exibir as primeiras imagens de cada categoria
for category in categories:
    # Define o caminho da pasta da categoria
    folder_path = os.path.join(base_folder, category)
    # Lista os arquivos de imagem na pasta
    image_files = [f for f in os.listdir(
        folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    print(f"Mostrando imagens da categoria: {category}")

    # Loop para mostrar até 5 imagens
    for img_file in image_files[:5]:
        img_path = os.path.join(folder_path, img_file)
        try:
            # Abre a imagem no modo RGB
            img = Image.open(img_path).convert("RGB")
            # Exibe a imagem com título
            plt.imshow(img)
            plt.title(f"Categoria: {category}")
            plt.axis('off')  # Remove os eixos
            plt.show()
        except Exception as e:
            print(f"Erro ao carregar imagem {img_path}: {e}")

# Classe para dataset que ignora imagens corrompidas


class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        while True:  # Tenta carregar a imagem até conseguir
            try:
                # Caminho e rótulo da imagem
                path, target = self.samples[index]
                sample = Image.open(path).convert(
                    "RGB")  # Carrega a imagem como RGB
                if self.transform is not None:  # Aplica as transformações se existirem
                    sample = self.transform(sample)
                return sample, target  # Retorna imagem e rótulo
            except Exception as e:
                print(f"Erro ao carregar imagem: {
                      self.samples[index]}. Ignorando...")
                # Vai para a próxima imagem
                index = (index + 1) % len(self.samples)


# Define as transformações a serem aplicadas nas imagens
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Redimensiona para 64x64
    transforms.ToTensor()  # Converte para tensor
])

# Cria o dataset usando a classe segura
data_dir = "/content/dataset/PetImages"
safe_dataset = SafeImageFolder(data_dir, transform=transform)

# Divide o dataset em treino (80%) e validação (20%)
train_size = int(0.8 * len(safe_dataset))
val_size = len(safe_dataset) - train_size
train_dataset, val_dataset = random_split(safe_dataset, [train_size, val_size])

# Cria DataLoaders para carregar os datasets em batches
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Exibe o número de imagens nos conjuntos de treino e validação
print(f"Número de imagens de treino: {len(train_dataset)}")
print(f"Número de imagens de validação: {len(val_dataset)}")

# Define uma rede neural simples com uma camada convolucional e uma totalmente conectada


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Camada convolucional
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        # Pooling para reduzir a dimensão
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Camada totalmente conectada (classificação)
        self.fc1 = nn.Linear(16 * 32 * 32, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # Aplica ReLU e pooling
        # Redimensiona para entrada na camada totalmente conectada
        x = x.view(-1, 16 * 32 * 32)
        x = self.fc1(x)  # Saída final
        return x


# Instancia o modelo
model = SimpleCNN()
# Define a função de perda
criterion = nn.CrossEntropyLoss()
# Define o otimizador (Adam)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Loop de treinamento
for epoch in range(5):  # Número de épocas
    model.train()  # Coloca o modelo em modo de treinamento
    running_loss = 0.0

    for inputs, labels in train_loader:  # Loop pelos batches de treino
        try:
            optimizer.zero_grad()  # Zera os gradientes acumulados
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Calcula a perda
            loss.backward()  # Backward pass (cálculo do gradiente)
            optimizer.step()  # Atualiza os pesos
            running_loss += loss.item()  # Acumula a perda
        except Exception as e:
            print(f"Erro ao processar um batch. Ignorando... Detalhes: {e}")
            continue

    # Exibe a perda média por época
    print(f"Época {epoch+1}, Perda média: {running_loss/len(train_loader):.4f}")
