# Importa funções para dividir datasets e criar dataloaders
from torch.utils.data import random_split, DataLoader
# Importa ferramentas para manipular datasets e aplicar transformações
from torchvision import datasets, transforms
# Importa módulo para manipulação de arquivos e diretórios
import os
# Biblioteca para manipulação de imagens
from PIL import Image
# Biblioteca para visualização de gráficos e imagens
import matplotlib.pyplot as plt

# Define o diretório base onde estão as imagens
base_folder = "/content/train"
# Define as categorias de imagens
categories = ["dog", "cat"]

for category in categories:                           # Itera sobre cada categoria
    # Cria o caminho completo para a pasta da categoria
    folder_path = os.path.join(base_folder, category)
    image_files = [f for f in os.listdir(             # Lista arquivos na pasta que são imagens
        folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # Exibe o nome da categoria
    print(f"Mostrando imagens da categoria: {category}")

    # Mostra as 5 primeiras imagens de cada categoria
    # Itera sobre os 5 primeiros arquivos de imagem
    for img_file in image_files[:5]:
        # Cria o caminho completo da imagem
        img_path = os.path.join(folder_path, img_file)
        img = Image.open(img_path)                     # Abre a imagem
        plt.imshow(img)                                # Exibe a imagem
        # Adiciona o título com o nome da categoria
        plt.title(f"Categoria: {category}")
        # Remove os eixos da imagem
        plt.axis('off')
        plt.show()                                     # Mostra a imagem

# Transformações para redimensionar e converter as imagens em tensores
transform = transforms.Compose([                      # Define a sequência de transformações
    # Redimensiona as imagens para 64x64
    transforms.Resize((64, 64)),
    # Converte as imagens para tensores
    transforms.ToTensor()
])

# Carregar o dataset com transformações
# Diretório onde está o dataset
data_dir = "/content/train"
# Carrega o dataset e aplica as transformações
dataset = datasets.ImageFolder(data_dir, transform=transform)

# Dividir em treino e validação
# Calcula o tamanho do conjunto de treino (80%)
train_size = int(0.8 * len(dataset))
# Calcula o tamanho do conjunto de validação (restante)
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(           # Divide o dataset em treino e validação
    dataset, [train_size, val_size]
)

# Criar dataloaders
train_loader = DataLoader(                           # Cria o dataloader para o conjunto de treino
    # Define o tamanho do lote e embaralhamento
    train_dataset, batch_size=32, shuffle=True
)
val_loader = DataLoader(                             # Cria o dataloader para o conjunto de validação
    # Define o tamanho do lote sem embaralhamento
    val_dataset, batch_size=32
)

# Exibe o número de imagens no conjunto de treino
print(f"Número de imagens de treino: {len(train_dataset)}")
# Exibe o número de imagens no conjunto de validação
print(f"Número de imagens de validação: {len(val_dataset)}")
