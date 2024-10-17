import torch
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from net.resnet_multi_view import ResNet_GCN_two_views

# Configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Model parameters
AU_num = 12
AU_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
fusion_mode = 0
database = 0
use_web = 0
lambda_co_regularization = 100
lambda_multi_view = 400

model_path = '../model/EmotioNet_model.pth.tar'
net = ResNet_GCN_two_views(AU_num=AU_num, AU_idx=AU_idx, output=2, fusion_mode=fusion_mode, database=database)
temp = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
net.load_state_dict(temp['net'])
# net.cuda()
net.cpu()

# Define image transformations
transform_test = transforms.Compose([
    transforms.Resize(240),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5355, 0.4249, 0.3801), (0.2832, 0.2578, 0.2548)),
])

AU_values = {
        'AU1': [],
        'AU2': [],
        'AU4': [],
        'AU5': [],
        'AU6': [],
        'AU9': [],
        'AU12': [],
        'AU17': [],
        'AU20': [],
        'AU25': [],
        'AU26': [],
        'AU43': []
    }

def get_all_file_paths(directory):
    return [str(file) for file in Path(directory).rglob('*') if file.is_file()]

def predict_AU():
    folder_path = './image_dataset'
    all_files = get_all_file_paths(folder_path)

    for file in all_files:
        # Open and preprocess the image
        img = Image.open(file).convert('RGB')
        img = transform_test(img)

        # img = Variable(img).cuda()
        img = Variable(img).cpu()
        img = img.view(1, img.size(0), img.size(1), img.size(2))

        # Make predictions
        AU_view1, AU_view2, AU_fusion = net(img)
        AU_view1 = torch.sigmoid(AU_view1)
        AU_view2 = torch.sigmoid(AU_view2)

        # Convert to numpy for easier manipulation
        AU_view1 = AU_view1.cpu().detach().numpy()
        AU_view2 = AU_view2.cpu().detach().numpy()

        for index, key in zip(range(12), AU_values.keys()):
            mean_au = (AU_view1[0][index] + AU_view2[0][index]) / 2
            AU_values[key].append(mean_au)

    return AU_values


def correlation_analysis():
    predicted_aus = predict_AU()
    correlation_matrix = np.zeros((12, 12))

    for i, key_i in zip(range(12), AU_values.keys()):
        for j, key_j in zip(range(12), AU_values.keys()):
            correlation_matrix[i][j] = np.corrcoef(predicted_aus[key_i], predicted_aus[key_j])[0][1]

    return correlation_matrix


def visualize_correlation_matrix():
    # correlation_matrix = correlation_analysis()
    predicted_aus = predict_AU()
    AU_df = pd.DataFrame(predicted_aus)
    correlation_matrix = AU_df.corr()
    print(correlation_matrix)

    plt.figure(figsize=(12, 10))
    # Generate a heatmap
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
    # Customize the plot labels
    plt.title('Correlation Map of Different AUs')
    plt.show()


visualize_correlation_matrix()