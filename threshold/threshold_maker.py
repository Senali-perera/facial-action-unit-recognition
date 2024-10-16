import os
from werkzeug.utils import secure_filename
import torch
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
import numpy as np

from net.resnet_multi_view import ResNet_GCN_two_views

# Configuration
UPLOAD_FOLDER = './static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Model parameters
AU_num = 12
AU_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
fusion_mode = 0
database = 0
use_web = 0
lambda_co_regularization = 100
lambda_multi_view = 400

# Load the model
model_path = './model/EmotioNet_model.pth.tar'
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


from pathlib import Path

def get_all_file_paths(directory):
    return [str(file) for file in Path(directory).rglob('*') if file.is_file()]

folder_path = './threshold/image_dataset'
all_files = get_all_file_paths(folder_path)

AU1 = []
AU2 = []
AU3 = []
AU4 = []
AU5 = []
AU6 = []
AU7 = []
AU8 = []
AU9 = []
AU10 = []
AU11 = []
AU12 = []

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

    
    AU1.append(AU_view1[0][0])
    AU2.append(AU_view1[0][1])
    AU3.append(AU_view1[0][2])
    AU4.append(AU_view1[0][3])
    AU5.append(AU_view1[0][4])
    AU6.append(AU_view1[0][5])
    AU7.append(AU_view1[0][6])
    AU8.append(AU_view1[0][7])
    AU9.append(AU_view1[0][8])
    AU10.append(AU_view1[0][9])
    AU11.append(AU_view1[0][10])
    AU12.append(AU_view1[0][11])


    AU1.append(AU_view2[0][0])
    AU2.append(AU_view2[0][1])
    AU3.append(AU_view2[0][2])
    AU4.append(AU_view2[0][3])
    AU5.append(AU_view2[0][4])
    AU6.append(AU_view2[0][5])
    AU7.append(AU_view2[0][6])
    AU8.append(AU_view2[0][7])
    AU9.append(AU_view2[0][8])
    AU10.append(AU_view2[0][9])
    AU11.append(AU_view2[0][10])
    AU12.append(AU_view2[0][11])

print(np.mean(AU1), np.mean(AU2), np.mean(AU3))
print(np.mean(AU4),np.mean(AU5),np.mean(AU6),np.mean(AU7),np.mean(AU8),np.mean(AU9),np.mean(AU10),np.mean(AU11),np.mean(AU12))