import os
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import torch
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image

# Import your model and loss function
from net.resnet_multi_view import ResNet_GCN_two_views
# from loss.loss_multi_view_final import MultiView_all_loss

app = Flask(__name__)


# Configuration
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def transfrom_img_test(img):
    img = transform_test(img)
    return img

# Main page with upload form
@app.route('/')
def upload_file():
    return render_template('index.html')

# Handling the file upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Open and preprocess the image
        img = Image.open(filepath).convert('RGB')
        img = transfrom_img_test(img)

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

        # Define Action Unit names
        AU_names = {
            1: "1. Inner Brow Raiser AU1",
            2: "2. Outer Brow Raiser AU2",
            3: "3. Brow Lowerer AU4",
            4: "4. Upper Lid Raiser AU5",
            5: "5. Cheek Raiser AU6",
            6: "6. Nose Wrinkler AU9",
            7: "7. Lip Corner Puller AU12",
            8: "8. Chin Raiser AU17",
            9: "9. Lip Stretcher AU20",
            10: "10. Lips part AU25",
            11: "11. Jaw Drop AU26",
            12: "12. Eyes Closed AU43"
        }

        # Create a response
        result = {
            'AU_view1': {AU_names[i+1]: AU_view1[0][i] for i in range(len(AU_view1[0]))},
            'AU_view2': {AU_names[i+1]: AU_view2[0][i] for i in range(len(AU_view2[0]))},
            'AU_fusion': AU_fusion.tolist()
        }

        print(AU_fusion)

        # format decimal places:
        fmt_AU_view1 = {AU_name: f"{value:.8f}" for AU_name, value in result['AU_view1'].items()}
        fmt_AU_view2 = {AU_name: f"{value:.8f}" for AU_name, value in result['AU_view2'].items()}

        # Update the result to formatted values
        result['AU_view1'] = fmt_AU_view1
        result['AU_view2'] = fmt_AU_view2

        return render_template('result.html', result=result, filename=filename)

# Serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    # return redirect(url_for('static', filename=f'uploads/{filename}'))
    return send_from_directory('uploads', filename)


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
