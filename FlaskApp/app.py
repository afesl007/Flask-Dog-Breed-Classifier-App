from flask import Flask, redirect, render_template, request, session, url_for
from flask_dropzone import Dropzone
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class

import os

app = Flask(__name__)
dropzone = Dropzone(app)


app.config['SECRET_KEY'] = 'supersecretkeygoeshere'

# Dropzone settings
app.config['DROPZONE_UPLOAD_MULTIPLE'] = True
app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image/*'
app.config['DROPZONE_REDIRECT_VIEW'] = 'results'

# Uploads settings
app.config['UPLOADED_PHOTOS_DEST'] = os.getcwd() + '/uploads'

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)  # set maximum file size, default is 16MB

breed = "global"

@app.route('/', methods=['GET', 'POST'])
def index():
    
    # set session for image results
    if "file_urls" not in session:
        session['file_urls'] = []
    # list to hold our uploaded image urls
    file_urls = session['file_urls']

    # handle image upload from Dropszone
    if request.method == 'POST':
        file_obj = request.files
        for f in file_obj:
            file = request.files.get(f)
            
            # save the file with to our photos folder
            filename = photos.save(
                file,
                name=file.filename    
            )

            breed = run_app(file)
            session['key'] = breed
            # append image urls
            file_urls.append(photos.url(filename))
            
            
        session['file_urls'] = file_urls
        return "uploading..."
    # return dropzone template on GET request    
    return render_template('index.html')


@app.route('/results')
def results():
    
    # redirect to home if no images to display
    if "file_urls" not in session or session['file_urls'] == []:
        
        return redirect(url_for('index'))
        
    # set the file_urls and remove the session variable
    file_urls = session['file_urls']
    #test = ''.join(file_urls)
   
    #breed = run_app(url_for('index'))
    
    
    session.pop('file_urls', None)
    name = request.args.get("name", "World")
    session_var_value = session.get('key')
    return render_template('results.html', file_urls=file_urls, message=session_var_value)


import cv2                
import matplotlib.pyplot as plt                        



def run_app(img_path):
    ## handle cases for a human face, dog, and neither
    
    if dog_detector(img_path):
        dog_breed, idx = predict_breed_transfer(img_path)
        return dog_breed
        
       
    elif face_detector(img_path):
        human_breed, idx = predict_breed_transfer(img_path)
        
        return human_breed
        
    else:
        return "Nothing"


import numpy as np
from glob import glob
import torchvision.transforms as transforms

def face_detector(img):
    #img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

from PIL import Image

def load_image(image):
    ''' Load in and transform an image, making sure the image
       is <= 400 pixels in the x-y dims.'''
    
    #image = torch.from_numpy(image).type(torch.FloatTensor) 
    image = Image.open(image).convert('RGB')
    
    #image = transforms.ToPILImage()(image)
    #image = cv2.imread(image,1)
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        
    data_transform = transforms.Compose([transforms.RandomResizedCrop(224), 
                                         transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize])

    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = data_transform(image).unsqueeze(0)
    
    
    return image


def VGG16_predict(img_path):
    '''
    Use pre-trained VGG-16 model to obtain index corresponding to 
    predicted ImageNet class for image at specified path
    
    Args:
        img_path: path to an image
        
    Returns:
        Index corresponding to VGG-16 model's prediction
    '''
    
    ## TODO: Complete the function.
    ## Load and pre-process an image from the given img_path
    
    pre_img = load_image(img_path)
    
    
    ## Return the *index* of the predicted class for that image
    VGG16 = models.vgg16(pretrained=True)
    VGG16 = VGG16.cpu()
        
    output = VGG16(pre_img)
    ps = torch.exp(output)
    top_p, top_class = ps.topk(1, dim=1)

    idx = top_class.data.cpu().numpy()[0]
    idx = idx.item(0)
    
    return idx

def dog_detector(img):
    ## TODO: Complete the function.
    #img = cv2.imread(img)
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    dog = VGG16_predict(img)
    #print(dog)
    if (dog >= 151) & (dog <= 268):
        return True
    else:
        return False

def imshow(img):
    inputp = np.transpose(img, (1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inputp = std * inputp + mean
    inputp = np.clip(inputp, 0, 1)
    plt.imshow(inputp)

import torch
from torchvision import datasets
#data_dir = 'D:/Kurs/Flask/flaskr/flask-multiple-file-upload-master/flask-multiple-file-upload-master/dogImages'
data_dir = './dogImages/'
train_dir = os.path.join(data_dir, 'train/')
#train_data = datasets.ImageFolder(train_dir, transform=None)

from PIL import Image
import torchvision.models as models
import torch.nn as nn


model_transfer = models.resnet18(pretrained=True)


for param in model_transfer.parameters():
    param.requires_grad = False
    
    
model_transfer.fc = nn.Linear(512, 133, bias=True)    
fc_parameters = model_transfer.fc.parameters()

for param in fc_parameters:
    param.requires_grad = True

model_transfer.load_state_dict(torch.load('model_transfer.pt', map_location=torch.device('cpu')))
#model_transfer.load_state_dict(torch.load('model_transfer.pt'))
#model_transfer.load_state_dict(torch.load('model_transfer.pt',map_location=lambda storage, loc: storage))

def fast_scandir(dir):
    subfolders= [f.path for f in os.scandir(dir) if f.is_dir()]
    for dir in list(subfolders):
        subfolders.extend(fast_scandir(dir))
    return subfolders
class_names = fast_scandir(train_dir)
class_names = [item[4:].replace("_", " ") for item in class_names]


def predict_breed_transfer(img_path):
    # load the image and return the predicted breed
    
    #pre_img = load_image(img_path)
    #output = model_transfer(pre_img)
    #ps = torch.exp(output)
    #top_p, top_class = ps.topk(1, dim=1)
    
    #idx = top_class.data.cpu().numpy()[0]
    #idx = idx.item(0)
    image = Image.open(img_path).convert('RGB')
    prediction_transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                     transforms.ToTensor(), 
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = prediction_transform(image)[:3,:,:].unsqueeze(0)
    #image = image.cuda()
    
    
    
    model_transfer.eval()
    idx = torch.argmax(model_transfer(image))
    
    return class_names[idx], idx