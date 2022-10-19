from __future__ import print_function

from flask import Flask
app = Flask(__name__)

from PIL import Image

import torch

from torchvision import models
from torchvision import transforms

import torch.utils.data
from torch import nn
import torchvision
from torchvision import transforms
from PIL import Image



@app.route('/')
def hello():

    checkpoint = '/Users/gqlxj1987/workSpace/nsfw-resnet/checkpoint/model_0_200.pth'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transformation = transforms.Compose([
                        transforms.Resize((224, 224)),
                        #transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,])
    '''

    transformation = transforms.Compose([
                        transforms.Resize(224),
                        transforms.ToTensor(),
                        normalize,])
    '''

    classes = torch.load(checkpoint)['classes']
    model = models.resnet101(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(classes))

    model = nn.DataParallel(model, device_ids=[0])
    device = torch.device('cpu')
    model.to(device)
    model.load_state_dict(torch.load(checkpoint)['model'])
    model.eval()

    test_path = '/Users/gqlxj1987/workSpace/nsfw-resnet/data/300/sticker_wa_0084caeee68e8500.webp'

    image_tensor = transformation(Image.open(test_path).convert('RGB')).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = image_tensor.to(device)
    # input = image_tensor.cuda()
    output = model(input)

    index = output.data.cpu().numpy().argmax()
    #print(output.data.cpu().numpy())
    # label = classes[index]
    print('{}\t{}\t{}'.format(test_path, classes[index], index))
    
    return classes[index]

@app.route('/predict', methods=['POST'])
def predict():
    return 'Hello World!'