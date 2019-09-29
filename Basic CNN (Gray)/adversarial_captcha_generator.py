from __future__ import print_function
import torch.nn as nn
import torch
from torch.autograd import Variable
import captcha_setting as setting
from captcha_cnn_model import CNN
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt


# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


epsilon = 0.05
learning_rate = 0.001
directory = 'adv'


def main():
    if os.path.exists(directory):
        shutil.rmtree(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cnn = CNN()
    # cnn.to(device)
    cnn.eval()
    cnn.load_state_dict(torch.load('model.pkl'))
    print("Model Loaded")

    criterion = nn.MultiLabelSoftMarginLoss()
    data_loader = setting.get_test_data_loader()

    correct = 0
    total = 0
    count = 1

    for i, (images, labels) in enumerate(data_loader):
        image = images
        variable = Variable(image)
        # variable = Variable(image).to(device)
        variable.requires_grad = True
        predict_label = cnn(variable)
        # predict_label = cnn(variable).to(device)

        loss = criterion(predict_label, labels.float())
        cnn.zero_grad()
        loss.backward()

        data_grad = variable.grad.data
        perturbed_image = fgsm_attack(variable, epsilon, data_grad)
        predict_label = cnn(perturbed_image)

        numpy_image = perturbed_image.squeeze().detach().cpu().numpy()
        plt.imshow(numpy_image)
        fig = plt.gcf()
        fig.savefig(directory + '/' + str(count) + '.png')
        count += 1

        c0 = setting.character_set[np.argmax(predict_label[0, 0:setting.character_set_length].data.numpy())]
        c1 = setting.character_set[np.argmax(predict_label[0, setting.character_set_length:2 * setting.character_set_length].data.numpy())]
        c2 = setting.character_set[np.argmax(predict_label[0, 2 * setting.character_set_length:3 * setting.character_set_length].data.numpy())]
        c3 = setting.character_set[np.argmax(predict_label[0, 3 * setting.character_set_length:4 * setting.character_set_length].data.numpy())]

        predict_label = '%s%s%s%s' % (c0, c1, c2, c3)
        true_label = setting.decode(labels.numpy()[0])
        total += labels.size(0)

        if predict_label == true_label:
            correct += 1
        if total % 20 == 0:
            print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))

    print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))


main()
