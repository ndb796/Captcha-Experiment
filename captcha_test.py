import numpy as np
import torch
from torch.autograd import Variable
import captcha_setting as setting
from captcha_cnn_model import CNN


def main():
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cnn = CNN()
    # cnn.to(device)
    cnn.eval()
    cnn.load_state_dict(torch.load('model.pkl'))
    print("Model Loaded")

    data_loader = setting.get_test_data_loader()
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(data_loader):
        image = images
        variable = Variable(image)
        # variable = Variable(image).to(device)
        predict_label = cnn(variable)
        # predict_label = cnn(variable).to(device)

        c0 = setting.character_set[np.argmax(predict_label[0, 0:setting.character_set_length].data.numpy())]
        c1 = setting.character_set[np.argmax(predict_label[0, setting.character_set_length:2 * setting.character_set_length].data.numpy())]
        c2 = setting.character_set[np.argmax(predict_label[0, 2 * setting.character_set_length:3 * setting.character_set_length].data.numpy())]
        c3 = setting.character_set[np.argmax(predict_label[0, 3 * setting.character_set_length:4 * setting.character_set_length].data.numpy())]

        predict_label = '%s%s%s%s' % (c0, c1, c2, c3)
        true_label = setting.decode(labels.numpy()[0])
        total += labels.size(0)

        if predict_label == true_label:
            correct += 1
        if total % 200 == 0:
            print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))

    print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))


main()
