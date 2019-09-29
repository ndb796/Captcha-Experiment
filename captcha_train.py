import torch
import torch.nn as nn
import captcha_setting as setting
from torch.autograd import Variable
from captcha_cnn_model import CNN


# Hyper Parameters
num_epochs = 30
batch_size = 100
learning_rate = 0.001


def main():
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cnn = CNN()
    # cnn.to(device)
    cnn.train()
    print('Model Initialization')
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

    # Train the Model
    train_data_loader = setting.get_train_data_loader()
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_data_loader):
            images = Variable(images)
            # images = Variable(images).to(device)
            labels = Variable(labels.float())
            # labels = Variable(labels.float()).to(device)
            predict_labels = cnn(images)
            loss = criterion(predict_labels, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 10 == 0:
                print("epoch:", epoch, "step:", i, "loss:", loss.item())
            if (i + 1) % 100 == 0:
                torch.save(cnn.state_dict(), "./model.pkl")
                print("Model Saved")
        print("epoch:", epoch, "step:", i, "loss:", loss.item())
    torch.save(cnn.state_dict(), "./model.pkl")
    print("Last Model Saved")


main()
