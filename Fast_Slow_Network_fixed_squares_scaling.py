import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributions import Categorical
import random


class SquareDataSet(Dataset):

    def __init__(self, size, line, random_sample, len_data_set, square_per):
        self.samples = []
        for i in range(len_data_set):
            if i <= len_data_set * square_per:
                x = (create_squares(size, line, random_sample), 1)
                self.samples.append(x)
            if len_data_set * square_per < i <= len_data_set * ((1 - square_per) / 2):
                x = (create_random(size, line, random_sample), 0)
                self.samples.append(x)
            if i > len_data_set * ((1 - square_per) / 2):
                x = (create_random_lines(size, line, random_sample), 0)
                self.samples.append(x)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class IndexSquareDataSet(Dataset):

    def __init__(self, size, line, random_sample, len_data_set, square_per):
        self.samples = []
        for i in range(len_data_set):
            if i <= len_data_set * square_per:
                image = image_to_index(create_squares(size, line, random_sample).unsqueeze(0))
                x = (image, 1)
                self.samples.append(x)
            if len_data_set * square_per < i <= len_data_set * ((1 - square_per) / 2):
                image = image_to_index(create_random(size, line, random_sample).unsqueeze(0))
                x = (image, 0)
                self.samples.append(x)
            if i > len_data_set * ((1 - square_per) / 2):
                image = image_to_index(create_random_lines(size, line, random_sample).unsqueeze(0))
                x = (image, 0)
                self.samples.append(x)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class IndexCornersDataSet(Dataset):

    def __init__(self, size=28, line=2, random_sample=False, len_data_set=100):
        self.samples = []
        for i in range(len_data_set):
            self.samples.append((image_to_index(create_corners(size, line, random_sample).unsqueeze(0)), 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def create_corners(size=28, line=2, random_sample=False):
    per = 70
    base = np.random.uniform(0.0, 0.0, (size, size))
    if random_sample is True:
        h_start = random.randint(1, int(size - 2 * line - 2))
        h_finish = random.randint(int(h_start + line + 1), int(size - line - 1))
        w_start = random.randint(1, int(size - 2 * line - 2))
        w_finish = random.randint(int(w_start + line + 1), int(size - line - 1))
    else:
        h_start = 4
        h_finish = 24
        w_start = 4
        w_finish = 24
    for i in range(w_start, w_finish + line):
        if i < w_start + line + 1 or i > w_finish - 2:
            for j in range(0, line):
                base[h_start + j, i] = 1.0
                base[h_finish + j, i] = 1.0
        else:
            a = random.randint(1, 100)
            b = random.randint(1, 100)
            for j in range(0, line):
                if a > per:
                    base[h_start + j, i] = 1.0
                if b > per:
                    base[h_finish + j, i] = 1.0
    for i in range(h_start, h_finish + line):
        if i < h_start + line + 1 or i > h_finish - 2:
            for j in range(0, line):
                base[i, w_start + j] = 1.0
                base[i, w_finish + j] = 1.0
        else:
            a = random.randint(1, 100)
            b = random.randint(1, 100)
            for j in range(0, line):
                if a > per:
                    base[i, w_start + j] = 1.0
                if b > per:
                    base[i, w_finish + j] = 1.0
    x = torch.from_numpy(base)
    x = x.float()
    return x


def create_random_lines(size=28, line=2, random_sample=False):
    base0 = np.random.uniform(0.0, 0.0, (size, size))
    base = np.random.uniform(0.0, 0.0, (size, size))
    if random_sample is True:
        h_start = random.randint(1, int(size - 2 * line - 2))
        h_finish = random.randint(int(h_start + line + 1), int(size - line - 1))
        w_start = random.randint(1, int(size - 2 * line - 2))
        w_finish = random.randint(int(w_start + line + 1), int(size - line - 1))
    else:
        h_start = 4
        h_finish = 24
        w_start = 4
        w_finish = 24
    for i in range(w_start, w_finish + line):
        for j in range(0, line):
            base[h_start + j, i] = 1.0
            base[h_finish + j, i] = 1.0
    for i in range(h_start, h_finish + line):
        for j in range(0, line):
            base[i, w_start + j] = 1.0
            base[i, w_finish + j] = 1.0
    count = 0
    for i in range(0, size):
        for j in range(0, size):
            if base[i, j] == 1.0:
                count += 1
    while count > 0:
        a = random.randint(5, 20)
        b = random.randint(0, 1)
        if b == 0:
            h0 = random.randint(0, size - a)
            w0 = random.randint(0, size - line)
            for i in range(0, a):
                for j in range(0, line):
                    if base0[h0 + i, w0 + j] == 0.0:
                        base0[h0 + i, w0 + j] = 1.0
                        count += -1
                        if count == 0:
                            x = np.array(base0)
                            x = torch.from_numpy(x)
                            x = x.float()
                            return x
                    else:
                        base0[h0 + i, w0 + j] = 1.0
        else:
            h0 = random.randint(0, size - line)
            w0 = random.randint(0, size - a)
            for i in range(0, a):
                for j in range(0, line):
                    if base0[h0 + j, w0 + i] == 0.0:
                        base0[h0 + j, w0 + i] = 1.0
                        count += -1
                        if count == 0:
                            x = torch.from_numpy(base0)
                            x = x.float()
                            return x
                    else:
                        base0[h0 + j, w0 + i] = 1.0
    x = torch.from_numpy(base0)
    x = x.float()
    return x


def create_random(size=28, line=2, random_sample=False):
    base0 = np.random.uniform(0.0, 0.0, (size, size))
    base = np.random.uniform(0.0, 0.0, (size, size))
    if random_sample is True:
        h_start = random.randint(1, int(size - 2 * line - 2))
        h_finish = random.randint(int(h_start + line + 1), int(size - line - 1))
        w_start = random.randint(1, int(size - 2 * line - 2))
        w_finish = random.randint(int(w_start + line + 1), int(size - line - 1))
    else:
        h_start = 4
        h_finish = 24
        w_start = 4
        w_finish = 24
    for i in range(w_start, w_finish + line):
        for j in range(0, line):
            base[h_start + j, i] = 1.0
            base[h_finish + j, i] = 1.0
    for i in range(h_start, h_finish + line):
        for j in range(0, line):
            base[i, w_start + j] = 1.0
            base[i, w_finish + j] = 1.0
    count = 0
    for i in range(0, size):
        for j in range(0, size):
            if base[i, j] == 1.0:
                count += 1
    while count > 0:
            h0 = random.randint(0, size - 1)
            w0 = random.randint(0, size - 1)
            if base0[h0, w0] == 0.0:
                base0[h0, w0] = 1.0
                count -= 1
                if count == 0:
                    x = torch.from_numpy(base0)
                    x = x.float()
                    return x
            else:
                base0[h0, w0] = 1.0


def create_squares(size=28, line=2, random_sample=False):
    base = np.random.uniform(0.0, 0.0, (size, size))
    if random_sample is True:
        h_start = random.randint(1, int(size - 2 * line - 2))
        h_finish = random.randint(int(h_start + line + 1), int(size - line - 1))
        w_start = random.randint(1, int(size - 2 * line - 2))
        w_finish = random.randint(int(w_start + line + 1), int(size - line - 1))
    else:
        h_start = 4
        h_finish = 24
        w_start = 4
        w_finish = 24
    for i in range(w_start, w_finish + line):
        for j in range(0, line):
            base[h_start + j, i] = 1.0
            base[h_finish + j, i] = 1.0
    for i in range(h_start, h_finish + line):
        for j in range(0, line):
            base[i, w_start + j] = 1.0
            base[i, w_finish + j] = 1.0
    x = torch.from_numpy(base)
    x = x.float()
    return x


def index_data_set_save_files(data_set, colab):
    if colab is True:
        torch.save(data_set, "/content/drive/My Drive/Colab Notebooks/fast_slow_network/"
                   "data_set_index_squares.pth.tar")
    else:
        torch.save(data_set, "content/drive/My Drive/Colab Notebooks/fast_slow_network/"
                   "data_set_index_squares.pth.tar")


def index_data_set_from_files(colab):
    if colab is True:
        data_set = torch.load("/content/drive/My Drive/Colab Notebooks/fast_slow_network/"
                              "data_set_index_squares.pth.tar")
    else:
        data_set = torch.load("content/drive/My Drive/Colab Notebooks/fast_slow_network/"
                              "data_set_index_squares.pth.tar")
    return data_set


def data_set_save_files(data_set, colab):
    if colab is True:
        torch.save(data_set, "/content/drive/My Drive/Colab Notebooks/fast_slow_network/"
                   "data_set_squares.pth.tar")
    else:
        torch.save(data_set, "content/drive/My Drive/Colab Notebooks/fast_slow_network/"
                   "data_set_squares.pth.tar")


def data_set_from_files(colab):
    if colab is True:
        data_set = torch.load("/content/drive/My Drive/Colab Notebooks/fast_slow_network/"
                              "data_set_squares.pth.tar")
    else:
        data_set = torch.load("content/drive/My Drive/Colab Notebooks/fast_slow_network/"
                              "data_set_squares.pth.tar")
    return data_set


def create_data_sets(len_data_set, show_image, random_sample, square_per):
    index_data_set = IndexSquareDataSet(size=28, line=2, random_sample=random_sample, len_data_set=len_data_set, square_per=square_per)
    if show_image:
        for i in range(10):
            a = random.randint(0, len_data_set)
            image = index_to_image(index_data_set[a][0].unsqueeze(0))
            show_image(image=image.unsqueeze(0),
                       mini_step=1, step=i, decision=index_data_set[a][1])
    print("index_data_set finished")
    index_data_set_save_files(index_data_set, False)
    print("index_data_set saved")


class FastNet(nn.Module):
    def __init__(self, categories):
        super(FastNet, self).__init__()
        self.fc1 = nn.Linear(3 * 784, int(10 * 4.3 * 4.3))
        self.fc2 = nn.Linear(int(10 * 4.3 * 4.3), int(10 * 4.3))
        self.fc3 = nn.Linear(int(10 * 4.3), categories)

    def forward(self, x):
        x = x.view(-1, 3 * 784).float()
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        output = x.float()
        return output

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True


class TopDownNet(nn.Module):
    def __init__(self, categories):
        super(TopDownNet, self).__init__()
        self.fc1 = nn.Linear(categories, int(10 * 4.3))
        self.fc2 = nn.Linear(int(10 * 4.3), int(10 * 4.3 * 4.3))
        self.fc3 = nn.Linear(int(10 * 4.3 * 4.3), 784 * 3)

    def forward(self, x):
        x = F.softmax(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = x.view(-1, 784, 3)
        x = x.float()
        return x


class SlowNet(nn.Module):

    def __init__(self, fast_net, top_down_net):
        super(SlowNet, self).__init__()
        self.fast = fast_net
        self.top_down = top_down_net

    def forward(self, x):

        output_fast = self.fast(x)
        output_top_down = self.top_down(output_fast)
        return output_fast, output_top_down

    def freeze_fast(self):
        for p in self.fast.parameters():
            p.requires_grad = False

    def unfreeze_fast(self):
        for p in self.fast.parameters():
            p.requires_grad = True


def image_to_index(image):
    image = image[0]
    index_image = []
    for i in range(image.size()[0]):
        for j in range(image.size()[1]):
            index_image.append([image[i, j], i, j])
    index_image = torch.tensor(index_image).float()
    index_image[:, 1:3] = index_image[:, 1:3] / 27
    index_image[:, 0] = index_image[:, 0]
    return index_image


def index_to_image(index_image):
    index_image = index_image[0]
    index_image[:, 1:3] = (index_image[:, 1:3]) * 27
    index_image[:, 0] = index_image[:, 0]
    image = np.zeros((28, 28))
    for index in index_image:
        if 0 <= index[1] <= 27 and 0 <= index[2] <= 27:
            image[int(index[1]), int(index[2])] = index[0]
    return image


def biased_loss(output_fast, output_check):
    # criterion = nn.CrossEntropyLoss()
    # loss = criterion(output_fast, torch.tensor([1]))
    criterion = nn.MSELoss()
    loss = criterion(output_fast, output_check)/1000
    return loss


def fast_net_from_files(fast_net):
    fast_net.load_state_dict(torch.load("content/drive/My Drive/Colab Notebooks/"
                                        "fast_slow_network/fast_net_parameters_random_squares.pt"))
    return fast_net


def fast_net_save_files(fast_net):
    torch.save(fast_net.state_dict(), "content/drive/My Drive/Colab Notebooks/fast_slow_network/"
               "fast_net_parameters_random_squares.pt")


def fast_optimizer_save_files(fast_optimizer):
    torch.save(fast_optimizer, "content/drive/My Drive/Colab Notebooks/fast_slow_network/"
               "fast_optimizer_parameters_random_squares.pth.tar")


def fast_optimizer_from_files():
    fast_optimizer = torch.load("content/drive/My Drive/Colab Notebooks/"
                                "fast_slow_network/fast_optimizer_parameters_random_squares.pth.tar")
    return fast_optimizer


def check_fast(slow_net, line):
    test_image = image_to_index(create_squares(line=line).unsqueeze(0))
    output_fast, __ = slow_net(test_image)
    output_fast_array = np.array(output_fast.detach())
    print("Check_" + str(output_fast_array))
    return output_fast


def save_image(image, mini_epochs_index, decision):
    image = image[0].detach()
    plt.imshow(image, cmap="gray")
    plt.savefig("content/drive/My Drive/Colab Notebooks/fast_slow_network/images/image_batch_"
                + str(mini_epochs_index) + "_decision_" + str(decision) + ".png")


def data_set_from_files(colab):
    if colab is True:
        data_set = torch.load("/content/drive/My Drive/Colab Notebooks/fast_slow_network/"
                              "data_set_index_squares.pth.tar")
    else:
        data_set = torch.load("content/drive/My Drive/Colab Notebooks/fast_slow_network/"
                              "data_set_index_squares.pth.tar")
    return data_set


def net_decision(output):
    return np.array(torch.argmax(output.detach()))


def criterion_output_loss(x):
    loss0 = (Categorical(probs=F.softmax(x[0])).entropy() - 1.192093048096*10**-7)
    loss = loss0
    return loss


def supervised_train(data_set, fast_net, num_epochs, print_values, lr_fast, show_image,
                     from_files, save_files, batch_size):
    print("Supervised Training")
    fast_net.unfreeze()
    fast_net.train()
    fast_net_optimizer = optim.RMSprop(fast_net.parameters(), lr=lr_fast)
    criterion = nn.CrossEntropyLoss()
    if from_files is True:
        print("Loading parameters")
        fast_net = fast_net_from_files(fast_net)
        fast_net_optimizer = fast_optimizer_from_files()
    for j in range(num_epochs):
        error_count = 0
        if print_values is True:
            print("Epoch No." + str(j + 1))
            data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)
        for batch_index, (image, target) in enumerate(data_loader):
            #  fast
            fast_net_optimizer.zero_grad()
            output_fast = fast_net(image)
            decision = np.array(torch.argmax(output_fast.detach()))
            loss_fast = criterion(output_fast, target)
            loss_fast.backward(retain_graph=True)
            fast_net_optimizer.step()
            if print_values is True:
                print("loss_fast_" + str(round(loss_fast.item(), 10)))
            if show_image is True:
                save_image(image=image, mini_epochs_index=batch_index, decision=decision)
            # restart_epoch = 5
            # if loss_fast > 0.6 and j > restart_epoch:
            #     print("restart net")
            #     fast_net = FastNet(2)
            #     fast_net_optimizer = optim.Adam(fast_net.parameters(), lr=lr_fast)
            #     j = 0
        if print_values:
            error = error_count/len(data_set)
            print("Error_" + str(error))
    if save_files is True:
        print("Saving parameters")
        fast_net_save_files(fast_net)
        fast_optimizer_save_files(fast_net_optimizer)
    return fast_net


def unsupervised_train_iterations(data_set, slow_net, line, num_epochs,
                                  editing_steps, lr, threshold, show_image, print_values):
    print("Unsupervised Training")
    check_fast(slow_net=slow_net, line=line)
    slow_net.freeze_fast()
    slow_net_optimizer = optim.Adam(filter(lambda p: p.requires_grad, slow_net.parameters()), lr=lr)
    slow_net.unfreeze_fast()
    for j in range(num_epochs):
        data_loader = DataLoader(data_set, batch_size=1, shuffle=True)
        for batch_index, (image, label) in enumerate(data_loader):
            mini_epochs_index = editing_steps
            output_fast, output_top_down = slow_net(image)
            loss_output = criterion_output_loss(output_fast)
            if show_image:
                a = np.array(image.detach())
                save_image(image=torch.tensor([index_to_image(a)]),
                           mini_epochs_index=mini_epochs_index+10, decision=1)
            while loss_output.item() > threshold and mini_epochs_index >= 0:
                slow_net.train()
                slow_net_optimizer.zero_grad()
                image = (image + output_top_down)/torch.max(image + output_top_down)
                output_fast, output_top_down = slow_net(image)
                loss_output = criterion_output_loss(output_fast)
                loss_output.backward(retain_graph=True)
                slow_net_optimizer.step()
                if print_values is True:
                    print("output_fast_" + str(np.array(output_fast.detach())))
                    print("LossOutput_" + str(np.array(loss_output.detach())))
                    print("max_" + str(torch.max(image[:, :, 0])))
                decision = net_decision(output_fast)
                if show_image:
                    a = np.array(image.detach())
                    save_image(image=torch.tensor([index_to_image(a)]),
                               mini_epochs_index=mini_epochs_index, decision=decision)
                mini_epochs_index += -1
                check_fast(slow_net=slow_net, line=line)


def __main__(categories=2, line=2,
             count_unsupervised=1,
             lr_supervised=0.001, lr_unsupervised=0.01,
             random_sample=False, batch_size=25,
             num_epochs_supervised=1, print_values_supervised=True,
             from_files_supervised=False, save_files_supervised=True,
             num_epochs_unsupervised=1, threshold=-1,
             editing_steps=20,
             show_image=True, print_values_unsupervised=True,
             super_epochs=1):
    fast_net = FastNet(categories=categories)
    top_down_net = TopDownNet(categories=categories)
    for epoch in range(super_epochs):
        data_set = index_data_set_from_files(False)
        fast_net = supervised_train(data_set=data_set, fast_net=fast_net,
                                    num_epochs=num_epochs_supervised, print_values=print_values_supervised,
                                    lr_fast=lr_supervised, show_image=False,
                                    from_files=from_files_supervised, save_files=save_files_supervised,
                                    batch_size=batch_size)
        slow_net = SlowNet(fast_net, top_down_net)
        data_set = IndexCornersDataSet(line=line, len_data_set=count_unsupervised, random_sample=random_sample)
        unsupervised_train_iterations(data_set=data_set, slow_net=slow_net,
                                      num_epochs=num_epochs_unsupervised,
                                      print_values=print_values_unsupervised,
                                      editing_steps=editing_steps,
                                      lr=lr_unsupervised, show_image=show_image,
                                      line=line, threshold=threshold)


# create_data_sets(50, False)
__main__()
