import numpy as np
import torch
import torchvision as vision
import torch.nn.functional as F
from PIL import Image
import torch.nn as nn


def content_loss(output_1_list, output_2_list, output_index_list):
    assert len(output_1_list) == len(output_2_list)
    loss = 0
    for i in output_index_list:
        n, m = output_1_list[i].shape
        loss += 1 / (4 * n * m) * F.mse_loss(output_1_list[i], output_2_list[i], reduction='sum')
    return loss


def style_loss(output_1_list, output_2_list, output_index_list, weights_list):
    assert len(output_1_list) == len(output_2_list)
    assert len(output_index_list) == len(weights_list)
    loss = 0
    for i in output_index_list:
        n, m = output_1_list[i].shape
        a_gram_matrix = torch.mm(output_1_list[i], output_1_list[i].t())
        b_gram_matrix = torch.mm(output_2_list[i], output_2_list[i].t())
        loss += weights_list[i] * 1 / ((2 * n * m) ** 2) * F.mse_loss(a_gram_matrix, b_gram_matrix, reduction='sum')
    return loss


def nst_loss(output_1_list, output_content_list, output_style_list, content_layers_ind_list, style_layers_ind_list, style_loss_weights_list, alpha, betta):
    s_loss = alpha * style_loss(output_1_list, output_style_list, style_layers_ind_list, style_loss_weights_list)
    c_loss = betta * content_loss(output_1_list, output_content_list, content_layers_ind_list)
    # print('c_loss {}'.format(c_loss))
    # print('s_loss {}'.format(s_loss))
    # print('\n')
    return s_loss + c_loss


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean).reshape(1, -1, 1, 1)
        self.std = torch.tensor(std).reshape(1, -1, 1, 1)

    def forward(self, img):
        device = img.device
        return (img - self.mean.to(device)) / self.std.to(device)

class VGG_Custom(nn.Module):
    def __init__(self, param_path_str):
        super().__init__()
        self._initialize_net(param_path_str)

        self.normilizer = Normalization([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self._output_layers = [0, 5, 10, 19, 28]

    def _initialize_net(self, path_str):
        model = vision.models.vgg19()
        state_dict = torch.load(path_str)
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        for p in model.parameters():
            p.requires_grad_(False)

        self.layers = torch.nn.ModuleList(model.features.children())

        for ind, m in enumerate(self.layers):
            if isinstance(m, torch.nn.MaxPool2d):
                kernel_size = m.kernel_size
                stride = m.stride
                pad = m.padding
                ceil_mode = m.ceil_mode
                self.layers[ind] = torch.nn.AvgPool2d(kernel_size, stride, pad, ceil_mode)

    def forward(self, x):
        outputs = []
        out = self.normilizer(x)
        for i, m in enumerate(self.layers):
            if i > max(self._output_layers):
                break

            out = m(out)

            if i in self._output_layers:
                #print('out_shape {}'.format(out.shape))
                outputs.append(out.reshape(out.shape[1], -1))

        return outputs


def image_preprocess(file_path):
    with Image.open(file_path) as img_file:
        img_data = np.array(img_file)
        img_data = img_data.transpose((2, 0, 1))
        img_data = img_data / 255
        img_data = np.expand_dims(img_data, 0)
    return torch.from_numpy(img_data).float()
