import torch
import matplotlib.pyplot as plt
import time
import nst_utils as nst
import os

# https://pytorch.org/tutorials/advanced/neural_style_tutorial.html#

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

my_model = nst.VGG_Custom('pre_trained_model/vgg19-dcbb9e9d.pth').to(device)
filename = 'louvre_small.jpg'
img_content = nst.image_preprocess(
    'images/'+filename).requires_grad_(False).to(device)
img_style = nst.image_preprocess(
    'images/monet.jpg').requires_grad_(False).to(device)

out_img_content = my_model(img_content)
out_img_style = my_model(img_style)

x = torch.randn(img_content.shape).requires_grad_(True)
optimizer = torch.optim.Adam((x,), 0.01)

start = time.time()
iter_number = 200
info_update_iter_number = 10

for i in range(iter_number):
    x.data.clamp_(0, 1)
    out_x = my_model(x.to(device))
    loss = nst.nst_loss(out_x, out_img_content, out_img_style, [0], [
                        0, 1, 2, 3, 4], [0.2, 0.2, 0.2, 0.2, 0.2], 40, 10)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i % info_update_iter_number == 0):
        end = time.time()
        eft = (iter_number - i) * (end - start) / info_update_iter_number
        print('{}. | loss: {:.6f} | time: {:.4f} sec. | Time left: {:.0f} sec. ({:.1f}) min.'.format(i, loss.item(), end - start, eft, eft / 60))
        start = time.time()
print('Training has been finished.')

# plt.imshow(x.detach().squeeze(0).numpy().transpose(1, 2, 0))
plt.imsave(os.path.join("out", 'generated_'+filename),
           x.detach().squeeze(0).numpy().transpose(1, 2, 0))
