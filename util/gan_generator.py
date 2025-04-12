import numpy
import json
import torch
from torch.autograd import Variable
import util.models.dcgan as dcgan
import toml


boundary_value = 5.12
nz = 32

imageSize = 64
ngf = 64
ngpu = 1
n_extra_layers = 0
features = len(json.load(open('GANTrain/index2str.json')))

generator = dcgan.DCGAN_G(imageSize, nz, features, ngf, ngpu, n_extra_layers)


def gan_generate(x,batchSize,nz,model_path):
    generator.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    latent_vector = torch.FloatTensor(x).view(batchSize,nz, 1,1)
    with torch.no_grad():
        levels = generator(Variable(latent_vector))
    levels.data = levels.data[:, :, :16, :56]
    im = levels.data.cpu().numpy()
    im = numpy.argmax( im, axis = 1)
    #ÐÞ¸´ START_NO_GROUND
    """
    if numpy.all(im[0][-2:, 0:3] == 0):
        im[0][-1][1] = 1
        print("AutoFix START_NO_GROUND")
    """
    #from IPython import embed
    #embed()
    return json.dumps(im[0].tolist())