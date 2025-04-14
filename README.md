Video Link
# MarioGAN-LEVEL-STRUCTURE-CONTROL
In this work, we extend the CMA-ME framework with structure-aware evaluation and control. We propose a structure failure detection module that identifies and classifies unplayable levels into interpretable categories. We also design a structure scoring system that quantifies the plausibility of level layouts. These signals are embedded into the fitness function and the mutation dynamics of the search process. Additionally, structurally invalid samples are filtered from the elite map.

# Training the GAN

Training the GAN is unnecessary as we include a *[pretrained model](https://github.com/icaros-usc/MarioGAN-LSI/blob/master/GANTrain/samples/netG_epoch_4999_7684.pth)* in the repo.

# Running LSI Experiments
```
-We suggest you to use Anaconda prompt(I use this to run the code on my computer)
-Make sure you have python in your own environment(python --version)
-Make sure you have Java in your own environment(java --version)
-Make sure you have downloaded Mario-AI-Framework(https://github.com/amidos2006/Mario-AI-Framework.git / git clone https://github.com/amidos2006/Mario-AI-Framework.git)
-Make sure you have downloaded torch(pip install torch torchvision torchaudio)
-Make sure you have downloaded pyjnius(pip install pyjnius)
```
Experiments can be run with the command:
```
Make sure you are in the correct folder(For example, use command like "cd C:\GitHub\MarioGAN-LSI")
python search/run_search.py -w 1 -c search/config/experiment/experiment.tml / python3 search/run_search.py -w 1 -c search/config/experiment/experiment.tml
```
```
Besure to use worker ID 1!!! If you use different worker ID, it may lead you to different trial with different algorithm(eg.CMA-ES, Random etc.). Note that we are talking about CMA-ME in this project, so just use the command I provide
(python search/run_search.py -w 1 -c search/config/experiment/experiment.tml / python3 search/run_search.py -w 1 -c search/config/experiment/experiment.tml)
```
```
The first command(python search/run_search.py -w 1 -c search/config/experiment/experiment.tml) works for me, but if this doesn't work for you, try the second command(python3 search/run_search.py -w 1 -c search/config/experiment/experiment.tml)
```

The w parameter specifies a worker id which specifies which trial to run from a given experiment file. This allows for parallel execution of all trials on a high-performance cluster.
