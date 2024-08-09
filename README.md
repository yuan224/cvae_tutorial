# cvae_tutorial
This is a tutorial for understanding CVAE. We start from generating data (gencvaedata.py), then develop three different models, autoencoder (aa.py), VAE (vae.py), and CVAE (cvae.py). The autoencoder (aa.py) illustrates the compression and decompression of the data. The VAE (vae.py) illustrates regulation of compressed data (latent space z) to a standard normal. The CVAE (cvae.py) illustrates the inclusion of the label information in the latent space.
There are four python files in this tutorial:
1. gencvaedata.py:
   This script generates the data for CVAE. The data are stored as dataforCVAE.csv
   The data are gaussian distributions with different means, variances, and magnitudes.
   p(x) = a1*exp( -b1 * (x-c1)^2 ) 
   The first 100 columns store p(x), the following 3 columns store a1, b1, and c1 
2. aa.py:
   This script trains the autoencoder. The training data are stored as dataforCVAE.csv
   The model parameters are saved in aa_model.pth
   The script allows you to generate new distributions by sampling from the latent space.
   The script also compares the generated distributions with the original distributions.
3. vae.py:
   This script trains the VAE. The training data are stored as dataforCVAE.csv
   The data are spliit into training data and testing data.
   The model parameters are saved in vae_model.pth
   The model can generate new Gaussian distributions by sampling from the latent space.
   The script also compares the generated distributions with the original distributions.
4. cvae.py:
   This script trains the CVAE. The training data are stored as dataforCVAE.csv
   The data are spliit into training data and testing data.
   The conditions are take from one of the three Gaussian parameters (a1, b1, c1).
   The example in this script uses c1 as the condition.
   The CVAE is trained for 100 epochs.
   The model parameters are saved in cvae31_model.pth
   The model can generate new Gaussian distributions with specific mean (c1) and varying magnitufe and variances (a1, b1)
   The script also compares the generated distributions with the original distributions.


