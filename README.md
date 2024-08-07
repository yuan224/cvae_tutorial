# cvae_tutorial
This is a tutorial for understanding CVAE.
There are three python files in this tutorial:
1. gencvaedata.py:
   This script generates the data for CVAE. The data are stored as dataforCVAE.csv
   The data are gaussian distributions with different means, variances, and magnitudes.
   p(x) = a1*exp( -b1 * (x-c1)^2 ) 
   The first 100 columns store p(x), the following 3 columns store a1, b1, and c1 
2. vae.py:
   This script trains the VAE. The training data are stored as dataforCVAE.csv
   The data are spliit into training data and testing data.
   The model parameters are saved in vae_model.pth
   The model can generate new Gaussian distributions by sampling from the latent space.
3. cvae.py:
   This script trains the CVAE. The training data are stored as dataforCVAE.csv
   The data are spliit into training data and testing data.
   The conditions are take from one of the three Gaussian parameters (a1, b1, c1).
   The example in this script uses c1 as the condition.
   The CVAE is trained for 100 epochs.
   The model parameters are saved in cvae31_model.pth
   The model can generate new Gaussian distributions with specific mean (c1) and varying magnitufe and variances (a1, b1)


