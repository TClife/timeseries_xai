
1. vqvae_official.py : VQ-VAE Process
2. classification_official.py : Convolutional classifier of quantized VQ-VAE vectors using pre-trained VQ-VAE model
3. timseries_demo.py : LIME of instances using both VQ-VAE and Classifier model 
4. timeseries_xai/dalle_vqvae/vqvae_unofficial.py : Addition of Lucidrain VQVAE based model 

```
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install lime
```
