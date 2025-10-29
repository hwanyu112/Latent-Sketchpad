conda env create -f yaml/environment.yml
source activate vit-vae
pip install -e torchscale/
pip install xformers==0.0.28
pip install lightning
pip install torchdata
pip install transformers==4.54.0
pip install timm==1.0.13