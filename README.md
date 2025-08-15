The code used for all our experiments and plots can be found in **main.ipynb**. 

We cannot guarantee that all code will run immediately, as some code requires specific hardware (running the models on slurm clusters for example), some code has conflicting version types (python 3.7 vs 3.12), and we often refer to datasets such as imagenet1k or imagenette and use checkpoints which cannot be uploaded to the repository due to size constraints. 

The model checkpoints can be found in the following Google Drive folder: [checkpoints.tar.gz](https://drive.google.com/file/d/1iaYoDliQixlOYfVQkYKtyp1Z9ygcLAYn/view). Place the downloaded and unzipped folder named 'checkpoints' in this directory in order to load the models.

While ImageNet1k was used for training and test classification when training the DeiT-III model with and without regs on Snellius, most of our other analyses use the much smaller Imagenette. This dataset can be downloaded here https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz. Unpack the folder named 'imagenette2' into a folder named 'datasets' located in this directory.
