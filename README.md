###  Typical Learned Image Compression Models (PyTorch + CompressAI-based):

This repository implements 4 typical end-to-end image compression models based on the CompressAI framework, including Factorized Model, Hyperprior Model, Joint Autoregressive Model, and Channel AR Model. The models are modular, extensible, and follow standard training and inference pipelines. It is intended both for research reproduction and practical experimentation.

### 1 Supported Compression Models

Factorized Model
Paper: https://arxiv.org/pdf/1611.01704
Core components: EntropyBottleneck, g_a, g_s, compress, decompress, forward.

Hyperprior Model
https://arxiv.org/pdf/1802.01436
Core components: Adds h_a, h_s, GaussianConditional, extends compress and forward.

Joint Autoregressive Model
Paper: https://arxiv.org/pdf/1809.02736
Core components: Adds entropy parameter network, more complex compress/decompress.

Channel AR Model
Paper: https://arxiv.org/pdf/2007.08739
Core components: Conditional channel modeling with mean/scale transforms and context modeling.

### 2 Environment Setup

We recommend using the Bitahub cluster: https://bitahub.ustc.edu.cn/ with the following pre-configured Docker base image: `0.11.3.8:5000/bitahub/deepo:py38_cu113`.

Alternatively, you can also run the code using other platforms with a PyTorch 1.12 and Python 3.8 environment.

In addition to the base image, you’ll need the following Python packages:
```
pip install pytorch_msssim einops compressai==1.1.8 opencv-python h5py tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3 Project Structure and Usage

Model implementations are located in: `/src/models/`

There are 4 supported models:
```python
from .factorized import Factorized
from .hyperprior import Hyperprior
from .joint_autogressive import JointAutoregressive
from .channel_ar import Channel_AR
```

### 4 Training

Use train.py to train a model. Available model names are listed in `/src/zoo/__init__.py`. The training loss follows the formulation: `L=R+λD`, where R is bitrate, D is distortion (MSE) and λ controls the rate-distortion trade-off. We train 4 rate points using: `λ=[0.0018, 0.0054, 0.0162, 0.0483]`.

Training command:
```
python train.py --model [MODEL_NAME] --lambda [LAMBDA_VALUE]
```

### 5 Testing

After training, use `test.py` to evaluate a model. Available model names are listed in /src/zoo/__init__.py. Provide the path to the trained checkpoint.

Testing command:
```
python test.py --model_name [MODEL_NAME] --path [MODEL_PATH]
```

### Tips

You’re encouraged to modify the model architectures or training hyperparameters to suit your own tasks or datasets.
You can extend the framework by adding new model classes to /src/models/ and registering them in /src/zoo/__init__.py.
Feel free to reach out or open issues if you encounter any problems. Happy compressing!

