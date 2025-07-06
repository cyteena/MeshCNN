conda create -n meshcnn python=3.7.16 -y

conda activate meshcnn

pip install numpy==1.21.6 pillow==9.5.0 typing-extensions==4.7.1

pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html