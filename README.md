# AI_hackathon
ai_hackathon: 22. 9 ~ 22. 10

Using TPU, train Efficient-b8 network for multi labeling noisy letter MNIST.

Make sure to check your environment, and a way recommended is here

If you want multi TPUs not single, edit the code.
<hr>
## Step

### TPU config

```
export XRT_TPU_CONFIG="localservice;0;localhost:51011"
```

### Config & Load dataset

```
source cmd.txt
```

### Check train script working

```
python3 train.py --input sample --label rsc/dirty_mnist_2nd_answer.csv --output train_output --seed 2022 --kfold 0
```

### Train

```
python3 train.py --input rsc/dirty_mnist --label rsc/dirty_mnist_2nd_answer.csv --output train_output --seed 2022 --kfold
```

### Albumentaion Error

```
sudo apt-get update. 
sudo apt-get install ffmpeg libsm6 libxext6  -y
```
