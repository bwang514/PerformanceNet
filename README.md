
# PerformanceNet
![Model image](https://github.com/bwang514/PerformanceNet/blob/master/model.jpg)

**[Update 2/17]** Data download and pre-process parts are uploaded, expected to upload the full model within a week. 

PerformanceNet is a deep convolutional model that learns in an end-to-end manner the score-to-audio mapping between musical scores and the correspondant real audio performance. Our model represents a tiny yet valuable step towards the dream of **The AI performer**.

Find more details in our AAAI '19 [paper](https://arxiv.org/abs/1811.04357)!


## Prerequisites

> __Below we assume the working directory is the repository root.__

### Install dependencies

  ```sh
  # Install the dependencies
  pip install -r requirements.txt
  ```

### Prepare training data

> PerformanceNet utilizes the [MusicNet](https://homes.cs.washington.edu/~thickstn/start.html) dataset
, which provides musical scores and the correspondant performance audio data.

```sh
# Download the training data
./scripts/download_data.sh
```
You can also download the training data manually
([musicnet.npz](https://homes.cs.washington.edu/~thickstn/media/musicnet.npz)).

> Pre-process the dataset into pianorolls and spectrogram used for training PerformanceNet.

```sh
# Pre-process the dataset
./scripts/process_data.sh
```


# Sound examples

1. Violin: https://www.youtube.com/watch?v=kAEbbNUEEgI
2. Flute: https://www.youtube.com/watch?v=Y38Z2De1NFo
3. Cello: https://www.youtube.com/watch?v=3LzN3GvMNeU
4. 吳萼洋 蜂蜜檸檬 cover: https://youtu.be/k0-cT6GxS3g

# Attribution

If you use this code in your research, please cite the following paper:

__PerformanceNet: Score-to-Audio Music Generation with Multi-Band Convolutional Residual Network__<br>
Bryan Wang, Yi-Hsuan Yang. _To Appear in Proceedings of the 33rd AAAI Conference on Artificial Intelligence (AAAI), 2019_. [[arxiv](https://arxiv.org/abs/1811.04357)]

