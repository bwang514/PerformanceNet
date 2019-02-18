# PerformanceNet

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

