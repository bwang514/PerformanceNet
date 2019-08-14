
# PerformanceNet


![Model image](https://github.com/bwang514/PerformanceNet/blob/master/model.jpg)

PerformanceNet is a deep convolutional model that learns in an end-to-end manner the score-to-audio mapping between musical scores and the correspondent real audio performance. Our work represents a humble yet valuable step towards the dream of **The AI Musician**. Find more details in our AAAI 2019 [paper](https://arxiv.org/abs/1811.04357)!


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
## Scripts

> __Below we assume the working directory is the repository root.__

We provide the scripts for easy managing the experiments.

### Train a new model

1. Run the following command to set up a new experiment. (~= 300 epoch to obtain good results)

> The arguments are (in order) __1. instrument 2.training epoch 3. testing frequency 4. experiment name.__

   ```sh
   # Set up a new experiment
   ./scripts/train_model.sh cello 300 10 cello_exp_1
   ```

### Inference and generate audio

We use the Griffin-Lim algorithm to convert the output spectrogram into audio waveform. (__Note:__ it can take very long time to synthesize a longer audio)

1. Synthesizing with test data split from the Musicnet dataset (New folder that contains generated audio would be created automatically in your exp directory.)

> The arguments are (in order) 1. experiment directory 2. data resource (TEST_DATA means using the test data split from training dataset.)

   ```sh
   # Generating 5 * 5 seconds audio clip by default
   ./scripts/synthesize_audio.sh cello_exp_1 TEST_DATA
   ```

2. Synthesizing audio from your own midi file:

> Please manually create a directory called "midi" in you experiment directory, then put the midi files into it before executing this script

   ```sh
   # Generating one audio clip, length depends on your midi score. 
   ./scripts/synthesize_audio.sh cello_exp_1 YOUR_MIDI_FILE.midi
   ```

Our model can perform any solo music given the score. Therefore we provide a convenient script to convert any .midi file to the input for our model. The quality could vary in different keys, as some notes may never appear in training data. Common keys (C, D, G) should work well though. Also it's important to make sure the note range is within the instrument's range.


## Sound examples

1. Violin: https://www.youtube.com/watch?v=kAEbbNUEEgI
2. Flute: https://www.youtube.com/watch?v=Y38Z2De1NFo
3. Cello: https://www.youtube.com/watch?v=3LzN3GvMNeU
4. 吳萼洋 蜂蜜檸檬 cover: https://youtu.be/k0-cT6GxS3g

## Attribution

If you use this code in your research, please cite the following papers:

__PerformanceNet: Score-to-Audio Music Generation with Multi-Band Convolutional Residual Network__<br>
Bryan Wang, Yi-Hsuan Yang. _in Proceedings of the 33rd AAAI Conference on Artificial Intelligence (AAAI), 2019_. [[paper](https://wvvw.aaai.org/ojs/index.php/AAAI/article/view/3911)]
__Demonstration of PerformanceNet: A Convolutional Neural Network Model for Score-to-Audio Music Generation__<br>
Yu-Hua Chen, Bryan Wang, Yi-Hsuan Yang. _in Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence (IJCAI), Demos, 2019_. [[paper](https://www.ijcai.org/proceedings/2019/938)]


