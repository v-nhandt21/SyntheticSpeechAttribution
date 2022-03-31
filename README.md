# Quickly Reproduce Result

``` bash predict.sh ```

# Synthetic Speech Attribution

2022 IEEE Signal Processing Cup

This competition is sponsored by the IEEE Signal Processing Society and MathWorks

The IEEE Signal Processing Society’s 2022 Signal Processing Cup (SP Cup) will be a synthetic
speech attribution challenge. Teams will be requested to design and develop a system for
synthetic speech attribution. This means, given an audio recording representing a
synthetically generated speech track, to detect which method among a list of candidate
ones has been used to synthesize the speech. The detector must rely on the analysis of the
speech signal through signal processing and machine learning techniques.

# Usage

## Dataset Prepare

Download data:

- Seen Data:

``` https://www.dropbox.com/s/36yqmymkva2bwdi/spcup_2022_training_part1.zip?dl=1 ```

- Unseen Data:

``` https://www.dropbox.com/s/wsmlthhri29fb79/spcup_2022_unseen.zip?dl=1 ```

- Prepare two script file for training: train.txt, val.txt with format:

``` <path to wav file>,<label>```

Example:

```wav/329cc66d02e4962ac4ead01e2bdef2c3.wav,4```

[Option] Check Config hparameter for training and melspectrogram feature, especially segment_size which is length of audio sample:

``` config.json ```

## Training

Install requirement: pytorch, librosa, onnxruntime-gpu, matlab

Training model using efficientNet b0

``` python train.py ```

## Prediction

``` 
python predict.py --file_out <part_scores.csv> \
--evaluation_folder <path_to_spcup_2022_eval_part1>\
--checkpoint_path_unseen <checkpoint unseen>  \
--checkpoint_path <checkpoint algorithm> \
--config config.json 
```

## Export Onnx
Export to onnx with static input (prefer)

``` python onnx_export.py [Option: model_path] ```

[Option] Export to onnx with dynamic input (need to handle in matlab)

## Inference Onnx with Matlab

Extract melspectrogram for inference in matlab (suppose test set from testing file: Static/data/test.txt)

``` python meldataset.py ```

Inspect matlab script to evaluation then submit final result, Check the script at:

``` matlab_inference.mlx ```