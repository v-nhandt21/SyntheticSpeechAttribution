rawlabels = readtable("/home/noahdrisort/Desktop/SCCup/Static/data/labels.csv");
filesCSV = string(rawlabels.track);
labels = int8(rawlabels.algorithm);

ffmpeg_path = '/usr/bin/';

id =fopen("/home/noahdrisort/Desktop/SCCup/Static/data_augment/train.txt", "a+")
count = 0

for idx = 1:length(filesCSV)
    audio_path = "/home/nhandt23/Desktop/SCCup2022-Synthetic-Speech-Attribution/Static/data/wav/" + filesCSV(idx)
    origin_path = "/home/nhandt23/Desktop/SCCup2022-Synthetic-Speech-Attribution/Static/data_augment/wav/" + filesCSV(idx)
    noise_path = "/home/nhandt23/Desktop/SCCup2022-Synthetic-Speech-Attribution/Static/data_augment/wav/noise_" + filesCSV(idx)
    reverb_path = "/home/nhandt23/Desktop/SCCup2022-Synthetic-Speech-Attribution/Static/data_augment/wav/reverb_" + filesCSV(idx)
    compress_path = "/home/nhandt23/Desktop/SCCup2022-Synthetic-Speech-Attribution/Static/data_augment/wav/compress_" + filesCSV(idx)

    [audioIn, fs] = audioread(audio_path);

    audiowrite(origin_path, audioIn, fs);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Noise injection
    noise_probability = 1;
    SNR_min = 10;
    SNR_max = 20;
    SNR_value = (SNR_max-SNR_min).*rand(1000,1) + SNR_min;
    augmenter = audioDataAugmenter( ...
        "AugmentationParameterSource","specify", ...
        "AddNoiseProbability", noise_probability, ...
        "SNR", SNR_value, ...
        "ApplyTimeStretch", false,...
        "ApplyVolumeControl", false, ...
        "ApplyPitchShift", false, ...
        "ApplyTimeStretch", false, ...
        "ApplyTimeShift", false);
    data = augment(augmenter, audioIn, fs);
    audioAug = data.Audio{1};
    audiowrite(noise_path, audioAug, fs);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Add reverberation
    predelay = 0;
    high_cf = 20000;
    diffusion = 0.5;
    decay = 0.5;
    hifreq_damp = 0.9;
    wetdry_mix = 0.25;
    fsamp = 16000;
    reverb = reverberator( ...
	    "PreDelay", predelay, ...
	    "HighCutFrequency", high_cf, ...
	    "Diffusion", diffusion, ...
	    "DecayFactor", decay, ...
        "HighFrequencyDamping", hifreq_damp, ...
	    "WetDryMix", wetdry_mix, ...
	    "SampleRate", fsamp);
    audioRev = reverb(audioIn);
    % Stereo to mono
    audioRev = .5*(audioRev(:,1) + audioRev(:,2));
    audiowrite(reverb_path, audioRev, fs);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Add compression (using ffmpeg)
    bitrate = 6;
    cmd = sprintf('%sffmpeg -y -i %s -b:a %dk %s', ffmpeg_path, audio_path, bitrate, compress_path);
    system(cmd);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    fprintf(id,'%s %d \n',origin_path, labels(idx))
    fprintf(id,'%s %d \n',noise_path, labels(idx))
    fprintf(id,'%s %d \n',reverb_path, labels(idx))
    fprintf(id,'%s %d \n',compress_path, labels(idx))
end
fclose(id);

