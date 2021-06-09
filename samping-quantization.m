%% Author: Mohammed Kashwah
clear
clc

[original_sound, Fs] = audioread('ELE725_lab1.wav');
counter = 1;
t = 0:(1/Fs): (length(original_sound)-1)*(1/Fs);
%% Audio File Properties
wav_info = audioinfo('ELE725_lab1.wav')
file_size_calc = (wav_info.TotalSamples * wav_info.BitsPerSample * wav_info.NumChannels) / 8;
fprintf("This calculated size of the file is: %d Bytes\n", file_size_calc);
fprintf("Calculated file size in KB is: %d KB\n",file_size_calc/1024);
%There are two sizes for the file when checked by windows explorer:
%First one is Size and this is equal to 466988 Bytes, meaning that there's
%44 bytes difference between it and the calculated one. The other size is
%"Size on disk" and this is equal to 471040 bytes, showing a wide
%discrepancy in size between this one and the calculated size.

%% Fourier Transform for original signal
fft_org = fft(original_sound);
n_org = length(original_sound);
f_org = (-n_org/2:n_org/2-1)*(Fs/n_org);
fft_org_shftd = fftshift(fft_org);
figure(counter)
titleString = sprintf('Fourier transform of original signal');
plot(f_org, abs(fft_org_shftd))
title(titleString)
xlabel('frequency')
ylabel('amplitude')
counter = counter + 1;

%% Sampling
N = [2,4,8];
pf = [0,1];    %


for kN = 1:length(N)
    for kpf = 1:length(pf)
        [downsampled, Fds] = DownSample('ELE725_lab1.wav', 'ELE725_lab1 ', N(kN), pf(kpf));
        %graphing fourier transform for sampled down signal
        fft_dwn_snd = fft(downsampled);
        n = length(downsampled);
        f = (-n/2:n/2-1)*(Fs/n);
        fft_dwn_snd_shftd = fftshift(fft_dwn_snd);
        
        figure(counter)
        titleString = sprintf('Fourier transform of Downsampled with N = %d and pf = %d',N(kN), pf(kpf));
        plot(f, abs(fft_dwn_snd_shftd))
        title(titleString)
        xlabel('frequency')
        ylabel('amplitude')
        counter = counter + 1;
        
        [reconstructedSignal] = Reconstructed(downsampled, N(kN));
        
        %function getReconstructedWav(downsampledSignal,pf, N, Fs)
        getReconstructedWav(reconstructedSignal,pf(kpf), N(kN), Fds)
        
        %graphing fourier transform for sampled up
        fft_up_snd = fft(reconstructedSignal);
        n = length(reconstructedSignal);
        f = (-n/2:n/2-1)*(Fs/n);
        fft_up_snd_shftd = fftshift(fft_up_snd);
        
        figure(counter)
        titleString = sprintf('Fourier transform of Reconstruced for N = %d & pf = %d',N(kN), pf(kpf));
        plot(f, abs(fft_up_snd_shftd))
        title(titleString)
        xlabel('frequency')
        ylabel('amplitude')
        counter = counter + 1;
        
%         f = (0:length(downsampled_fft)-1)*/length(downsampled_fft);
    end
    
%     [reconstructedLeft] = Reconstructed(original_sound(:,1), N(kN));
%     [reconstructedRight] = Reconstructed(original_sound(:,2), N(kN));
%     reconstructedStereo = [reconstructedLeft,reconstructedRight];
%     getConstructedStereo(reconstructedStereo, N(kN), Fs)
    %they all sound the same,, ask the TA!!
    
    
    
end


%% Quantization

%% Uniform Quantization
t = 0:(1/Fs): (length(original_sound)-1)*(1/Fs);


for kN = 1:length(N)
    
        [S_q, MSE] = uniformQuant('ELE725_lab1.wav','Uniformly Quantized ', N(kN));
        %graphing fourier transform for Uniformly Quantized signal
%         fft_uni_qnt = fft(S_q);
%         n = length(S_q);
%         f = (-n/2:n/2-1)*(Fs/n);
%         fft_dwn_snd_shftd = fftshift(fft_uni_qnt);
        
        
        figure(counter)
        subplot(3,1,1)
        plot(t,original_sound)
        title('Original WAV')
        xlabel('time (s)')
        ylabel('Amplitude')
        
        titleString_quantized = sprintf('Uniformly Quantized with N = %d vs Original signal',N(kN));
        subplot(3,1,2)
        plot(t, S_q)
        title(titleString_quantized)
        xlabel('time (s)')
        ylabel('Amplitude')
        
           
        subplot(3,1,3)
        plot(t, original_sound - S_q)
        title('difference between quantized and original')
        xlabel('time (s)')
        ylabel('\Delta Amplitude')
       
        counter = counter + 1;
    
end
    

%% Mu-law quantizer
Mu = 100;

for kN = 1:length(N)
    
        [x_hat, MSE] = MulawQuant('ELE725_lab1.wav', 'Mu-Law quantized', N(kN), Mu);
        %graphing fourier transform for Uniformly Quantized signal
        fft_nonuni_qnt = fft(x_hat);
        n = length(x_hat);
        f = (-n/2:n/2-1)*(Fs/n);
        fft_nonuni_qnt_shftd = fftshift(fft_nonuni_qnt);
        
        
        figure(counter)
        titleString_quantized = sprintf('Mu-Law Quantized with N = %d vs Original signal',N(kN));
        subplot(3,1,1)
        plot(f, abs(fft_nonuni_qnt_shftd))
        title(titleString_quantized)
        xlabel('time (s)')
        ylabel('Amplitude')
        
        subplot(3,1,2)
        plot(f_org,abs(fft_org_shftd))
        title('Original WAV')
        xlabel('Frequnecy')
        ylabel('Amplitude')
        
        subplot(3,1,3)
        histogram(x_hat)
        title('Histogram of the non uniformly sampled')
        xlabel('symbol')
        ylabel('occurance')
       
        counter = counter + 1;
    
end

    
    






















%% Reconstructed (upsampling) function

function [upsampled] = Reconstructed(original_wave, N)
upsampled = interp(original_wave, N);
% filename = sprintf('Reconstructed N%d.wav', N);
% audiowrite(filename, upsampled, Fs*8)
end

function getReconstructedWav(downsampledSignal,pf, N, Fs)
filename = sprintf('Reconstructed N = %d pf = %d .wav', N, pf);
audiowrite(filename, downsampledSignal, Fs*N)

end


%% Quantization Functions:

function [S_q, MSE] = uniformQuant(inFile,outFile, N)
    %This quantizer is midrise
    
    %Reading the audio file
    [audio_array, Fs] = audioread(inFile);
    
    %N is the number of bits
    num_of_levels = 2^N;
    
    %finding V_max & V_min
    V_max = max(audio_array);
    V_max_channelA = V_max(:,1);
    V_max_channelB = V_max(:,2);
    
    V_min = min(audio_array);
    V_min_channelA = V_min(:,1);
    V_min_channelB = V_min(:,2);
    
    %step size
    q_channelA = (V_max_channelA - V_min_channelA)/num_of_levels;
    q_channelB = (V_max_channelB - V_min_channelB)/num_of_levels;
    
    %quantization
    
    %quantized levels
    quantized_levels = [(floor(audio_array(:,1) /q_channelA) + 0.5),(floor(audio_array(:,2) /q_channelB) + 0.5)];
 
    %reconstructed signal
    S_q =[(quantized_levels(:,1) * q_channelA),(quantized_levels(:,2) * q_channelB)];
    
    %calculating MSE
    square_of_difference = (audio_array - S_q).^2;
    MSE = (1/(length(S_q))) * sum(square_of_difference);
    
    
    %returning wav file into outFile
    filename = sprintf('%s N = %d.wav', outFile, N);
    audiowrite(filename, S_q, Fs)
    
end


function [x_hat, MSE] = MulawQuant(inFile, outFile, N, Mu)
    %This is Mu-law nonuniform quantizer
    % x -> y(mu law transformed) -> y_hat (unformaly quantized y) -> x_hat (expanding, inverse mu law)
    
    
    %Reading the audio file
    [x, Fs] = audioread(inFile);
    
    %normalizing x (x/X_max)
    x_normalized = [abs(x(:,1)/ max(x(:,1))), abs(x(:,2)/ max(x(:,2))) ];
    
    %y (Mu law transformation)
    %included channel A and B
    y = [max(x(:,1)) .* sign(x(:,1)) .* (log(1 + Mu.*x_normalized(:,1)))/(log(1+Mu)), max(x(:,2)) .* sign(x(:,2)) .* (log(1 + Mu.*x_normalized(:,2)))/(log(1+Mu))];
    
    %y_hat (uniformly quantized y with N bits)
    %the quantizer used here is midrise from UniformQuant()
    
    %N is the number of bits
    num_of_levels = 2^N;
    
    
    %finding V_max & V_min
    V_max = max(y);
    V_max_channelA = V_max(:,1);
    V_max_channelB = V_max(:,2);
    
    V_min = min(y);
    V_min_channelA = V_min(:,1);
    V_min_channelB = V_min(:,2);
    
    %step size
    q_channelA = (V_max_channelA - V_min_channelA)/num_of_levels;
    q_channelB = (V_max_channelB - V_min_channelB)/num_of_levels;
    
    %quantization
    
    %quantized levels
    quantized_levels = [(floor(y(:,1) /q_channelA) + 0.5),(floor(y(:,2) /q_channelB) + 0.5)];
    
    %y_hat
    y_hat =[(quantized_levels(:,1) * q_channelA),(quantized_levels(:,2) * q_channelB)];
    
    %x_hat (inverse Mu law)
    x_hat = [
        ((max(x(:,1))/Mu) .* (10.^((log(1+Mu) .* abs(y_hat(:,1)) )./(max(x(:,1)))) - 1) .* sign(y_hat(:,1))),((max(x(:,1))/Mu) .* (10.^((log(1+Mu) .* abs(y_hat(:,1)) )./(max(x(:,1)))) - 1) .* sign(y_hat(:,1)))
    ];

    %calculating MSE
    square_of_difference = (x - x_hat).^2;
    MSE = (1/(length(x_hat))) * sum(square_of_difference);
    
    %returning wav file into outFile
    filename = sprintf('%s N = %d Mu = %d.wav', outFile, N, Mu);
    audiowrite(filename, x_hat, Fs)
    
    
    
    
end


%% Trash

%[downsampled] = DownSample('ELE725_lab1.wav', 'ds-no-filter-N10.wav', 10, false);
%sound(original_sound, Fs)



