%% Author: Mohammed Kashwah
function [downsampled, Fds] = DownSample(inFile, outFile, N, pf)

%reading the audio
[original_sound , fs] = audioread(inFile);
aud_info = audioinfo(inFile);   %reads the info of the file
n_samples = aud_info.TotalSamples;  %keeps the total number of samples

%retrurn fs: sampling frequency
Fds = fs;

%check if pf is active !=0 then apply 

if pf == 1
    downsampled = decimate(original_sound, N);
    Fds = round(fs / N);
elseif pf == 0
   %Fds = round(fs / N);    %might be REMOVED
   
   for i = N:N:n_samples
       downsampled(i) = original_sound(i);
   end
   %Fds = fs / N;

end


%saving the audio
filename = sprintf('%s downsampled pf %d N%d.wav',outFile, pf, N);
audiowrite(filename, downsampled, Fds)




end