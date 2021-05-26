import glob
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import librosa
import libfmp.c2
from PIL import Image, ImageStat


class SpectrogramCreator:
    def __init__(self, out_path, density_thresh):
        self.output_folder = out_path
        self.density_thresh = density_thresh

    # converts all wav files in the mono_audio folder to spectograms. used to regenerate the spectrograms when the wav files are still valid
    def convert_wavs(self):
        files = glob.glob(self.output_folder + "/mono_audio/*.wav")
        # success, count, min_info_dens, info_dens_avg, and max_info_dens keep track of the statistics over the conversion process so I have more detailed info on the dataset to tweak the threshold values
        success = 0
        count = 0
        min_info_dens = 1.0
        info_dens_avg = 0.0
        max_info_dens = 0.0
        for file in files:
            count += 1
            path = Path(file)
            original_name = path.name.split('.')[0]
            age_category = original_name.split('_')[2]
            # this if statement makes sure undefined age clips are sorted as adult, just in case. if they weren't, the neural network would be training for 8 classes instead of 6
            if age_category == "undefined":
                age_category = "adult"
            gender_category = original_name.split('_')[3]
            save_path = "/voice_spectrograms/" + str(age_category) + "_" + str(gender_category) + "/" + str.lower(original_name) + ".jpg"
            print("Converting \"" + file + "\" to spectrogram file \"" + save_path + "\".")
            info_density = self.plot_spec(file, save_path)
            if info_density < min_info_dens:
                min_info_dens = info_density
            if info_density > max_info_dens:
                max_info_dens = info_density
            info_dens_avg += info_density
            if info_density >= self.density_thresh:
                string_prefix = "Spectrogram beats minimum info density: " + str(info_density) + " "
                success += 1
            else:
                string_prefix = "Spectrogram fails minimum info density: " + str(info_density) + " "
            print(string_prefix + str(round(100 * success / count, 2)) + "%(" + str(success) + "/" + str(count) + ") meet or beat minimum.")
        print("COMPLETE: Lowest info density = " + str(min_info_dens) + " Average info density = " + str(info_dens_avg / count) + " Highest info density = " + str(max_info_dens))

    # creates a single spectrogram using the input wav file path
    def plot_spec(self, input_wav, output):
        Fs = 22050
        x, Fs = librosa.load(input_wav, sr=Fs)

        # Compute Magnitude STFT
        N = 4096
        H = 1024
        X, T_coef, F_coef = libfmp.c2.stft_convention_fmp(x, Fs, N, H)
        Y = np.abs(X) ** 2

        # Plot spectrogram
        # this figure size(with additional settings below) ends up being 370x170px, which is 62,900 input neurons: this should be a reasonable number
        fig = plt.figure(figsize=(4, 2), frameon=False)
        eps = np.finfo(float).eps
        plt.imshow(10 * np.log10(eps + Y), origin='lower', aspect='auto', cmap='gray_r',
                   extent=[T_coef[0], T_coef[-1], F_coef[0], F_coef[-1]])
        plt.clim([-30, 30])
        plt.ylim([0, 4500])
        # turning off the axis and using a tight layout helps reduce the image to the 370x170 size
        plt.axis('off')
        plt.tight_layout()
        fig.canvas.draw()
        # convert image to grayscale to calculate "information density"
        im = Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb()).convert('L')
        stat = ImageStat.Stat(im)
        info_density = 1 - (stat.rms[0] / 255)
        # created this boolean in case I need to add additional requirements for culling unusable data
        beats_thresh = info_density >= self.density_thresh
        if beats_thresh:
            # saving with these settings ensures there's no borders on the image, reducing the size as much as possible
            plt.savefig(self.output_folder + output, bbox_inches='tight', pad_inches=0.0)
        plt.close()
        return info_density