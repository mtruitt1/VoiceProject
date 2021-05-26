import glob
from pathlib import Path
from pydub import AudioSegment
import soundfile as sf
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from SpectrogramCreator import SpectrogramCreator

# This class exists to convert audio files into mono-channel 5 second wav files, and then also turn them into spectrograms
class AudioConverter:
    def __init__(self, clip_length, output_folder):
        self.clip_length = clip_length
        self.output_folder = output_folder

    # used for data sets(though there's only one data set)
    def import_audio(self, folder, dens_thresh, continue_point=0):
        files = glob.glob(folder + '/**/*.wav', recursive=True) \
                + glob.glob(folder + '/**/*.ogg', recursive=True) \
                + glob.glob(folder + '/**/*.mp3', recursive=True)
        average = 0.0
        smallest = 1000.0
        largest = 0.0
        count = 0
        meets_min = 0
        converter = SpectrogramCreator(self.output_folder, dens_thresh)
        for file in files:
            sound_file, sample = sf.read(file)
            clip_length = len(sound_file) / sample
            average += clip_length
            if clip_length > largest:
                largest = clip_length
            if clip_length < smallest:
                smallest = clip_length
            count += 1
            if clip_length < self.clip_length or count < continue_point:
                continue
            sf.write(self.output_folder + "/temp_files/wavconvert.wav", sound_file, sample)
            print("Processing " + str(file))
            meets_min += 1
            path = Path(file)
            sound = AudioSegment.from_file(self.output_folder + "/temp_files/wavconvert.wav")
            sound = sound.set_channels(1)
            file_name = str.lower(path.name.split('.')[0])
            direct_parent = str.lower(path.parts[-2])
            # undefined age will be assumed to be an adult
            age = "undefined"
            if "kid" in file_name or "kid" in direct_parent \
                    or "child" in file_name or "child" in direct_parent \
                    or "young" in file_name or "young" in direct_parent:
                age = "child"
            if "old" in file_name or "old" in direct_parent \
                    or "elder" in file_name or "elder" in direct_parent:
                age = "old"
            if age == "undefined":
                age = "adult"
            # gender starts out undefined but assuming all voice clip sets are properly labeled, by the end none should be undefined
            gender = "undefined"
            if "male" in file_name or "male" in direct_parent \
                    or "npcm" in file_name or "npcm" in direct_parent:
                gender = "male"
            if "female" in file_name or "female" in direct_parent \
                    or "npcf" in file_name or "npcf" in direct_parent:
                gender = "female"
            clip_num = "clip_" + str(meets_min) + "_"
            demographics = age + "_" + gender
            export_path = self.output_folder + "/mono_audio/" + clip_num + demographics
            sound.export(self.output_folder + "/temp_files/monowav.wav", format="wav")
            half_diff = (clip_length - self.clip_length) / 2
            ffmpeg_extract_subclip(self.output_folder + "/temp_files/monowav.wav", half_diff, self.clip_length + half_diff, targetname=export_path + ".wav")
            converter.plot_spec(export_path + ".wav", "/voice_spectrograms/" + demographics + "/" + clip_num + demographics + ".jpg")
            print(str(round((count / len(files)) * 100, 2)) + "% complete (" + str(count) + "/" + str(len(files)) + ") Total spectrograms: " + str(meets_min) + "(" + str(round(meets_min / count), 2) + "% of checked clips)\n")
        print("COMPLETE: Shortest = " + str(smallest) + " Average = " + str(average / len(files)) + " Longest = " + str(largest))

    # not used for data sets, only for prediction. returns whether or not the clip is valid, and why
    def import_single(self, file, dens_thresh):
        try:
            sound_file, sample = sf.read(file)
        except:
            print("File is not an acceptable data type. Please try a different format.")
            return False, "File is not an acceptable data type. Please try a different format."
        clip_length = len(sound_file) / sample
        if clip_length < self.clip_length:
            print("Clip is not long enough. Needs to be at least 5 seconds long.")
            return False, "Clip is not long enough. Needs to be at least 5 seconds long."
        sf.write(self.output_folder + "/temp_files/wavconvert.wav", sound_file, sample)
        print("Processing " + str(file))
        sound = AudioSegment.from_file(self.output_folder + "/temp_files/wavconvert.wav")
        sound = sound.set_channels(1)
        export_path = self.output_folder + "/prediction_audio/prediction"
        sound.export(self.output_folder + "/temp_files/monowav.wav", format="wav")
        half_diff = (clip_length - self.clip_length) / 2
        ffmpeg_extract_subclip(self.output_folder + "/temp_files/monowav.wav", half_diff, self.clip_length + half_diff, targetname=export_path + ".wav")
        dens_return = SpectrogramCreator(self.output_folder, dens_thresh).plot_spec(export_path + ".wav", "/prediction_spectrograms/prediction.jpg")
        if dens_return < dens_thresh:
            print("Clip is not info-dense enough. Please speak more or speak louder.")
            return False, "Clip is not info-dense enough. Please speak more or speak louder."
        return True, "Clip is valid."
