import glob
from pathlib import Path
import random
import os
import PredictionImageCreator as ic


# Tests single files or 1000 files for partial correctness. Running a single file outputs an image to the predictions folder
class CategoryTester:
    # Run a single clip through the neural network and output the "demographics" image. if it doesn't work, it returns the reason why
    def run_single(self, import_path, converter, neural, output_folder, density_threshold):
        if not os.path.exists(import_path):
            print("File does not exist.")
        file_good, reason = converter.import_single(import_path, density_threshold)
        if file_good:
            main_predict, all_predicts, all_names = neural.predict(output_folder + "\\prediction_spectrograms\\prediction.jpg")
            ic.GenerateImage(all_predicts, main_predict, all_names, output_folder)
            return None
        else:
            return reason

    # tests all of the training and testing data by hand to determine more in a more granular manner how correct/incorrect it is with partial correctness
    def run_full_test(self, converter, neural, output_folder, density_threshold):
        total_age_wrong = 0
        total_gender_wrong = 0
        total_both_wrong = 0
        total_valid = 0
        files = glob.glob(output_folder + "\\mono_audio" + '/**/*.wav', recursive=True) \
                + glob.glob(output_folder + "\\mono_audio" + '/**/*.ogg', recursive=True) \
                + glob.glob(output_folder + "\\mono_audio" + '/**/*.mp3', recursive=True)
        while total_valid < 1000:
            file = random.choice(files)
            print("Testing file \"" + file + "\", assessed " + str(total_valid) + "/1000")
            file_good, reason = converter.import_single(file, density_threshold)
            if file_good:
                path = Path(file)
                file_name = str.lower(path.name.split('.')[0])
                age_category = file_name.split('_')[2]
                if age_category == "undefined":
                    age_category = "adult"
                gender_category = file_name.split('_')[3]
                total_valid += 1
                result = neural.predict(output_folder + "/prediction_spectrograms/prediction.jpg")[0]
                age_result = str.lower(result.split('_')[0])
                gender_result = str.lower(result.split('_')[1])
                if not age_result == age_category and not gender_result == gender_category:
                    total_both_wrong += 1
                elif not age_result == age_category:
                    total_age_wrong += 1
                elif not gender_result == gender_category:
                    total_gender_wrong += 1
        total_correct = 1000 - (total_both_wrong + total_age_wrong + total_gender_wrong)
        print("COMPLETE: Of 1000 files, " + str(total_correct) + "(" + str(
            round(100 * total_correct / 1000, 2)) + "%) were correctly predicted.\n" + \
              "\t" + str(total_both_wrong) + "(" + str(
            round(100 * total_both_wrong / 1000, 2)) + "%) were wrong for both gender and age.\n" + \
              "\t" + str(total_age_wrong) + "(" + str(
            round(100 * total_age_wrong / 1000, 2)) + "%) were wrong for age alone.\n" + \
              "\t" + str(total_gender_wrong) + "(" + str(
            round(100 * total_age_wrong / 1000, 2)) + "%) were wrong for gender alone.")