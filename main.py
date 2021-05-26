import os
from AudioConverter import AudioConverter
from SpectrogramAnalyzer import ConvNeuralNet
from SpectrogramCreator import SpectrogramCreator
# still leaving this import in in case the SpectrogramCreator class is needed
from SeparatedCategoryTester import CategoryTester
from DiscordBot import VoiceAI

if __name__ == '__main__':
    density_threshold = 0.10
    print("Please select one of the following options:\n\t1 - Recreate dataset\n\t2 - Train new model\n\t3 - Run single test\n\t4 - Run partial correctness test\n\t5 - Run Discord bot\n\t6 - Stop program")
    selection = input()
    converter = AudioConverter(5.0, "output")
    while not "6" in selection:
        if "1" in selection:
            print("Please input an unprocessed data set path:")
            input_path = input() # "input\\data_set" for this project
            converter.import_audio(input_path, density_threshold)
            # this is not necessary since the converter will automatically convert to spectrogram images. Only used if the spectrograms need to be regenerated
            # SpectrogramCreator("output", density_threshold).convert_wavs()
        elif not "7" in selection:
            if "2" in selection:
                print("Enter a unique name for the new model (Optional, empty response overwrites default):")
            else:
                print("Select a model to use (Optional, empty response selects default):")
            model_name = input()
            model_folder = ""
            if model_name:
                model_folder = "models\\"
                model_name = "_" + model_name
            else:
                print("Using default model")
            neural = ConvNeuralNet(model_folder + "model" + model_name + ".json", model_folder + "model" + model_name + ".h5", model_folder + "names" + model_name + ".json")
            if "2" in selection:
                neural.train_new((170, 370, 3), "output\\voice_spectrograms", 65, 11)
            else:
                tester = CategoryTester()
                if "3" in selection:
                    print("Please enter an input path:")
                    import_path = input()
                    while not os.path.exists(import_path):
                        print("File does not exist, select another path:")
                        import_path = input()
                    tester.run_single(import_path, converter, neural, "output", density_threshold)
                elif "4" in selection:
                    tester.run_full_test(converter, neural, "output", density_threshold)
                else:
                    bot_instance = VoiceAI(converter, density_threshold, neural, tester, "output")
                    bot_instance.run(os.getenv('TOKEN'))
                    print("VoiceAI is no longer running")
        else:
            print("Please select one of the following options:\n\t1 - Recreate dataset\n\t2 - Train new model\n\t3 - Run single test\n\t4 - Run partial correctness test\n\t5 - Run Discord bot\n\t6 - Stop program")
        print("To reshow options, enter 7")
        selection = input()
