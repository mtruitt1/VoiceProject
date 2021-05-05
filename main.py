import os
from AudioConverter import AudioConverter
from SpectrogramAnalyzer import ConvNeuralNet
from SpectrogramCreator import SpectrogramCreator
from SeparatedCategoryTester import CategoryTester
import PredictionImageCreator as ic

if __name__ == '__main__':
    output_folder = "output"
    density_threshold = 0.10
    converter = AudioConverter(5.0, output_folder)
    # converter.import_audio("input\\data_set", density_threshold)
    # SpectrogramCreator(output_folder, density_threshold).convert_wavs()
    neural = ConvNeuralNet("model.json", "model.h5", "names.json")
    # neural.train_new((170, 370, 3), output_folder + "\\voice_spectrograms", 65, 11)
    tester = CategoryTester()
    import_path = "input\\single_clip\\import.wav"
    if not os.path.exists(import_path):
        import_path = "input\\single_clip\\import.ogg"
    tester.run_single(import_path, converter, neural, output_folder, density_threshold)
    # tester.run_full_test(converter, neural, output_folder, density_threshold)
    # ic.GenerateImage(None, None, None, output_folder)
