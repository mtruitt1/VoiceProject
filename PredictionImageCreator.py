from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from textwrap import wrap


# Generates the image output. Put this function into its own file because it's a mess of MatPlotLib stuff
def GenerateImage(data, best, class_names, output_folder):
    gender_scale = [0.5, 2, 3.5]
    if not data == None:
        nparr = np.array(data)
        scores = [(0, 0), (0, 0), (0, 0)]
        age_score_dict = {
            "child": 0,
            "adult": 1,
            "old": 2
        }
        for x in range(6):
            class_name = class_names[x]
            result = nparr.item(x)
            score_ind = age_score_dict[class_name.split('_')[0]]
            if "female" in str.lower(class_name):
                scores[score_ind] = (scores[score_ind][0], result)
            else:
                scores[score_ind] = (result, scores[score_ind][1])
        for x in range(len(scores)):
            score = scores[x]
            total_len = (score[0] + score[1])
            gender_scale[x] = 2 + ((score[1] / total_len) * 1.5) - ((score[0] / total_len) * 1.5)
    plt.rcdefaults()
    fig, ax = plt.subplots()
    age_groups = ('Child', 'Adult', 'Old')
    y_pos = np.arange(len(age_groups))
    bars = 4
    bars = ax.barh(y_pos, bars, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(age_groups)
    gender_classes = ('', 'Masculine', '', '', 'Androgynous', '', '', 'Feminine', '')
    ax.set_xticklabels(gender_classes)
    ax.invert_yaxis()
    if "child" in best:
        ax.set_title("\n".join(wrap(
        'Based on your voice clip, the AI has made the following predictions by age group. You are most likely in the child age group.',
        60)))
    elif "adult" in best:
        ax.set_title("\n".join(wrap(
        'Based on your voice clip, the AI has made the following predictions by age group. You are most likely in the adult age group.',
        60)))
    else:
        ax.set_title("\n".join(wrap(
        'Based on your voice clip, the AI has made the following predictions by age group. You are most likely in the old age group.',
        60)))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    N = 256
    vals = np.ones((N, 4))
    vals[:, 0] = np.linspace(127.5 / 256, 255.0 / 256, N)
    vals[:, 1] = np.linspace(178.5 / 256, 127.5 / 256, N)
    vals[:, 2] = np.linspace(255.0 / 256, 178.5 / 256, N)
    newcmp = ListedColormap(vals)
    ax = bars[0].axes
    lim = ax.get_xlim() + ax.get_ylim()
    for bar in bars:
        bar.set_edgecolor("black")
        bar.set_linewidth(2)
        bar.set_zorder(1)
        bar.set_facecolor("none")
        x, y = bar.get_xy()
        w, h = bar.get_width(), bar.get_height()
        grad = np.atleast_2d(np.linspace(0, 1 * w / 4, 256))
        ax.imshow(grad, extent=[x, x + w, y, y + h], aspect="auto", zorder=0, cmap=newcmp)
    ax.axis(lim)
    plt.scatter(gender_scale, age_groups, color='black', s=100)
    plt.savefig(output_folder + "\\prediction_image\\prediction.jpg")