import matplotlib.pyplot as plt
import numpy as np


def gerenate_data_hist(current_idx, scaled_x_values, digital,
                       plot_title, plot_name):

    nbins = 30

    spread = int(np.linalg.norm(np.max(digital) - np.min(digital)))

    if 0 < spread and spread < nbins:
        nbins = spread
    else:
        nbins = nbins

    hist, bins = np.histogram(data, bins=nbins)

    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2

    # plot data
    fig = plt.figure(figsize=(13, 5))
    ax = fig.add_subplot(121)
    ax.bar(center, hist, align="center", width=width, label="hist digital")
    ax.legend()

    idx = current_idx + (slice(None),)

    ax = fig.add_subplot(122)
    ax.plot(scaled_x_values, analog, ".", markersize=0.5, label="analog")
    ax.plot(scaled_x_values, digital, ".", markersize=0.5, label="digital")

    ax.legend()
    fig.suptitle(plot_title)
    fig.savefig(plot_name)
    fig.clf()
    plt.close(fig)


def generate_data_plot(current_idx, scaled_x_values, analog,
                       digital, plot_title, plot_name):

    idx = current_idx + (slice(None),)

    fig = plt.figure(figsize=None)
    plt.plot(scaled_x_values, analog, ".", markersize=0.5, label="analog")
    plt.plot(scaled_x_values, digital, ".", markersize=0.5, label="digital")

    plt.legend()
    fig.suptitle(plot_title)
    fig.savefig(plot_name)
    fig.clf()
    plt.close(fig)


def remove_legend_dubplicates():
    # Remove duplicates in legend
    # https://stackoverflow.com/questions/26337493/pyplot-combine-multiple-line-labels-in-legend
    handles, labels = plt.gca().get_legend_handles_labels()
    i =1
    while i<len(labels):
        if labels[i] in labels[:i]:
            del(labels[i])
            del(handles[i])
        else:
            i +=1

    return handles, labels


def generate_fit_plot(current_idx, x_values, data_to_fit, slopes, offsets,
                      n_intervals, plot_title, plot_name):

    fig = plt.figure(figsize=None)
    for i in np.arange(len(x_values)):
        array_idx = current_idx + (i,)

        m = slopes[array_idx]
        b = offsets[array_idx]
        x = x_values[i]

        plt.plot(x, data_to_fit[i], 'o', color="C0",
                 label="Original data", markersize=1)

        plt.plot(x, m * x + b, "r",
                 label="Fitted line ({} intervals)".format(n_intervals))

    handles, labels = remove_legend_dubplicates()
    plt.legend(handles, labels)

    fig.suptitle(plot_title)
    fig.savefig(plot_name)
    fig.clf()
    plt.close(fig)


def generate_combined_plot(current_idx, scaled_x_values, analog, digital,
                           fit_cutoff_left, fit_cutoff_right, x_values, slopes,
                           offsets, plot_title, plot_name):

    fig = plt.figure(figsize=None)
    plt.plot(scaled_x_values, analog, ".", markersize=0.5, label="analog")
    plt.plot(scaled_x_values, digital, ".", markersize=0.5, label="digital")

    for gain in ["high", "medium", "low"]:
        # if there are multiple sub-fits some are cut off
        if len(x_values[gain]) >= fit_cutoff_left + fit_cutoff_right + 1:
            # display the cut off fits on the left side
            for i in np.arange(fit_cutoff_left):
                array_idx = current_idx + (i,)

                m = slopes[gain][array_idx]
                b = offsets[gain][array_idx]
                x = x_values[gain][i]

                plt.plot(x, m * x + b, "r", alpha=0.3, label="Fitted line (unused)")
            # display the used fits
            for i in np.arange(fit_cutoff_left, len(x_values[gain]) - fit_cutoff_right):
                array_idx = current_idx + (i,)

                m = slopes[gain][array_idx]
                b = offsets[gain][array_idx]
                x = x_values[gain][i]

                plt.plot(x, m * x + b, "r", label="Fitted line")
            # display the cut off fits on the right side
            for i in np.arange(len(x_values[gain]) - fit_cutoff_right,
                               len(x_values[gain])):
                array_idx = current_idx + (i,)

                m = slopes[gain][array_idx]
                b = offsets[gain][array_idx]
                x = x_values[gain][i]

                plt.plot(x, m * x + b, "r", alpha=0.3, label="Fitted line (unused)")
        else:
            # display all fits
            for i in np.arange(len(x_values[gain])):
                array_idx = current_idx + (i,)

                m = slopes[gain][array_idx]
                b = offsets[gain][array_idx]
                x = x_values[gain][i]

                plt.plot(x, m * x + b, "r", label="Fitted line")

    handles, labels = remove_legend_dubplicates()
    plt.legend(handles, labels)

    fig.suptitle(plot_title)
    fig.savefig(plot_name)
    fig.clf()
    plt.close(fig)


def generate_all_plots(current_idx, scaled_x_values, analog,
                       digital, x_values, data_to_fit, slopes, offsets,
                       n_intervals, fit_cutoff_left, fit_cutoff_right,
                       plot_title_prefix, plot_name_prefix, plot_ending):

    plot_title = "{} data".format(plot_title_prefix)
    plot_name = "{}_data{}".format(plot_name_prefix, plot_ending)
    generate_data_plot(current_idx,
                       scaled_x_values,
                       analog,
                       digital,
                       plot_title,
                       plot_name)

    for gain in ["high", "medium", "low"]:
        plot_title = "{} fit {}".format(plot_title_prefix, gain)
        plot_name = "{}_fit_{}{}".format(plot_name_prefix, gain, plot_ending)

        generate_fit_plot(current_idx,
                          x_values[gain],
                          data_to_fit[gain],
                          slopes[gain],
                          offsets[gain],
                          n_intervals[gain],
                          plot_title,
                          plot_name)

    plot_title = "{} combined".format(plot_title_prefix)
    plot_name = "{}_combined{}".format(plot_name_prefix, plot_ending)
    generate_combined_plot(current_idx,
                           scaled_x_values,
                           analog,
                           digital,
                           fit_cutoff_left,
                           fit_cutoff_right,
                           x_values,
                           slopes,
                           offsets,
                           plot_title,
                           plot_name)

