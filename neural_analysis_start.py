'''
Neural analysis of working memory (with simulated data)
'''

# import packages that we'll need
import numpy as np
from matplotlib.pyplot import *
import scipy.ndimage
from neural_analysis_functions import generate_simulated_data, generate_time_axis, generate_rectangles, \
                                        generate_figure_legend, generate_colors


'''
SET THE PARAMETERS FOR THE ANALYSIS AND SIMULATION
'''

# First, enter the parameters
number_of_trials = 20   # number of trials per stimulus
max_firing_rate = 50    # maximum firing rate in spikes/sec
smoothing_duration = 80 # smoothing duration in ms
brain_region = 'brain region 1'  # can be 'brain region 1' or 'brain region 2'

    
'''
GENERATE THE SIMULATED DATA
'''
print('simulating data')

# create a list of times when the neuron spiked, as well as an array of when the neuron was (1) or was not (0) spiking
spike_times, spike_array, time_axis = generate_simulated_data(max_firing_rate, number_of_trials, brain_region)

# smooth the spike array and divide by time bin duration in order to get firing rates (spikes per second)
firing_rates = scipy.ndimage.filters.gaussian_filter1d(spike_array, smoothing_duration, mode = 'reflect', axis = 1	) / .001


'''
PLOT THE DATA
'''
print('plotting data')

# generate the colors to be used during plotting
plot_colors, plot_colors_light, plot_colors_raster = generate_colors(number_of_trials)

# create the figure
neural_activity_figure = figure()

# create rectangle to show stimulus timing
rectangle1, rectangle2, rectangle3, rectangle4 = generate_rectangles()

# create a 2x1 subput and enter the first plot into the upper subplot
subplot_1 = subplot(211)

# raster plot and labels
eventplot(spike_times, color=plot_colors_raster, linelengths = 1) 
subplot_1.set(title = 'Raster plot', ylabel = 'trial number', xticks = [])

# add the rectangles
subplot_1.add_patch(rectangle1); subplot_1.add_patch(rectangle2)

# create a 2x1 subput and enter the first plot into the upper subplot
subplot_2 = subplot(212)

# TO DO: plot activity for single trials

    
# set up the subplot
subplot_2.set(title = 'Firing rate plot', xlabel = 'time (s)', ylabel = 'firing rate(spike/s)')
subplot_2.set(ylim = [0, np.max(firing_rates[:])] )

# add the rectangles
subplot_2.add_patch(rectangle3); subplot_2.add_patch(rectangle4)
    
# add a figure legend
generate_figure_legend(subplot_2, plot_colors)
