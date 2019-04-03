"""
NEURAL ANALYSIS FUNCTIONS
"""

# import packages that we'll need
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage


def generate_simulated_data(max_firing_rate, number_of_trials, brain_region):
    '''
    Generate neural data responding to one of two stimuli (AA or BA), and then turning left or right
    '''
    # set parameters
    duration = 6000
    time_resolution = .001
    
    # generate the fake data
    spike_times, spike_array, _, time_resolution = \
        generate_spike_array(max_firing_rate, duration, number_of_trials, time_resolution = .001)
        
    # generate the stimulus responses:
    stimulus_modulators = generate_stimulus_modulators(duration, time_resolution = .001)
    
    # modify firing rate according to the effect of the stimulus
    spike_times, spike_array = \
        modify_spikes_by_stimulus(stimulus_modulators, spike_array, number_of_trials, brain_region, time_resolution = .001)
        
    # make a time axis for plotting
    time_axis = generate_time_axis(duration, number_of_trials, time_resolution = .001)

    return spike_times, spike_array, time_axis
        
        
def generate_trial_spike_array(max_firing_rate = 100, duration = 6000, time_resolution = .001):
    
    '''
    Make the array of timepoints indicating when the neurons spikes, for a single trial
    '''
    # find the spike ISIs by drawing from the exponential distribution
    spike_ISIs = np.random.exponential(scale = 1/max_firing_rate, size = int(duration * max_firing_rate * time_resolution))
    
    # get the spike times from the ISIs
    trial_spike_times = np.cumsum(spike_ISIs)
    trial_spike_times = trial_spike_times[trial_spike_times < (duration*time_resolution)]
    
    # get integer indices to populate the spike array
    trial_spike_index = (trial_spike_times / time_resolution).astype(int)
    
    # generate an array of spikes
    trial_spike_array = np.zeros( int(duration))
    trial_spike_array[trial_spike_index] = 1
    
    return trial_spike_times, trial_spike_array, trial_spike_index
    

def generate_spike_array(max_firing_rate, duration, number_of_trials, time_resolution = .001):
    
    '''
    Make the array of timepoints indicating when the neurons spikes, for all trials
    '''
    
    # choose a random firing rate that is less than the max firing rate (25-100%)
    random_firing_rate = int(max_firing_rate * (np.random.random(1)*.75 + .25))
    
    
    # initialize array of spikes (number of trials x number of time points)
    spike_array = np.zeros( (2 * number_of_trials, int(duration) ) )
    spike_times = []; spike_index = []
    
    # loop through each trial, adding the trials data to thte overall array
    for trial in range(2 * number_of_trials):
        trial_spike_times, trial_spike_array, trial_spike_index = generate_trial_spike_array(random_firing_rate, duration, time_resolution)
        spike_times.append(trial_spike_times)
        spike_index.append(trial_spike_index)
        spike_array[trial, :] = trial_spike_array
       
    return spike_times, spike_array, spike_index, time_resolution


def generate_time_axis(duration, number_of_trials, time_resolution = .001):
    
    '''
    create the x-axis for the plots (time in seconds)
    '''
    
    # make a time axis for plotting
    time_axis = np.arange(0	, duration*time_resolution, time_resolution)

    return time_axis


def generate_stimulus_modulators(duration, time_resolution = .001, baseline_rate = .1):
    
    '''
    create sequences of probabilities for modulating the firing rates
    '''

    # set the stimulus onset times
    stimulus_onset = [ int(1/time_resolution), int(3/time_resolution) ]
    stimulus_offset = [ int(2/time_resolution), int(4/time_resolution) ]
    array_length = duration
    none = np.zeros( array_length ) + baseline_rate	
    
    # generate the stimulus on array
    stimulus_on = np.zeros( array_length ) + baseline_rate	
    stimulus_on[stimulus_onset[0]:stimulus_offset[0]] = 1
    stimulus_on[stimulus_onset[1]:stimulus_offset[1]] = 1

    # generate the stimulus 1 on array
    stimulus_1_on = np.zeros( array_length ) + baseline_rate
    stimulus_1_on[stimulus_onset[0]:stimulus_offset[0]] = 1
    
    # generate the stimulus 2 on array
    stimulus_2_on = np.zeros( array_length ) + baseline_rate
    stimulus_2_on[stimulus_onset[1]:stimulus_offset[1]] = 1

    # generate the stimulus transient array
    stimulus_transient = np.zeros( array_length ) + baseline_rate
    stimulus_transient[stimulus_onset[0]:stimulus_onset[0] + int(.75 / time_resolution)] = \
        np.flip(np.linspace(0,1, int(.75 / time_resolution) ))
    stimulus_transient[stimulus_onset[1]:stimulus_onset[1] + int(.75 / time_resolution)] = \
        np.flip(np.linspace(0,1, int(.75 / time_resolution) ))
    
    # generate the stimulus 1 transient array
    stimulus_1_transient = np.zeros( array_length ) + baseline_rate
    stimulus_1_transient[stimulus_onset[0]:stimulus_onset[0] + int(.75 / time_resolution)] = \
        np.flip(np.linspace(0,1, int(.75 / time_resolution) ))
    
    # generate the stimulus 2 transient array
    stimulus_2_transient = np.zeros( array_length ) + baseline_rate
    stimulus_2_transient[stimulus_onset[1]:stimulus_onset[1] + int(.75 / time_resolution)] = \
        np.flip(np.linspace(0,1, int(.75 / time_resolution) ))
        
    # generate the stimulus persistent array
    stimulus_persistent = np.zeros( array_length ) + baseline_rate
    stimulus_persistent[stimulus_onset[0]:stimulus_onset[0]+int(2 / time_resolution)] = 1
    stimulus_persistent[stimulus_onset[1]:stimulus_onset[1]+int(2 / time_resolution)] = 1
    
    # generate the stimulus 1 persistent array
    stimulus_1_persistent = np.zeros( array_length ) + baseline_rate
    stimulus_1_persistent[stimulus_onset[0]:stimulus_onset[0]+int(2 / time_resolution)] = 1
    
    # generate the stimulus 2 persistent array
    stimulus_2_persistent = np.zeros( array_length ) + baseline_rate
    stimulus_2_persistent[stimulus_onset[1]:stimulus_onset[1]+int(2 / time_resolution)] = 1
    
    # generate the ramping motor array
    ramping_motor = np.zeros( array_length ) + baseline_rate
    ramping_motor[stimulus_onset[0]:stimulus_offset[1]] = np.linspace(0,.4, int(3 / time_resolution) )
    ramping_motor[stimulus_offset[1]:stimulus_offset[1] + int(2 / time_resolution)] = \
        np.flip(np.linspace(0,1, int(2 / time_resolution) ))
    
    # generate the ramping then off array
    ramping_then_off = np.zeros( array_length ) + baseline_rate
    ramping_then_off[stimulus_onset[0]:stimulus_offset[1]] = np.linspace(0,.4, int(3 / time_resolution) )
    
    return [stimulus_on, stimulus_1_on, stimulus_2_on, stimulus_transient, stimulus_1_transient, stimulus_2_transient, \
    stimulus_persistent, stimulus_1_persistent, stimulus_2_persistent, ramping_motor, ramping_then_off, none]
    
    
def modify_spikes_by_stimulus(stimulus_modulators, spike_array, number_of_trials, brain_region, time_resolution = .001):
    '''
    take away spikes, based on the stimulus-dependent firing rate
    '''
    
    # first, choose the cell class (and plotting color) based on the brain region
    rn = np.random.random(1)[0]
    if brain_region == 'brain region 1':
        if rn > .55: cell_classes = [9,10]
        elif rn > .1: cell_classes = [10, 9]
        else: cell_classes = [11, 11]
        
    elif brain_region == 'brain region 2':
        if rn > .9: cell_classes = [0, 0]
        elif rn > .8: cell_classes = [0, 2]
        elif rn > .7: cell_classes = [11, 1]
        elif rn > .6: cell_classes = [3, 3] 
        elif rn > .5: cell_classes = [3, 5]
        elif rn > .4: cell_classes = [11, 4]
        elif rn > .3: cell_classes = [6 , 6]
        elif rn > .2: cell_classes = [6 , 8]
        elif rn > .1: cell_classes = [11 , 7]
        else: cell_classes = [11, 11]
    else:
        print('brain region not recognized')

    
    # modulate firing rates for each stimulus type
    spike_times = []
    
    for stimulus in range(2):
        
        # get the cell type
        cell_class = cell_classes[stimulus]
    
        # generate a zero or a one at each point, with probability p
        # expand the stimulus on array to match no of trials
        stimulus_on_all_trials = np.tile(stimulus_modulators[cell_class], number_of_trials)
        stimulus_on_all_trials = np.reshape(stimulus_on_all_trials, (number_of_trials, len(stimulus_modulators[cell_class])))
        
        # now generate random numbers
        random_numbers = np.random.random((number_of_trials, len(stimulus_modulators[cell_class]))) - (1 - stimulus_on_all_trials) 
        
        # and turn it into a binary decision
        take_out_spikes_array = random_numbers > 0
        
        # modify the spikes array using this new array
        spike_array[number_of_trials * (stimulus) : number_of_trials * (stimulus + 1), :] = \
            spike_array[number_of_trials * (stimulus) : number_of_trials * (stimulus + 1), :] * take_out_spikes_array
        
        # convert this back into spike times
        spike_times_array = np.where(spike_array[number_of_trials * (stimulus) : number_of_trials * (stimulus + 1), :])
        
        # loop across trials, turning this array into a list of lists
        for trial_number in range(number_of_trials):
            spike_times.append(spike_times_array[1][spike_times_array[0]==trial_number] * time_resolution)  
                   
    return spike_times, spike_array
    
def generate_colors(number_of_trials):
    '''
    Generate colors to be used during plotting
    '''

    color1 = np.tile([1, 0, 0], number_of_trials)
    color1 = np.reshape(color1, (number_of_trials, 3))
    color2 = np.tile([0, 0, 1], number_of_trials)
    color2 = np.reshape(color2, (number_of_trials, 3))
    
    plot_colors_raster = np.concatenate((color1, color2), axis = 0)
    plot_colors = [[1, 0, 0, .8], [0, 0, 1., .6]]
    plot_colors_light = [[1, .7, .7, .9], [.7, .7, 1, .7]]   
    
    return plot_colors, plot_colors_light, plot_colors_raster



def generate_rectangles(number_of_rectangles = 4):
    
    '''
    create the rectangle objects used to shade the data
    '''
    rectangle1 = plt.Rectangle( (1,-1), 1, 100, color = [.8,.8,.8], alpha = .5)
    rectangle2 = plt.Rectangle( (3,-1), 1, 100, color = [.8,.8,.8], alpha = .5)
    rectangle3 = plt.Rectangle( (1,-1), 1, 100, color = [.8,.8,.8], alpha = .5)
    rectangle4 = plt.Rectangle( (3,-1), 1, 100, color = [.8,.8,.8], alpha = .5)

    if not number_of_rectangles == 4:
        print('number of rectangles was not 4, but you get 4 rectangles anyway :p')

    return rectangle1, rectangle2, rectangle3, rectangle4

def generate_figure_legend(axis, plot_colors):
    '''
    Make a legend for the firing rate plot
    '''
    
    plot1 = axis.plot([0,0], [1,1], color = plot_colors[0], linewidth = 6)
    plot2 = axis.plot([0,0], [1,1], color = plot_colors[1], linewidth = 6)
    
    
    axis.legend((plot1[0], plot2[0]), ('Odor A - Odor A (Left)' , 'Odor B - Odor A (Right)' ))
    