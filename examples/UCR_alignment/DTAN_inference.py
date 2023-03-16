import os
import glob
import sys
module_path = os.path.abspath(os.path.join('..\\..'))
module_path1 = os.path.abspath(os.path.join('..\\..\\..'))
module_path2 = os.path.abspath(os.path.join('..\\..\\DTAN'))
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

if module_path not in sys.path:
    sys.path.append(module_path)
if module_path1 not in sys.path:
    sys.path.append(module_path1)
if module_path2 not in sys.path:
    sys.path.append(module_path2)

from models.train_utils import ExperimentsManager, DTAN_args
from DTAN.DTAN_layer import DTAN
from UCR_alignment import argparser
import csv
from scipy.signal import find_peaks, butter, filtfilt
from tslearn.preprocessing import TimeSeriesResampler
from sklearn.metrics import mean_squared_error
import statistics
import math
from re import search
sys.path.append("D:\\M.Sc_study\\github\\thesis\\ilan_computerVisionML_AI\\mediapipe_eval\\data\\")
from process_data import gen_feature

SIGNAL_LENGTH = 800

PLOT_ENABLED = 0
FWRITE_ENABLED = 0
output_feature_vec = "features_ex234full_wNoseHipDistwAngle.txt"

#a dictionary with scores of each exercise. scores_dict['h1'][i] returns the score of exercise i of patient h1
#the first entry indicates the side, 0 indicates left side, while 1 indicates right side
scores_dict = {
'h1': [0,7,2,2,2,7,7,7,7,7,7,7,7,7,2,2,2],
'h2': [0,7,2,2,2,7,7,7,7,7,7,7,7,7,2,2,2],
'p3': [1,7,1,2,1,7,7,7,7,7,7,7,7,7,1,1,2],
'h4': [0,7,2,2,1,7,7,7,7,7,7,7,7,7,2,2,2], #MP mis detection
'p5': [0,7,2,2,2,7,7,7,7,7,7,7,7,7,2,2,2],
'p6': [0,7,0,0,0,7,7,7,7,7,7,7,7,7,2,0,0],
'p7': [0,7,2,2,1,7,7,7,7,7,7,7,7,7,2,1,2],
'p8': [0,7,1,1,1,7,7,7,7,7,7,7,7,7,2,2,1],
'p9': [1,7,1,1,0,7,7,7,7,7,7,7,7,7,2,2,2],
'p10':[1,7,1,1,1,7,7,7,7,7,7,7,7,7,2,1,1],
'h11':[0,7,2,2,1,7,7,7,7,7,7,7,7,7,2,2,2],
'h12':[0,7,1,1,1,7,7,7,7,7,7,7,7,7,2,2,2],
'p13':[1,7,1,1,1,7,7,7,7,7,7,7,7,7,2,1,2],
'p14':[1,7,1,1,1,7,7,7,7,7,7,7,7,7,2,2,2],
'h15':[0,7,2,2,2,7,7,7,7,7,7,7,7,7,2,2,2],
'p16':[1,7,2,1,1,7,7,7,7,7,7,7,7,7,2,2,2],
'p17':[0,7,1,1,1,7,7,7,7,7,7,7,7,7,2,2,2],
'p18':[1,7,1,0,0,7,7,7,7,7,7,7,7,7,2,1,0],
'p19':[1,7,1,0,1,7,7,7,7,7,7,7,7,7,2,1,1],
'p20':[0,7,1,1,0,7,7,7,7,7,7,7,7,7,2,2,2],
'p21':[1,7,2,1,1,7,7,7,7,7,7,7,7,7,2,2,2],
'p22':[1,7,1,0,0,7,7,7,7,7,7,7,7,7,2,2,1],
'p23':[0,7,2,2,0,7,7,7,7,7,7,7,7,7,2,0,2],
'p24':[0,7,2,2,2,7,7,7,7,7,7,7,7,7,7,7,7],
'p25':[1,7,1,2,1,7,7,7,7,7,7,7,7,7,7,7,7],
'p26':[1,7,2,1,0,7,7,7,7,7,7,7,7,7,7,7,7],
'h27':[0,7,2,2,2,7,7,7,7,7,7,7,7,7,2,2,2],
'p28':[0,7,2,1,2,7,7,7,7,7,7,7,7,7,7,7,7],
'p29':[1,7,1,1,1,7,7,7,7,7,7,7,7,7,2,2,2],
'p30':[1,7,0,0,1,7,7,7,7,7,7,7,7,7,2,1,1],
'p31':[0,7,2,2,2,7,7,7,7,7,7,7,7,7,2,2,2],
'p32':[0,7,1,1,0,7,7,7,7,7,7,7,7,7,2,0,0],
'p33':[0,7,2,2,2,7,7,7,7,7,7,7,7,7,7,7,7],
'p34':[0,7,0,0,0,7,7,7,7,7,7,7,7,7,2,0,0],
'p35':[1,7,2,2,1,7,7,7,7,7,7,7,7,7,7,7,7],
'p36':[1,7,2,2,2,7,7,7,7,7,7,7,7,7,2,2,0],
'p37':[0,7,1,1,1,7,7,7,7,7,7,7,7,7,7,7,7],
'p38':[0,7,1,1,0,7,7,7,7,7,7,7,7,7,7,7,7],
'p39':[0,7,2,0,1,7,7,7,7,7,7,7,7,7,2,2,2],
'p40':[0,7,1,2,2,7,7,7,7,7,7,7,7,7,7,7,7],
'p41':[1,7,2,2,0,7,7,7,7,7,7,7,7,7,7,7,7],
'p42':[1,7,1,1,0,7,7,7,7,7,7,7,7,7,7,7,7],
}  #ex 0,1,2,3,4,5,6,7,8,9,A,B,C,D,E,F,0


def my_inter(list1, resample_len):
    if (list1.size == resample_len):
        return list1.tolist()
    tmp =  list1.reshape(list1.size)
    resample_list1 = TimeSeriesResampler(sz=resample_len).fit_transform(tmp)
    tmp = resample_list1.reshape(resample_len)
    return tmp

def split_signal2iterations(data, ex_num):

    b, a = butter(1, 0.03)
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    smooth = filtfilt(b, a, data)
    if (PLOT_ENABLED):
       plt.plot(data[:], label="full signal")
       plt.plot(smooth[:], label="smoothed signal")
       plt.legend()
       plt.show()

    splits = []
    if (ex_num > 1 and ex_num < 5):
        min_indices, _ = find_peaks(-smooth, distance=70, height=27)
        max_indices, _ = find_peaks(smooth, distance=70, height=-25)
        if (min_indices.size == 0 or max_indices.size == 0):
            print("failed to perform")
            return 0,0
        while (len(max_indices) > 1):
            if (max_indices[1] < min_indices[0]):
                max_indices = max_indices[1:]
                continue
            break
        while (len(min_indices) > 1):
            if (min_indices[1] < min_indices[0]):
                min_indices = min_indices[1:]
                continue
            break
        ##if the last detected peak is a max, then we set the last series point as min peak
        if (max_indices[-1] > min_indices[-1]):
            min_indices = np.append(min_indices, len(data)-1)
        min_idx = 0
        max_idx = 0
        split_list = []
        tmp_list = []
        
        ##advance the min index point to a min which has at least one max peak before it
        ##e.g. min0,max0,min1 >> min1
        while (min_indices[min_idx] <= max_indices[0]):
            min_idx = min_idx + 1
        
        ##if the min peak before the current one is before the first max peak, then we remove all the series
        ##from the beginning until that point (min peak before the current one)
        if (min_indices[min_idx-1] <= max_indices[0]):
            print("removing redundant head")
            plt.plot(data[:], label="full signal")
            data = data[min_indices[min_idx-1]:]
            plt.plot(data[:], label="cut signal")
            plt.legend()
            plt.show()
            #data = my_inter(data, SIGNAL_LENGTH)
            return split_signal2iterations(data, ex_num)
        ##the last min index is not the last point in the series, we remove all the points after it (it's a redundant tail)
        if (min_indices[-1] != len(data)-1):
            print("removing redundant tail")
            plt.plot(data[:], label="full signal")
            data = data[:min_indices[-1]]
            plt.plot(data[:], label="cut signal")
            plt.legend()
            plt.show()
            #data = my_inter(data, SIGNAL_LENGTH)
            return split_signal2iterations(data, ex_num)

        while (min_idx < len(min_indices) and max_idx < len(max_indices)):
            tmp_list.append(min_indices[min_idx])
            try:
                while (min_indices[min_idx+1] <= max_indices[max_idx+1]):
                    min_idx = min_idx + 1
                    tmp_list.append(min_indices[min_idx])
                    continue
            except:
                print("out of bounds")
            split_list.append(int(np.mean(tmp_list)))
            min_idx = min_idx + 1
            max_idx = max_idx + 1
            tmp_list.clear()
    if (ex_num > 13 and ex_num < 17):
        if (PLOT_ENABLED):
            plt.plot(data)
            plt.plot(smooth)
            plt.show()
        #find max an min peaks
        min_indices, _ = find_peaks(-smooth, distance=70, height=-20)
        max_indices, _ = find_peaks(smooth, distance=70, height=23)

        if (min_indices.size == 0 or max_indices.size == 0):
            print("failed to perform")
            return 0,0
            
        while (len(min_indices) > 1):
            if (min_indices[1] < min_indices[0]):
                min_indices = min_indices[1:]
                continue
            break
        while (len(max_indices) > 1):
            if (max_indices[1] < min_indices[0]):
                max_indices = max_indices[1:]
                continue
            break
            
        ##if the last detected peak is a min, then we set the last series point as max peak
        if (min_indices[-1] > max_indices[-1]):
            max_indices = np.append(max_indices, len(data)-1)
        min_idx = 0
        max_idx = 0
        split_list = []
        tmp_list = []
        
        ##advance the max index point to a max which has at least one min peak before it
        ##e.g. max0,min0,max1 >> max1
        while (max_indices[max_idx] <= min_indices[0]):
            max_idx = max_idx + 1
        
        ##if the min peak before the current one is before the first max peak, then we remove all the series
        ##from the beginning until that point (min peak before the current one)
        if (max_indices[max_idx-1] <= min_indices[0]):
            data = data[max_indices[max_idx-1]:]
            #data = my_inter(data, SIGNAL_LENGTH)
            return split_signal2iterations(data, ex_num)
        ##the last min index is not the last point in the series, we remove all the points after it (it's a redundant tail)
        if (max_indices[-1] != len(data)-1):
            print("removing redundant tail")
            data = data[:max_indices[-1]]
            #data = my_inter(data, SIGNAL_LENGTH)
            return split_signal2iterations(data, ex_num)

        while (max_idx < len(max_indices) and min_idx < len(min_indices)):
            tmp_list.append(max_indices[max_idx])
            try:
                while (max_indices[max_idx+1] <= min_indices[min_idx+1]):
                    max_idx = max_idx + 1
                    tmp_list.append(max_indices[max_idx])
                    continue
            except:
                print("out of bounds")
            split_list.append(int(np.mean(tmp_list)))
            min_idx = min_idx + 1
            max_idx = max_idx + 1
            tmp_list.clear()
        
        
        
        ##first split is the first minimum which its index is greater than the first max peak
        splits.append(max_indices[np.argmin(max_indices<min_indices[0])])
        
        second_split = splits[0]
        idx = 1
        while (second_split == splits[0]):
            second_split = max_indices[np.argmin(max_indices<min_indices[idx])]
            idx = idx + 1
    largest_values = smooth[max_indices]
    smallest_values = smooth[min_indices]
    if (PLOT_ENABLED):
        plt.scatter(max_indices, largest_values, s = 50, color='red', marker = 'D')
        plt.scatter(min_indices, smallest_values, s = 50, color='blue', marker = 'D')
        plt.plot(smooth)
        plt.show()

    iterations = [data[:split_list[0]]]
    iterations[0] = np.array([float(x) for x in iterations[0]])
    if (PLOT_ENABLED):
        plt.plot(data[:], label="full signal")
        plt.plot(np.arange(1, len(iterations[0])+1), iterations[0], label="iteration 1")
    length = 0
    for idx in range(1, len(split_list)):
        iterations.append(data[split_list[idx-1]:split_list[idx]])
        iterations[idx] = np.array([float(x) for x in iterations[idx]])
        length = length + len(iterations[idx-1])
        iter = np.linspace(length, length + len(iterations[idx])+1, len(iterations[idx]))
        if (PLOT_ENABLED):
            plt.plot(iter, iterations[idx], label=f"iteration {idx+1}")
        
    ###np.linspace is used to shift the signal starting point at the x axes in the plot
    #second = np.linspace(len(iterations[0]), len(iterations[0])+len(iterations[1])+1, len(iterations[1]))
    #plt.plot(second, iterations[1], label="iteration 2")
    #
    ###np.linspace is used to shift the signal starting point at the x axes in the plot
    #third = np.linspace(len(iterations[0])+len(iterations[1]), len(iterations[0])+len(iterations[1])+len(iterations[2])+1, len(iterations[2]))
    #plt.plot(third, iterations[2], label="iteration 3")
    if (PLOT_ENABLED):
        plt.legend()
        plt.show()

    return iterations, split_list

#https://stackoverflow.com/questions/9542738/python-find-in-list
#https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
#--dataset ex234WInter --dpath "..\..\..\..\UCR_TS_Archive_2015\UCR_TS_Archive_2015"  --n_recurrences 1
def find_nearest(array, value, input_shape):
    array = np.asarray(array)
    array = array.reshape(array.size)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]%input_shape
    

def calculate_speeds(inference_ex, identity, std_mean, input_shape, identityTF_numpy, interpolated_X_test, transformed_input_numpy):
    DTAN_mean = inference_ex.get_DTAN_mean()
    main_start_end_idx = []
    secondary_start_end_idx = []
    speeds = []
    for iter in range(len(std_mean)):
        main_begin = inference_ex.get_points()[0][0]
        main_end = inference_ex.get_points()[0][1] 
        secondary_end = inference_ex.get_points()[1][1] 
        
        found_idx_main_begin = find_nearest(identity[iter], identityTF_numpy[iter][main_begin], input_shape)
        found_idx_main_end = find_nearest(identity[iter], identityTF_numpy[iter][main_end], input_shape)
        found_idx_secondary_end = find_nearest(identity[iter], identityTF_numpy[iter][secondary_end], input_shape)

        x_frames = [main_begin, main_end, secondary_end]
        y_frames = [DTAN_mean[main_begin], DTAN_mean[main_end], DTAN_mean[secondary_end]]
        if PLOT_ENABLED:
            plt.plot(DTAN_mean, label="Mean", color="black")
            plt.scatter(x_frames, y_frames, c='red', marker='o')
            plt.plot(nb, transformed_input_numpy[iter], label="Aligned", color="green")
            plt.scatter(main_begin, transformed_input_numpy[iter][main_begin], c='red', marker='h')
            plt.scatter(main_end, transformed_input_numpy[iter][main_end], c='red', marker='h')
            plt.scatter(secondary_end, transformed_input_numpy[iter][secondary_end], c='magenta', marker='h')
            plt.plot(interpolated_X_test[iter], label="Orig", color="blue")
            plt.scatter(found_idx_main_begin[0], interpolated_X_test[iter][found_idx_main_begin[0]], c='red', marker='^')
            plt.scatter(found_idx_main_end[0], interpolated_X_test[iter][found_idx_main_end[0]], c='red', marker='^')
            plt.scatter(found_idx_secondary_end[0], interpolated_X_test[iter][found_idx_secondary_end[0]], c='magenta', marker='^')
            plt.legend()
            plt.show()

        # These lists will be used to calculate speed and knee angle, in the interval of found indexes
        main_start_end_idx.append([found_idx_main_begin[0], found_idx_main_end[0]])
        secondary_start_end_idx.append([found_idx_main_end[0], found_idx_secondary_end[0]])

        distance = interpolated_X_test[iter][main_start_end_idx[iter][1]] - interpolated_X_test[iter][main_start_end_idx[iter][0]]  # Distance in meters
        time = main_start_end_idx[iter][1] - main_start_end_idx[iter][0]  # Time in frames
        speeds.append(distance / time)
    
    return speeds, main_start_end_idx, secondary_start_end_idx


def generate_features_vector_ex234(identity, identityTF_numpy, interpolated_X_test, transformed_input_numpy,
                                   split_list, std_mean, min_iter, inference_ex, input_shape, landmarks_json_path):
    speeds, main_start_end_idx, secondary_start_end_idx = calculate_speeds(inference_ex, identity, 
                                                            std_mean, input_shape, identityTF_numpy, 
                                                            interpolated_X_test, transformed_input_numpy)
    
    #np.savetxt(f'identity_{dataset}.txt', identity[0][0], delimiter=',',fmt='%1.15f')
    #np.savetxt(f'identity_aligned_{dataset}.txt', identityTF_numpy[0], delimiter=',',fmt='%1.15f')
    #print(interpolated_X_test[min_iter])
    
    ##start generating feature vector, consists of:
    ##speed of best iteration, std_mean of best iteration, min_height and max_height for best iter, 
    ##mean of speed for all iterations, mean of std_mean of all iterations
    ##calculate movement speed in best iteration

    distance = interpolated_X_test[min_iter][main_start_end_idx[min_iter][1]] - interpolated_X_test[min_iter][main_start_end_idx[min_iter][0]] # distance in meters
    time = main_start_end_idx[min_iter][1] - main_start_end_idx[min_iter][0] # time in frames

    speed = distance / time #+ std_mean[min_iter]

    ##knee angle: look at the frames [main_end:secondary_end], those are the frames where patient leg is on bed, calculate the angle in this interval
    ##take the average of the angle or maybe average and min,max or something
    joint_positions = gen_feature(landmarks_json_path, joint0=12, joint1=24)
    stability_measure = np.mean(joint_positions)

    ##left side is odd, right side is even which is joint+1 e.g. left_hip=23, right_hip=24=left_hip+1
    side = inference_ex.get_side()
    knee_angle = np.array(gen_feature(landmarks_json_path, "A", joint0=23+side, 
                                                                                        joint1=25+side,
                                                                                        joint2=27+side))
    best_iteration_angles = knee_angle[split_list[min_iter]:split_list[min_iter+1]]
    best_iteration_angles = my_inter(best_iteration_angles, SIGNAL_LENGTH)
    secondary_angles_interval = best_iteration_angles[secondary_start_end_idx[min_iter][0]:secondary_start_end_idx[min_iter][1]]
    #now, calculate the mean knee angle in the interval between reaching the bed and starting to go down
    knee_angle_mean = np.mean(secondary_angles_interval)
    #calculate the percentage of values greater than 140 
    knee_angle_percentage = np.count_nonzero(secondary_angles_interval > 140) / secondary_angles_interval.size * 100

    #create the feature vector
    feature_vec = [speed, std_mean[min_iter], min(interpolated_X_test[min_iter]), max(interpolated_X_test[min_iter]),
                  (sum(speeds)/len(speeds)), (sum(std_mean)/len(std_mean)),
                  knee_angle_mean, knee_angle_percentage]

    #append stability measures for certain joints to the feature vector
    joint_positions = gen_feature(landmarks_json_path, joint0=12, joint1=24)
    stability_measure = np.mean(joint_positions)
    feature_vec.append(stability_measure)
    joint_positions = gen_feature(landmarks_json_path, joint0=11, joint1=23)
    stability_measure = np.mean(joint_positions)
    feature_vec.append(stability_measure)

    print(feature_vec)
    if (FWRITE_ENABLED):
        output_file_h = open(output_feature_vec, 'a', encoding="utf8")
        output_file_h.write("[\"{}_f{}\",{},{}],\n".format(inference_ex.get_patient_name(), ex_num, 
                                                           label_score, str(feature_vec)))
        output_file_h.close()

    print("done")


def generate_features_vector_ex141516(identity, identityTF_numpy, interpolated_X_test, transformed_input_numpy,
                                   split_list, std_mean, min_iter, inference_ex, input_shape, landmarks_json_path):
    speeds, main_start_end_idx, secondary_start_end_idx = calculate_speeds(inference_ex, identity, 
                                                            std_mean, input_shape, identityTF_numpy, 
                                                            interpolated_X_test, transformed_input_numpy)  

    distance = interpolated_X_test[min_iter][main_start_end_idx[min_iter][1]] - interpolated_X_test[min_iter][main_start_end_idx[min_iter][0]] # distance in meters
    time = main_start_end_idx[min_iter][1] - main_start_end_idx[min_iter][0] # time in frames

    speed = distance / time

    ##hip angle: take the average of the angle or maybe average and min,max or something
    ##left side is odd, right side is even which is joint+1 e.g. left_hip=23, right_hip=left_hip+1=24
    side = inference_ex.get_side()

    #we always take the left side here, since it's the side closest to the camera
    hip_angle = np.array(gen_feature(landmarks_json_path, "A", joint0=11, 
                                                               joint1=23,
                                                               joint2=25))
    
    best_iteration_angles = hip_angle[split_list[min_iter]:split_list[min_iter+1]]
    best_iteration_angles = my_inter(best_iteration_angles, SIGNAL_LENGTH)
    secondary_angles_interval = best_iteration_angles[secondary_start_end_idx[min_iter][0]:secondary_start_end_idx[min_iter][1]]
    #now, calculate the mean hip angle in the interval between reaching the ground and starting to go up
    hip_angle_mean = np.mean(secondary_angles_interval)
    #calculate the percentage of angles less than 45 
    hip_angle_percentage = np.count_nonzero(secondary_angles_interval < 45) / secondary_angles_interval.size * 100

    #create the feature vector
    feature_vec = [speed, std_mean[min_iter], min(interpolated_X_test[min_iter]), min(interpolated_X_test[min_iter]),
                  (sum(speeds)/len(speeds)), (sum(std_mean)/len(std_mean)),
                  hip_angle_mean, hip_angle_percentage]

    print(feature_vec)
    if (FWRITE_ENABLED):
        output_file_h = open(output_feature_vec, 'a', encoding="utf8")
        output_file_h.write("[\"{}_f{}\",{},{}],\n".format(inference_ex.get_patient_name(), ex_num, 
                                                           label_score, str(feature_vec)))
        output_file_h.close()

    print("done")


class Exercise:
    def __init__(self, dataset, DTAN_model = 0):
        ##list of all exercises and features, with frame numbers of each exercise for each feature
        ##exercises_points[exercise][feature], e.g. exercises_points[2][0] is the frame numbers for the first 
        ##feature of exercise 3. first feature ([0]) always indicates the golden feature
        ##feature mapping can be converted from numbers to strings using a dict (a nice to have TODO)
        ##
        ex_mapping = {2:0,3:0,4:0,
                      14:1,15:1,16:1}


        ex234_class2_mean = np.array([-54.912395477294922,-54.887119293212891,-54.861858367919922,-54.836585998535156,-54.811298370361328,-54.785888671875000,-54.760711669921875,-54.745605468750000,-54.738735198974609,-54.732044219970703,-54.721035003662109,-54.706596374511719,-54.693729400634766,-54.678699493408203,-54.662151336669922,-54.643516540527344,-54.624263763427734,-54.606094360351562,-54.588577270507812,-54.574214935302734,-54.561672210693359,-54.548900604248047,-54.535942077636719,-54.522579193115234,-54.508045196533203,-54.492195129394531,-54.474876403808594,-54.456401824951172,-54.441574096679688,-54.427963256835938,-54.415485382080078,-54.402183532714844,-54.388778686523438,-54.374004364013672,-54.360248565673828,-54.347434997558594,-54.335655212402344,-54.324848175048828,-54.315250396728516,-54.305374145507812,-54.295536041259766,-54.285606384277344,-54.276088714599609,-54.262336730957031,-54.248046875000000,-54.234497070312500,-54.221042633056641,-54.207763671875000,-54.199657440185547,-54.191204071044922,-54.183536529541016,-54.175441741943359,-54.166179656982422,-54.156780242919922,-54.147975921630859,-54.140232086181641,-54.132587432861328,-54.127540588378906,-54.122562408447266,-54.118412017822266,-54.114192962646484,-54.110221862792969,-54.106773376464844,-54.102973937988281,-54.099315643310547,-54.094512939453125,-54.088787078857422,-54.082050323486328,-54.075538635253906,-54.069450378417969,-54.063243865966797,-54.057369232177734,-54.051490783691406,-54.047176361083984,-54.042091369628906,-54.036571502685547,-54.031127929687500,-54.025550842285156,-54.019393920898438,-54.011260986328125,-54.002635955810547,-53.992855072021484,-53.980476379394531,-53.967460632324219,-53.954833984375000,-53.943199157714844,-53.931125640869141,-53.920387268066406,-53.913467407226562,-53.907089233398438,-53.903636932373047,-53.901096343994141,-53.898010253906250,-53.895473480224609,-53.892913818359375,-53.889759063720703,-53.887763977050781,-53.886051177978516,-53.884151458740234,-53.882762908935547,-53.881980895996094,-53.880863189697266,-53.879226684570312,-53.877002716064453,-53.874736785888672,-53.872608184814453,-53.870136260986328,-53.867656707763672,-53.865200042724609,-53.861171722412109,-53.856384277343750,-53.851474761962891,-53.846469879150391,-53.841594696044922,-53.837768554687500,-53.834041595458984,-53.830154418945312,-53.827327728271484,-53.824718475341797,-53.822166442871094,-53.819396972656250,-53.817173004150391,-53.815650939941406,-53.812644958496094,-53.808624267578125,-53.804302215576172,-53.799232482910156,-53.792926788330078,-53.785182952880859,-53.777462005615234,-53.770229339599609,-53.763759613037109,-53.756633758544922,-53.749649047851562,-53.743389129638672,-53.737384796142578,-53.731571197509766,-53.726364135742188,-53.721046447753906,-53.716018676757812,-53.711723327636719,-53.707317352294922,-53.703556060791016,-53.700748443603516,-53.697395324707031,-53.693458557128906,-53.690849304199219,-53.688270568847656,-53.685264587402344,-53.682262420654297,-53.679241180419922,-53.676349639892578,-53.673622131347656,-53.671409606933594,-53.668952941894531,-53.665851593017578,-53.662734985351562,-53.661540985107422,-53.659515380859375,-53.659267425537109,-53.659191131591797,-53.658077239990234,-53.656349182128906,-53.651355743408203,-53.643585205078125,-53.635795593261719,-53.626148223876953,-53.613185882568359,-53.599838256835938,-53.585609436035156,-53.570816040039062,-53.554870605468750,-53.539791107177734,-53.528495788574219,-53.517318725585938,-53.510101318359375,-53.505165100097656,-53.500122070312500,-53.495960235595703,-53.491371154785156,-53.485004425048828,-53.477462768554688,-53.465476989746094,-53.450569152832031,-53.435588836669922,-53.421432495117188,-53.404823303222656,-53.385604858398438,-53.364807128906250,-53.344367980957031,-53.327529907226562,-53.312091827392578,-53.298542022705078,-53.291049957275391,-53.285034179687500,-53.279834747314453,-53.274456024169922,-53.269573211669922,-53.263969421386719,-53.259319305419922,-53.255901336669922,-53.252429962158203,-53.246204376220703,-53.237361907958984,-53.222087860107422,-53.207805633544922,-53.197891235351562,-53.188793182373047,-53.171997070312500,-53.148967742919922,-53.126552581787109,-53.115520477294922,-53.106212615966797,-53.091354370117188,-53.066265106201172,-53.039066314697266,-53.017562866210938,-53.002117156982422,-52.988582611083984,-52.973491668701172,-52.955097198486328,-52.933315277099609,-52.913650512695312,-52.905666351318359,-52.898494720458984,-52.892845153808594,-52.881088256835938,-52.873580932617188,-52.877758026123047,-52.881965637207031,-52.888198852539062,-52.892578125000000,-52.896606445312500,-52.894931793212891,-52.892463684082031,-52.885375976562500,-52.873493194580078,-52.858150482177734,-52.844337463378906,-52.838897705078125,-52.837104797363281,-52.828613281250000,-52.805221557617188,-52.778186798095703,-52.754161834716797,-52.720359802246094,-52.686328887939453,-52.659183502197266,-52.653358459472656,-52.621803283691406,-52.588615417480469,-52.567695617675781,-52.552371978759766,-52.513938903808594,-52.467716217041016,-52.430313110351562,-52.388751983642578,-52.320354461669922,-52.270385742187500,-52.225658416748047,-52.184570312500000,-52.159507751464844,-52.117546081542969,-52.071453094482422,-52.031314849853516,-51.987850189208984,-51.938354492187500,-51.894233703613281,-51.830139160156250,-51.744022369384766,-51.656665802001953,-51.563503265380859,-51.460361480712891,-51.363491058349609,-51.263313293457031,-51.152580261230469,-51.048686981201172,-50.974269866943359,-50.911037445068359,-50.826477050781250,-50.756996154785156,-50.659591674804688,-50.552410125732422,-50.454875946044922,-50.357067108154297,-50.256874084472656,-50.141895294189453,-50.010292053222656,-49.875350952148438,-49.729667663574219,-49.576816558837891,-49.419757843017578,-49.262981414794922,-49.139339447021484,-49.014602661132812,-48.904491424560547,-48.777111053466797,-48.606010437011719,-48.435527801513672,-48.276683807373047,-48.127296447753906,-47.980064392089844,-47.833545684814453,-47.673980712890625,-47.498233795166016,-47.307239532470703,-47.110843658447266,-46.886646270751953,-46.647109985351562,-46.422405242919922,-46.184078216552734,-45.934226989746094,-45.692977905273438,-45.432132720947266,-45.174354553222656,-44.890907287597656,-44.602375030517578,-44.332637786865234,-44.062168121337891,-43.781963348388672,-43.474620819091797,-43.161712646484375,-42.845985412597656,-42.528148651123047,-42.212558746337891,-41.885330200195312,-41.547164916992188,-41.210693359375000,-40.885467529296875,-40.562576293945312,-40.256565093994141,-39.932033538818359,-39.604793548583984,-39.272552490234375,-38.909301757812500,-38.517517089843750,-38.156814575195312,-37.809463500976562,-37.471179962158203,-37.137310028076172,-36.771003723144531,-36.365123748779297,-35.902671813964844,-35.421672821044922,-34.910354614257812,-34.339630126953125,-33.746047973632812,-33.173576354980469,-32.647624969482422,-32.107051849365234,-31.519123077392578,-30.985935211181641,-30.422058105468750,-29.836790084838867,-29.242158889770508,-28.613958358764648,-28.017213821411133,-27.455751419067383,-26.879508972167969,-26.234361648559570,-25.611633300781250,-25.050880432128906,-24.490226745605469,-23.860795974731445,-23.195337295532227,-22.522199630737305,-21.848001480102539,-21.202907562255859,-20.578701019287109,-19.988220214843750,-19.425790786743164,-18.882579803466797,-18.334674835205078,-17.774417877197266,-17.171983718872070,-16.580371856689453,-15.986240386962891,-15.404841423034668,-14.862066268920898,-14.300154685974121,-13.737215995788574,-13.175838470458984,-12.616444587707520,-12.098246574401855,-11.571227073669434,-11.065671920776367,-10.600976943969727,-10.139617919921875,-9.645271301269531,-9.154519081115723,-8.650514602661133,-8.199810028076172,-7.815958023071289,-7.452412605285645,-7.124792575836182,-6.815603733062744,-6.504616737365723,-6.162858009338379,-5.790024280548096,-5.446753501892090,-5.136656761169434,-4.828695774078369,-4.565013408660889,-4.351739406585693,-4.170156478881836,-4.011196136474609,-3.867655754089355,-3.748766660690308,-3.658476352691650,-3.572677612304688,-3.460507154464722,-3.357185363769531,-3.250992298126221,-3.133094072341919,-2.938410043716431,-2.730198383331299,-2.555971860885620,-2.399153470993042,-2.241688013076782,-2.098327875137329,-1.962128400802612,-1.822000384330750,-1.661295771598816,-1.492465615272522,-1.335305452346802,-1.194364786148071,-1.084503054618835,-0.997315585613251,-0.917861163616180,-0.865597903728485,-0.828778028488159,-0.803827762603760,-0.776134073734283,-0.744932889938354,-0.740107953548431,-0.755195379257202,-0.762534499168396,-0.772886097431183,-0.812405467033386,-0.866000950336456,-0.905605137348175,-0.931271255016327,-0.961690068244934,-1.000764012336731,-1.046558976173401,-1.078985214233398,-1.132739901542664,-1.197230577468872,-1.281005978584290,-1.379612088203430,-1.463162422180176,-1.507607579231262,-1.552023530006409,-1.597123265266418,-1.636391282081604,-1.690895915031433,-1.735951185226440,-1.769762992858887,-1.800212144851685,-1.828166246414185,-1.854852199554443,-1.876586675643921,-1.898451089859009,-1.923711538314819,-1.944038033485413,-1.943997263908386,-1.958856582641602,-1.983642816543579,-2.013885021209717,-2.051555156707764,-2.102218627929688,-2.197436094284058,-2.316930055618286,-2.434287548065186,-2.544811725616455,-2.661620855331421,-2.789701223373413,-2.920058250427246,-3.049659729003906,-3.187317848205566,-3.310427427291870,-3.375691413879395,-3.423758506774902,-3.487057924270630,-3.555698871612549,-3.625658988952637,-3.706215858459473,-3.761449098587036,-3.790397405624390,-3.812332391738892,-3.823425292968750,-3.808557033538818,-3.803860902786255,-3.823807001113892,-3.879531145095825,-3.944971799850464,-3.969629526138306,-3.968930721282959,-3.969494342803955,-3.948501348495483,-3.950651645660400,-3.964333534240723,-3.960989475250244,-3.964169979095459,-3.955817937850952,-3.947220087051392,-3.949578523635864,-3.976626634597778,-3.972972869873047,-3.952782392501831,-3.915101528167725,-3.872095823287964,-3.833684206008911,-3.797381877899170,-3.779043674468994,-3.757640123367310,-3.776784181594849,-3.805270671844482,-3.809720993041992,-3.814469337463379,-3.852092742919922,-3.909806013107300,-3.963366985321045,-3.992319107055664,-4.014924526214600,-4.040279388427734,-4.027639389038086,-3.987214565277100,-3.957388401031494,-3.932076454162598,-3.899275064468384,-3.886312484741211,-3.876655578613281,-3.854738235473633,-3.791628599166870,-3.705718040466309,-3.690243721008301,-3.706938743591309,-3.706118345260620,-3.688243389129639,-3.616727352142334,-3.511440992355347,-3.402806997299194,-3.298174619674683,-3.183025836944580,-3.061317920684814,-3.019694089889526,-2.992300510406494,-2.986166238784790,-2.871604204177856,-2.779083013534546,-2.754312276840210,-2.794124126434326,-2.832082748413086,-2.751963138580322,-2.716826200485229,-2.775320529937744,-2.929482936859131,-3.091369152069092,-3.245195150375366,-3.384593963623047,-3.504414558410645,-3.668747663497925,-3.848278045654297,-4.108495235443115,-4.506964206695557,-4.959867477416992,-5.353964328765869,-5.655378818511963,-5.929183959960938,-6.426780700683594,-7.111298084259033,-7.706236839294434,-8.136401176452637,-8.715195655822754,-9.361445426940918,-10.138720512390137,-11.051081657409668,-12.081626892089844,-12.953159332275391,-13.954274177551270,-14.856128692626953,-15.716116905212402,-16.618463516235352,-17.461601257324219,-18.285446166992188,-19.193582534790039,-20.258583068847656,-21.467420578002930,-22.654396057128906,-23.792995452880859,-24.699470520019531,-25.613660812377930,-26.707475662231445,-27.717071533203125,-28.576826095581055,-29.459865570068359,-30.351184844970703,-31.367042541503906,-32.238384246826172,-33.003757476806641,-33.849182128906250,-34.606761932373047,-35.331012725830078,-36.003696441650391,-36.591438293457031,-37.204227447509766,-37.816154479980469,-38.275146484375000,-38.883083343505859,-39.496341705322266,-40.047229766845703,-40.644657135009766,-41.217079162597656,-41.709983825683594,-42.167137145996094,-42.656673431396484,-43.145568847656250,-43.595241546630859,-44.076828002929688,-44.486640930175781,-44.854534149169922,-45.214061737060547,-45.542873382568359,-45.831161499023438,-46.097225189208984,-46.355098724365234,-46.631584167480469,-46.929389953613281,-47.192562103271484,-47.422634124755859,-47.660568237304688,-47.900154113769531,-48.117084503173828,-48.343955993652344,-48.585247039794922,-48.809978485107422,-49.014633178710938,-49.197696685791016,-49.413509368896484,-49.619228363037109,-49.809665679931641,-49.982841491699219,-50.166206359863281,-50.355266571044922,-50.527065277099609,-50.692413330078125,-50.862640380859375,-51.012306213378906,-51.152954101562500,-51.316543579101562,-51.479084014892578,-51.618671417236328,-51.756992340087891,-51.891056060791016,-52.032623291015625,-52.166091918945312,-52.295669555664062,-52.424057006835938,-52.547599792480469,-52.643196105957031,-52.718811035156250,-52.781173706054688,-52.831604003906250,-52.868202209472656,-52.909793853759766,-52.959373474121094,-53.008670806884766,-53.048507690429688,-53.078208923339844,-53.094173431396484,-53.113849639892578,-53.126937866210938,-53.130233764648438,-53.143711090087891,-53.152004241943359,-53.160629272460938,-53.186038970947266,-53.222545623779297,-53.257896423339844,-53.299438476562500,-53.346046447753906,-53.403869628906250,-53.464111328125000,-53.520538330078125,-53.575702667236328,-53.630809783935547,-53.682090759277344,-53.726280212402344,-53.769458770751953,-53.818641662597656,-53.862529754638672,-53.895374298095703,-53.921703338623047,-53.937320709228516,-53.942928314208984,-53.965518951416016,-54.014347076416016,-54.074367523193359,-54.139682769775391,-54.201385498046875,-54.255203247070312,-54.296443939208984,-54.331619262695312,-54.360755920410156,-54.386035919189453,-54.403984069824219,-54.415313720703125,-54.423118591308594,-54.438400268554688,-54.454811096191406,-54.469768524169922,-54.485874176025391,-54.501483917236328,-54.513156890869141,-54.526493072509766,-54.539966583251953,-54.551651000976562,-54.563667297363281,-54.577606201171875,-54.590221405029297,-54.601604461669922,-54.616180419921875,-54.633068084716797,-54.650753021240234,-54.672252655029297,-54.689155578613281,-54.704349517822266,-54.718055725097656,-54.730628967285156,-54.739967346191406,-54.744804382324219,-54.740600585937500,-54.733631134033203,-54.729217529296875,-54.728137969970703,-54.726741790771484,-54.727241516113281,-54.725429534912109,-54.719127655029297,-54.707752227783203,-54.696640014648438,-54.685775756835938,-54.678840637207031,-54.678997039794922,-54.679542541503906,-54.678646087646484,-54.676635742187500,-54.677852630615234,-54.677162170410156,-54.674827575683594,-54.668762207031250,-54.657897949218750,-54.645671844482422,-54.634487152099609,-54.621902465820312,-54.606517791748047,-54.589775085449219,-54.573276519775391,-54.558200836181641,-54.545909881591797,-54.534362792968750,-54.519649505615234,-54.503715515136719,-54.489456176757812,-54.476966857910156,-54.465045928955078,-54.452812194824219,-54.442779541015625,-54.435852050781250,-54.430873870849609,-54.430480957031250,-54.431598663330078,-54.431358337402344,-54.428508758544922,-54.425254821777344,-54.418182373046875,-54.408939361572266,-54.402080535888672,-54.393135070800781,-54.385929107666016,-54.383449554443359,-54.381893157958984,-54.378978729248047,-54.372295379638672,-54.368309020996094,-54.365436553955078,-54.363807678222656,-54.361373901367188,-54.360279083251953,-54.361206054687500,-54.362628936767578,-54.365203857421875,-54.366744995117188,-54.363403320312500,-54.351417541503906,-54.339645385742188,-54.327747344970703,-54.316013336181641,-54.304267883300781])
        ex234_class2_std = np.array([4.967250823974609,4.975839614868164,4.988574028015137,5.005420684814453,5.026359558105469,5.049267768859863,5.073655128479004,5.088273525238037,5.093230724334717,5.097934246063232,5.106286048889160,5.117267131805420,5.126485347747803,5.135589122772217,5.139670848846436,5.140980243682861,5.141490936279297,5.142237186431885,5.142951011657715,5.141512393951416,5.138693809509277,5.135926723480225,5.132469654083252,5.128373146057129,5.126749992370605,5.127780437469482,5.132312297821045,5.135107517242432,5.140632152557373,5.145726203918457,5.149401664733887,5.153276443481445,5.157012939453125,5.160775184631348,5.165282726287842,5.170673370361328,5.175803184509277,5.178859710693359,5.181085586547852,5.183277606964111,5.184807300567627,5.185530662536621,5.186439990997314,5.192412376403809,5.201178073883057,5.209831237792969,5.217564582824707,5.224858760833740,5.228829383850098,5.230106830596924,5.228445529937744,5.226670265197754,5.225436210632324,5.225289344787598,5.225309371948242,5.223748683929443,5.222887039184570,5.221627235412598,5.220087051391602,5.219184398651123,5.218540668487549,5.217818737030029,5.216560363769531,5.216374397277832,5.215879917144775,5.215646743774414,5.215500831604004,5.213841438293457,5.211719989776611,5.208801746368408,5.204876422882080,5.201478004455566,5.198093414306641,5.196035385131836,5.192183494567871,5.188688755035400,5.185358524322510,5.182358264923096,5.179678916931152,5.175023078918457,5.170299053192139,5.164783954620361,5.157685756683350,5.151165008544922,5.144761562347412,5.137933731079102,5.131852149963379,5.128170967102051,5.126400947570801,5.124210357666016,5.123268127441406,5.123016357421875,5.122654438018799,5.121860980987549,5.121349811553955,5.121078014373779,5.120461940765381,5.119200706481934,5.117784023284912,5.115952968597412,5.113912582397461,5.111698627471924,5.109211921691895,5.105803966522217,5.102483749389648,5.099180221557617,5.095982074737549,5.093014240264893,5.090205192565918,5.087579250335693,5.085334777832031,5.082771301269531,5.080181598663330,5.078045845031738,5.076187610626221,5.074647426605225,5.073492050170898,5.073952198028564,5.074389934539795,5.075264930725098,5.076209068298340,5.077292442321777,5.078581809997559,5.077623844146729,5.076049327850342,5.074930191040039,5.072295665740967,5.068400382995605,5.065392971038818,5.062444210052490,5.059542655944824,5.055361270904541,5.050138473510742,5.045609474182129,5.040694236755371,5.035823345184326,5.030904769897461,5.025970935821533,5.021399497985840,5.016643047332764,5.011730194091797,5.007132053375244,5.002992630004883,4.999971389770508,4.997459888458252,4.995536327362061,4.991657733917236,4.987415790557861,4.983309745788574,4.978733539581299,4.973758697509766,4.969449043273926,4.964833736419678,4.961253166198730,4.958401679992676,4.955196857452393,4.951777935028076,4.945863723754883,4.939624786376953,4.932638168334961,4.925523757934570,4.916970729827881,4.908766269683838,4.899828433990479,4.893650054931641,4.887586116790771,4.881935596466064,4.871254920959473,4.860715389251709,4.852821350097656,4.842085838317871,4.831295013427734,4.821529865264893,4.815467834472656,4.809595584869385,4.805121421813965,4.796981811523438,4.788822650909424,4.780601501464844,4.773147583007812,4.765107154846191,4.756510734558105,4.749393939971924,4.739176273345947,4.729631423950195,4.721326828002930,4.715672016143799,4.712575912475586,4.715694904327393,4.721944332122803,4.732484340667725,4.744414329528809,4.759511947631836,4.772056579589844,4.782522201538086,4.793728351593018,4.802356719970703,4.803503513336182,4.802701950073242,4.802042484283447,4.802939891815186,4.802793979644775,4.800483226776123,4.797747135162354,4.787921905517578,4.783557891845703,4.792234897613525,4.810376644134521,4.833294868469238,4.857587337493896,4.880250930786133,4.896678924560547,4.910502433776855,4.922914505004883,4.939778804779053,4.963840007781982,4.986626625061035,5.012035369873047,5.039781570434570,5.071947097778320,5.107858657836914,5.144235134124756,5.169886112213135,5.183828353881836,5.198163032531738,5.214128017425537,5.242326259613037,5.270633697509766,5.291644573211670,5.308124542236328,5.321657657623291,5.343873500823975,5.365623950958252,5.379139900207520,5.385998725891113,5.389361858367920,5.391330718994141,5.392048835754395,5.397031307220459,5.407573699951172,5.429773330688477,5.464153766632080,5.502849102020264,5.539356708526611,5.567569732666016,5.591678142547607,5.619302749633789,5.643895149230957,5.660936355590820,5.672740459442139,5.681729793548584,5.695512294769287,5.699346542358398,5.676652908325195,5.660778999328613,5.652897834777832,5.647253036499023,5.657286643981934,5.670133590698242,5.687960624694824,5.695579528808594,5.710960388183594,5.729814052581787,5.741963386535645,5.761099338531494,5.783183574676514,5.799745559692383,5.829602718353271,5.882001399993896,5.937329769134521,6.005683898925781,6.070572376251221,6.129720211029053,6.193808555603027,6.261742591857910,6.324728012084961,6.391581058502197,6.457763671875000,6.513551712036133,6.561433315277100,6.614273548126221,6.673457622528076,6.741276264190674,6.814258575439453,6.887757778167725,6.947128772735596,6.999788761138916,7.046741485595703,7.092483520507812,7.133162498474121,7.174765110015869,7.218038558959961,7.240036487579346,7.300031185150146,7.366593837738037,7.459840774536133,7.538106918334961,7.557877540588379,7.586197853088379,7.635463714599609,7.683093547821045,7.741381168365479,7.807598114013672,7.875003814697266,7.951243400573730,8.025416374206543,8.095326423645020,8.156127929687500,8.202201843261719,8.269275665283203,8.352782249450684,8.442217826843262,8.530278205871582,8.621592521667480,8.741000175476074,8.889998435974121,9.029843330383301,9.154765129089355,9.288268089294434,9.404001235961914,9.516563415527344,9.624922752380371,9.695422172546387,9.759167671203613,9.785219192504883,9.827726364135742,9.919754981994629,10.063990592956543,10.199877738952637,10.305718421936035,10.364016532897949,10.444787979125977,10.594749450683594,10.786584854125977,11.032354354858398,11.263174057006836,11.435145378112793,11.617639541625977,11.821256637573242,12.030835151672363,12.180034637451172,12.310956001281738,12.422966957092285,12.526993751525879,12.612919807434082,12.637093544006348,12.709236145019531,12.803256034851074,12.909494400024414,12.960175514221191,12.965800285339355,12.926865577697754,12.885206222534180,12.801857948303223,12.699404716491699,12.638473510742188,12.596208572387695,12.544191360473633,12.475368499755859,12.441953659057617,12.409146308898926,12.372008323669434,12.369309425354004,12.369669914245605,12.320661544799805,12.261816978454590,12.193796157836914,12.153768539428711,12.144495010375977,12.155532836914062,12.194373130798340,12.217905998229980,12.199621200561523,12.130863189697266,12.051774024963379,11.967187881469727,11.865825653076172,11.782656669616699,11.718454360961914,11.678381919860840,11.637665748596191,11.619432449340820,11.623924255371094,11.640707969665527,11.657012939453125,11.664640426635742,11.637484550476074,11.563156127929688,11.438295364379883,11.335732460021973,11.235480308532715,11.133006095886230,11.051515579223633,10.998445510864258,10.959739685058594,10.904208183288574,10.844265937805176,10.780322074890137,10.737465858459473,10.704674720764160,10.664954185485840,10.626180648803711,10.571082115173340,10.499642372131348,10.409967422485352,10.294527053833008,10.177969932556152,10.067762374877930,9.962399482727051,9.872960090637207,9.765267372131348,9.645450592041016,9.528663635253906,9.418033599853516,9.230538368225098,8.992281913757324,8.790047645568848,8.617024421691895,8.479837417602539,8.374382972717285,8.285324096679688,8.152411460876465,7.962856292724609,7.744034290313721,7.534538745880127,7.343095779418945,7.225693225860596,7.140231609344482,7.060421466827393,6.981357097625732,6.887928485870361,6.797703266143799,6.736209392547607,6.682074069976807,6.626247882843018,6.563285350799561,6.507894039154053,6.456830501556396,6.419803619384766,6.375713348388672,6.308655738830566,6.249334812164307,6.225196361541748,6.222601890563965,6.221872329711914,6.201769351959229,6.156675815582275,6.108387947082520,6.079218387603760,6.066800117492676,6.054319858551025,5.988941192626953,5.916278362274170,5.855669021606445,5.819603443145752,5.797814846038818,5.773691654205322,5.737600326538086,5.708529472351074,5.691607475280762,5.669276237487793,5.641047954559326,5.604503631591797,5.554409503936768,5.501516342163086,5.428918361663818,5.362350463867188,5.302621841430664,5.264510631561279,5.226120471954346,5.176902294158936,5.146851062774658,5.119312763214111,5.120434761047363,5.147938728332520,5.158279895782471,5.090811729431152,5.030261039733887,4.997862339019775,5.034145355224609,5.097373962402344,5.026574134826660,4.914320468902588,4.844807624816895,4.798109054565430,4.782680034637451,4.829996109008789,4.851108551025391,4.780149936676025,4.699617385864258,4.626884460449219,4.540028572082520,4.442646503448486,4.347493171691895,4.335340499877930,4.352534294128418,4.303999423980713,4.231421470642090,4.187922954559326,4.164196968078613,4.161715984344482,4.190659046173096,4.196704864501953,4.184081554412842,4.174217700958252,4.158133983612061,4.146560668945312,4.148229122161865,4.131157398223877,4.108975887298584,4.035457611083984,3.921121597290039,3.824353218078613,3.763711214065552,3.745786905288696,3.755051612854004,3.820370435714722,3.819959640502930,3.796881198883057,3.716379404067993,3.702134609222412,3.763025522232056,3.852619886398315,3.899652957916260,3.890413761138916,3.841056108474731,3.758039712905884,3.712119579315186,3.675970077514648,3.642591714859009,3.671782016754150,3.730222463607788,3.820172548294067,3.927342653274536,3.979892253875732,4.022863864898682,4.147732734680176,4.274947643280029,4.330847740173340,4.378749370574951,4.399205684661865,4.434498310089111,4.476202487945557,4.536744117736816,4.533425331115723,4.442706108093262,4.389002323150635,4.400430679321289,4.462772846221924,4.477552890777588,4.505796432495117,4.601646423339844,4.733809947967529,4.934429168701172,4.815742969512939,4.673584938049316,4.719190597534180,4.900602340698242,5.270493507385254,5.650305747985840,5.860616683959961,6.025705814361572,6.211218833923340,6.426295757293701,6.766153812408447,7.210365772247314,7.701918125152588,8.163361549377441,8.317446708679199,8.331032752990723,8.410386085510254,8.591182708740234,8.757582664489746,8.873217582702637,9.085048675537109,9.305638313293457,9.686599731445312,10.112338066101074,10.772291183471680,11.077277183532715,11.528722763061523,11.682065010070801,11.910692214965820,12.190708160400391,12.598879814147949,12.922795295715332,13.168378829956055,13.569618225097656,14.023753166198730,14.434729576110840,14.963546752929688,15.212292671203613,15.383792877197266,15.665549278259277,15.959129333496094,16.045928955078125,16.183153152465820,16.211273193359375,16.166589736938477,16.068443298339844,15.968262672424316,15.847568511962891,15.692958831787109,15.553006172180176,15.349737167358398,15.148150444030762,14.992700576782227,14.790844917297363,14.533531188964844,14.356827735900879,14.072775840759277,13.746688842773438,13.488968849182129,13.229818344116211,12.999344825744629,12.844485282897949,12.701662063598633,12.474884986877441,12.242556571960449,12.109392166137695,11.982228279113770,11.862581253051758,11.725476264953613,11.552036285400391,11.299898147583008,11.061742782592773,10.856314659118652,10.674612998962402,10.498144149780273,10.338490486145020,10.153298377990723,9.957339286804199,9.799833297729492,9.659771919250488,9.542078018188477,9.410679817199707,9.275730133056641,9.139700889587402,9.003455162048340,8.872183799743652,8.730531692504883,8.576863288879395,8.414276123046875,8.258967399597168,8.150975227355957,8.056682586669922,7.969748973846436,7.859433650970459,7.723304748535156,7.580330848693848,7.429591178894043,7.294116973876953,7.177041053771973,7.064651489257812,6.951172351837158,6.835390090942383,6.721502304077148,6.615817546844482,6.523767471313477,6.462653636932373,6.384339809417725,6.294068336486816,6.194204807281494,6.105031490325928,6.020966529846191,5.944420337677002,5.870284557342529,5.790615081787109,5.711858272552490,5.636820793151855,5.571314334869385,5.515630722045898,5.463914871215820,5.403617382049561,5.337999820709229,5.274630069732666,5.205309867858887,5.137258529663086,5.077537536621094,5.022357463836670,4.967229366302490,4.903712272644043,4.840069293975830,4.784541130065918,4.748188495635986,4.732269287109375,4.730269908905029,4.733112335205078,4.740354061126709,4.745314598083496,4.754373073577881,4.765054225921631,4.773756027221680,4.784882545471191,4.784261703491211,4.762232780456543,4.747881412506104,4.757143020629883,4.797953128814697,4.852407455444336,4.914259433746338,4.960544109344482,4.983034133911133,5.006079196929932,5.027617931365967,5.047439575195312,5.058347225189209,5.049508571624756,5.035178184509277,5.039586544036865,5.046986103057861,5.054087638854980,5.052189350128174,5.055358886718750,5.071825504302979,5.093952178955078,5.110785007476807,5.125187397003174,5.139640331268311,5.156218528747559,5.171985149383545,5.187937259674072,5.201303005218506,5.216678619384766,5.231701850891113,5.246028900146484,5.259452342987061,5.271413326263428,5.283185482025146,5.293581962585449,5.293412208557129,5.283962249755859,5.273189067840576,5.260622501373291,5.250647544860840,5.241805076599121,5.234564304351807,5.222686767578125,5.211853981018066,5.201045989990234,5.189259529113770,5.174319267272949,5.160471916198730,5.151184558868408,5.150746345520020,5.151503086090088,5.152854442596436,5.153058052062988,5.152826309204102,5.152260303497314,5.151272296905518,5.148958206176758,5.144312858581543,5.139403343200684,5.136547565460205,5.132636547088623,5.126588821411133,5.120756626129150,5.115336894989014,5.111860752105713,5.109864234924316,5.106749057769775,5.103148937225342,5.100293636322021,5.097828388214111,5.095407009124756,5.094638824462891,5.094941139221191,5.097133159637451,5.100564956665039,5.105216026306152,5.108319282531738,5.111969947814941,5.117643833160400,5.125150680541992,5.131722450256348,5.135263919830322,5.136389732360840,5.136982440948486,5.136862277984619,5.136836528778076,5.139200210571289,5.142301559448242,5.146812915802002,5.146733283996582,5.143676280975342,5.140727519989014,5.138195991516113,5.137362480163574,5.136950969696045,5.135814666748047,5.134786128997803,5.135399818420410,5.136360645294189,5.140963554382324,5.153757095336914,5.168745517730713,5.185648918151855,5.204764842987061,5.226162910461426])
        
        ex141516_class2_mean = np.array([58.360755920410156,58.131629943847656,57.941783905029297,57.791107177734375,57.568153381347656,57.585845947265625,57.941417694091797,58.724609375000000,58.099517822265625,58.643486022949219,58.499931335449219,57.976005554199219,58.146270751953125,57.919311523437500,57.954387664794922,58.278480529785156,58.127799987792969,57.916984558105469,57.952217102050781,57.892784118652344,57.865558624267578,57.883190155029297,57.874622344970703,57.756385803222656,57.704833984375000,58.135902404785156,58.675804138183594,59.360858917236328,59.508811950683594,59.475166320800781,59.472789764404297,59.591075897216797,59.674732208251953,59.710670471191406,59.834362030029297,59.766899108886719,59.654308319091797,59.652915954589844,59.655937194824219,59.559532165527344,59.453449249267578,59.370361328125000,59.323585510253906,59.286811828613281,59.125041961669922,58.967933654785156,58.823944091796875,58.722312927246094,58.650188446044922,58.560955047607422,58.535118103027344,58.533172607421875,58.487567901611328,58.365486145019531,58.261577606201172,58.169944763183594,58.109210968017578,58.066875457763672,58.030632019042969,58.001903533935547,57.985404968261719,57.958385467529297,57.929340362548828,57.897125244140625,57.873573303222656,57.851020812988281,57.813220977783203,57.756416320800781,57.684661865234375,57.614463806152344,57.547138214111328,57.485790252685547,57.410663604736328,57.335552215576172,57.263439178466797,57.194896697998047,57.131042480468750,57.059722900390625,56.995185852050781,56.931621551513672,56.859687805175781,56.786315917968750,56.721752166748047,56.658184051513672,56.604896545410156,56.553913116455078,56.506919860839844,56.459266662597656,56.407470703125000,56.348571777343750,56.284191131591797,56.220069885253906,56.152477264404297,56.093513488769531,56.034656524658203,55.974163055419922,55.922809600830078,55.876640319824219,55.831169128417969,55.786602020263672,55.742298126220703,55.699970245361328,55.660045623779297,55.623260498046875,55.584423065185547,55.541801452636719,55.500724792480469,55.463756561279297,55.427581787109375,55.391429901123047,55.363323211669922,55.337272644042969,55.306549072265625,55.275978088378906,55.247966766357422,55.221607208251953,55.193382263183594,55.164329528808594,55.132068634033203,55.098064422607422,55.065284729003906,55.029041290283203,54.993652343750000,54.958099365234375,54.928886413574219,54.901100158691406,54.872905731201172,54.846946716308594,54.822425842285156,54.808372497558594,54.804412841796875,54.814231872558594,54.813110351562500,54.804119110107422,54.788349151611328,54.762798309326172,54.735088348388672,54.706157684326172,54.680103302001953,54.657672882080078,54.638233184814453,54.627807617187500,54.633563995361328,54.639884948730469,54.647167205810547,54.630748748779297,54.617309570312500,54.605552673339844,54.582904815673828,54.556304931640625,54.531818389892578,54.500385284423828,54.457283020019531,54.416339874267578,54.376380920410156,54.331939697265625,54.292774200439453,54.245422363281250,54.201824188232422,54.163806915283203,54.124420166015625,54.039878845214844,53.947246551513672,53.823364257812500,53.692070007324219,53.580352783203125,53.476753234863281,53.390209197998047,53.333972930908203,53.110908508300781,52.982715606689453,52.946380615234375,52.936664581298828,52.849620819091797,52.738639831542969,52.630725860595703,52.532695770263672,52.466400146484375,52.497154235839844,52.595527648925781,52.727283477783203,52.834495544433594,52.834980010986328,52.829174041748047,52.813129425048828,52.801445007324219,52.807292938232422,52.829231262207031,52.856494903564453,52.864948272705078,52.848258972167969,52.861236572265625,52.887714385986328,52.927124023437500,52.940185546875000,52.940620422363281,52.940216064453125,52.934009552001953,52.917274475097656,52.890697479248047,52.860481262207031,52.827568054199219,52.799633026123047,52.776042938232422,52.754718780517578,52.734752655029297,52.706054687500000,52.645057678222656,52.569896697998047,52.490257263183594,52.416149139404297,52.342887878417969,52.275402069091797,52.222370147705078,52.157104492187500,52.093833923339844,52.017993927001953,51.948509216308594,51.877746582031250,51.802295684814453,51.725608825683594,51.646545410156250,51.576805114746094,51.504600524902344,51.439056396484375,51.369445800781250,51.292785644531250,51.204765319824219,51.109336853027344,50.978984832763672,50.817855834960938,50.704277038574219,50.616725921630859,50.532981872558594,50.449134826660156,50.363548278808594,50.276344299316406,50.188758850097656,50.096771240234375,49.999732971191406,49.864467620849609,49.691528320312500,49.520542144775391,49.357212066650391,49.185546875000000,49.011817932128906,48.806911468505859,48.578147888183594,48.371841430664062,48.174091339111328,48.012546539306641,47.858657836914062,47.730411529541016,47.592136383056641,47.436485290527344,47.276058197021484,47.104610443115234,46.934455871582031,46.771888732910156,46.606101989746094,46.444252014160156,46.273460388183594,46.116100311279297,45.960750579833984,45.808609008789062,45.646709442138672,45.474014282226562,45.291820526123047,45.115665435791016,44.936634063720703,44.758460998535156,44.573017120361328,44.390972137451172,44.200473785400391,44.013576507568359,43.819541931152344,43.621776580810547,43.446166992187500,43.284400939941406,43.127433776855469,42.981990814208984,42.831787109375000,42.687072753906250,42.544979095458984,42.404594421386719,42.253379821777344,42.100364685058594,41.946327209472656,41.798377990722656,41.657230377197266,41.521720886230469,41.393608093261719,41.258701324462891,41.126651763916016,40.996841430664062,40.871063232421875,40.751010894775391,40.639698028564453,40.526500701904297,40.408702850341797,40.289203643798828,40.160453796386719,40.028255462646484,39.897907257080078,39.770244598388672,39.642730712890625,39.520149230957031,39.398101806640625,39.281730651855469,39.166828155517578,39.068382263183594,38.962081909179688,38.854454040527344,38.734031677246094,38.612804412841797,38.504928588867188,38.378986358642578,38.266395568847656,38.163051605224609,38.064113616943359,37.985580444335938,37.903362274169922,37.807579040527344,37.696090698242188,37.591003417968750,37.484249114990234,37.324474334716797,37.144081115722656,36.956413269042969,36.800163269042969,36.661312103271484,36.558284759521484,36.447444915771484,36.323852539062500,36.213516235351562,36.116477966308594,36.030803680419922,35.940029144287109,35.822872161865234,35.669845581054688,35.502994537353516,35.315437316894531,35.123508453369141,34.952686309814453,34.821456909179688,34.696079254150391,34.594238281250000,34.505668640136719,34.418312072753906,34.328487396240234,34.231830596923828,34.121135711669922,34.005718231201172,33.889076232910156,33.774524688720703,33.656677246093750,33.553443908691406,33.451618194580078,33.353202819824219,33.248264312744141,33.140251159667969,33.024497985839844,32.900844573974609,32.762290954589844,32.619674682617188,32.479026794433594,32.353000640869141,32.234645843505859,32.120586395263672,32.006416320800781,31.890768051147461,31.771291732788086,31.643014907836914,31.517467498779297,31.382160186767578,31.240978240966797,31.078664779663086,30.905094146728516,30.725347518920898,30.551523208618164,30.370128631591797,30.186655044555664,29.994007110595703,29.775030136108398,29.557144165039062,29.336696624755859,29.105829238891602,28.871089935302734,28.625198364257812,28.383005142211914,28.138284683227539,27.900173187255859,27.675750732421875,27.455480575561523,27.238582611083984,27.022455215454102,26.811229705810547,26.602102279663086,26.416065216064453,26.242193222045898,26.085119247436523,25.932561874389648,25.797283172607422,25.673824310302734,25.546997070312500,25.418802261352539,25.280523300170898,25.140037536621094,24.994762420654297,24.844390869140625,24.690082550048828,24.522094726562500,24.340389251708984,24.160751342773438,23.983652114868164,23.811105728149414,23.628595352172852,23.442150115966797,23.262676239013672,23.079622268676758,22.893436431884766,22.703037261962891,22.487773895263672,22.266170501708984,22.027889251708984,21.788158416748047,21.534490585327148,21.269460678100586,21.006229400634766,20.751747131347656,20.471912384033203,20.186496734619141,19.896911621093750,19.620761871337891,19.361190795898438,19.127790451049805,18.912792205810547,18.701259613037109,18.503019332885742,18.327112197875977,18.160818099975586,17.998994827270508,17.835664749145508,17.687129974365234,17.536746978759766,17.361116409301758,17.162836074829102,16.954837799072266,16.726215362548828,16.490806579589844,16.261892318725586,16.049293518066406,15.852540969848633,15.665281295776367,15.511251449584961,15.373393058776855,15.261342048645020,15.138613700866699,14.959665298461914,14.757028579711914,14.460615158081055,14.181364059448242,13.911941528320312,13.664716720581055,13.413917541503906,13.155853271484375,12.944905281066895,12.809511184692383,12.650897979736328,12.361013412475586,12.055007934570312,11.788067817687988,11.624895095825195,11.515563964843750,11.358108520507812,11.148619651794434,10.987457275390625,10.902082443237305,10.794120788574219,10.676751136779785,10.598304748535156,10.581212997436523,10.589361190795898,10.632731437683105,10.647575378417969,10.595169067382812,10.538465499877930,10.453225135803223,10.369780540466309,10.384903907775879,10.403676986694336,10.537890434265137,10.671781539916992,10.701121330261230,10.865413665771484,11.017736434936523,11.119486808776855,11.129032135009766,11.142699241638184,11.245037078857422,11.346651077270508,11.258409500122070,11.180767059326172,11.184949874877930,11.137389183044434,11.083795547485352,10.996148109436035,10.953874588012695,10.982431411743164,10.997757911682129,10.989600181579590,10.966450691223145,10.941596984863281,10.917523384094238,10.913988113403320,10.920657157897949,10.926538467407227,10.916605949401855,10.916990280151367,10.923172950744629,10.966333389282227,11.061815261840820,11.162408828735352,11.269421577453613,11.402085304260254,11.577955245971680,11.713823318481445,11.846687316894531,12.014701843261719,12.161510467529297,12.306002616882324,12.445192337036133,12.597333908081055,12.728685379028320,12.830790519714355,12.941436767578125,13.020871162414551,13.177502632141113,13.370336532592773,13.516868591308594,13.688196182250977,13.779457092285156,14.016163825988770,14.077684402465820,14.110000610351562,14.039785385131836,14.091105461120605,14.377057075500488,14.542849540710449,14.676341056823730,14.825599670410156,14.970944404602051,15.129228591918945,15.253819465637207,15.406588554382324,15.592175483703613,15.793467521667480,15.978976249694824,16.157173156738281,16.335327148437500,16.510211944580078,16.721490859985352,16.925928115844727,17.081060409545898,17.261192321777344,17.458324432373047,17.683179855346680,17.955833435058594,18.183912277221680,18.364902496337891,18.579589843750000,18.739742279052734,18.963258743286133,19.195726394653320,19.390659332275391,19.581338882446289,19.744169235229492,19.830486297607422,19.972072601318359,20.047496795654297,20.164472579956055,20.298175811767578,20.470020294189453,20.709257125854492,21.010303497314453,21.288000106811523,21.525506973266602,21.777639389038086,22.092700958251953,22.394710540771484,22.638572692871094,22.887855529785156,23.153919219970703,23.403606414794922,23.669076919555664,23.961822509765625,24.258876800537109,24.532915115356445,24.809375762939453,25.089494705200195,25.356294631958008,25.635143280029297,25.892456054687500,26.116558074951172,26.312231063842773,26.501098632812500,26.675802230834961,26.852575302124023,27.047756195068359,27.257221221923828,27.467433929443359,27.694116592407227,27.939804077148438,28.181976318359375,28.419078826904297,28.652002334594727,28.897878646850586,29.165353775024414,29.392904281616211,29.656560897827148,29.962152481079102,30.301868438720703,30.663177490234375,31.001846313476562,31.321279525756836,31.605869293212891,31.865329742431641,32.128337860107422,32.403678894042969,32.680675506591797,32.929725646972656,33.206912994384766,33.518363952636719,33.829216003417969,34.101192474365234,34.390220642089844,34.653205871582031,34.919868469238281,35.198211669921875,35.491909027099609,35.827682495117188,36.125797271728516,36.468601226806641,36.831089019775391,37.191040039062500,37.554653167724609,37.888999938964844,38.194198608398438,38.482521057128906,38.750740051269531,39.017963409423828,39.270053863525391,39.527000427246094,39.795711517333984,40.092910766601562,40.407276153564453,40.710002899169922,40.988288879394531,41.259147644042969,41.548618316650391,41.869354248046875,42.175399780273438,42.453174591064453,42.690975189208984,42.898765563964844,43.089172363281250,43.272697448730469,43.466938018798828,43.635738372802734,43.731704711914062,43.868518829345703,44.057548522949219,44.242565155029297,44.454582214355469,44.663433074951172,44.878906250000000,45.106739044189453,45.322807312011719,45.523639678955078,45.710884094238281,45.884181976318359,46.042079925537109,46.204269409179688,46.372611999511719,46.517616271972656,46.663436889648438,46.807666778564453,46.924121856689453,47.040683746337891,47.157295227050781,47.277561187744141,47.405128479003906,47.518600463867188,47.624355316162109,47.740425109863281,47.866382598876953,47.998863220214844,48.144260406494141,48.302845001220703,48.460182189941406,48.608150482177734,48.756843566894531,48.947322845458984,49.157859802246094,49.377296447753906,49.568130493164062,49.760204315185547,49.981773376464844,50.184097290039062,50.387683868408203,50.575721740722656,50.763652801513672,50.914901733398438,51.083148956298828,51.299552917480469,51.510040283203125,51.723205566406250,51.934577941894531,52.206146240234375,52.449756622314453,52.615974426269531,52.685352325439453,52.778472900390625,52.957714080810547,53.292419433593750,53.426322937011719,53.622730255126953,53.707752227783203,53.615295410156250,53.729442596435547,53.835491180419922,53.792785644531250,53.757957458496094,53.816764831542969,53.915489196777344,53.947608947753906,53.831806182861328,53.639030456542969,53.507240295410156,53.235767364501953,52.667381286621094,52.457092285156250,52.665710449218750,53.278190612792969,53.572711944580078,53.621746063232422,53.652759552001953,53.938159942626953,53.911052703857422,54.024269104003906,54.232147216796875,54.479644775390625,54.637351989746094,54.852344512939453,55.051170349121094,55.234710693359375,55.425773620605469,55.584281921386719,55.785175323486328,56.010063171386719,56.240661621093750,56.464244842529297,56.694538116455078,56.921314239501953,57.126884460449219,57.335807800292969,57.545654296875000,57.758415222167969,57.966888427734375,58.164733886718750,58.326160430908203,58.444957733154297,58.560920715332031,58.691791534423828,58.806114196777344,58.840263366699219,58.858936309814453,58.842689514160156,58.893302917480469,59.005317687988281,59.116191864013672,59.224250793457031,59.320911407470703,59.428417205810547,59.565162658691406,59.708362579345703,59.835945129394531,59.922142028808594,59.980865478515625,60.034656524658203,60.098011016845703,60.166370391845703,60.289215087890625,60.443042755126953,60.561870574951172,60.591468811035156,60.580821990966797,60.617736816406250,60.671825408935547,60.721275329589844])
        ex141516_class2_std = np.array([16.570016860961914,16.366165161132812,16.420372009277344,16.413867950439453,16.436830520629883,16.243225097656250,16.042444229125977,16.798627853393555,16.246946334838867,16.811840057373047,16.383903503417969,16.087665557861328,16.198345184326172,15.961622238159180,15.639066696166992,15.454683303833008,15.611071586608887,15.711094856262207,15.798887252807617,15.799748420715332,15.939246177673340,16.190361022949219,16.150793075561523,16.457035064697266,16.573362350463867,16.459213256835938,15.378038406372070,14.129621505737305,13.853063583374023,13.841444969177246,13.862308502197266,13.943418502807617,13.986906051635742,13.936250686645508,13.985451698303223,13.877668380737305,13.720039367675781,13.601524353027344,13.498889923095703,13.336436271667480,13.212346076965332,13.164339065551758,13.213277816772461,13.262636184692383,13.230908393859863,13.140976905822754,13.065377235412598,13.002140045166016,12.966054916381836,12.960309982299805,13.017662048339844,13.086346626281738,13.136231422424316,13.171704292297363,13.191836357116699,13.199817657470703,13.201103210449219,13.195430755615234,13.191225051879883,13.184243202209473,13.193826675415039,13.215371131896973,13.239835739135742,13.274751663208008,13.316382408142090,13.337648391723633,13.342857360839844,13.311139106750488,13.280448913574219,13.233362197875977,13.186885833740234,13.143389701843262,13.094650268554688,13.047860145568848,12.991605758666992,12.936334609985352,12.879808425903320,12.819106101989746,12.747404098510742,12.667517662048340,12.572484016418457,12.462589263916016,12.361677169799805,12.245519638061523,12.134018898010254,12.013665199279785,11.871726989746094,11.720795631408691,11.560764312744141,11.381281852722168,11.194674491882324,11.017552375793457,10.840245246887207,10.691300392150879,10.546825408935547,10.399682998657227,10.290989875793457,10.196733474731445,10.105537414550781,10.021091461181641,9.943855285644531,9.870510101318359,9.799948692321777,9.736612319946289,9.675130844116211,9.618131637573242,9.562080383300781,9.511669158935547,9.460886955261230,9.411490440368652,9.371873855590820,9.331600189208984,9.282869338989258,9.232642173767090,9.185888290405273,9.142615318298340,9.096076011657715,9.051409721374512,9.002076148986816,8.951943397521973,8.903372764587402,8.844808578491211,8.786224365234375,8.727241516113281,8.677337646484375,8.629658699035645,8.583766937255859,8.541471481323242,8.500376701354980,8.481134414672852,8.485568046569824,8.514419555664062,8.526450157165527,8.530747413635254,8.522461891174316,8.497097015380859,8.469477653503418,8.442023277282715,8.425420761108398,8.420086860656738,8.426801681518555,8.450022697448730,8.509715080261230,8.570343017578125,8.631465911865234,8.619598388671875,8.618896484375000,8.631426811218262,8.622394561767578,8.608459472656250,8.610226631164551,8.601308822631836,8.574315071105957,8.540327072143555,8.503891944885254,8.447975158691406,8.399866104125977,8.347470283508301,8.304083824157715,8.262043952941895,8.212866783142090,8.162218093872070,8.118149757385254,8.040254592895508,7.966089725494385,7.939568519592285,7.930666446685791,7.945940494537354,7.960151672363281,7.925753116607666,7.886363506317139,7.813295841217041,7.770958423614502,7.781956672668457,7.804989814758301,7.842267513275146,7.895329475402832,7.932951927185059,7.965075969696045,8.075579643249512,8.235938072204590,8.392423629760742,8.318672180175781,8.255081176757812,8.139106750488281,7.945771694183350,7.729640960693359,7.559292316436768,7.455814361572266,7.399431705474854,7.358868598937988,7.349984169006348,7.357307910919189,7.379566192626953,7.403178215026855,7.432712554931641,7.472709655761719,7.513696670532227,7.547148704528809,7.563512802124023,7.577751159667969,7.593162059783936,7.627288818359375,7.665351867675781,7.706882476806641,7.744190692901611,7.758923053741455,7.752388477325439,7.742132663726807,7.741020202636719,7.752603530883789,7.768521785736084,7.794408321380615,7.839470386505127,7.862679004669189,7.889887809753418,7.882682323455811,7.871448040008545,7.859854698181152,7.846729755401611,7.831392765045166,7.821972370147705,7.828466892242432,7.839935302734375,7.858574390411377,7.875802993774414,7.886618614196777,7.887051105499268,7.876335144042969,7.856744289398193,7.835621833801270,7.834554195404053,7.842476844787598,7.854387283325195,7.869678020477295,7.884137630462646,7.895144462585449,7.905234336853027,7.912266254425049,7.916014194488525,7.886558532714844,7.830403327941895,7.762292861938477,7.699037551879883,7.637791156768799,7.586807727813721,7.533357620239258,7.481655597686768,7.441068172454834,7.425587177276611,7.428554534912109,7.452500343322754,7.495974063873291,7.545096874237061,7.585340976715088,7.628597259521484,7.660209655761719,7.702733516693115,7.755377769470215,7.809015750885010,7.867561340332031,7.923968315124512,7.957803726196289,7.992352485656738,8.036121368408203,8.074822425842285,8.103343009948730,8.128086090087891,8.136872291564941,8.150815963745117,8.175073623657227,8.194304466247559,8.228549003601074,8.271643638610840,8.322128295898438,8.389406204223633,8.468190193176270,8.529125213623047,8.589260101318359,8.643301963806152,8.687764167785645,8.731287002563477,8.773627281188965,8.805580139160156,8.826134681701660,8.848367691040039,8.871480941772461,8.899345397949219,8.935787200927734,8.970132827758789,9.007300376892090,9.051445007324219,9.106653213500977,9.172584533691406,9.241725921630859,9.315500259399414,9.400662422180176,9.491624832153320,9.584620475769043,9.678675651550293,9.771075248718262,9.865019798278809,9.968473434448242,10.073766708374023,10.183334350585938,10.297355651855469,10.410762786865234,10.521318435668945,10.616037368774414,10.708951950073242,10.808044433593750,10.908102989196777,10.997865676879883,11.073007583618164,11.134614944458008,11.140284538269043,11.141203880310059,11.154147148132324,11.198204994201660,11.251941680908203,11.302873611450195,11.318961143493652,11.330305099487305,11.333600044250488,11.344191551208496,11.356474876403809,11.199183464050293,11.033866882324219,10.843857765197754,10.704338073730469,10.605917930603027,10.567617416381836,10.510808944702148,10.444579124450684,10.417702674865723,10.441663742065430,10.504636764526367,10.555358886718750,10.550582885742188,10.502364158630371,10.447477340698242,10.362834930419922,10.313225746154785,10.289345741271973,10.323439598083496,10.367620468139648,10.436718940734863,10.562689781188965,10.697624206542969,10.824373245239258,10.931730270385742,11.022438049316406,11.101984024047852,11.182376861572266,11.275640487670898,11.375102043151855,11.463808059692383,11.530851364135742,11.596568107604980,11.661597251892090,11.731163024902344,11.792445182800293,11.853663444519043,11.891880989074707,11.914173126220703,11.933178901672363,11.951601982116699,11.987991333007812,12.031616210937500,12.064869880676270,12.068065643310547,12.074176788330078,12.090677261352539,12.137962341308594,12.196447372436523,12.259742736816406,12.272279739379883,12.265677452087402,12.256953239440918,12.257131576538086,12.249279022216797,12.242218017578125,12.229393005371094,12.187797546386719,12.143206596374512,12.096380233764648,12.050478935241699,12.002151489257812,11.957782745361328,11.941814422607422,11.934972763061523,11.942514419555664,11.960655212402344,11.995686531066895,12.046158790588379,12.104657173156738,12.167174339294434,12.236166954040527,12.291409492492676,12.334215164184570,12.353785514831543,12.372141838073730,12.392930984497070,12.419634819030762,12.443841934204102,12.470557212829590,12.512452125549316,12.558826446533203,12.612140655517578,12.669428825378418,12.724987030029297,12.789969444274902,12.863928794860840,12.956974029541016,13.048711776733398,13.125717163085938,13.195884704589844,13.253709793090820,13.302594184875488,13.342864036560059,13.364153861999512,13.371173858642578,13.366030693054199,13.365094184875488,13.353053092956543,13.339282035827637,13.331482887268066,13.333291053771973,13.344848632812500,13.372111320495605,13.426938056945801,13.503391265869141,13.586560249328613,13.656438827514648,13.717433929443359,13.769892692565918,13.816885948181152,13.858922004699707,13.887540817260742,13.906723022460938,13.924897193908691,13.956970214843750,13.988107681274414,14.009445190429688,14.032852172851562,14.050145149230957,14.082983016967773,14.127737998962402,14.189775466918945,14.233279228210449,14.276523590087891,14.316246986389160,14.326801300048828,14.328576087951660,14.352573394775391,14.412931442260742,14.560271263122559,14.808115959167480,14.977073669433594,15.170167922973633,15.186238288879395,15.326768875122070,15.522613525390625,15.639069557189941,15.727498054504395,15.801304817199707,15.787012100219727,15.852021217346191,15.932849884033203,16.073415756225586,16.297632217407227,16.424587249755859,16.373533248901367,16.422080993652344,16.585103988647461,16.656867980957031,16.757143020629883,16.821851730346680,16.877424240112305,16.941917419433594,17.027664184570312,17.144830703735352,17.281227111816406,17.553331375122070,17.913860321044922,18.091264724731445,18.387918472290039,18.719078063964844,18.908224105834961,19.317987442016602,19.721841812133789,20.135511398315430,20.412067413330078,20.509471893310547,21.018802642822266,21.252035140991211,21.324712753295898,21.108222961425781,20.861677169799805,20.886545181274414,21.117879867553711,20.890743255615234,20.779989242553711,20.965307235717773,20.983150482177734,20.935401916503906,20.687160491943359,20.551399230957031,20.591098785400391,20.610717773437500,20.538326263427734,20.417676925659180,20.287916183471680,20.126142501831055,20.016208648681641,19.944520950317383,19.879808425903320,19.776943206787109,19.694210052490234,19.592039108276367,19.507699966430664,19.455278396606445,19.421051025390625,19.378406524658203,19.391206741333008,19.575695037841797,19.713666915893555,19.890663146972656,20.116664886474609,20.159105300903320,20.117265701293945,20.034341812133789,19.980920791625977,19.876649856567383,19.747859954833984,19.678180694580078,19.524248123168945,19.554567337036133,19.641489028930664,19.581521987915039,19.530078887939453,19.291830062866211,19.390729904174805,19.164768218994141,18.889080047607422,18.427129745483398,18.141574859619141,18.275985717773438,18.178678512573242,18.012516021728516,17.869546890258789,17.695713043212891,17.522855758666992,17.334754943847656,17.211709976196289,17.107257843017578,16.978309631347656,16.821434020996094,16.619909286499023,16.374252319335938,16.089126586914062,15.866895675659180,15.645238876342773,15.303269386291504,15.058835029602051,14.908010482788086,14.815352439880371,14.781064033508301,14.673149108886719,14.491416931152344,14.492847442626953,14.377763748168945,14.383639335632324,14.425402641296387,14.408911705017090,14.362083435058594,14.268236160278320,14.077718734741211,13.900617599487305,13.698082923889160,13.561114311218262,13.479228019714355,13.424734115600586,13.395451545715332,13.376895904541016,13.322940826416016,13.249657630920410,13.157778739929199,13.076030731201172,12.996610641479492,12.923950195312500,12.855772972106934,12.779510498046875,12.690637588500977,12.605026245117188,12.518915176391602,12.437923431396484,12.365949630737305,12.283109664916992,12.188183784484863,12.092744827270508,11.997024536132812,11.874445915222168,11.755282402038574,11.651371955871582,11.553909301757812,11.474163055419922,11.401107788085938,11.337265968322754,11.276540756225586,11.197583198547363,11.117959022521973,11.028044700622559,10.936644554138184,10.849807739257812,10.767732620239258,10.683375358581543,10.598371505737305,10.549673080444336,10.504121780395508,10.447659492492676,10.391219139099121,10.339878082275391,10.303062438964844,10.287135124206543,10.270953178405762,10.257062911987305,10.254649162292480,10.247427940368652,10.237663269042969,10.225654602050781,10.213171005249023,10.198953628540039,10.178773880004883,10.149932861328125,10.127980232238770,10.102303504943848,10.080545425415039,10.063409805297852,10.049753189086914,10.047271728515625,10.023541450500488,10.054537773132324,10.130648612976074,10.264346122741699,10.446241378784180,10.619310379028320,10.715728759765625,10.707406997680664,10.615447998046875,10.529953002929688,10.406824111938477,10.270236015319824,10.196152687072754,10.234350204467773,10.308512687683105,10.344932556152344,10.341187477111816,10.331125259399414,10.382517814636230,10.560093879699707,10.734829902648926,10.828474998474121,10.843824386596680,10.824274063110352,10.739628791809082,10.640275001525879,10.581332206726074,10.500330924987793,10.340250015258789,10.243091583251953,10.201990127563477,10.183226585388184,10.195496559143066,10.199295997619629,10.217824935913086,10.290300369262695,10.365786552429199,10.396173477172852,10.403020858764648,10.377459526062012,10.348912239074707,10.362875938415527,10.366374015808105,10.359305381774902,10.358547210693359,10.354527473449707,10.329335212707520,10.307523727416992,10.282725334167480,10.257747650146484,10.240124702453613,10.152898788452148,10.024385452270508,9.898546218872070,9.834340095520020,9.790169715881348,9.774087905883789,9.761786460876465,9.741018295288086,9.714452743530273,9.681412696838379,9.713591575622559,9.780865669250488,9.846034049987793,9.868823051452637,9.884562492370605,9.970936775207520,10.001669883728027,10.018263816833496,10.014023780822754,10.005675315856934,9.957871437072754,9.948460578918457,10.023350715637207,10.062514305114746,10.108064651489258,10.142740249633789,10.200565338134766,10.285672187805176,10.335817337036133,10.212944030761719,10.299369812011719,10.567416191101074,10.826381683349609,10.978982925415039,11.439563751220703,11.813832283020020,12.487153053283691,13.171720504760742,13.526101112365723,13.629242897033691,13.978600502014160,14.512338638305664,14.889068603515625,14.852226257324219,15.161656379699707,15.035285949707031,14.180328369140625,13.130412101745605,12.150096893310547,12.565107345581055,12.951509475708008,14.308024406433105,13.882252693176270,13.309473037719727,12.734990119934082,11.594950675964355,11.258274078369141,10.825196266174316,10.542204856872559,10.620296478271484,10.587382316589355,10.450853347778320,10.391335487365723,10.376425743103027,10.349369049072266,10.412838935852051,10.474328041076660,10.543589591979980,10.590559959411621,10.713720321655273,10.792061805725098,10.824166297912598,10.842563629150391,10.897347450256348,10.976055145263672,11.037585258483887,11.080087661743164,11.131536483764648,11.194657325744629,11.275796890258789,11.375813484191895,11.516784667968750,11.663830757141113,11.717406272888184,11.735116958618164,11.721779823303223,11.755358695983887,11.805956840515137,11.803980827331543,11.763766288757324,11.722147941589355,11.727976799011230,11.781064033508301,11.827228546142578,11.862681388854980,11.948155403137207,12.027210235595703,12.098163604736328,12.154377937316895,12.205136299133301,12.269984245300293,12.325514793395996,12.349973678588867,12.340440750122070,12.330597877502441,12.340264320373535,12.374560356140137,12.422119140625000])
        
        features_func_mapping = {2:generate_features_vector_ex234,
                      3:generate_features_vector_ex234,
                      4:generate_features_vector_ex234,
                      14:generate_features_vector_ex141516,
                      15:generate_features_vector_ex141516,
                      16:generate_features_vector_ex141516
                       }

        camera_positions = {2:'f',3:'f',4:'f',
                      14:'s',15:'s',16:'s'}
        
        ##list of points on the aligned signal for each exercise
        ##the indexes in [0] is the golden feature's  points, other points are points for additional features, e.g.
        ##[272,420]: [raise leg beginning, leg raised on bed], 
        ##[420,560]: [leg raised one bed, leg on_bed_end, start going down]
        exercises_points = [[[272,420],[420,560]],
                            [[220,472],[472,570]]]
        DTAN_means = [ex234_class2_mean, ex141516_class2_mean]
        DTAN_stds = [ex234_class2_std, ex141516_class2_std]

        DTAN_models = ['../../checkpoints/1673108357_ex234_split_iter_modelstate_dict.pth',
                       '../../checkpoints/1677755522_ex1516_manual_modelstate_dict.pth']
        ex_num = int(dataset.split("_")[1][1:])
        patient_name = (dataset.split("_")[0])
        label_score = str(scores_dict[patient_name][ex_num])
        ##if DTAN model is provided as a parameter, use it, otherwise use the globally defined one
        if (DTAN_model != 0):
            self.DTAN_model = DTAN_model
        else:
            self.DTAN_model = DTAN_models[self.ex_num]
        self.ex_num = ex_mapping[ex_num]
        self.points = exercises_points[self.ex_num]
        self.DTAN_mean = DTAN_means[self.ex_num]
        self.DTAN_std = DTAN_stds[self.ex_num]
        self.side = scores_dict[patient_name][0]
        self.patient_name = patient_name
        self.camera_position = camera_positions[ex_num]
        self.generate_feature_vec = features_func_mapping[ex_num]

    def get_patient_name(self):
        return self.patient_name

    def get_golden_points(self):
        return self.points[0]

    def get_points(self):
        return self.points

    def get_DTAN_mean(self):
        return self.DTAN_mean

    def get_DTAN_std(self):
        return self.DTAN_std

    def get_side(self):
        return self.side

def track_joint1D(joint_positions):
    position_deltas = [joint_positions[i+1][0] - joint_positions[i][0] for i in range(len(joint_positions)-1)]
    stability_measure = statistics.stdev(position_deltas)
    return stability_measure * 100

def track_joint2D(joint_positions):
    position_deltas = [math.sqrt((joint_positions[i+1][0] - joint_positions[i][0])**2 + 
                                 (joint_positions[i+1][1] - joint_positions[i][1])**2) for i in range(len(joint_positions)-1)]
    stability_measure = statistics.stdev(position_deltas)
    return stability_measure * 100

def track_joint3D(joint_positions):
    position_deltas = [math.sqrt((joint_positions[i+1][0] - joint_positions[i][0])**2 + 
                                 (joint_positions[i+1][1] - joint_positions[i][1])**2 + 
                                 (joint_positions[i+1][2] - joint_positions[i][2])**2) for i in range(len(joint_positions)-1)]
    stability_measure = statistics.stdev(position_deltas)
    return stability_measure * 100

def inference_signal(interpolated_X_test, loaded_model, input_shape, exercise_obj):
    X = torch.Tensor(interpolated_X_test).to("cpu")

    transformed_input_tensor, thetas = loaded_model(X, return_theta=True)
    transformed_input_numpy = transformed_input_tensor.data.cpu().numpy()
    transformed_input_numpy = transformed_input_numpy.reshape(int(transformed_input_numpy.size/input_shape), input_shape)
    mse = []
    msem = []
    std_mean = []
    t = range(transformed_input_numpy[0].size)
    for iter_num in range(transformed_input_numpy.shape[0]):
        mse.append((exercise_obj.get_DTAN_mean() - transformed_input_numpy[iter_num]) ** 2)
        msem.append(np.mean((exercise_obj.get_DTAN_mean() - transformed_input_numpy[iter_num]) ** 2))
        
        std_mean.append(np.mean(np.abs(mse[-1] / exercise_obj.get_DTAN_std() ** 2)))
        print(f"MeanSquaredError={msem[iter_num]}, std_mean={std_mean[iter_num]}")
        ##Plot inference result compared to mean signal
        ##
        upper_t = exercise_obj.get_DTAN_mean() + exercise_obj.get_DTAN_std()
        lower_t = exercise_obj.get_DTAN_mean() - exercise_obj.get_DTAN_std()

        if (PLOT_ENABLED):
            plt.plot(t, transformed_input_numpy[iter_num], label="Aligned", color ="green")
            plt.plot(t, exercise_obj.get_DTAN_mean(), label="mean", color ="blue")
            plt.fill_between(t, upper_t, lower_t, color='#539caf', alpha=0.6, label=r"$\pm\sigma$")
            plt.legend()
            plt.show()

    return transformed_input_numpy, thetas, std_mean

def read_MPoutput_file1(file_name, ex_num, has_header=1):
    with open(f"{file_name}_landmarks.json", "r") as file:
        lines = file.readlines()
        list_from_file = list(map(float, lines[1].strip().split(",")))
    return list_from_file

def read_MPoutput_file(file_name, ex_num, has_header=1):
    #f_name = r"D:\M.Sc_study\github\thesis\ilan_computerVisionML_AI\mediapipe_eval\data\p7\h15_f2_out.txt"
    values=[]         #an empty list to store the second column
    with open(file_name+'.txt', 'r') as rf:
        reader = csv.reader(rf, delimiter=',')
        if (has_header):
            next(reader) #uncomment if input file has a header line
        for row in reader:
            #if (ex_num > 1 and ex_num < 5):
            values.append([float(x) for x in row])
            #if (ex_num > 13 and ex_num < 17):
            #values.append(float(row[1])) #change to row[1] if input file has two columns
    #P#print(values)
    return values
model_init = 0
loaded_model = 0
loaded_model_recurrence = 0




def inference(args, dataset="ECGFiveDays"):

    # Print args
    #print(args)

    # Data
    datadir = args.dpath #"data/UCR/UCR_TS_Archive_2015"
    device = 'cpu'
    exp_name = f"{dataset}_exp"
    
    # Init an instance of the experiment class. Holds results
    # and trainning param such as lr, n_epochs etc
    expManager = ExperimentsManager()
    expManager.add_experiment(exp_name, n_epochs=args.n_epochs, batch_size=args.batch_size, lr=args.lr, device=device)
    Experiment = expManager[exp_name]

    # DTAN args
    DTANargs1 = DTAN_args(tess_size=args.tess_size,
                          smoothness_prior=args.smoothness_prior,
                          lambda_smooth=args.lambda_smooth,
                          lambda_var=args.lambda_var,
                          n_recurrences=args.n_recurrences,
                          zero_boundary=True,
                          )
    DTANargs2 = DTAN_args(tess_size=args.tess_size,
                          smoothness_prior=args.smoothness_prior,
                          lambda_smooth=args.lambda_smooth,
                          lambda_var=args.lambda_var,
                          n_recurrences=args.n_recurrences+1,
                          zero_boundary=True,
                          )
    expManager[exp_name].add_DTAN_arg(DTANargs1)
    CHANNELS = 1
    DTANargs = Experiment.get_DTAN_args()

    ex_num = int(dataset.split("_")[1][1:])
    patient_name = (dataset.split("_")[0])
    f_name = os.path.join(datadir, dataset)
    label_score = str(scores_dict[patient_name][ex_num])
    inference_ex = Exercise(dataset, DTAN_model = '../../checkpoints/1673108357_ex234_split_iter_modelstate_dict.pth')

    channels = CHANNELS
    input_shape = SIGNAL_LENGTH
    global model_init
    global loaded_model
    global loaded_model_recurrence
    if (model_init == 0):
        model_init = 1
        loaded_model = DTAN(input_shape, channels, tess=[DTANargs.tess_size,], n_recurrence=DTANargs.n_recurrences,
                        zero_boundary=DTANargs.zero_boundary, device=device).to(device)
        loaded_model.load_state_dict(torch.load(inference_ex.DTAN_model))
        loaded_model.eval()
    
        loaded_model_recurrence = DTAN(input_shape, channels, tess=[DTANargs.tess_size,], n_recurrence=DTANargs.n_recurrences + 1,
                        zero_boundary=DTANargs.zero_boundary, device=device).to(device)
        loaded_model_recurrence.load_state_dict(torch.load(inference_ex.DTAN_model))
        loaded_model_recurrence.eval()

    try:
        features = read_MPoutput_file1(f_name, ex_num, has_header=1)
    except:
        print(f"error reading file {f_name}")
        return
    X_test = np.array([features, features])

    #P#print(X_test)
    #X_test = np.loadtxt(file_name+'.txt',delimiter=',') #for csv input in one row [e.g. 4,1,5,2,1,1]

    # add a third channel for univariate data
    if len(X_test.shape) < 3:
        X_test = np.expand_dims(X_test, -1)
    # Switch channel dim ()
    # Torch data format is  [N, C, W] W=timesteps
    X_test = np.swapaxes(X_test, 2, 1)

    interpolated_X_test = []
    
    iterations, split_list = split_signal2iterations(X_test[0].reshape(X_test[0].size), ex_num)
    #iterations = split_signal2iterations(my_inter(X_test[0], input_shape), ex_num)
    if (iterations == 0):
        feature_vec = [-9999, 9999, 9999, -9999, -9999, 9999, 9999, 9999, 9999, 9999]
        if (FWRITE_ENABLED):
            output_file_h = open(output_feature_vec, 'a', encoding="utf8")
            output_file_h.write("[\"{}_f{}\",{},{}],\n".format(patient_name, ex_num, label_score, str(feature_vec)))
            output_file_h.close()
        return 9999
    
    split_list = np.insert(split_list, 0, 0.)
    b, a = butter(2, 0.8)
    for iter in iterations:
        interpolated_sig = my_inter(iter, SIGNAL_LENGTH)
        smoothed_iteration = filtfilt(b, a, interpolated_sig)
        interpolated_X_test.append(smoothed_iteration)
    interpolated_X_test = np.array(interpolated_X_test)
    ## add a third channel for univariate data
    if len(interpolated_X_test.shape) < 3:
        interpolated_X_test = np.expand_dims(interpolated_X_test, -1)
    interpolated_X_test = np.swapaxes(interpolated_X_test, 2, 1)

    ##start inference of input signal and verify std_mean values
    ##if an iteration has std_mean > 2 then we try to inference it once again
    transformed_input_numpy, thetas, std_mean = inference_signal(interpolated_X_test, loaded_model, input_shape, inference_ex)
    for iter_num in range(len(std_mean)):
        if (std_mean[iter_num] > 2):
            transformed_input_numpy_recurrence, thetas_recurrence, std_mean_recurrence = inference_signal(interpolated_X_test, 
                                                                         loaded_model_recurrence, input_shape, inference_ex)
            for iter_num in range(len(std_mean_recurrence)):
                if std_mean[iter_num] > std_mean_recurrence[iter_num]:
                    transformed_input_numpy[iter_num] = transformed_input_numpy_recurrence[iter_num]
                    #thetas[iter_num] = thetas_recurrence[iter_num]
                    std_mean[iter_num] = std_mean_recurrence[iter_num]
            break
    min_std_mean = min(std_mean)
    min_iter = std_mean.index(min_std_mean) + 1
    print(f"patient score is {min_std_mean} in iteration {min_iter}")
    orig = interpolated_X_test.reshape(interpolated_X_test.size)
    aligned = transformed_input_numpy.reshape(transformed_input_numpy.size)
    nb = np.arange(input_shape)

    identity = np.arange(input_shape * len(std_mean))
    identity = identity.reshape(len(std_mean), input_shape)
    ## add a third channel for univariate data
    if len(identity.shape) < 3:
        identity = np.expand_dims(identity, -1)
    identity = np.swapaxes(identity, 2, 1)
    identityTF = torch.Tensor(identity).to("cpu")
    identityTF_tensor = loaded_model.T.transform_data(identityTF, thetas[0], outsize=(loaded_model.input_shape,))
    identityTF_numpy = identityTF_tensor.data.cpu().numpy()
    try:
        if (thetas_recurrence):
            identityTF_tensor_recurrence = loaded_model_recurrence.T.transform_data(identityTF, thetas_recurrence[-1], outsize=(loaded_model_recurrence.input_shape,))
            identityTF_numpy_recurrence = identityTF_tensor_recurrence.data.cpu().numpy()
            ##in case an iteration got better score with recurrence DTAN, update its aligned signal in final array
            for iter_num in range(len(std_mean_recurrence)):
                if std_mean[iter_num] > std_mean_recurrence[iter_num]:
                    identityTF_numpy[iter_num] = identityTF_numpy_recurrence[iter_num]
    except NameError:
        print("no need for thetas_recurrence")
    ####
    landmarks_json_path = f"{datadir}/{patient_name}_{inference_ex.camera_position}{ex_num}_landmarks.json"
    identityTF_numpy = identityTF_numpy.reshape(len(std_mean), input_shape)
    interpolated_X_test = interpolated_X_test.reshape(len(std_mean), input_shape)
    transformed_input_numpy = transformed_input_numpy.reshape(len(std_mean), input_shape)
    
    #subtract 1 from min_iter to make it zero-indexed
    inference_ex.generate_feature_vec(identity, identityTF_numpy, interpolated_X_test, transformed_input_numpy,
                                   split_list, std_mean, min_iter-1, inference_ex, input_shape, landmarks_json_path)


if __name__ == "__main__":
    #15#s1516_split()
    print(output_feature_vec)
    args = argparser()
    if (args.dataset == 'ALL'):
        # Get a list of all .mp4 files in the directory
        mp4_files = glob.glob(f"{args.dpath}/*.mp4")
        for file_path in mp4_files:
            # Get the file name without the extension
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            match = search(r'\d+', file_name)
            if match:
                if (int(match.group()) > 38):
                    continue
            print(file_name)
            inference(args, dataset=file_name)
    else:
         inference(args, dataset=args.dataset)

