# 2016-07-07: Changed to up to date fps. ident parameter added to Positioner() creation

from __future__ import division

# Import from the standard Python libraries
import math
import re
#import simplejson
import json as simplejson
import numpy as np
import ntpath
import os

import matplotlib
#import matplotlib.pyplot as plt
from timeit import default_timer as timer
matplotlib.use("TkAgg")


# Import functions from the DNF and FPS shared library

try:
    import mocpath.conflict.fps_shared as fps
    import mocpath.path.pa_dnf as dnf
    import mocpath.path.pa_animate as animate
except ImportError:
    import fps_shared as fps
    import pa_dnf as dnf
    import pa_animate as animate

import parameters as param

positioners = []  # List of positioners

def path_generator(config_file, target_file, result_folder, if_animate=False, if_write_to_file=False):

    # mechanical properties
    target_from_file = True

    # Define some constants
    not_a_number = 10000       # Defined by Stefano for positioners without target #TODO: change the non-assigned positioners tag to avoid hard coding

    # Define simulation constants #TODO: Put the constants from hardware in a separate library
    sim_length = param.sim_length            # Number of steps
    dt = param.dt                      # Time step duration
    max_speed_alpha = param.max_speed_alpha  # Maximum RPM #0.2911
    min_speed_alpha = param.min_speed_alpha  # Minimum RPM #0.0764
    max_speed_beta = param.max_speed_beta  # Maximum RPM #0.3927
    min_speed_beta = param.min_speed_beta  # Minimum RPM #0.1026

    # Open the config file, locate the R and theta coordinations lines
    if config_file.lower().endswith('.cfg'):
        fr = open(config_file , 'r')
    else:
        print("Wrong configuration file type")


    #TODO : Change configuration file type to cvs to avoid hard coding
    ident = 0
    positioner_configs =[]

    for i, line in enumerate(fr): #search the lines to find a new row positioner data
        if 'R coordinate of positioner' in line:
            r_centre_focal = float(re.findall("\d+.\d+", line[17:35])[0])
            print(r_centre_focal)
        elif 'THETA coordinate of positioner' in line:
            theta_centre_focal = math.radians(float(re.findall("-?\d+.\d+", line[16:35])[0]))
            print(theta_centre_focal)
        elif 'As assembled orientation' in line:
            orient = float(re.findall("\d+.\d+", line[17:35])[0])
        elif 'Column number of positioner' in line:
            column = int(re.findall("\d+", line[20:35])[0])
        elif 'Row number of positioner' in line:
            row = int(re.findall("\d+", line[20:35])[0])
            ident += 1
            config = (ident,r_centre_focal,theta_centre_focal,orient,column,row)
            positioner_configs.append(config)
    fr.close()
    positioner_grid = fps.PositionerGrid(positioner_configs)

    # Define two motors for each positioner by assigning the target points
    if target_from_file:
        # read the targets from the file
        if target_file.lower().endswith('.txt'):
            fr2 = open(target_file ,'r')
        else:
            print("Wrong target file type")

        for i, line in enumerate(fr2):
            if i > positioner_grid.positioner_count: #Check if they are more targets than positioners
                print("The target files does not correspond to the configuration file ")
            # x1: R coord of the positioner in focal plane (x1 is not used here)
            # x2: THETA coordinate of positioner in focal plane (x2 is not used here)
            # x3: Radial distance of the target in focal plane
            # x4: Theta coordinate of positioner in focal plane
            # x5: For parity
            # x6: For parity
            # x7: Priority of the target/positioner (is assigned to the positioner in set_target...)

            x1,x2,x3,x4,x5,x6,x7,x8 = map(float, line.split())
            if x3 < not_a_number:  # there is a target assigned
                if x5 == 1 and x6==0:
                    parity = 1
                elif x5==0 and x6==1:
                    parity = -1

                j=i
                #positioner_grid.get_positioner(i+1).set_target(x3,math.radians(x4),parity,x7)
                positioner_grid.get_positioner(j + 1).set_target(x3, math.radians(x4), parity, x7, x8, j + 1)


                (alpha_position_target, beta_position_target) = positioner_grid.get_positioner(i+1).get_arm_angles()
                positioner_grid.positioners[i+1].add_motors(sim_length, math.radians(alpha_position_target), math.radians(beta_position_target),\
                                                            max_speed_alpha,min_speed_alpha,max_speed_beta,min_speed_beta)

            else:
                positioner_grid.positioners[i+1].add_motors(sim_length, not_a_number, not_a_number,\
                                                            max_speed_alpha,min_speed_alpha,max_speed_beta,min_speed_beta)
    else:
        print('No target file found')

    motionPlanning_positioner=[]

    startTime=timer()
    # Simulate motion and collision avoidance
    for i in range(sim_length-2):
        for positioner in positioner_grid.positioners[1:]: #The positioner array does not have an attribute on the first element
            if i==0:
                motionPlanning_positioner.append(dnf.Motion_Planning(positioner))
            (motor1_velocity, motor2_velocity) = motionPlanning_positioner[positioner.ident-1].pas_config_dnf(i)
            #(motor1_velocity, motor2_velocity) = dnf.pas_config_dnf(positioner,i)

            positioner.motor1.speed_array[i] = motor1_velocity
            positioner.motor2.speed_array[i] = motor2_velocity

            positioner.motor1.position_array[i+1] = positioner.motor1.position_array[i] + dt * motor1_velocity
            positioner.motor2.position_array[i+1] = positioner.motor2.position_array[i] + dt * motor2_velocity


    endTime=timer()
    ProcessTime=endTime-startTime
    print('Time taken is:',ProcessTime)

    print(len(positioners))
    print(positioners)

    in_target = 0
    target_achived = []
    have_target = 0

    for positioner in positioner_grid.positioners[1:]:
        if math.fabs(positioner.motor1.position_array[-1] - positioner.motor1.position_array[-2]) < 0.08 and\
                        math.fabs(positioner.motor2.position_array[-1] - positioner.motor2.position_array[-2]) < 0.08:
            in_target += 1
            target_achived.append(1)
        else:
            target_achived.append(0)
        if positioner.motor1.position_array[-1] < not_a_number:
            have_target += 1

    print('number of targets reached the goal',in_target ,'from',have_target)
    # Add one line to Stefano's file and save it with the same name_PA in the result folder

    file = os.path.join(result_folder, ntpath.basename(target_file)[0:-4] + '_PA.txt')
    fr3 = open(file ,'w')
    fr2 = open(target_file ,'r')

    for i, line in enumerate(fr2):
        fr3.write(line.rstrip('\n')+' '+str(target_achived[i])+'\n')

    fr2.close()
    fr3.close()

    # Write motor waveforms
    if if_write_to_file:

        fw = open(os.path.join(result_folder, 'motor_commands.txt'),'w')

        np.set_printoptions(precision=3)

        for positioner in positioner_grid.positioners[1:]:
            simplejson.dump(('Positioner ' + str(positioner.ident)), fw)
            fw.write('\n')
            array = np.array(positioner.motor1.position_array) * 180/math.pi
            array = np.round(array, decimals=4)
            simplejson.dump(array.tolist(), fw)
            fw.write('\n')
            array = np.array(positioner.motor2.position_array) * 180/math.pi
            array = np.round(array, decimals=4)
            simplejson.dump(array.tolist(), fw)
            fw.write('\n \n')
        fw.close()

    #annimate the positiners
    if if_animate:
        animate.run_animation(positioner_grid.positioners[1:])

    return
