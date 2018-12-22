# -*- coding: utf-8 -*-
"""
Created on Fri Mar 2 14:43:23 2018

@author: Hang Su

This file is written to control the simulation, and serves as the driver of the programme.

Must be used accompanied with the predefined class Gas in the file TR_gas.py.
"""

import numpy as np
import scipy as sp
import pylab as pl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import TR_gas
import sys
import pickle


kb = 1.38064852e-23


if __name__ == "__main__":

    ################################################################################
    # Systematic parameters
    # Parameters in this section will be applied to both new generated and loaded instances
    ################################################################################
    # No. of frames that you want to simulate for
    no_of_frame = 100

    # Length of time for each frame
    time_step = 1.e-9

    # After N ball-ball collisions, sample the distribution of speed
    sample_step = 60

    # Eanble/disable animation
    enable_anim = False

    # Enable/disable the echo function to keep tracking the physical quantitities for each frame
    enable_echo = True

    # No. of bins in histogram
    no_of_bin = 500

    # Enable/disable plotting the variation of pressure in time
    pressure_variation = True

    # Enable/disable loading
    enable_loading = False

    # The name of the file that you want to load
    load_file = 'TR_save'

    ################################################################################
    # Parameters for generating a new gas
    # No need to change parameters in this section if you are loading saved file
    ################################################################################
    # No. of dimensions of the atmosphere, either 2 or 3
    dimension = 2

    # No. of particles along the diameter of the circular container
    p_no_density = 6

    # Particle radius/ Append elements to generate molecules of different radii/masses
    particle_size = np.array( [ 6.e-9 ] )

    # Particle mass/ must have the same shape as particle_size
    particle_mass = np.array( [ 4.82e-26 ] )

    # Particle average speed/ must have the same shape as particle_size
    particle_speed = np.array( [ 1000 ] )

    # container radius / particle radius(The first one in the radius array)
    size_ratio = 1.e2

    # Enable/disable generating a big heavy particle at the center of the container
    brownian = False


################################################################################

   # load = input('Load the instance from last time? - y/n \n')

    if ( not enable_loading ):
        # Create a new instance
        current_instance = TR_gas.Gas( dimension, p_no_density, particle_size, particle_mass, particle_speed, size_ratio, brownian )
    else:
        # Load the previous instance
        try:
            with open( load_file, 'rb' ) as input:
                current_instance = pickle.load(input)
                print( 'Successfully loaded' )

        except:
            print( "Unexpected error:", sys.exc_info()[0] )
            raise

    ################################################################################
    # Initialization
    ################################################################################
    current_instance.set_time_step( time_step )

    current_instance.set_sample_step( sample_step )

    current_instance.set_anim( enable_anim )

    current_instance.set_echo( enable_echo )

    ################################################################################
    # Simulation
    ################################################################################

    # Move the instance in the dimension of time
    if ( current_instance.enable_anim() ):

        fig = plt.figure()

        ax = plt.axes( xlim=(- current_instance.container_radius(), current_instance.container_radius() ), ylim=( -current_instance.container_radius(), current_instance.container_radius() ) )

        ax.axes.set_aspect('equal')

        patches = current_instance.gas_patch()

        for pch in patches:

            ax.add_patch( pch )

        anim = animation.FuncAnimation( fig,
                                        current_instance.next_frame,
                                        frames = no_of_frame,
                                        interval = 10, # ms
                                        blit = True,
                                        repeat = False # stop the animation after frames
                                        )

        plt.show()


        current_instance.reset_patch()

    if ( not current_instance.enable_anim() ):

        i = 0

        while i < no_of_frame:

            current_instance.next_frame( i )

            i += 1


    #Save the instance
    with open( 'TR_save', 'wb' ) as output:

        pickle.dump( current_instance, output, pickle.HIGHEST_PROTOCOL )

    print ( current_instance )

    ################################################################################
    # Analysis part
    ################################################################################

    # Functions
    def Maxwell_boltzmann( v, dimension, T, mass ):
        '''
        Maxwell boltzmann distribution function
        '''
        if ( dimension == 2 ):
            return v * mass / kb / T * np.exp( - ( v**2 *.5 * mass ) / ( kb * T )  )
        if ( dimension == 3 ):
            return ( mass / 2. / np.pi / kb / T )**1.5 * 4. * np.pi *  v**2 * np.exp( - ( v**2 *.5 * mass ) / ( kb * T )  )

    def Maxwell_boltzmann_variance( dimension, T, mass ):
        '''
        Return the variance of M-B function
        '''
        if ( dimension == 2 ):
            return kb * T / mass * ( 4. - np.pi ) / 2.
        if ( dimension == 3 ):
            return kb * T / mass * ( 3. * np.pi - 8 ) / np.pi

    def Maxwell_boltzmann_mean( dimension, T, mass ):
        '''
        Return the mean of M-B function
        '''
        if ( dimension == 2 ):
            return ( np.pi / 2. )**.5 * ( mass / kb / T ) ** -.5
        if ( dimension == 3 ):
            return 2. * (kb * T / mass)**.5 * ( 2. / np.pi )


    # Plot speed distribution
    if current_instance.equilibrium():

        fig = plt.figure()

        ax1 = fig.add_subplot( 1,1,1 )

        ax1.tick_params( axis='x')

        ax1.tick_params( axis='y' )

        # Histogram
        speed_distribution = current_instance.speed_stat()[1:]

        plt.hist( speed_distribution, bins=no_of_bin, color='black' )

        # Plot Maxwell-Boltzmann distribution curve
        bin_width = ( max( speed_distribution ) - min( speed_distribution ) ) / no_of_bin

        plot_x = np.linspace( min( speed_distribution ), max( speed_distribution ), 200 )

        plot_y = np.zeros( np.shape( plot_x ) )

        # Plot Maxwell Boltzmann distribution for particles of different masses
        i = 0
        while( i < len( current_instance.no_of_particle() ) ):
            p_y = Maxwell_boltzmann( plot_x, current_instance.dimension(),\
             current_instance.temperature(),\
             current_instance.particle_mass()[i] ) * current_instance.sample() * current_instance.no_of_particle()[i] * bin_width

            plot_y += p_y

            plot_y_sep, = plt.plot( plot_x, p_y, 'y--', linewidth=2.0, label = 'Separate' )

            i += 1

        curve_superposition = plt.plot( plot_x, plot_y, 'r', linewidth=3.0, label = 'Superposition' )

        plt.legend( handles= [ curve_superposition[0], plot_y_sep ], loc = 'upper right')

        plt.text( .53 * plt.xlim()[1],  .36 * plt.ylim()[1] , current_instance )

        plt.xlabel( 'speed(ms^-1)' )
        plt.ylabel( 'count')
        plt.title( 'Speed distribution' )

        plt.show()

        # Compare the variance and mean
        if len( current_instance.no_of_particle() ) == 1:
            mb_mean = Maxwell_boltzmann_mean( \
                                        current_instance.dimension(), \
                                        current_instance.temperature(), \
                                        current_instance.particle_mass() * current_instance.no_of_particle() / np.sum( current_instance.no_of_particle() ) \
                                        )
            mb_var = Maxwell_boltzmann_variance( \
                                        current_instance.dimension(), \
                                        current_instance.temperature(), \
                                        current_instance.particle_mass()* current_instance.no_of_particle() / np.sum( current_instance.no_of_particle() ) \
                                        )

            print ( 'variance of data: ' + str( np.var( speed_distribution  ) ) )
            print ( 'variance of m-b distribution: ' + str(mb_var ) )
            print ( 'mean of data: : ' + str( np.mean( speed_distribution ) ) )
            print ( 'mean of m-b distribution: ' + str(mb_mean ) )
    else:
        print ( 'The system has not yet reached the equilibrium.' )


    # Plot the variation of pressure in time
    if pressure_variation:
        fig = plt.figure()

        time_list = current_instance.time_list()

        pressure_list = current_instance.pressure_list()

        plt.plot( time_list, pressure_list )

        plt.xlabel( 't/s' )
        plt.ylabel( 'pressure/Pa' )
        plt.title( 'Pressure variation in time' )

        plt.show()


    # Plot Brownian motion trails
    if current_instance.brownian():
        fig = plt.figure()

        r_list = current_instance.brownian_pos()

        plt.plot( r_list[:,0], r_list[:,1], '+' )

        plt.xlabel( 'x/m' )
        plt.ylabel( 'y/m' )
        plt.title( 'Brownian motion trail' )

        plt.show()
