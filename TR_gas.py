# -*- coding: utf-8 -*-
"""
Created on Fri Mar 2 14:43:23 2018

@author: Hang Su

This file is written to define class Gas.

Must be used accompanied with the predefined class Ball in the file TR_ball.py

This section will be used in the file TR_driver.py.
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import random
import TR_ball


kb = 1.38064852e-23

class Gas:

    def __init__( self, dimension, p_no_density, p_size, p_mass, p_speed, radius_ratio, brownian=False ):

        #################################### Parameters

        # No. of dimension of the atmosphere, either 2 or 3
        self.__dimension = dimension

        # No. of particles per diameter of the atmosphere
        self.__p_no_density = p_no_density

        # Particle radius/ Append elements to generate molecules of different radii/masses
        self.__particle_size = p_size

        # Particle mass
        self.__particle_mass = p_mass

        # Particle average speed
        self.__particle_speed = p_speed

        # container radius / particle radius(The first one in the radius array)
        self.__size_ratio = radius_ratio

        # Brownian
        self.__brownian = brownian


        # Time in each frame( with default value )
        self.__time_step = 5.e-10

        # After N collisions, sample the distribution of speed
        self.__sample_step = 100

        # Animation enable boolean
        self.__enable_anim = False

        # Echo enable boolean
        self.__enable_echo = True

        ################################### Initialization

        self.__container_size = self.__particle_size[0] * self.__size_ratio

        self.__no_of_p = np.zeros( np.shape( self.__particle_size ) )

        if self.__brownian :
            self.__no_of_p = np.append( self.__no_of_p, 1 )
            self.__particle_mass = np.append( self.__particle_mass, self.__particle_mass[ 0 ] * 1500 )
            self.__particle_size = np.append( self.__particle_size, self.__particle_size[ 0 ] * 15 )
            self.__particle_speed = np.append( self.__particle_speed, 0 )


        self.__gas = self.generate_gas()

        self.__ke_list = np.zeros( np.shape( self.__gas ) )

        self.__speed_list = np.zeros( np.shape( self.__gas ) )

        self.__speed_stat = np.array([])

        self.__mass_list = np.zeros( np.shape( self.__gas ) )

        self.__radius_list = np.zeros( np.shape( self.__gas ) )


        self.__pp_collision_accumulation = 0

        self.__no_of_collision = 0

        self.__no_of_pp_collision = 0

        self.__frame_number = 0

        self.__pressure_change = 0

        self.__equilibrium = False


        self.__time_list = np.array([])

        self.__pressure_list = np.array([])

        #################################### Physical variables

        self.__time_passed = 1.e-20

        self.__equilibrium_time_passed = 1.e-20

        self.__impulse_acc = float()

        self.__equilibrium_impulse_acc = float()

        self.__sample = 0



        self.__circumference = 2.* np.pi * self.__container_size

        self.__surface = np.pi * self.__container_size**2

        self.__sphere_surface = 4. * np.pi * self.__container_size**2

        self.__volume = 4./3. * np.pi * self.__container_size**3

        self.__pressure = float()

        self.__ke = float()

        self.__temperature = float()

        self.__ang_mom = np.zeros( np.shape( self.__gas[1].ang_mom() ) )

        self.__b = float()

        self.__alpha = 2.

        self.__beta = 1.

        self.__b_theoretical = np.sum( np.pi* self.__particle_size**2 * self.__alpha * self.__no_of_p ) + np.pi * self.__container_size * np.sum(self.__particle_size * self.__no_of_p) / np.sum(self.__no_of_p) * self.__beta

        if self.__brownian:
            self.__brownian_pos = self.__gas[1].pos()

        #################################### Initialization

        self.physical_update()


    def __repr__(self):
        return '\nDimension = %d \nTemperature = %3.1f K \nPressure = %.5e Nm^-%s \nParticle no. = %s \nSample = %d \nratio = %d\n ' % ( self.__dimension, self.__temperature, self.__pressure, str( self.__dimension-1 ), str(self.__no_of_p), self.__sample, self.__size_ratio )

################################################################################ Debuging mod functions
    def enable_anim( self ):
        return self.__enable_anim

    def set_anim( self, state ):
        self.__enable_anim = state

    def enable_echo( self ):
        return self.__enable_echo

    def set_echo( self, state ):
        self.__enable_echo = state

    def set_time_step( self, t ):
        self.__time_step = t

    def set_sample_step( self, c ):
        self.__sample_step = c

    def brownian( self ):
        return self.__brownian

    def equilibrium( self ):
        return self.__equilibrium

################################################################################ State functions
    def dimension( self ):
        return self.__dimension

    def particle_radius( self ):
        return self.__particle_size

    def particle_mass( self ):
        return self.__particle_mass

    def container_radius( self ):
        return self.__container_size

    def brownian_pos( self ):
        if self.brownian():
            return self.__brownian_pos
        else: return None

    def ke_list( self ):
        return self.__ke_list

    def speed_list( self ):
        return self.__speed_list

    def speed_stat( self ):
        return self.__speed_stat

    def mass_list( self ):
        return self.__mass_list

    def radius_list( self ):
        return self.__radius_list

    def no_of_collision( self ):
        return self.__no_of_collision

    def no_of_particle( self ):
        return self.__no_of_p

    def sample( self ):
        return self.__sample

    def temperature( self ):
        return self.__temperature

    def pressure( self ):
        return self.__pressure

    def time_list( self ):
        return self.__time_list

    def pressure_list( self ):
        return self.__pressure_list

################################################################################

    def append_speed_stat( self, arr ):
        '''
        Append the current sample to the total distribution
        '''
        self.__speed_stat = np.append( self.__speed_stat, arr )

    def gas_patch( self ):
        '''
        Return the patches of all instances
        '''
        patches = []
        for b in self.__gas:
            pch = b.sprite()
            patches.append( pch )
        return patches

    def reset_patch( self ):
        '''
        Reset patches so that they can be added to a new figure.
        '''
        for b in self.__gas:
            b.reset_sprite()

    def echo_physical_quantities( self ):
        print ( 'Frame ' + str.format('{:.0f}', self.__frame_number ) + '\n' +
                'Time ' + str.format('{:.2e}', self.__time_passed ) + ' s' + '\n' +
                'Dimension ' + str( self.__dimension ) + '\n' +
                'No. of particles ' + str( self.__no_of_p ) + '\n' +
                'No. of collision ' + str.format('{:.0f}', self.__no_of_collision ) + '\n' +
                'No. of p-p collision ' + str.format('{:.0f}', self.__no_of_pp_collision ) + '\n' +
                'Ratio ' + str( self.__size_ratio ) + '\n' +
                'Sample ' + str.format('{:.0f}', self.__sample ) + '\n' +
                'KE ' + str.format('{:.3e}', self.__ke ) + ' J' + '\n' +
                'Pressure ' + str.format('{:.5e}', self.__pressure ) + ' Nm^-' + str( self.__dimension-1 ) + '\n' +
                'Temperature ' + str.format('{0:.3f}', self.__temperature ) + ' K' + '\n' +
                'Angular momentum ' + str( self.__ang_mom ) + ' kgm^2s^-1' + '\n' +
                'b ' + str.format('{0:.5e}', self.__b) + ' m^' + str( self.__dimension ) + '\n' +
                'b(theoretical) ' + str.format('{0:.5e}', self.__b_theoretical) + ' m^' + str( self.__dimension ) + '\n' +
                'Equilibrium ' + str( self.__equilibrium ) + '\n'
               )
        return 0

    def overlap_check( self, gas, x, r ):
        '''
        Used when generate new gas to avoid particles overlappings
        '''
        overlap = False
        i = 1
        while ( i < len( gas ) ):
            displacement = gas[ i ].pos( )- x
            radius_sum = gas[ i ].radius() + r
            if ( np.dot( displacement, displacement ) < radius_sum**2 ):
                overlap = True
            i += 1
        return overlap

    def generate_gas( self ):
        '''
        Uniformly generate the gas in the container given.
        '''
        # Create positions arrays
        dim = list()
        i = 0
        while ( i < self.__dimension ):
            dim.append( np.linspace( -self.__container_size, self.__container_size, self.__p_no_density ) )
            i += 1

        gas = list()

        # Constuct 2d atmosphere
        if ( self.__dimension == 2 ):

            # Container
            gas.append( TR_ball.Ball( (0, 0), (0, 0), mass = float('Inf'), radius = self.__container_size, objtype = 'container' ) )

            # Big particle at the center if brownian boolean is True
            if self.__brownian :
                gas.append( TR_ball.Ball( (0, 0), ( 0, 0 )
                , mass = self.__particle_mass[ -1 ]
                , radius = self.__particle_size[ -1 ]
                ) )

            # Uniformly generate particles
            for i in dim[0]:

                for j in dim[1]:

                    # Randomly choose the type of molecule
                    if self.__brownian: molecule = int( random.random() * ( len( self.__particle_size ) - 1 ) )
                    else: molecule = int( random.random() * len( self.__particle_size ) )

                    # Randomize the velocity direction
                    phi = random.random() * 2.*np.pi

                    # Only generate a new particle when overlapping does not happen
                    if ( i**2 + j**2 < ( self.__container_size - self.__particle_size[ molecule ] )**2\
                        and not self.overlap_check( gas, np.array([i, j]), self.__particle_size[ molecule ] )):

                        gas.append( TR_ball.Ball( ( i, j )
                        , velocity = (  self.__particle_speed[ molecule ] * np.cos( phi ),
                                        self.__particle_speed[ molecule ] * np.sin( phi ),
                                        )
                        , mass = self.__particle_mass[ molecule ]
                        , radius = self.__particle_size[ molecule ]
                        ) )

                        self.__no_of_p[ molecule ] += 1

        #Constuct 3d atmosphere
        if ( self.__dimension == 3 ):

            gas.append( TR_ball.Ball( (0, 0, 0), (0, 0, 0), mass = float('Inf'), radius = self.__container_size, objtype = 'container' ) )

            if self.__brownian :
                gas.append( TR_ball.Ball( (0, 0, 0), ( 0, 0, 0 )
                , mass = self.__particle_mass[ 0 ] * 8000
                , radius = self.__particle_size[ 0 ] * 20
                ) )

            for i in dim[0]:

                for j in dim[1]:

                    for k in dim[2]:

                        if self.__brownian: molecule = int( random.random() * ( len( self.__particle_size ) - 1 ) )
                        else: molecule = int( random.random() * len( self.__particle_size ) )

                        phi = random.random() * 2.*np.pi

                        theta = random.random() * np.pi

                        if ( i**2 + j**2 + k**2 < ( self.__container_size - self.__particle_size[ molecule ] )**2\
                        and not self.overlap_check( gas, np.array([i, j, k]), self.__particle_size[ molecule ] )):

                            gas.append( TR_ball.Ball( ( i, j, k )
                            , velocity = (  self.__particle_speed[ molecule ] * np.sin( theta ) * np.cos( phi ),
                                            self.__particle_speed[ molecule ] * np.sin( theta ) * np.sin( phi ),
                                            self.__particle_speed[ molecule ] * np.cos( theta ) )
                            , mass = self.__particle_mass[ molecule ]
                            , radius = self.__particle_size[ molecule ]
                            ) )

                            self.__no_of_p[ molecule ] += 1

        return gas

################################################################################

    def physical_update( self ):
        '''
        Update the physical values for each frame
        Including time elapsed, kE, temperature, pressure and gas laws,
        '''
        i = 1

        self.__ang_mom *= 0

        # Recording gas properties in the current frame
        while i < len( self.__gas ):

            p = self.__gas[i]

            self.__ke_list[i] = p.kinetic_energy()
            self.__speed_list[i] = p.speed()
            self.__mass_list[i] = p.mass()
            self.__radius_list[i] = p.radius()

            self.__ang_mom += p.ang_mom()

            i += 1

        self.__ke = np.sum( self.__ke_list )

        # Record the time passed
        if ( self.__equilibrium == True ):
            self.__equilibrium_time_passed += self.__time_step

        self.__time_passed += self.__time_step

        # Sample the speed distribution
        while ( self.__pp_collision_accumulation >= self.__sample_step ):
            self.append_speed_stat( self.__speed_list[1:] )

            self.__pp_collision_accumulation -= self.__sample_step

            self.__sample += 1

        # Store the previous pressure for calculating fluctuation
        previous_pressure = self.__pressure

        if ( self.__dimension == 2 ):
            self.__temperature = self.__ke / np.sum( self.__no_of_p ) / kb
            if ( self.__equilibrium == True ):
                self.__pressure = self.__equilibrium_impulse_acc / self.__equilibrium_time_passed / self.__circumference
            else:
                self.__pressure = self.__impulse_acc / self.__time_passed / self.__circumference
            self.__b = self.__surface - kb * np.sum( self.__no_of_p ) * self.__temperature / self.__pressure

        if ( self.__dimension == 3 ):
            self.__temperature = self.__ke / np.sum( self.__no_of_p ) / kb * 2. / 3.
            if ( self.__equilibrium == True ):
                self.__pressure = self.__equilibrium_impulse_acc / self.__equilibrium_time_passed / self.__sphere_surface
            else:
                self.__pressure = self.__impulse_acc / self.__time_passed / self.__sphere_surface
            self.__b = self.__volume - kb * np.sum( self.__no_of_p ) * self.__temperature / self.__pressure


        self.__pressure_change = self.__pressure - previous_pressure

        if self.__brownian :
            self.__brownian_pos = np.vstack( ( self.__brownian_pos, self.__gas[1].pos() ) )


        # Justify if the system has returned to equilibrium
        if ( abs( self.__pressure_change ) <= 1e-4 * self.__pressure
        and self.__no_of_pp_collision >= np.sum( self.__no_of_p )
        and self.__equilibrium == False):
            self.__equilibrium = True


################################################################################

    def next_frame( self, framenumber ):
        '''
        The update function
        Each frame has a fixed time period, stored in __time_step
        In each frame multiple collisions may occur.
        If next collision happens within this frame, move the time forward to the collision.
        After the collision, if the time passed less than the assigned time step, the frame is not finished
        Detections and implementations of collisions will continue untile the time passed in this frame equal the time step
        '''

        dt = self.__time_step

        # Repeat the collision detections until the time passed in this frame is equal to the defined time_per_frame, rather than a floating dt.
        while dt > 1.e-16:

            dt -= self.per_collision( dt )

        # Calculations for physical values
        self.physical_update()

        # Print physical values
        if ( self.__enable_echo == True ):
            self.echo_physical_quantities()

        # Record the pressure & no. of collision in each frame
        self.__time_list = np.append( self.__time_list, self.__time_passed )

        self.__pressure_list = np.append( self.__pressure_list, self.__pressure )

        self.__frame_number += 1

        # Handle drawings
        if ( self.__enable_anim ):

            patches = []

            for b in self.__gas:

                patches.append( b.sprite() )

            return patches



    def per_collision( self, dt ):
        '''
        This function handles individual collisions happenned in the frame.
        Returns the time to the collision
        So that time can be moved forward in the next_frame() function
        '''
        collide_check = bool()

        ttc_min = float('Inf')

        p_min = list()

        # Check collisions between particles, and find the next collision
        for i in range( len( self.__gas ) ):
            for j in range( i+1, len( self.__gas ) ):
                a = self.__gas[i]
                b = self.__gas[j]
                ttc = a.time_to_collision( b )

                if ( ttc < ttc_min ):
                    ttc_min = ttc
                    p_min = [ a, b ]

        # If time to next collision is within dt
        if ( ( ttc_min - dt ) < -1.e-16 and ttc_min > 1.e-16 ):
            dt = ttc_min - 1.e-16
            collide_check = True

        for b in self.__gas:
            b.move( dt - 1.e-16 ) # The small number is there to avoid escapings and overlappings.

        if ( collide_check ):
            self.__no_of_collision += 1
            impulse = p_min[0].collide( p_min[1] )
            self.__impulse_acc += impulse
            if ( self.__equilibrium == True ):
                self.__equilibrium_impulse_acc += impulse

            if ( impulse == 0 ):
                self.__no_of_pp_collision += 1
                if ( self.__equilibrium == True ): self.__pp_collision_accumulation += 1

        # Return the value to next_frame() to move forward
        return dt - 1.e-16
