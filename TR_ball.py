# -*- coding: utf-8 -*-
"""
Created on Fri Mar 2 14:43:23 2018

@author: Hang Su

This file is written to define class Ball, including both particles and the container.

This section will be used in the file TR_gas.py.
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math

class Ball:


    def __init__( self, position, velocity, density = 1., mass = 0., radius = 1., clr='r', objtype = 'ball' ):

        self.__radius = radius
        self.__density = density
        if ( mass == 0 ):self.__mass = np.pi * self.__radius**2 * self.__density
        else: self.__mass = mass

        self.__position = np.array( position )
        self.__velocity = np.array( velocity )
        self.__speed = np.sqrt( np.dot( self.__velocity, self.__velocity ) )
        self.__sprite = plt.Circle( self.__position, self.__radius, fill = ( objtype == 'ball' ), fc=clr )

        self.__clr = clr
        self.__type = objtype

        self.min_time_to_colli = float()

    def __repr__( self ):
        return self.__type

    def pos( self ):
        return self.__position

    def vel( self ):
        return self.__velocity

    def mass( self ):
        return self.__mass

    def density( self ):
        return self.__density

    def radius( self ):
        return self.__radius

    def sprite( self ):
        self.__sprite.center = self.__position
        return self.__sprite

    def speed( self ):
        return np.sqrt( np.dot( self.__velocity, self.__velocity ) )

    def kinetic_energy( self ):
        return .5 * self.__mass * np.dot( self.__velocity, self.__velocity )

    def momentum( self ):
        return self.__mass * self.__velocity

    def ang_mom( self ):
        return np.cross( self.__position, self.momentum() )


    def move( self, dt ):
        self.__position = self.__position + self.__velocity * dt

    def set_vel( self, new_vel ):
        self.__velocity  = new_vel

    def set_pos( self, new_pos ):
        self.__position  = new_pos

    def set_sprite( self, new_sprite ):
        self.__sprite = new_sprite

    def reset_sprite( self ):
        self.__sprite = plt.Circle( self.__position, self.__radius, fill = ( self.__type == 'ball' ), fc=self.__clr )

    def time_to_collision( self, other ):
        '''Calculate the time to the next collision with another object
        by solving the quadratic equation for collision condition
        return the length of time
        '''

        container_presence = ( other.__repr__() == 'container' or self.__repr__() == 'container' )

        contact_distance = float()

        delta_pos = self.pos() - other.pos()

        delta_vel = self.vel() - other.vel()


        if ( not container_presence ):

            contact_distance = self.radius() + other.radius()

        else:

            contact_distance = abs( self.radius() - other.radius() )

        # Construct the quadratic equation and solve
        a = np.dot( delta_vel, delta_vel )
        b = 2. * np.dot( delta_pos, delta_vel )
        c = np.dot( delta_pos, delta_pos ) - contact_distance**2
        delta = b**2 - 4. * a * c

        # Test delta
        if ( delta >= 0 ):

            first_sol = ( -b - np.sqrt( delta ) ) / ( 2. * a )

            second_sol = ( -b + np.sqrt( delta ) ) / ( 2. * a )

        else: return float('Inf')

        # Solutions one-negative case for 2 balls: Overlaps
        if ( first_sol < 1.e-16 and not container_presence ): return float('Inf')

        # Solutions all-negative case
        if ( first_sol < 1.e-16 and second_sol < 1.e-16 ): return float('Inf')

        if ( container_presence ):

            return second_sol

        return first_sol


    def collide( self, other ):
        '''Handle the collision between two objects
        '''

        container_presence = ( other.__repr__() == 'container' or self.__repr__() == 'container' )

        rel_pos = self.pos() - other.pos()

        rel_pos_norm = rel_pos / np.sqrt( np.dot( rel_pos, rel_pos ) )

        rel_vel = self.vel() - other.vel()

        rel_vel_parallel = np.dot( rel_vel, rel_pos_norm )

        if ( container_presence ):

            if ( self.__repr__() == 'container' ):
                # self is always the container
                other.set_vel( other.vel() + 2.* rel_vel_parallel * rel_pos_norm  )

                impulse = 2.* rel_vel_parallel * other.mass()

            return impulse

        else:

            #Construct and solve the quadratic equation
            a = ( self.__mass / other.mass() + 1. )
            b = -2. * self.__mass / other.mass() * rel_vel_parallel
            c = ( self.__mass / other.mass() - 1. ) * rel_vel_parallel**2
            delta = b**2 - 4. * a * c

            #Test delta
            if ( delta >= 0 ):

                first_sol = ( -b - np.sqrt( delta) ) / ( 2. * a )

                second_sol = ( -b + np.sqrt( delta ) ) / ( 2. * a )

            else: return None

            rel_vel_parallel_after = second_sol

            change_of_rel_vel_parallel = rel_vel_parallel_after - rel_vel_parallel #The magnitude of the change of relative parallel velocity

            change_rel_vel = change_of_rel_vel_parallel * rel_pos_norm #The vector of the change of relative parallel velocity

            change_of_other_vel = - self.mass() / other.mass() * change_rel_vel

            self.set_vel( self.vel() + change_rel_vel )

            other.set_vel( other.vel() + change_of_other_vel )

            return 0
