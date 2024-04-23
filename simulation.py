import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import pandas as pd
import os
from matplotlib.animation import FuncAnimation  
from celluloid import Camera
import astropy.units as u

class particles(): 
    def __init__(self, file_name): 
        #write code to read in csv file 
        file = pd.read_csv(file_name, header = 0)
        
        #defining variables that will be used throughout the code 
        position = file['Radius/pc']
        time = file['Time/yr']
        #velocity =
        
        self.position = position*u.pc.to(u.m)
        self.time = time
        #self.velocity = velocity 
        
    def pos(self, angle): 
        angle = np.radians(angle)
        xpos = [self.position * np.cos(angle)]
        ypos = [self.position * np.sin(angle)]
        self.xpos = xpos
        self.ypos = ypos
        return self.xpos, self.ypos
    
    def vel(self, pos_x, pos_y):
        posx_m = pos_x*u.pc.to(u.m) #internally to function convert the position data from pc to m
        posy_m = pos_y*u.pc.to(u.m)
        time_s = self.time*u.yr.to(u.s) #internally to function convert the time data from yr to s



        vx = []
        vy = []
        vx.append(0)
        vy.append(0)
        #print(len(self.time))
        for i in range(len(self.time)-1):
            vx.append((pos_x[i+1]-pos_x[i])/(self.time[i+1]-self.time[i]))
            vy.append((pos_y[i+1]-pos_y[i])/(self.time[i+1]-self.time[i]))
        vx_array = np.array(vx)
        vy_array = np.array(vy)

        self.vx_array = vx_array
        self.vy_array = vy_array
        return self.vx_array, self.vy_array

def getAcc( pos, mass, G, softening):
    softening = 1.0
    G = 10
    
    # positions r = [x,y,z] for all particles
    x = pos[:,0:1] 
    y = pos[:,1:2]
    
    #indexing by pos[:,0:1] means that I am grabbing only the 0th row from that array 
    #indexing by pos[:,1:2] means that I am grabbing the 1st row from that array 

    # matrix that stores all pairwise particle separations: r_j - r_i
    dx = x.T - x  #x.T means that you are doing a transformation of x (linear algebra)
    dy = y.T - y

    # matrix that stores 1/r^3 for all particle pairwise particle separations 
    inv_r3 = (dx**2 + dy**2 + softening**2)
    inv_r3[inv_r3>0] = inv_r3[inv_r3>0]**(-1.5)

    ax = G * (dx * inv_r3) @ mass #@ symbol is the matrix multiplier 
    ay = G * (dy * inv_r3) @ mass
    
    acceleration = np.hstack((ax, ay))
    return acceleration

def Nbody(Nparticles, positions_list, velocity_list, time, j): 
    # Simulation parameters
    N         = Nparticles    # Number of particles
    t         = 0      # current time of the simulation
    tEnd      = time[len(time)-1]  # time at which simulation ends
    dt        = 0.01   # timestep
    softening = 0.1    # softening length
    G         = 1.0    # Newton's Gravitational Constant
    plotRealTime = True # switch on for plotting as the simulation goes along

    # Generate Initial Conditions
    nbody_positions_x = []
    nbody_positions_y = []
    nbody_vel_x = []
    nbody_vel_y = []

    #need to create a list of positions and velocities of every particle at a certain timestep b/c Nbody is based on 
    #current positions, not positions over time

    #also when we input this into the animation function, we will be generation new values of position and velocity 
    #each time
    pos = np.ones((N, 2)) #creates N x 2 matrix filled with ones that will change once in the for loop below 
    vel = np.ones((N, 2)) 
    
    #filling N x 2 matrix with values from our position and velocity lists 
    #in the 0th value are the x values and in the 1st value are the y values 
    
    #the j corresponds to the time step from our animation function inside of the for loop 
    #remember that the outside for loop in the animation function represents looping through each time 
    for i in range(N): 
        pos[i, 0] = position_list[i][1][0][j] #x position
        pos[i, 1] = position_list[i][0][0][j] #y position

        vel[i, 0] = velocity_list[i][0][j] #x velocity
        vel[i, 1] = velocity_list[i][1][j] #y velocity
    
    #creating 1D array of mass 
    mass = 1.6735575*10**(-24)*np.ones((N, 1))/N  # total mass is Hydrogen 

    # Convert to Center-of-Mass frame
    vel -= np.mean(mass * vel,0) / np.mean(mass)

    # calculate initial gravitational accelerations
    acc = getAcc( pos, mass, G, softening)
    #print(acc)
    

    # (1/2) kick
    vel += acc * dt/2.0

    # drift
    pos += vel * dt
    
    #position returns two arrays: one for x and one for y position
    return pos

#creating list of positions and velocities 
position_list = []
velocity_list = []
#setting the number of particles 
Nparticles = 5 

#reading in the file and initializing the animation object 
file = 's(0) radius red - s(0) radius red.csv'
animation_sne = particles(file)
time = animation_sne.time

#getting list of positions and velocities from particles class 
for i in np.linspace(0, 360, num = Nparticles): 
    print("angle:", i)
    position = animation_sne.pos(i)
    velocity = animation_sne.vel(position[0][0], position[1][0])
    
    position_list.append(position)
    velocity_list.append(velocity)



#running the simulation
fig, ax = plt.subplots(nrows=1, ncols=1)
#need this to create an animation to save 
camera = Camera(fig)
ax.set_facecolor('black')
ax.set_xticks([])
ax.set_yticks([])
t = 0
#looping over the time length*2 because we are alternating between the SNR output and the Nbody sim
while t < len(time)*2: 
    ax.set_title(f"Years After Explosion: {time[int(t/2)]}")
    #plotting the SNR output
    if t%2 == 0: 
        t_ = t/2
        data = np.ones((2, Nparticles))
        for i in range(Nparticles):  
            data[0, i] = (position_list[i][1][0][t_]) #x position
            data[1, i] = (position_list[i][0][0][t_]) #y position
        ax.scatter(data[0], data[1], c = 'white')
        camera.snap()

    #plotting the Nbody sim 
    else: 
        #j tells us what position at time t are are on, hence why it is an argument in the Nbody 
        j = int((t-1)/2)
        #position_list and velocity_list are global variables which are read in the Nbody sim which retrieves the positions and velocities for specific time t 
        Nbody_pos = Nbody(Nparticles, position_list, velocity_list, time, j)
        ax.scatter(Nbody_pos[:,0], Nbody_pos[:,1], c = 'white')
        camera.snap()
    t+= 1

#saving animation 
anim = camera.animate(blit=True)
anim.save('supernova1987A.mp4')
