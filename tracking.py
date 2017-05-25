#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
## @author Claire Burke
## Created 08/05/2017

####Test bit for object tracking in sucessive video frames.

import numpy as np
import numpy.ma as ma
import matplotlib as mpl
import matplotlib.cm as mpl_cm
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import scipy as sp
import scipy.stats as ss
import commands

##################################

def map_plot(plt_frame,obj_coords_x,obj_coords_y):
    #colour_map = mpl_cm.get_cmap('Greys')
    #mpl.pyplot.pcolormesh(long_array,lat_array,plot_var, norm=plt.Normalize(),cmap=colour_map,vmin=min_val,vmax=max_val)
    plt.figure()
    plt.pcolormesh(plt_frame, norm=plt.Normalize(),vmin=np.median(plt_frame),vmax=np.amax(plt_frame)) #,cmap=colour_map
    plt.scatter(obj_coords_x,obj_coords_y,color='white',s=40, marker='>')
    plt.gca().invert_yaxis()
    plt.show(block=False)
    

def gps_path(ratio):
    gps_file_path='/Users/Claire/Documents/astro-eco/data/tanzania_split/flight_path/flight_area2/70m/area2_70m.csv'
    gps=np.loadtxt(gps_file_path,skiprows=1, usecols=(0,1,2), delimiter=',')
    lat=gps[:,1]
    lon=gps[:,0]
    tstep=gps[:,2]
    tstep_gps=np.arange(0,len(tstep),1)
    tstep_frames=np.arange(0,len(tstep_gps),1/ratio)
    frame_lat_array=np.interp(tstep_frames,tstep_gps,lat)
    frame_lon_array=np.interp(tstep_frames,tstep_gps,lon)
    return frame_lat_array,frame_lon_array

def lat_lon_to_meters(latitude):
    #Calcultaes the size of a degree in meters for lat and lon given the observed latitude
    deg_lat=2*np.pi*6371000/360
    deg_lon=np.cos(latitude)*deg_lat
    
    
def id_objects_in_frame(object_coords,min_spacing,expect_min_size):
    #min_spacing = minimun distance apart in pix 2 bright "objects" have to be to be defined as separate
    x_coords= object_coords[0]
    y_coords= object_coords[1]    

    x_obj_centers=np.zeros(1)
    y_obj_centers=np.zeros(1)
    
    i=0    
    while i != len(x_coords)+1:
        n=i+1
        while n!=0:
            #print x_coords[i],x_coords[n]
            if n >= len(x_coords):
                if np.amax(x_coords[i:n])-np.amin(x_coords[i:n]) >= expect_min_size:
                    x_obj_centers=np.append(x_obj_centers,np.mean(x_coords[i:n]))
                    y_obj_centers=np.append(y_obj_centers,np.mean(y_coords[i:n]))
                i=len(x_coords)+1
                n=0
            else:
                if x_coords[n]-min_spacing < x_coords[i] < x_coords[n]+min_spacing:
                    if y_coords[n]-min_spacing < y_coords[i] < y_coords[n]+min_spacing:
                        #second check for same obj, just in case more than one object with same x coords.
                        n+=1
                    else:
                        x_obj_centers=np.append(x_obj_centers,np.mean(x_coords[i:n]))
                        y_obj_centers=np.append(y_obj_centers,np.mean(y_coords[i:n]))
                        i=n
                        n=0
                else:
                    if np.amax(x_coords[i:n])-np.amin(x_coords[i:n]) >= expect_min_size:
                        x_obj_centers=np.append(x_obj_centers,np.mean(x_coords[i:n]))
                        y_obj_centers=np.append(y_obj_centers,np.mean(y_coords[i:n]))
                    i=n
                    n=0
                    
    return x_obj_centers[1:],y_obj_centers[1:]
                
        


#basic information about camera, frame rate and fov in pixels and metres.
fps=9
dim_x=640
dim_y=512 #480????
jiggle_size= 40 # guess at size of image wobbles as drone moves around in pixels
height=70
fov_x_m=height*np.tan(25/2)*2
fov_y_m=height*np.tan(19/2)*2


#parameters for object identification, minimum size of object in metres
obj_size_metres=0.3
pix_scale=fov_x_m/dim_x
obj_expect_size_pix=obj_size_metres*pix_scale#****problem here!

#Since drone moves so that objects appear at top of fov and move downwards, y fov== -y map. 
#But at top of fov y==0 in camera coords....

#infomation about movement of drone (or if stationary camera, movement of objects in image)
#vx,vy in pix per second?
#may be an array of values depending on how drone is moving 
#vx=#+,0,-,0,+
#vy=#0,+,0,+,0
#theta=#0,90,180,90,0 rotation of drone fov througout path

lat,lon=gps_path(9.0/4.0) #Input ratio of number of fps in TIR camera to 'fps' for the gps readout 


file_dir='/Users/Claire/Documents/astro-eco/data/tanzania_split/tracking_test70'
frame_list=commands.getoutput('ls -rtd '+file_dir+'/frame_0403.csv').split('\n')
n_frames= len(frame_list)#number of frames in the data

###*** Shell script here to fix the extra ';' at the end of lines in the csv file
### commands.whatever(run script)


#transfer frame numbers from read in so that we can map the gps to the frame numbers
frame_id=np.empty(len(frame_list))
frame_lat=np.empty(len(frame_list))
frame_lon=np.empty(len(frame_list))
for f,frame in enumerate(frame_list):
    temp=np.array([frame[-8:-4]])
    frame_id[f]=temp.astype(int)
    frame_lat[f]=lat[frame_id[f]]
    frame_lon[f]=lon[frame_id[f]]
    
#for frame in frame_list:    
frame_orig=np.loadtxt(frame_list[0],delimiter=';') 


frame=frame_orig-np.median(frame_orig)
#print np.amin(frame), np.amax(frame)

mask = np.ma.masked_where(frame<=np.percentile(frame,99.98), frame, copy=True)
object_coords = np.where(frame>np.percentile(frame,99.98))


print frame[np.where(frame>np.percentile(frame,99.98))]
print object_coords[0]
print object_coords[1]


x_obj_centers,y_obj_centers = id_objects_in_frame(object_coords, min_spacing=10, expect_min_size=2)

print x_obj_centers
print y_obj_centers

map_plot(frame_orig,y_obj_centers-15,x_obj_centers)



#for x in object_coords[0]:


    

#object_array=np.nans([n_frames,2]) ##n dim array which can be appended each time a new source is detected. 
##Contains x,y position of each source in each frame. len(n-frames)x(x)x(y), add a new dim each time a new source is detected. 
##default value = 'nan'. intiger x,y, only in frames where source is detected.
##continue to guess where the sources are after they leave the frame, in case they come back intoview and so we can make a map of where stuff is.

##do object detection for first frame and location of sources
#n_obj_fov=    #number of sources
#x_array=    #arrays containing loaction of sources
#y_array=

#for m in range(0,n_obj_fov):
    #new_obj=np.nans([n_frames,2])
    #new_obj[0]=[x_array[m],y_array[m]]
    #object_array=np.array([object_array,new_obj])


#for f,frame in enumerate(frames):

    ##distance all objects in frame should have moved by cf previous frame
    #dy=vy/fps
    #dx=vx/fps
    
    ## Use gps loactions here to get expected x,y ?
    ## Record expected location of centre of fov an distance of objects from there to make a map
    #x_expect=x_array+dx 
    #y_expect=y_array+dy
    
    
    ##check to see if objects in previous frame should have moved out of image including padding for image jiggling
    ##if source not expected to be in frame any more change obj_in_image for that source==0
    ##alternatively, could have a buffer zone around the edge of the fov where we ignore any sources that we see...
    #for n,obj in enumerate(x_array):
        ##check where source is expected to be => dim_change+/-jiggle
        #xmin=x_expect[n]-jiggle_size
        #xmax=x_expect[n]+jiggle_size
        #ymin=y_expect[n]-jiggle_size
        #ymax=y_expect[n]+jiggle_size    
    
        ##is the expected location still within the fov?
        #if 0< xmax & xmin < dim_x:
            #if 0< ymax & ymin > dim_y:
        
            ##cut out box of size = jiggle_size around expected location
            ##is there a source in the box?
            #if source ==True:
                ##new x,y location of source goes in object array.
                #object_array[f,n]=[x,y] #location of measured source in cut outside
            #else:
            ##if source not found, expected values go in object array
            ##Decrease the number of expected sources by 1.
                #object_array[f,n]=[x_expect,y_expect] 
                #n_obj_fov-=1

        #else:
            #object_array[f,n]=[x_expect,y_expect] 
            #n_obj_fov-=1
        ##if source is outside of fov we don't look for it but we do record where we think it is in case it comes back. Decrease the number of expected sources by 1.
        
        ##how to deal with multiple sources in the expected area?

    
    ##Now look for new sources entering the field of view
    ##number of expected objects in the field after tracking ones we already know
    
    ##source detection happens on whole frame and location of sources
    ##get num_obj_detected, x_array, y_array
    #if num_obj_detected != n_obj_fov:
        ##extract the new sources
        #???
        
        ##add new dim(s) to the object array for each newly found source.
        ##function this, def add_obj_dim(object_array,frame_number, new_x, new_y):
        #for i in range (0, num_obj_detected-n_obj_fov):
            #x=np.nans(n_frames)
            #y=np.nans(n_frames) 
            #x[0]=x_array[m]
            #y[0]=y_array[m]    
            #object_array[0]=[x_array[m],y_array[m]]


#def rot_theta(theta,rot_centre,dist): 
    ##rot_centre == x,y of centre of rotation
    ##dist == distnace from centre of rotation
    ##theta == 90 or 180 rotation angle
    #if theta == 180:
        #x_fov == -xmap
        #y_fov= -ymap
   #return(x_fov, y_fov)
   
    ##if theta == 90:
        ## y map = x(crentre rot point)-x(centre fov) camera
        ## x map = y camera
        
    ##if theta == 270: # don't need this case presently as drone doesn't turn this way.

