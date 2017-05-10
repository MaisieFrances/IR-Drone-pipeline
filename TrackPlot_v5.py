#################################
import cv2
import six
import PIL
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
##################################

def numpy_fillna(data):
    # Get lengths of each row of data
    lens = np.array([len(i) for i in data])
    # Mask of valid places in each row
    mask = np.arange(lens.max()) < lens[:,None]
    # Setup output array and put elements from data into masked positions
    out = np.zeros(mask.shape, dtype=np.uint64)
    out[mask] = np.concatenate(data)
    return out
    
############## Defining colours (not neccesary currently) ##############   
# def col():
# 	colors_ = list(six.iteritems(colors.cnames))
# 	# Add the single letter colors.
# 	for name, rgb in six.iteritems(colors.ColorConverter.colors):
# 		hex_ = colors.rgb2hex(rgb)
# 		colors_.append((name, hex_))
# 	# Transform to hex color values.
# 	hex_ = [color[1] for color in colors_]
# 	# Get the rgb equivalent.
# 	rgb = [colors.hex2color(color) for color in hex_]
# 	rgb = np.asarray(rgb)*255
# 	rgb = rgb.astype(int)
# 	bgr = rgb[:,::-1]
# 	# alternate list of colors
# 	colors_py = [(255,0,0), (0,0,255), (0,255,0), (0,0,0), (255,165,0),(255,255,255),(0,165,255), (255,0,255),(128,0,128)] #['red','blue','green','black','orange','white','cyan','magenta','purple']# for matplotlib this is RGB
# 	colors_cv2 = [(0,0,255), (255,0,0), (0,255,0), (0,0,0),(0,165,255),(255,255,255), (255,165,0), (255,0,255), (128,0,128)] #for cv2 thi is BGR
# 	return colors_py, colors_cv2, bgr, rgb


######### Load GPS files ###### 
# def load_GPS():
# 	with open ('/Volumes/TheBananaStand/Flight_2_GPS.csv', 'r') as f:
# 		GPS = []                              
#      	for line in f:
#         	GPS.append([float(i) for i in line.strip('\n').split(',')])
#      	GPS = np.transpose(GPS)
#     	print(GPS[:][:])                     # [column][row]
#  
#  	with open ('/Volumes/TheBananaStand/Flight_2_Vid_Data.csv', 'r') as f:
#      	Data = [] #create an empty list
#      	for line in f:
#     		Data.append([float(i) for i in line.strip('\n').split(',')])
#     	Data = np.transpose(Data)
# 		print(Data[:][:])                    # [column][row]
# 	return GPS, Data

######## Calculate average height from GPS #########		
# def GPS(ifile):
# 	GPS, Data = load_GPS()
# 	height = []
# 	x = np.where((ifile+1) == Data) #finding the framecount in Data
# 	y = Data[1,x[1]] #accessing the timestamp for that frame
# 	for j in range(len(GPS[0]) - 1):
# 		z = np.where(GPS[1,:] == y) #finding the instances where the timestamp in Data matches those in GPS
# 		z = np.asarray(z) #converting to a numpy array for easier transformations	
# 	for i in np.nditer(z):
# 		h = GPS[0,i] #finding heights corresponding to timestamps
# 		height.append(h)
# 		s = sum(height)
# 		avg = s/len(height) #calculating average height from multiple height values
# 	return avg

######### Load Directory ########
def load_dir(directory):
	dirlist=os.listdir(directory)
	filelist = sorted([s for s in dirlist if ".tif" in s])	
	return filelist
######### Set limits for frame display #######	
def limits(frame):
	maxframe = frame.max()
	minframe = frame.min()
	return maxframe, minframe

####### Load and threshold frames ########
def load_frame(ifile, file, directory):
	frame = cv2.imread(directory+'/'+file, -1)
	framegray = frame.copy()
	framegray = framegray*(1/256.) #resim0.convertTo(resim0,CV_8U,1/256.)
	framegray = np.uint8(framegray)
#   	framegray[:350,:] = 0 #block top in y
#	framegray[300:, :] = 0 #block bottom in y
	framecount = ifile #+= 1	
# 	GPS_height = GPS(ifile)
# 	avgheight.append(GPS_height)
	thresh = np.percentile(framegray.ravel(),95.) #real adaptive threshold
	print frame
	mask = np.where(framegray<=thresh)
	resim0 = framegray.copy()
	resim0[mask] = 0 
	return frame, framegray, framecount, resim0, thresh
    
########### Find contours and calculate parameters #########	
def cont(resim0, framegray, frame, framecount, maxframe, minframe):
	intlist_frame = []
	arealist_frame = []
	print intlist
	kernel = np.ones((10,10),np.uint8)
	resim = cv2.dilate(resim0,kernel,iterations = 1) # resim0 after dilation 
	image, contours, hierarchy = cv2.findContours(resim,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
 	contours = [contour for contour in contours if cv2.contourArea(contour)>=110.]
	print len(contours)
	for i in range(len(contours)):
		# Create a mask image that contains the contour filled in
		cimg = np.zeros_like(resim)
		cv2.drawContours(cimg, contours, i, color=255, thickness=-1)
		# Access the image pixels, count them, and create a 1D numpy array then add to list
		pts = np.where(cimg == 255)		
		area = len(pts[0])
		int = np.sum(framegray[pts])
		intlist_frame.append(int)
		print intlist_frame
		arealist_frame.append(area)
	for i in range(len(contours)): 
		#cv2.drawContours(frame, contours, i, bgr[i], 1)#with full list of colors
		cv2.drawContours(frame, contours, i,255, 1)#short&defined list of colours colors_cv2[i]
	plot_cont(frame, framecount, maxframe, minframe)
	return resim, contours, intlist_frame, arealist_frame

########## Display frame with contours ###########
def plot_cont(frame, framecount, maxframe, minframe):
	plt.figure('Track')
 	plt.imshow(frame, vmin=minframe,vmax=maxframe)
	plt.title('Frame '+str(framecount))
	plt.axis('off')
	plt.show()
	#image.set_data(frame)
	#fig.suptitle('Frame '+str(framecount))
	plt.draw()
	

######### Plotting ##############
def plotting(intlist, arealist):
# 	colors_py, colors_cv2, bgr, rgb = col()
	#Making intlist into array to plot
	intlist = np.asarray(intlist)
	intarray = numpy_fillna(intlist)
	#same with areas list
	arealist = np.asarray(arealist)
	arearray = numpy_fillna(arealist)
 	sns.set(color_codes=True)
 	sns.set(style='ticks')
 	sns.color_palette("colorblind")
 	sns.despine()
 	plt.figure(2)
 	for i in np.arange(intarray.shape[1]):
 		plt.plot(np.arange(intarray.shape[0]),intarray[:,i], label='contour'+str(i))
#  		plt.plot(np.arange(intarray.shape[0]),intarray[:,i], label='contour'+str(i),color=colors_py[i])
 	plt.grid()	
	plt.legend(loc=2)	
	plt.show()	
	
	
plt.ion()
######## Initializing empty lists #######
framecount = 0
intlist = []
arealist = []
meancont = []
framelist = []
cimglist = []
meancont_frame = []
arealistno_frame = []
output = [] 
avgheight = []


######## Calling functions #########
directory = "test"
filelist = load_dir(directory)
for ifile,file in enumerate(filelist):
	frame, framegray, framecount, resim0, thresh = load_frame(ifile, file, directory)
	framelist.append(frame)
	maxframe, minframe = limits(frame)
	resim, contours, intlist_frame, arealist_frame = cont(resim0, framegray, frame, framecount, maxframe, minframe)
	intlist.append(intlist_frame)
	arealist.append(arealist_frame)
plotting(intlist, arealist)




