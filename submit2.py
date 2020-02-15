import numpy as np
import random as rnd
import time as tm

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE SUBMIT.PY
# DO NOT INCLUDE PACKAGES LIKE SKLEARN, SCIPY, KERAS ETC IN YOUR CODE
# THE USE OF ANY MACHINE LEARNING LIBRARIES FOR WHATEVER REASON WILL RESULT IN A STRAIGHT ZERO
# THIS IS BECAUSE THESE PACKAGES CONTAIN SOLVERS WHICH MAKE THIS ASSIGNMENT TRIVIAL

# DO NOT CHANGE THE NAME OF THE METHOD "solver" BELOW. THIS ACTS AS THE MAIN METHOD AND
# WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THIS NAME WILL CAUSE EVALUATION FAILURES

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def getObjValue( X, y, wHat ):
	lassoLoss = np.linalg.norm( wHat, 1 ) + pow( np.linalg.norm( X.dot( wHat ) - y, 2 ), 2 )
	return lassoLoss
def solver2( X, y, timeout, spacing):# ,timeseries,objvalue2 ):
	(n, d) = X.shape
	t = 0
	totTime = 0
	
	
	# w is the model vector and will get returned once timeout happens
	w = np.zeros( (d,) )
	
	tic = tm.perf_counter()
	w_run = np.zeros( (d,) )
	h = np.zeros( (d,) )
################################
#  Non Editable Region Ending  #
################################

	# You may reinitialize w to your liking here
	# You may also define new variables here e.g. step_length, mini-batch size etc

################################
# Non Editable Region Starting #
################################
	while True:
		t = t + 1
		if t % spacing == 0:
			toc = tm.perf_counter()
			totTime = totTime + (toc - tic)
			if totTime > timeout:
				return (w, totTime)
			else:
				tic = tm.perf_counter()
		# if totTime>=0.1:
		# 	objvalue2.append(getObjValue(X,y,w))
		# 	timeseries.append(totTime)
		# w_run = w
		alpha = 0.09
		nt=0.1
		Y1=X.dot(w)
		delta=Y1-y
		h=alpha*h + nt*(2*X.T.dot(delta)+np.sign(w-alpha*h))
		h=h/np.sqrt(t)
		w_run=w_run-h
		# w=w_run
		
		w = (w * (t-1) + w_run)/t
		
################################
#  Non Editable Region Ending  #
################################

		# Write all code to perform your method updates here within the infinite while loop
		# The infinite loop will terminate once timeout is reached
		# Do not try to bypass the timer check e.g. by using continue
		# It is very easy for us to detect such bypasses which will be strictly penalized
		
		# Please note that once timeout is reached, the code will simply return w
		# Thus, if you wish to return the average model (as is sometimes done for GD),
		# you need to make sure that w stores the average at all times
		# One way to do so is to define a "running" variable w_run
		# Make all GD updates to w_run e.g. w_run = w_run - step * delw
		# Then use a running average formula to update w
		# w = (w * (t-1) + w_run)/t
		# This way, w will always store the average and can be returned at any time
		# In this scheme, w plays the role of the "cumulative" variable in the course module optLib
		# w_run on the other hand, plays the role of the "theta" variable in the course module optLib
		
	return (w, totTime) # This return statement will never be reached