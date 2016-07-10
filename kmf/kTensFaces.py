"
Compute tensorial kernel for CMU faces images
subjects x poses x illumination x images
K(X,x_i)=K_im(X,x_i)*K_pos(X,x_i)*K_ill(X,x_i)

Input:	X: Tensor (subjects x poses x illumination x images)
	param: Parameters for each kernel
"

import numpy as np
from sklearn.metrics import pairwise_kernels as K

def kTensFaces(X, *param):
	size_tensor = X.shape
	#compute kernel for subjects
	 





