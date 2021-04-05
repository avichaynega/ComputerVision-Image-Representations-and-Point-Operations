"""
		'########:'##::::'##::::'##:::
		 ##.....::. ##::'##:::'####:::
		 ##::::::::. ##'##::::.. ##:::
		 ######:::::. ###::::::: ##:::
		 ##...:::::: ## ##:::::: ##:::
		 ##:::::::: ##:. ##::::: ##:::
		 ########: ##:::. ##::'######:
		........::..:::::..:::......::
"""
from typing import List
# from skimage.color import rgb2yiq ,yiq2rgb
import cv2
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import numpy as np
LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int:
	"""
	Return my ID (not the friend's ID I copied from)
	:return: int
	"""
	return 309612307


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
	"""
	Reads an image, and returns the image converted as requested
	:param filename: The path to the image
	:param representation: GRAY_SCALE or RGB
	:return: The image object
	"""
	if representation != 1 and representation != 2:
		raise ValueError('wrong number of the representation. please enter 1 or 2')

	img = cv2.imread(filename)
	if img is None:
		raise ValueError('cannot open the image file.')
	img_color = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if representation == LOAD_GRAY_SCALE \
		else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	final_img = (img_color - img_color.min()) / (img_color.max() - img_color.min())  # normalization
	return final_img


def imDisplay(filename: str, representation: int):
	"""
	Reads an image as RGB or GRAY_SCALE and displays it
	:param filename: The path to the image
	:param representation: GRAY_SCALE or RGB
	:return: None
	"""
	img = cv2.imread(filename)
	if representation == LOAD_GRAY_SCALE:
		img_color = imReadAndConvert(filename, representation)
		plt.gray()
	else:  # RGB image
		img_color = imReadAndConvert(filename, representation)
	plt.imshow(img_color)
	plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
	"""
	Converts an RGB image to YIQ color space
	:param imgRGB: An Image in RGB
	:return: A YIQ in image color space
	"""
	yiq_from_rgb = np.array([[0.299,0.587,0.114],[0.59590059, -0.27455667, -0.32134392],
	[0.21153661, -0.52273617, 0.31119955]])

	img_yiq =  np.dot(imgRGB, yiq_from_rgb.T.copy())
	# yiq = cv2.normalize(img_yiq, 0, 255, cv2.NORM_MINMAX)

	return img_yiq

	pass


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
	"""
	Converts an YIQ image to RGB color space
	:param imgYIQ: An Image in YIQ
	:return: A RGB in image color space
	"""
	yiq_from_rgb = np.array([[0.299, 0.587, 0.114], [0.59590059, -0.27455667, -0.32134392],[0.21153661, -0.52273617, 0.31119955]])
	OrigShape=imgYIQ.shape
	yiq2rgb = np.dot(imgYIQ.reshape(-1,3), np.linalg.inv(yiq_from_rgb).transpose()).reshape(OrigShape)
	return yiq2rgb
	pass


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
	"""
		Equalizes the histogram of an image
		:param imgOrig: Original Histogram
		:return: (imgEq,histOrg,histEQ)
	"""
	if len(imgOrig.shape) != 3 and len(imgOrig.shape) != 2:
		raise Exception("please insert RGB or GRAYSCALE images.")
	
	is_color = False

	if len(imgOrig.shape) == 3: # if RGB image
		is_color = True
		yiqIm = transformRGB2YIQ(imgOrig)
		imgOrig = yiqIm[:, :, 0]
	imgOrig = cv2.normalize(imgOrig, None, 0, 255, cv2.NORM_MINMAX)
	imgOrig = imgOrig.astype('uint8')
	histOrig = np.histogram(imgOrig.flatten(), bins=256)[0]
	cs = np.cumsum(histOrig)
	imgNew = cs[imgOrig]
	imgNew = cv2.normalize(imgNew, None, 0, 255, cv2.NORM_MINMAX)
	imgNew = imgNew.astype('uint8')
	histNew = np.histogram(imgNew.flatten(), bins=256)[0]
	if is_color:
		yiqIm[:, :, 0] = imgNew / (imgNew.max() - imgNew.min())
		imgNew = transformYIQ2RGB(yiqIm)
	return imgNew, histOrig, histNew

	pass


def fix_q(z: np.array, image_hist: np.ndarray) -> np.ndarray:
	"""
		Calculate the new q using wighted average on the histogram
		:param image_hist: the histogram of the original image
		:param z: the new list of centers
		:return: the new list of wighted average
	"""
	q = [np.average(np.arange(z[k], z[k + 1] + 1), weights=image_hist[z[k]: z[k + 1] + 1]) for k in range(len(z) - 1)]
	return np.round(q).astype(int)

def fix_z(q: np.array) -> np.array:
	"""
		Calculate the new z using the formula from the lecture.
		:param q: the new list of q
		:param z: the old z
		:return: the new z
	"""
	z_new = np.array([round((q[i - 1] + q[i]) / 2) for i in range(1, len(q))]).astype(int)
	z_new = np.concatenate(([0], z_new, [255]))
	return z_new

def findBestCenters(histOrig: np.ndarray, nQuant: int, nIter: int) -> (np.ndarray, np.ndarray):
	"""
			Finding the best nQuant centers for quantize the image in nIter steps or when the error is minimum
			:param histOrig: hist of the image (RGB or Gray scale)
			:param nQuant: Number of colors to quantize the image to
			:param nIter: Number of optimization loops
			:return: return all centers and they color selected to build from it all the images.
		"""
	Z = []
	Q = []
	# head start, all the intervals are in the same length
	z = np.arange(0, 256, round(256 / nQuant))
	z = np.append(z, [255])
	Z.append(z.copy())
	q = fix_q(z, histOrig)
	Q.append(q.copy())
	for n in range(nIter):
		z = fix_z(q)
		if (Z[-1] == z).all():  # break if nothing changed
			break
		Z.append(z.copy())
		q = fix_q(z, histOrig)
		Q.append(q.copy())
	return Z, Q


def convertToImg(imOrig: np.ndarray, histOrig: np.ndarray, yiqIm: np.ndarray, arrayQuantize: np.ndarray) -> (
		np.ndarray, float):
	"""
		Executing the quantization to the original image
		:return: returning the resulting image and the MSE.
	"""
	imageQ = np.interp(imOrig, np.linspace(0, 1, 255), arrayQuantize)
	curr_hist = np.histogram(imageQ, bins=256)[0]
	err = np.sqrt(np.sum((histOrig.astype('float') - curr_hist.astype('float')) ** 2)) / float(
		imOrig.shape[0] * imOrig.shape[1])
	if len(yiqIm):  # if the original image is RGB
		yiqIm[:, :, 0] = imageQ / 255
		return transformYIQ2RGB(yiqIm), err
	return imageQ, err


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
	"""
		Quantized an image in to **nQuant** colors
		:param imOrig: The original image (RGB or Gray scale)
		:param nQuant: Number of colors to quantize the image to
		:param nIter: Number of optimization loops
		:return: (List[qImage_i],List[error_i])
	"""
	if(len(imOrig.shape) != 3 and len(imOrig.shape) !=2):
		 raise Exception('please insert RGB or Greyscale images only')
	if len(imOrig.shape) == 3:
		imYIQ = transformRGB2YIQ(imOrig)
		imY = imYIQ[:, :, 0].copy()  # take only the y chanel
	else:
		imY = imOrig
	histOrig = np.histogram(imY.flatten(), bins=256)[0]
	Z, Q = findBestCenters(histOrig, nQuant, nIter)
	image_history = [imOrig.copy()]
	E = []
	for i in range(len(Z)):
		arrayQuantize = np.array([Q[i][k] for k in range(len(Q[i])) for x in range(Z[i][k], Z[i][k + 1])])
		q_img, e = convertToImg(imY, histOrig, imYIQ if len(imOrig.shape) == 3 else [], arrayQuantize)
		image_history.append(q_img)
		E.append(e)
	# plt.plot(E)
	# plt.xlabel('iterations')
	# plt.ylabel('Error')	
	# plt.show()
	return image_history, E

	pass

##############check imDisplay and imread
# imDisplay('Red-Rose.jpg', 1)

##############check transformRGB2YIQ
# img = cv2.imread('Red-Rose.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# plt.imshow(transformRGB2YIQ(img))
# plt.show()

# plt.imshow(rgb2yiq(img))
# plt.show()

###############check transformYIQ2RGB

# img = cv2.imread('Red-Rose.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# yiq2 = rgb2yiq(img)
# plt.imshow(transformYIQ2RGB(yiq2))
# plt.show()

# plt.imshow(yiq2rgb(yiq2))
# plt.show()


###############check hsitogramEqualize(imgOrig: np.ndarray)

# img = cv2.imread('sample.jpeg')
# img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# hsitogramEqualize(img)


###############check quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int)
	# img = cv2.imread('water_bear.png')
	# img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	# img = np.array([1,2,3,4])
	# quantizeImage(img,5,5)

