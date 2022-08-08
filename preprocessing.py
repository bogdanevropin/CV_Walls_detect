"""
Preprocessing images
"""
import cv2
# import matplotlib as plt
import numpy as np
import os
from pathlib import Path
from copy import deepcopy
import random
import pytesseract as pytess
import re

pytess.pytesseract.tesseract_cmd = r'C:\Users\Bogdan\AppData\Local\Programs\Python\Python39\Lib\site-packages\Tesseract-OCR\tesseract.exe'


def all_paths(input_dir: str):
	"""
	Find all paths
	:param input_dir: in this directory
	"""
	all_images_paths = []  # finding all input images
	for root, dirs, files in os.walk(input_dir):
		for file in files:
			all_images_paths.append(input_dir + file)
	return all_images_paths


def clean_dir(c_dir: str):
	"""
	cleaning directory
	:param c_dir: path for cleaning directory
	"""
	for file in os.listdir(c_dir):
		os.remove(os.path.join(c_dir, file))


class Image:
	"""
	Processing images class
	"""
	start_image: np.ndarray
	name: str
	current_image: np.ndarray = None
	
	def __init__(self, image_path: str, output_dir: str = os.getcwd() + '/preprocessed_images/'):
		self.image_path = image_path
		self.output_dir = output_dir
		self.name = Path(self.image_path).stem
		self.start_image = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
		if self.current_image is None:
			self.current_image = self.start_image.copy()
	
	@staticmethod
	def inverting_img(img):
		"""
		inverting colours of image
		:param img: cv2 image object
		"""
		return cv2.bitwise_not(img)
	
	@staticmethod
	def scaling_img(img, max_size=1024):
		"""
		:param img: np.ndarray image
		:param max_size: max length of image side
		:return: resized np.ndarray image
		"""
		m_s = max(img.shape[:2])
		resized_img = img
		if m_s > max_size:
			width = int(round(img.shape[1] * max_size / m_s))
			height = int(round(img.shape[0] * max_size / m_s))
			dim = (width, height)
			resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)  # resize image
		return resized_img
	
	def resize_img(self):
		"""
		Resizing before showing image
		"""
		if self.current_image is None:
			self.current_image = deepcopy(self.start_image)
		self.current_image = Image.scaling_img(self.current_image)
		return self.current_image
	
	@staticmethod
	def show_img(img_path: str, name: str = 'Current Image'):
		"""
		:param img_path: image path or image object
		:param name: displayed name of image
		"""
		cv2.imshow(name, img_path)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	
	def show(self):
		"""
		Showing current Image class instance image after resizing
		"""
		self.resize_img()
		cv2.imshow(self.name, self.current_image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	
	def clean_output_dir(self):
		"""
		Clean output dir
		"""
		clean_dir(self.output_dir)
		return self.current_image
	
	def create_greyscale_img(self, write=False):
		"""
		making greyscale image from current
		:param write: Write or not image in outputs
		"""
		if self.current_image is None:
			self.current_image = deepcopy(self.start_image)
		if len(self.current_image.shape) != 2:
			self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)  # making image greyscale
		if write:
			cv2.imwrite(os.path.splitext(self.output_dir + self.name)[0] + '_greyscale_img.jpg', self.current_image)
		return self.current_image
	
	def gaussian_blur(self):
		"""
		Making gaussian bluring current image
		"""
		if self.current_image is None:
			self.current_image = deepcopy(self.start_image)
		self.current_image = cv2.GaussianBlur(self.current_image, (5, 5), 0)  # Gaussian filtering
		return self.current_image
	
	def create_binary_img(self, write=False):
		"""
		making image binary
		:param write: Writing in output dir or not
		"""
		self.create_greyscale_img()
		ret: float or int
		th_binary: np.ndarray  # dtype=unit8 shape tuple: 2
		ret, th_binary = cv2.threshold(self.current_image, 127, 255, cv2.THRESH_BINARY)  # apply binary thresholding
		self.current_image = th_binary
		if write:
			cv2.imwrite(self.output_dir + self.name + '_binary.jpg', self.current_image)
		return self.current_image
	
	def create_contours(self, write=False, ratio=150):
		"""
		Make contours, filtering them
		# :param epsilon: approximation
		:param write: write or not in output dir
		:param ratio: how small ratio particles are deleted (by area ob bounded turned rectangles)
		"""
		self.create_binary_img()  # apply binary thresholding
		contours: tuple or list  # tuple of np.ndarrays
		hierarchy: np.ndarray  # shape tuple 3
		contours, hierarchy = cv2.findContours(image=self.current_image, mode=cv2.RETR_TREE,
		                                       method=cv2.CHAIN_APPROX_SIMPLE)  # checks modes (external points mode)
		# if epsilon:
		# 	approximated = []
		# 	for contour in contours:
		# 		length = epsilon * cv2.arcLength(contour, True)
		# 		approx = cv2.approxPolyDP(contour, length, True)
		# 		approximated.append(approx)
		# 	contours = approximated
		max_rect_area = max(cv2.minAreaRect(contour)[1][0] * cv2.minAreaRect(contour)[1][1] for contour in contours)
		contours_filtered = []
		for contour in contours:
			if cv2.minAreaRect(contour)[1][0] * cv2.minAreaRect(contour)[1][1] > max_rect_area / ratio:
				contours_filtered.append(contour)
		contours = contours_filtered
		drawing = self.inverting_img(
			np.zeros((self.current_image.shape[0], self.current_image.shape[1], 3), dtype=np.uint8))
		self.current_image = cv2.drawContours(image=drawing, contours=contours, contourIdx=-1,
		                                      color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
		self.current_image = cv2.drawContours(drawing, contours, -1, (0, 0, 0), 2)
		self.current_image = cv2.drawContours(image=self.current_image, contours=contours, contourIdx=-1,
		                                      color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
		if write:
			cv2.imwrite(self.output_dir + self.name + '_RectR_Contours.jpg', self.current_image)
		return self.current_image
	
	def thick_lines_filtering(self, ratio=10, filtering=True, write=False):
		"""
		Remains only thickest lines
		:param filtering: OTSU
		:param write: write image or not
		:param ratio: ratio of thickness
		"""
		self.create_binary_img()  # apply binary thresholding
		contours: tuple or list  # tuple of np.ndarrays
		hierarchy: np.ndarray  # shape tuple 3
		if filtering:
			self.create_otsu()
		contours, hierarchy = cv2.findContours(image=self.current_image, mode=cv2.RETR_TREE,
		                                       method=cv2.CHAIN_APPROX_SIMPLE)  # checks modes (external points mode)
		max_cont_area = max(cv2.contourArea(contour) for contour in contours)
		contours_filtered = []
		for contour in contours:
			if cv2.contourArea(contour) > max_cont_area / ratio:
				contours_filtered.append(contour)
		contours = contours_filtered
		drawing = self.inverting_img(
			np.zeros((self.current_image.shape[0], self.current_image.shape[1], 3), dtype=np.uint8))
		self.current_image = cv2.drawContours(image=drawing, contours=contours, contourIdx=-1,
		                                      color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
		self.current_image = cv2.drawContours(drawing, contours, -1, (0, 0, 0), 2)
		self.current_image = cv2.drawContours(image=self.current_image, contours=contours, contourIdx=-1,
		                                      color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
		if write:
			cv2.imwrite(self.output_dir + self.name + '_Thick_lines.jpg', self.current_image)
		return self.current_image
	
	def create_canny_edges(self, write=False, white=False, filtering=True):
		"""
		Writing images and returning black or white edges
		:param filtering: blur or not
		:param write: True - need to write pics or False/None - don't need
		:param white: True - need white pics or False/None - don't need
		:return: black or white image of edges, depends on param white
		"""
		self.create_greyscale_img()
		if filtering:
			self.gaussian_blur()
		if white:
			self.current_image = cv2.Canny(self.current_image, 200, 300)
		else:
			self.current_image = self.inverting_img(cv2.Canny(self.current_image, 200, 300))
		if write and white:
			cv2.imwrite(self.output_dir + self.name + '_CANNY_EDGES_WHITE.jpg', self.current_image)
		if write and not white:
			cv2.imwrite(self.output_dir + self.name + '_CANNY_EDGES_BLACK.jpg', self.current_image)
		return self.current_image
	
	def create_adaptive_mean_c(self, write=False):
		"""
		Adaptive mean c filtering current image
		:param write: write in output or not
		"""
		self.create_greyscale_img()
		self.gaussian_blur()
		self.current_image = cv2.adaptiveThreshold(self.current_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
		                                           cv2.THRESH_BINARY, 11, 2)  # adaptive c
		if write:
			cv2.imwrite(self.output_dir + self.name + '_ADAPTIVE_THRESH_MEAN_C.jpg', self.current_image)
		return self.current_image
	
	def create_adaptive_gaussian_c(self, write=False):
		"""
		Adaptive gaussian c filtering
		:param write:
		"""
		self.create_greyscale_img()
		self.gaussian_blur()
		self.current_image = cv2.adaptiveThreshold(self.current_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
		                                           cv2.THRESH_BINARY, 11, 2)  # adaptive gaussian
		if write:
			cv2.imwrite(self.output_dir + self.name + '_ADAPTIVE_THRESH_GAUSSIAN_C.jpg', self.current_image)
		return self.current_image
	
	def create_otsu(self, write=False):
		"""
		Binary+OTSU filtering
		:param write: write or not in outputs
		"""
		self.create_greyscale_img()
		self.gaussian_blur()
		if len(self.current_image.shape) != 2:
			self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)  # making current image greyscale
		ret3, th_otsu = cv2.threshold(self.current_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
		self.current_image = th_otsu
		if write:
			cv2.imwrite(self.output_dir + self.name + '_THRESH_OTSU.jpg', self.current_image)
		return self.current_image
	
	def create_blob(self, write=False, m=0):
		"""
		Finding blobs
		:param m:
		:param write: write in outputs or not
		"""
		self.create_greyscale_img()
		# Setup SimpleBlobDetector parameters.
		params = cv2.SimpleBlobDetector_Params()  # Change thresholds
		params.minThreshold = 10
		params.maxThreshold = 200
		params.filterByArea = True  # Filter by Area.
		params.minArea = m
		# params.filterByCircularity = True  # Filter by Circularity
		# params.minCircularity = 0.1
		params.filterByConvexity = True  # Filter by Convexity
		params.minConvexity = 0.87
		params.filterByInertia = True  # Filter by Inertia
		params.minInertiaRatio = 0.01
		# Create a detector with the parameters
		# OLD: detector = cv2.SimpleBlobDetector(params)
		detector = cv2.SimpleBlobDetector_create(params)
		# Detect blobs.
		key_points = detector.detect(self.current_image)
		# Draw detected blobs as red circles.
		# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEY_POINTS ensures
		# the size of the circle corresponds to the size of blob
		self.current_image = cv2.drawKeypoints(self.current_image, key_points, np.array([]), (0, 0, 255),
		                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		if write:
			cv2.imwrite(self.output_dir + self.name + '_BLOB.jpg', self.current_image)
		return self.current_image
	
	def morph_trans(self, write=False):
		"""
		morp transform
		:param write:
		"""
		self.create_greyscale_img()
		kernel = np.ones((5, 5), np.uint8)
		self.current_image = cv2.erode(self.current_image, kernel, iterations=1)
		if write:
			cv2.imwrite(self.output_dir + self.name + '_Morph_trans.jpg', self.current_image)
		return self.current_image
	
	def test_all_funcs(self, forever=False):
		all_plan_funcs = (
			self.create_greyscale_img(write=True),
			self.create_canny_edges(write=True, white=False),
			self.create_blob(write=True),
			self.create_otsu(write=True),
			self.create_binary_img(write=True),
			self.create_adaptive_gaussian_c(write=True),
			self.create_adaptive_mean_c(write=True),
			self.create_contours(write=True),
			self.morph_trans(write=True),
			self.thick_lines_filtering(write=True)
		)
		if forever:
			while True:
				random.choice(all_plan_funcs)
		else:
			for _ in range(10):
				random.choice(all_plan_funcs)
	
	def preprocess_no_pytess(self, write=True, show=True):
		"""
		#  output_dir: str = os.getcwd(),
		preprocess image without pytesseract
		:param write: Write image or not
		:rtype: Image class object
		:param show: show if True
		"""
		self.create_binary_img()
		self.resize_img()
		if show:
			self.show()
		self.create_otsu()
		if show:
			self.show()
		self.create_contours(write=write)
		if show:
			self.show()
		return self.current_image
	
	def preprocess_pytess_floors(self, config=r'--psm 11 --oem 1', write=True, show=True):
		"""
		Preprocess with tesseract
		:param config: rte
		:param write: write or not
		:param show: show or not
		"""
		floor_names = ('basement', 'ground', 'first', 'second', 'third', 'loft', 'roof')
		output_images = []
		
		grey = cv2.cvtColor(self.start_image, cv2.COLOR_BGR2GRAY)
		mask = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
		
		text = pytess.image_to_string(mask, lang='eng', config=config)
		opo = re.sub(',', '', text).lower()
		floor_q = []
		for f in floor_names:
			if f == opo or f in opo:
				floor_q.append([f])
		
		floor_q = [value for value in floor_q if value != 2]
		floor_qty = len(floor_q)
		
		image_copy = self.start_image.copy()
		mask = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 5)
		contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
		
		# max_rect_area = max(cv2.minAreaRect(contour)[1][0] * cv2.minAreaRect(contour)[1][1] for contour in contours)
		# for contour in contours:
		# 	if cv2.minAreaRect(contour)[1][0] * cv2.minAreaRect(contour)[1][1] > max_rect_area * 0.95 or\
		# 		cv2.minAreaRect(contour)[1][0] * cv2.minAreaRect(contour)[1][1] < max_rect_area * 0.04:
		# 		cv2.drawContours(image_copy, [contour], -1, (255, 255, 255), 3)
		
		for contour in contours:
			area = cv2.contourArea(contour)
			if area > 5000000 or area < 500:
				cv2.drawContours(image_copy, [contour], -1, (255, 255, 255), 3)
		
		# fill detected text
		h, w = grey.shape[:2]
		boxes = pytess.image_to_boxes(mask, lang='eng', config=config)
		for b in boxes.splitlines():
			b = b.split(' ')
			cv2.rectangle(self.current_image, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (255, 255, 255),
			              -1)
		gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
		mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 5)
		contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		for cnt in contours:
			cv2.drawContours(image_copy, [cnt], -1, (0, 0, 0), 50)
		gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
		mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 5)
		contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
		floor_area = {}
		for cnt in contours:
			x, y, w, h = cv2.boundingRect(cnt)
			area = cv2.contourArea(cnt)
			floor_area.update({area: [x, y, w, h]})
		sorted_floor_area = sorted(floor_area.items(), key=lambda k: k[0], reverse=True)
		numbers = int(floor_qty * 2)
		number = sorted_floor_area[:numbers]
		floor_coordinates = []
		
		for i in number:
			floor_coordinates.append(i[-1])
		final_coord, weights = cv2.groupRectangles(floor_coordinates, groupThreshold=1, eps=0.05)
		print(final_coord)
		for i in final_coord:
			x, y, w, h = i[0], i[1], i[2], i[3]
			crop = gray.copy()
			final_image = crop[y:y + h, x:x + w]
			data = np.asarray(final_image)
			output_images.append(data)
			m_s = max(data.shape[:2])
			width = int(round(data.shape[1] * 1024 / m_s))
			height = int(round(data.shape[0] * 1024 / m_s))
			if write:
				cv2.imwrite(self.output_dir + f'{i}' + '_preprocessed.jpg', data)
			dim = (width, height)
			resized_img = cv2.resize(data, dim, interpolation=cv2.INTER_AREA)  # resize image
			cv2.imshow('floor', resized_img)
		if show:
			self.show()
		return self, output_images


def main_preprocessing(input_dir: str = os.getcwd() + '/input_images/',
                       output_dir: str = os.getcwd() + '/preprocessed_images/'):
	"""
	Cleaning input pictures
	:param input_dir: directory of input pictures for preprocessing
	:param output_dir: directory for outputs of preprocessing
	"""
	# curr_dir = os.getcwd()
	# input_dir = curr_dir + '/input_images/'
	# output_dir = curr_dir + '/preprocessed_images/'
	clean_dir(output_dir)
	all_images_paths = all_paths(input_dir)
	for img_p in all_images_paths:
		img = Image(image_path=img_p, output_dir=output_dir)
		img.preprocess_no_pytess()
		img.resize_img()
		# img.preprocess_pytess_floors()


if __name__ == '__main__':
	main_preprocessing()
