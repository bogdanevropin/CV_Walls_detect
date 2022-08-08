"""
Areas comparing and finding difference by homography matching and key-points
"""
import cv2
import imutils
from skimage.metrics import structural_similarity as ssim
import numpy as np
import os
from preprocessing import Image, all_paths


class Images_compare:
	"""
	All images comparing functions
	"""
	
	@staticmethod
	def homography_match(im_1: Image, im_2: Image, max_features=500, good_match_percent=0.5, write=True, output_dir=os.getcwd() + '/after_transforming/'):
		"""
		Comparing and return image and homography transform matrix
		:param write: need to save or not image
		:param output_dir: for saving match img
		:param good_match_percent: how many draw matches
		:param max_features: 500 default
		:param im_1: Image class obj
		:param im_2: Image class obj
		:return: image and transform matrix
		"""
		im_1.create_greyscale_img()
		im_2.create_greyscale_img()
		# Detect ORB features and compute descriptors.
		orb = cv2.ORB_create(max_features)
		key_points_before, descriptors_before = orb.detectAndCompute(im_1.current_image, None)  # Finding key_points
		key_points_after, descriptors_after = orb.detectAndCompute(im_2.current_image, None)
		matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)  # Match features.
		matches = list(matcher.match(descriptors_before, descriptors_after, None))
		matches.sort(key=lambda x: x.distance, reverse=False)  # Sort matches by score
		non_matches = int(len(matches) * good_match_percent)  # Remove not enough percent matches
		matches = matches[:non_matches]
		# Draw top matches
		# new_img = Image.inverting_img(np.zeros((im_after.current_image.shape[0], im_after.current_image.shape[1], 1), dtype=np.uint8))
		im_matches = cv2.drawMatches(im_1.current_image, key_points_before, im_2.current_image, key_points_after,
		                             matches, None)
		# im_matches = cv2.drawMatches(im_before.current_image, key_points_before, im_after.current_image, key_points_after, matches, new_img)
		cv2.imwrite(os.getcwd() + '/schemas/matches_' + im_1.name + '_and_' + im_2.name + '.jpg', im_matches)
		# Extract location of good matches
		points_before = Image.inverting_img(np.zeros((len(matches), 2), dtype=np.float64))
		points_after = Image.inverting_img(np.zeros((len(matches), 2), dtype=np.float64))
		
		for i, match in enumerate(matches):
			points_before[i, :] = key_points_before[match.queryIdx].pt
			points_after[i, :] = key_points_after[match.trainIdx].pt
		# Find homography transform matrix
		h_trans_matrix, mask = cv2.findHomography(points_before, points_after, cv2.RANSAC)
		# # Print estimated homography
		# print("Estimated homography : \n", h_trans_matrix)
		# Use homography
		height, width = im_2.current_image.shape[:2]
		im_1.current_image = cv2.warpPerspective(im_1.current_image, h_trans_matrix, (width, height),
		                                         flags=cv2.INTER_LINEAR, borderValue=255,
		                                         borderMode=cv2.BORDER_CONSTANT)
		if write:
			cv2.imwrite(output_dir + im_1.name + '.jpg', im_1.current_image)
			cv2.imwrite(output_dir + im_2.name + '.jpg', im_2.current_image)
		return im_1, im_2, h_trans_matrix
	
	@staticmethod
	def xor_comparing(im1: Image, im2: Image, filtering=True, write=True, output_dir: str = os.getcwd() + '/after_transforming/xor_compared/'):
		"""
		Make xor
		:param write: Write or not
		:param filtering: Make filtering before xor or not
		:param im1: Image class obj
		:param im2: Image class obj
		:param output_dir: for writing
		"""
		if filtering:
			im1.create_canny_edges()
			im2.create_canny_edges()
			im1.create_otsu()
			im2.create_otsu()
		compared = cv2.bitwise_not(cv2.bitwise_xor(im2.current_image, im1.current_image))
		if write:
			cv2.imwrite(output_dir + '/' + im1.name + '_Compared_xor.jpg', compared)
		return im1, im2, Image(output_dir + '/' + im1.name + '_Compared_xor.jpg', compared)
	
	@staticmethod
	def contours(im1, im2, full=True):
		"""
		Finding difference between two images
		:param im1: Image object 1 for comparing
		:param im2: Image object 2 for comparing
		:param full:
		"""
		if 'after' not in im2.name:
			im1, im2 = im2, im1
		im1.create_greyscale_img()
		im2.create_greyscale_img()
		score, diff = ssim(im1.current_image, im2.current_image, full=full)
		diff = Image.inverting_img((diff * 255).astype("uint8"))
		thresh = cv2.threshold(diff, 0, 255,
		                       cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
		contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		                            cv2.CHAIN_APPROX_SIMPLE)
		contours = imutils.grab_contours(contours)
		for c in contours:
			# compute the bounding box of the contour and then draw the
			# bounding box on both input images to represent where the two
			# images differ
			(x, y, w, h) = cv2.boundingRect(c)
			cv2.rectangle(im1.current_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
			cv2.rectangle(im2.current_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
		# show the output images
		cv2.imshow("Original", im1.current_image)
		cv2.imshow("Modified", im2.current_image)
		cv2.imshow("Diff", diff)
		cv2.imshow("Thresh", thresh)
		cv2.waitKey(0)
		return im1, im2, diff
	
	@staticmethod
	def find_diff(im1, im2, ratio=100):
		"""
		Finding difference between image objects
		:param ratio:
		:param im1: Image class object
		:param im2: Image class object
		"""
		if 'after' in im1.name:
			im2, im1 = im1, im2
		im1.create_greyscale_img()
		im2.create_greyscale_img()
		im1.create_binary_img()
		im2.create_binary_img()
		im_before = im1.current_image
		im_after = im2.current_image
		score, difference = ssim(im_before, im_after, full=True)
		# The diff image contains the actual image differences between the two images
		# and is represented as a floating point data type in the range [0,1]
		# so we must convert the array to 8-bit unsigned integers in the range
		# [0,255] before we can use it with OpenCV
		difference = (difference * 255).astype("uint8")
		diff_box = cv2.merge([difference, difference, difference])
		# Threshold the difference image, followed by finding contours to
		# obtain the regions of the two input images that differ
		r, thresh_otsu = cv2.threshold(difference, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
		contours_f, hierarchy = cv2.findContours(image=thresh_otsu, mode=cv2.RETR_TREE,
		                                         method=cv2.CHAIN_APPROX_SIMPLE)
		max_cont_area = max(cv2.contourArea(contour) for contour in contours_f)
		contours_filtered = []
		for contour in contours_f:
			if cv2.contourArea(contour) > max_cont_area / ratio:
				contours_filtered.append(contour)
		# contours_f = cv2.findContours(thresh_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		# contours_f = contours_f[0] if len(contours_f) == 2 else contours_f[1]
		mask = Image.inverting_img(np.zeros(im_before.shape, dtype='uint8'))
		filled_after = im_after.copy()
		
		for c in contours_filtered:
			# area = cv2.contourArea(c)
			x, y, w, h = cv2.boundingRect(c)
			cv2.rectangle(im_before, (x, y), (x + w, y + h), (36, 255, 12), 2)
			cv2.rectangle(im_after, (x, y), (x + w, y + h), (36, 255, 12), 2)
			cv2.rectangle(diff_box, (x, y), (x + w, y + h), (36, 255, 12), 2)
			cv2.drawContours(mask, [c], 0, (255, 255, 255), -1)
			cv2.drawContours(filled_after, [c], 0, (0, 255, 0), -1)
		
		cv2.imshow('before', im_before)
		cv2.imshow('after', im_after)
		cv2.imshow('diff', difference)
		cv2.imshow('diff_box', diff_box)
		cv2.imshow('mask', mask)
		cv2.imshow('filled after', filled_after)
		cv2.waitKey()
		return im1, im2, difference
	
	@staticmethod
	def mark_new(im1: Image, im2: Image, write=True,
	             output_dir: str = os.getcwd() + '/after_transforming/xor_compared/'):
		"""
		fill images with white colour in areas of size's deference
		:param im1: Image object
		:param im2: Image object
		:param write:
		:param output_dir:
		"""
		rows_1, cols_1 = im1.current_image.shape[:2]
		rows_2, cols_2 = im2.current_image.shape[:2]
		sizes = list(im1.current_image.shape)
		
		max_high = max(rows_1, rows_2)
		max_length = max(cols_1, cols_2)
		sizes[0] = max_high
		sizes[1] = max_length
		img1 = im1.current_image.copy()
		img2 = im2.current_image.copy()
		blank_image = np.zeros(sizes, np.uint8)
		blank_image[:, :] = (255, 255, 255)
		new_img1 = blank_image.copy()
		new_img2 = blank_image.copy()
		x_offset = y_offset = 0
		# Here, y_offset+height <= blank_image.shape[0] and x_offset+width <= blank_image.shape[1]
		new_img1[y_offset:y_offset + rows_1, x_offset:x_offset + cols_1] = img1.copy()
		new_img2[y_offset:y_offset + rows_2, x_offset:x_offset + cols_2] = img2.copy()
		
		cv2.imshow('img1', img1)
		cv2.imshow('img2', img2)
		im1.current_image = new_img1
		im2.current_image = new_img2
		cv2.destroyAllWindows()
	
	# @staticmethod
	# def fill_zeros(im1: Image, im2: Image, write=True,
	#                output_dir: str = os.getcwd() + '/after_transforming/xor_compared/'):
	# 	"""
	# 	fill images with white colour in areas of size's deference
	# 	:param im1: Image object
	# 	:param im2: Image object
	# 	:param write:
	# 	:param output_dir:
	# 	"""
	#
	# 	rows_1, cols_1 = im1.current_image.shape[:2]
	# 	rows_2, cols_2 = im2.current_image.shape[:2]
	# 	sizes = list(im1.current_image.shape)
	#
	# 	max_high = max(rows_1, rows_2)
	# 	max_length = max(cols_1, cols_2)
	# 	sizes[0] = max_high
	# 	sizes[1] = max_length
	# 	new_img_1 = Image.inverting_img(np.zeros(sizes))  # full size image with black dots
	# 	new_img_2 = Image.inverting_img(np.zeros(sizes))
	# 	# I want to put each image on top-left corner, So I create a ROI
	# 	img1 = im1.current_image.copy()
	# 	img2 = im2.current_image.copy()
	# 	roi_1 = new_img_1[0:rows_1, 0:cols_1]
	# 	roi_2 = new_img_2[0:rows_2, 0:cols_2]
	# 	# Now create a mask of each image and create its inverse mask also
	# 	im1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	# 	im2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
	# 	ret_1, mask_1 = cv2.threshold(im1_gray, 10, 255, cv2.THRESH_BINARY)
	# 	ret_2, mask_2 = cv2.threshold(im2_gray, 10, 255, cv2.THRESH_BINARY)
	# 	mask_inv_1 = cv2.bitwise_not(mask_1)
	# 	mask_inv_2 = cv2.bitwise_not(mask_2)
	# 	# Now black-out the area of images in ROI
	# 	im1_bg = cv2.bitwise_and(roi_1, roi_1, mask=mask_inv_1)
	# 	im2_bg = cv2.bitwise_and(roi_2, roi_2, mask=mask_inv_2)
	# 	# Take only region of images from start images.
	# 	im1_fg = cv2.bitwise_and(roi_1, roi_1, mask=mask_1)
	# 	im2_fg = cv2.bitwise_and(roi_2, roi_2, mask=mask_2)
	# 	# Put images in ROI and modify the main image
	# 	dst_1 = cv2.add(im1_bg, im1_fg)
	# 	dst_2 = cv2.add(im2_bg, im2_fg)
	#
	# 	# img1 = Image.inverting_img(img1)
	# 	# img2 = Image.inverting_img(img2)
	#
	# 	img1[0:rows_1, 0:cols_1] = dst_1
	# 	cv2.imshow('dst 1', dst_1)
	# 	img2[0:rows_2, 0:cols_2] = dst_2
	# 	cv2.imshow('window', img2)
	# 	cv2.waitKey(0)
	# 	new_img_1 = cv2.add(new_img_1, img1)
	# 	new_img_2 = cv2.add(new_img_2, img2)
	# 	cv2.imshow('im 1', new_img_1)
	# 	cv2.imshow('im 2', new_img_2)


def main_comparing():
	"""
	Making homograph transforms
	"""
	input_dir = os.getcwd() + '/preprocessed_images/'
	output_dir = os.getcwd() + '/after_transforming/'
	images_paths = all_paths(input_dir)
	image_1: Image = Image(images_paths[0], output_dir)
	image_2: Image = Image(images_paths[1], output_dir)
	if 'after' in image_1.name:
		image_1, image_2 = image_2, image_1
	image_1, image_2 = Images_compare.homography_match(im_1=image_1, im_2=image_2, write=True)[:2]
	compared = Images_compare.xor_comparing(im1=image_1, im2=image_2)[2]
	compared.thick_lines_filtering(ratio=1000)
	compared.show()
	cv2.imwrite(output_dir + 'final_compared/' + compared.name + '_final.jpg', compared.current_image)
	return image_1, image_2, compared


if __name__ == '__main__':
	main_comparing()
