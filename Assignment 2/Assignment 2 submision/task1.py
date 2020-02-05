import cv2
############################IMAGE READING#######################################
img1 = cv2.imread("D:/NUST/PhD/Computer Vision/Assignment 2/A2_task1_images/building_1.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("D:/NUST/PhD/Computer Vision/Assignment 2/A2_task1_images/building_3.jpg", cv2.IMREAD_GRAYSCALE)
'''
############################SIFT FEATURE DETECTOR#######################################
# SIFT detect key features
sift= cv2.xfeatures2d.SIFT_create()
kp1, des1= sift.detectAndCompute(img1, None)
kp2, des2= sift.detectAndCompute(img2, None)

###########################ORB FEATURE DETECTOR######################################
MAX_MATCHES = 50
# Detect ORB features and compute descriptors.
orb = cv2.ORB_create(MAX_MATCHES)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
# Match features.
matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
matches = matcher.match(des1, des2, None)
# Sort matches by score
matches.sort(key=lambda x: x.distance, reverse=False)
matching_output = cv2.drawMatches(img1, kp1,img2, kp2, matches [:10], None, flags=2)
'''
#########################SURF FEATURE DETECTOR###########################################
surf= cv2.xfeatures2d.SURF_create()
kp1, des1= surf.detectAndCompute(img1, None)
kp2, des2= surf.detectAndCompute(img2, None)
#########################MATCHING AND OUTPUT###########################################
# Brute Force Matching
# bf= cv2.BFMatcher(cv2.NORM_L1, crossCheck= False)
# ??matches= bf.match(kp1, kp2)
# Match features
matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE)
matches = matcher.match(des1, des2, None)
matches=sorted(matches, key= lambda x:x.distance)

matching_output = cv2.drawMatches(img1, kp1,img2, kp2, matches [:10], None, flags=2)
# Draw keypoints
img1= cv2.drawKeypoints(img1, kp1,None)
img2= cv2.drawKeypoints(img2, kp2,None)
cv2.imshow("Matching Output", matching_output )
cv2.waitKey(0)
cv2.destroyAllWindows()
