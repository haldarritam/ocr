# # Import required packages
# import cv2
# import pytesseract

# # Mention the installed location of Tesseract-OCR in your system
# # pytesseract.pytesseract.tesseract_cmd = 'ocr_env/bin/pytesseract'

# # Read image from which text needs to be extracted
# img = cv2.imread("viv_doc_img.jpeg")

# # Preprocessing the image starts

# # Convert the image to gray scale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Performing OTSU threshold
# ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

# # Specify structure shape and kernel size.
# # Kernel size increases or decreases the area
# # of the rectangle to be detected.
# # A smaller value like (10, 10) will detect
# # each word instead of a sentence.
# rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

# # Applying dilation on the threshold image
# dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)

# # Finding contours
# contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
# 												cv2.CHAIN_APPROX_NONE)

# # Creating a copy of image
# im2 = img.copy()

# # A text file is created and flushed
# file = open("recognized.txt", "w+")
# file.write("")
# file.close()

# # Looping through the identified contours
# # Then rectangular part is cropped and passed on
# # to pytesseract for extracting text from it
# # Extracted text is then written into the text file
# for cnt in contours:
# 	x, y, w, h = cv2.boundingRect(cnt)
	
# 	# Drawing a rectangle on copied image
# 	rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
	
# 	# Cropping the text block for giving input to OCR
# 	cropped = im2[y:y + h, x:x + w]
	
# 	# Open the file in append mode
# 	file = open("recognized.txt", "a")
	
# 	# Apply OCR on the cropped image
# 	text = pytesseract.image_to_string(cropped)
	
# 	# Appending the text into file
# 	file.write(text)
# 	file.write("\n")
	
# 	# Close the file
# 	file.close



import pytesseract, cv2
from pytesseract import Output
from PIL import Image
import pandas as pd

# Read image from which text needs to be extracted
img = cv2.imread("viv_doc_img.jpeg")

# Preprocessing the image starts

# Convert the image to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Performing OTSU threshold
# ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
# th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

blur = cv2.GaussianBlur(gray,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


# Using cv2.blur() method 
ksize = (1, 1)  
image = cv2.blur(gray, ksize) 

img = th3
# img = gray
# img = image

cv2.imwrite('test.jpeg', img)


custom_config = r'-c preserve_interword_spaces=1 --oem 1 --psm 1 -l eng+ita'
# d = pytesseract.image_to_data(Image.open(r'viv_doc_img.jpeg'), config=custom_config, output_type=Output.DICT)
d = pytesseract.image_to_data(img, config=custom_config, output_type=Output.DICT)
df = pd.DataFrame(d)
file = open("recognized.txt", "w+")

# clean up blanks
df1 = df[(df.conf!='-1')&(df.text!=' ')&(df.text!='')]

# sort blocks vertically
sorted_blocks = df1.groupby('block_num').first().sort_values('top').index.tolist()
for block in sorted_blocks:
    curr = df1[df1['block_num']==block]
    sel = curr[curr.text.str.len()>3]
    print("-------------")
    print(curr)
    print("-------------")
    char_w = (sel.width/sel.text.str.len()).mean()
    prev_par, prev_line, prev_left = 0, 0, 0
    text = ''
    for ix, ln in curr.iterrows():
        # add new line when necessary
        if prev_par != ln['par_num']:
            text += '\n'
            prev_par = ln['par_num']
            prev_line = ln['line_num']
            prev_left = 0
        elif prev_line != ln['line_num']:
            text += '\n'
            prev_line = ln['line_num']
            prev_left = 0

        added = 0  # num of spaces that should be added
        if ln['left']/char_w > prev_left + 1:
            added = int((ln['left'])/char_w) - prev_left
            text += ' ' * added 
        text += ln['text'] + ' '
        prev_left += len(ln['text']) + added + 1
    text += '\n'
    file.write(text)
file.close()