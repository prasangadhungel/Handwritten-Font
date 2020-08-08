# 0. If it says: " import cv2" ... "ImportError: DLL load failed: The specified module could not be found." then:
# 0.1 open anaconda and type:
# pip install opencv-python
# pip install opencv-contrib-python
# conda install -c menpo opencv
# pip install StringIO
# 1. Similarly if the error is on the line "import numpy as np" then open anaconda and type:
# conda install numpy

# pip install pdf2image
#conda install -c conda-forge poppler
import os, re, os.path
import cv2
import numpy as np
import re
import sys
from io import StringIO ## for Python 3
from pdf2image import convert_from_path
from pyzbar.pyzbar import decode
from PIL import Image

import scipy.signal
import numpy as np
import matplotlib.pyplot as plt

class ReadTemplate:
    
    # intializes object
    def __init__(self):
        ## Declare parameters
        # scanned filled image that is read in for processing
        self.image = None 
        # original scanned filled image that is read in staying unmodified for symbol extraction
        self.original = None 
        # black and white image
        self.gray = None
        # blurred black and white image
        self.blur  = None
        # blurred black and white image after filter with some threshold
        self.thresh = None
        # the contours that are detected by the algorithm
        self.cnts = None
        
        ## Specifications that are outputed by the template creating module
        self.nrOfSymbols = None
        self.boxWidth = None
        self.boxHeight = None
        self.nrOfBoxesPerLine = None
        self.nrOfBoxesPerLineMinOne = None
        self.nrOfLinesPerPage = None
        self.nrOfLinesPerPageMinOne = None
        
        ## hardcode script parameters
        # relative qr specification file location
        self.spec_filename = 'symbol_spec.txt'
        self.spec_loc = f'../template_creating/{self.spec_filename}'
        # output directory for contours (used for temporary manual inspection of code)
        self.output_dir = "out"
        # scanned template pdf name
        self.scanned_pdf_name = 'filled.pdf'
        # Input directory with scanned/photographed filled templates
        self.input_dir = "in"
        # output directory to which the fonts/letter/symbol images are being exported
        self.font_dir = 'font' 
        # False = do not overwrite symbols that are already captured and exported
        self.overwrite = False
        # The scanned/photographed filled template image that is being read.
        self.img_name = f'{self.input_dir}/p0.png'
        
        ## execute file reading steps
        # read the image specifications of the outputted pdf template
        self.read_image_specs() 
        # clear the contour output folder 
        self.clear_output_folder() 
        # the template is exported as pdf, and perhaps the scanned files are returned to pdf
        # the code however only evaluates images, so pdfs are converted to images
        self.nr_input_img = self.convert_pdf_to_img(f'{self.input_dir}/{self.scanned_pdf_name}') 
        # Load the (processed) image(s) from the input folder self.input_dir
        self.image, self.original,self.thresh = self.load_image(self.img_name)
        # Apply morphing to image, don't know why necessary but it works
        self.close,self.kernel = self.morph_image()
        # Finds the contours which the code thinks contain qr codes
        self.cnts = self.find_contours()
        # Returns the regions of interest that actually contain qr codes
        # ROI consists of 3 stacked blocks, on top the printed symbol, middle =written symbol,
        # bottom is qr code
        self.ROIs,self.ROIs_pos= self.loop_through_contours()
        # read the qr codes from the ROIs and export the found handwritten symbols
        self.read_qr_imgs()
        
    # clear the contour output folder 
    def clear_output_folder(self):
        for root, dirs, files in os.walk(self.output_dir):
            for file in files:
                os.remove(os.path.join(root, file))
        
    # converts pdf (from input path) to set of images
    def convert_pdf_to_img(self,path_to_pdf):
        
        # first convert the pdf into separate images        
        pages = convert_from_path(path_to_pdf, 500)
        count=0
        for page in pages:
            page.save(f'{self.input_dir}/out{count}.jpg', 'JPEG')
            count = count+ 1
        return count

    # loads image, makes a copy to keep as original, applies blur, 
    # outputs blurred image and applies a filtering threshold to image
    # returns the image for modification, the original image and the thresholded image
    def load_image(self,img_name):
        image = cv2.imread(img_name)
        original = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9,9), 0)
        cv2.imwrite(f'{self.output_dir}/blur.png', blur)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        return image,original,thresh
        
    # returns and generates a morphing kernal and something/close
    # based on the blurred thresholded image
    def morph_image(self):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        close = cv2.morphologyEx(self.thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        return close,kernel

    # Find contours and filter for QR code from the blurred and thresholded image
    def find_contours(self):
        cnts = cv2.findContours(self.close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        return cnts

    # returns the region of interest (ROI) and ROI pixelwise corner coordinates in original image
    def loop_through_contours(self):
        ROIs = []
        ROIs_pos = []
        
        # loop through contours
        for c in self.cnts:
            #if len(ROIs) <2: # for quick debugging on single succesfull qr code
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True) # number of sides of polygon?
            x,y,w,h = cv2.boundingRect(approx)
            area = cv2.contourArea(c)
            ar = w / float(h)
            ROIs,ROIs_pos= self.select_qr_contours(approx,ar,area,h,w,x,y,ROIs,ROIs_pos)
        return ROIs,ROIs_pos
        
    # TODO: change len(approx) to a standardized value from qr code sizes
    # TODO: change area size to output of box size in symbol_spec.txt
    # TODO: change ar size to output of box size in symbol_spec.txt    
    # Selects the contours based on their geometrics vs symbol spec file outputed by template creation
    def select_qr_contours(self,approx,ar,area,h,w,x,y,ROIs,ROIs_pos):
        
        # area is the nr of minimum pixels a ROI must have
        # ar is the aspect ratio (width/height)
        if area > 100 and (ar > .25 and ar < 0.4):
            cv2.rectangle(self.image, (x, y), (x + w, y + h), (36,255,12), 3)
            ROI = self.original[y:y+h, x:x+w]
            
            #ROI_pos =[left,top,width,height] measured from top to bottom and left edge to right
            ROI_pos =[x,y,w,h]
            
            # convert images to grayscale
            ROI_gray = self.convert_img_to_grayscale(ROI)
            
            # blur/smoothen roi
            ROI_gray = self.smoothen_img(ROI_gray)
            
            # export images
            cv2.imwrite(f'{self.output_dir}/ROI_{len(ROIs)}.png', ROI_gray)
            ROIs.append(ROI_gray)
            ROIs_pos.append(ROI_pos)
        return ROIs,ROIs_pos
            
    # applies a smoothing to the ROI to enhance qr detection rate in grainy images
    def smoothen_img(self,im):
        # make a smoothing kernel
        t = 1 - np.abs(np.linspace(-1, 1, 21))
        kernel = t.reshape(21, 1) * t.reshape(1, 21)
        kernel /= kernel.sum()   # kernel should sum to 1!  :) 

        # convolve 2d the kernel with each channel
        print(f'im.shape={im.shape}')
        im_out = scipy.signal.convolve2d(im, kernel, mode='same')
        
        return im_out
    
    # read image specs from the specifications txt file
    def read_image_specs(self):
        file1 = open(self.spec_loc, 'r') 
        Lines = file1.readlines() 
        self.nrOfSymbols = self.rhs_val_of_eq(Lines[0])
        self.boxWidth = self.rhs_val_of_eq(Lines[1])
        self.boxHeight = self.rhs_val_of_eq(Lines[2])
        self.nrOfBoxesPerLine = self.rhs_val_of_eq(Lines[3])
        self.nrOfBoxesPerLineMinOne = self.rhs_val_of_eq(Lines[4])
        self.nrOfLinesPerPage = self.rhs_val_of_eq(Lines[5])
        self.nrOfLinesPerPageMinOne = self.rhs_val_of_eq(Lines[6])
    
    # returns the integer value of the number on the rhs of the equation in the line
    def rhs_val_of_eq(self, line):
        start_index = line.find(' = ')
        rhs = line[start_index:]
        return ''.join(x for x in rhs if x.isdigit())
    
    # returns image that is converted from (width x height x 3) (rgb) to:
    # gray scale (width x height x 1) (blackwhite)
    def convert_img_to_grayscale(self,color_img):
        gray_img = np.zeros((color_img.shape[0],color_img.shape[1]),dtype=int)
        gray_img = color_img.mean(axis=2)
        return gray_img
    
    # reads the qr codes, finds written symbol and exports symbol as picture with the id contained in the qr code.
    def read_qr_imgs(self):
    
        # loop through all regions of interest (that might contain qr code)
        for i in range(0,len(self.ROIs)):
            
            # get qr code from ROI
            contour = self.ROIs[i]
            img = cv2.imread(f'{self.output_dir}/ROI_{i}.png')
            qrcode = self.preprocess_qrcode(img)
            print(f'nr of qr codes in ROI {i} is: {len(qrcode)}')
            
            # if the ROI contains a (legible) qr code
            if len(qrcode)>0:
                
                # get then encoded qr code content as decoded text
                qr_content = self.get_qr_content(qrcode)
            
                # Get the qr coordinates relative to the contour:
                left_qr_cont = qrcode[0].rect[0]
                top_qr_cont = qrcode[0].rect[1]
                width_qr_cont = qrcode[0].rect[2]
                height_qr_cont = qrcode[0].rect[3]
                print(f'relative qr code positions = left={left_qr_cont},top={top_qr_cont},width={width_qr_cont},height={height_qr_cont}')
        
                # get the contour positions wrt the original image
                left_ct = self.ROIs_pos[i][0]
                top_ct = self.ROIs_pos[i][1]
                width_ct = self.ROIs_pos[i][2]
                height_ct = self.ROIs_pos[i][3]
                bottom_ct = top_ct+height_ct
                right_ct = left_ct+width_ct        
                print(f'The original contour coordinates are= left={left_ct},top={top_ct},bottom={bottom_ct},right={right_ct}\n\n')
                
                # compute absolute qr position wrt original image
                top_qr = top_ct+top_qr_cont
                left_qr = left_ct+left_qr_cont
                bottom_qr =top_ct+top_qr_cont+height_qr_cont
                right_qr = left_ct+ left_qr_cont+width_qr_cont
                
                # trim parameters for the edges of the symbol square to remove black borders of 
                # the square around the handwritten symbol (space)
                vert_focus_margin = 0.95
                hori_focus_margin = 0.94
                
                # compute handwritten symbol position wrt original image
                top_sym = top_qr-height_qr_cont*vert_focus_margin
                left_sym = left_qr+width_qr_cont*(1-hori_focus_margin)
                bottom_sym = top_qr-height_qr_cont*(1-vert_focus_margin)
                right_sym = right_qr-width_qr_cont*(1-hori_focus_margin)
   
                # reload full image
                full_img=Image.open(self.img_name)
                
                # export the contour/ROI to the output folder for manual inspection
                contour = full_img.crop((left_ct,top_ct,right_ct,bottom_ct))
                contour.save(f'out/contour{i}.png')
                
                # crop the qr code from the original image and export it to output folder for manual inspection
                im_crop = full_img.crop((left_qr,top_qr,right_qr,bottom_qr))
                im_crop.save(f'out/crop{i}.png')
                
                # crop the handwritten symbol from the original image and export it to output folder for manual inspection
                sym_crop = full_img.crop((left_sym,top_sym,right_sym,bottom_sym))
                sym_crop.save(f'out/symbol_{i}_{qr_content}.png')
                
                # export the handwritten symbol to the font output directory
                self.export_to_font(sym_crop,qr_content)
    
    # export font symbol
    def export_to_font(self, img,symbol_nr):
        
        # first create font output folder if it doesnt exist yet
        if not os.path.exists(self.font_dir):
            os.makedirs(self.font_dir)
        
        # Check if the symbol is already captured or should be overwritten
        if self.overwrite or (not os.path.exists(f'{self.font_dir}/{symbol_nr}.png')):
            
            # export/store the handwritten symbol
            img.save(f'{self.font_dir}/{symbol_nr}.png')
            
    # Source of magic preprocessing settings: https://stackoverflow.com/questions/50080949/qr-code-detection-from-pyzbar-with-camera-image
    def preprocess_qrcode(self,image):
        # thresholds image to white in back then invert it to black in white
        #   try to just the BGR values of inRange to get the best result
        mask = cv2.inRange(image,(0,0,0),(200,200,200))
        thresholded = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
        inverted = 255-thresholded # black-in-white
        qrcode = decode(inverted)
        print(f'qrcode={qrcode}')
        return qrcode
    
    # returns the (decoded) content of the qr code as a string.
    def get_qr_content(self,qrcode):
        # format of qr content is given as b'content' so remove b'
        content= str(qrcode[0].data)[2:]
        # remove the last ' of the content format
        content = content[:-1]
        return content
                
    # unused method to export an image in ndarray format to a txt file.
    # (to numerically inspect qr code images)
    def ndarray_to_txt(self,arr):
        str_arr = ""
        text_file = open("foo.txt", "w")
        np.set_printoptions(threshold=sys.maxsize)
        text_file.write(str(arr))
        text_file.close()
    
    # Converts a ndarray image (qr code) to a binary string at a threshold
    # i.e. white = 0, black = 255, everything below 245 is mapped to 0, above to 1.
    # this yields string of 0 and 1's (binary) string.
    def binary_string(self,arr, isbinary=False):
        threshold = 245
        output = StringIO()
        for row in range(0,arr.shape[0]):
            for column in range(0,arr.shape[1]):
                if (isbinary and arr[row][column]):
                    output.write('X')
                elif arr[row][column]>threshold:
                    output.write('X')
                else:
                    output.write('_')
            output.write('\n')
            content = output.getvalue()
        output.close()    
        
        return content
        
# executes this main code
if __name__ == '__main__':
    main = ReadTemplate()