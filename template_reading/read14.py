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
from os import listdir
from os.path import isfile, join
import cv2
import re
import sys
from io import StringIO ## for Python 3
from pdf2image import convert_from_path
from pyzbar.pyzbar import decode
from PIL import Image

from QrCode import QrCode

import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean


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
        self.scanned_pdf_name = 'filled'
        # Input directory with scanned/photographed filled templates
        self.input_dir = "in"
        # output directory to which the fonts/letter/symbol images are being exported
        self.font_dir = 'font' 
        # False = do not overwrite symbols that are already captured and exported
        self.overwrite = False
        # define types of images that are analysed
        self.image_types = ['.png','jpg','jpeg']
        # minimum frac
        self.dist_frac=0.25
        
        ## Run configuration settings
        self.all_cnts = False
        self.no_cnts_filter = False
        
    def perform_runs(self):
        ## read the image specifications of the outputted pdf template
        self.read_image_specs()
        
        ## Convert scanned pdf file to images
        # the template is exported as pdf, and perhaps the scanned files are returned to pdf
        # the code however only evaluates images, so pdfs are converted to images
        self.convert_pdf_to_img(self.input_dir,self.scanned_pdf_name)
        
        ## clear the contour output folder 
        self.clear_output_folder() 
        
        ## perform run on all images, scanning an entire page at once
        self.loop_through_scanned_images(self.all_cnts,True)
        
        # Perform run that first finds all contours,
        # then filters the contours/ROIs on aspect ratio, 
        # then scans qr codes in ROIs
        self.loop_through_scanned_images(self.all_cnts,self.no_cnts_filter)
        
        # Perform run that finds all contours and scans qr codes in all of them
        if not self.found_all_symbols():
            self.loop_through_scanned_images(True,self.no_cnts_filter)
    
    # loops through all pages of a scanned template pdf
    def loop_through_scanned_images(self,all_cnts,no_cnts_filter):
        
        # get a list of all input images
        input_files = [f for f in listdir(self.input_dir) if isfile(join(self.input_dir, f))]
        
        for i in range(0,len(input_files)):
            
            # only analyse certain input image types
            if any(map(input_files[i].__contains__, self.image_types)):
            
                # The scanned/photographed filled template image that is being read.
                img_name = f'{self.input_dir}/{input_files[i]}'
                
                # find symbols by performing run
                self.perform_sym_extraction_run(img_name,all_cnts,no_cnts_filter)
        
    # performs a run on an image based on run configuration settings    
    def perform_sym_extraction_run(self,img_name,all_cnts,no_cnts_filter):
        
        ## execute file reading steps
        # Load the (processed) image(s) from the input folder self.input_dir
        self.image, self.original,self.thresh = self.load_image(img_name)
         
        # decode entire page at once if no_cnts_filter
        if (no_cnts_filter):
            print(f'img_name={img_name}')
            merged_qrcodes = self.decode_complete_page(img_name)
            
            # perform geometric inference
            self.geometric_inference(img_name,merged_qrcodes) 
            
        else: # first select ROIs then find qr codes in those ROIs only
            # Apply morphing to image, don't know why necessary but it works
            self.close,self.kernel = self.morph_image() 
            # Finds the contours which the code thinks contain qr codes
            self.cnts = self.find_contours()
            # Returns the regions of interest that actually contain qr codes
            # ROI consists of 3 stacked blocks, on top the printed symbol, middle =written symbol,
            # bottom is qr code
            self.ROIs,self.ROIs_pos= self.loop_through_contours(all_cnts,no_cnts_filter)
            # read the qr codes from the ROIs and export the found handwritten symbols
            self.read_qr_imgs(img_name)
        
    # decodes complete page at once for both original and thresholded image
    def decode_complete_page(self,img_name):
        # read qrs directly from entire page
        original = decode(self.image)
        qrcode_original= self.decode_image(img_name,original)
                
        # read thesholded image
        thresh = decode(self.thresh)
        qrcode_thresholded= self.decode_image(img_name,thresh)
        
        # return merged qrcodes with dupes removed
        #return list(set(qrcode_original+qrcode_thresholded)) 
        merged_list = qrcode_original+qrcode_thresholded
        print(f'merged_list={merged_list}')
        seen = []
        unique_list =[]
        for obj in merged_list:
            if not (obj.hashcode in seen):
                seen.append(obj.hashcode)
                unique_list.append(obj)
            
        print(f'unique_list={unique_list}')
        return unique_list
        
        
    # performs actual decoding of qr codes on an image
    def decode_image(self,img_name,qrcode):
        custom_qrcode_objects =[]
        # find list of qr code content
        for i in range(0,len(qrcode)):
            
            # get the decoded text of the qr code
            qr_content = self.get_qr_content([qrcode[i]])# need to repack it as list
        
            # check if a missing qr code is found
            missing_sym_indices = self.list_missing_symbols()
            found_missing = self.list_contains_string(self.list_missing_symbols(), int(qr_content))
            if found_missing:
                print(f'FOUND MISSING={qr_content}')
                
                # compute coordinates of qr code
                left_qr,top_qr,width_qr,height_qr = self.get_qr_coord(qrcode[i])
                bottom_qr =top_qr+height_qr
                right_qr = left_qr+width_qr
                
                # reload full image
                full_img=Image.open(img_name)
                
                # export symbol to font directory
                self.get_symbol(full_img,qr_content,top_qr,left_qr,bottom_qr,right_qr,width_qr,height_qr)
                
                custom_qrcode_object = QrCode(qr_content, top_qr,left_qr,width_qr,height_qr,self.nrOfSymbols,self.nrOfBoxesPerLine,self.nrOfLinesPerPage)
                custom_qrcode_objects.append(custom_qrcode_object)
                
        return custom_qrcode_objects
                
        
    # clear the contour output folder 
    def clear_output_folder(self):
        for root, dirs, files in os.walk(self.output_dir):
            for file in files:
                os.remove(os.path.join(root, file))
        
    # converts pdf (from input path) to set of images
    def convert_pdf_to_img(self,path_to_pdf,pdf_name):
        
        # if scanned input folder contains a pdf file:
        input_files = [f for f in listdir(self.input_dir) if isfile(join(self.input_dir, f))]
        list_of_input_pdfs = [s for s in input_files if '.pdf' in s]
        
        # first convert the pdf into separate images        
        for i in range(0,len(list_of_input_pdfs)):
            pages = convert_from_path(f'{path_to_pdf}/{list_of_input_pdfs[i]}', 500)
            count=0
            for page in pages:
                page.save(f'{self.input_dir}/{list_of_input_pdfs[i][:-4]}_{i}_{count}.jpg', 'JPEG')
                count = count+ 1

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
    def loop_through_contours(self,all_cnts,no_cnts_filter):
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
            ROIs,ROIs_pos= self.select_qr_contours(approx,ar,area,h,w,x,y,ROIs,ROIs_pos,all_cnts,no_cnts_filter)
        return ROIs,ROIs_pos
        
    # TODO: change len(approx) to a standardized value from qr code sizes
    # TODO: change area size to output of box size in symbol_spec.txt
    # TODO: change ar size to output of box size in symbol_spec.txt    
    # Selects the contours based on their geometrics vs symbol spec file outputed by template creation
    def select_qr_contours(self,approx,ar,area,h,w,x,y,ROIs,ROIs_pos,all_cnts,no_cnts_filter):
        
        # area is the nr of minimum pixels a ROI must have
        # ar is the aspect ratio (width/height)
        if all_cnts or (area > 100 and (ar > .25 and ar < 0.4)):
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
        im_out = scipy.signal.convolve2d(im, kernel, mode='same')
        
        return im_out
    
    # read image specs from the specifications txt file
    def read_image_specs(self):
        file1 = open(self.spec_loc, 'r') 
        Lines = file1.readlines() 
        self.nrOfSymbols = self.rhs_val_of_eq(Lines[0])
        self.boxWidth = self.rhs_val_of_eq(Lines[1])
        self.boxHeight = self.rhs_val_of_eq(Lines[2])
        self.nrOfBoxesPerLine = int(self.rhs_val_of_eq(Lines[3]))
        self.nrOfBoxesPerLineMinOne = self.rhs_val_of_eq(Lines[4])
        self.nrOfLinesPerPage = int(self.rhs_val_of_eq(Lines[5]))
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
    def read_qr_imgs(self,img_name):
    
        # reload full image
        full_img=Image.open(img_name)
    
        # loop through all regions of interest (that might contain qr code)
        for i in range(0,len(self.ROIs)):
            
            # get qr code from ROI
            contour = self.ROIs[i]
            img = cv2.imread(f'{self.output_dir}/ROI_{i}.png')
            qrcode = self.preprocess_qrcode(img)
            print(f'nr of qr codes in ROI {i} is: {len(qrcode)}')
            
             # get the contour positions wrt the original image
            left_ct = self.ROIs_pos[i][0]
            top_ct = self.ROIs_pos[i][1]
            width_ct = self.ROIs_pos[i][2]
            height_ct = self.ROIs_pos[i][3]
            bottom_ct = top_ct+height_ct
            right_ct = left_ct+width_ct
            
            # export the contour/ROI to the output folder for manual inspection
            contour = full_img.crop((left_ct,top_ct,right_ct,bottom_ct))
            contour.save(f'out/contour{i}.png')
            
            
            # if the ROI contains a (legible) qr code
            if len(qrcode)>0:
                
                # get then encoded qr code content as decoded text
                qr_content = self.get_qr_content(qrcode) 
            
                # TODO: Check if there is only 1 qr code in ROI
                for j in range(0,len(qrcode)):
                    left_qr_cont,top_qr_cont,width_qr_cont,height_qr_cont = self.get_qr_coord(qrcode[j])
            
                    # compute absolute qr position wrt original image
                    top_qr = top_ct+top_qr_cont
                    left_qr = left_ct+left_qr_cont
                    bottom_qr =top_ct+top_qr_cont+height_qr_cont
                    right_qr = left_ct+ left_qr_cont+width_qr_cont
                    self.get_symbol(full_img,qr_content,top_qr,left_qr,bottom_qr,right_qr,width_qr_cont,height_qr_cont,i)
        
    def get_qr_coord(self,single_qrcode):
    # Get the qr coordinates relative to the contour:
        left_qr_cont = single_qrcode.rect[0]
        top_qr_cont = single_qrcode.rect[1]
        width_qr_cont = single_qrcode.rect[2]
        height_qr_cont = single_qrcode.rect[3]
        return left_qr_cont,top_qr_cont,width_qr_cont,height_qr_cont
        
    # gets the qr code, finds the symbol and exports the symbol
    def get_symbol(self,full_img,qr_content,top_qr,left_qr,bottom_qr,right_qr,width_qr_cont,height_qr_cont,i=None):
        
        # trim parameters for the edges of the symbol square to remove black borders of 
        # the square around the handwritten symbol (space)
        vert_focus_margin = 0.95
        hori_focus_margin = 0.94
        
        # compute handwritten symbol position wrt original image
        top_sym = top_qr-height_qr_cont*vert_focus_margin
        left_sym = left_qr+width_qr_cont*(1-hori_focus_margin)
        bottom_sym = top_qr-height_qr_cont*(1-vert_focus_margin)
        right_sym = right_qr-width_qr_cont*(1-hori_focus_margin)
        
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
    
    # returns true if all symbols are found in the font directory
    def found_all_symbols(self):
        if (len(self.list_missing_symbols())==0):
            return True
        return False
     
    # returns list of integers, containing missing symbol indices
    def list_missing_symbols(self):
        
        missing_sym = []
        
        # find list of files in font directory
        font_files = [f for f in listdir(self.font_dir) if isfile(join(self.font_dir, f))]
        
        # symbol index starts at one.
        for i in range(1,int(self.nrOfSymbols)):
            if not self.list_contains_string(font_files, f'{i}.png'):
                missing_sym.append(i)
        return missing_sym
        
    # returns true if any list element equals the string, false otherwise
    def list_contains_string(self,lst, string):
        for i in range(0,len(lst)):
            if string == lst[i]:
                return True
        return False
        
    def geometric_inference(self,qrcodes):
        qr = QrCode(11,1,2,3,4,100,9)
        pass
        
    # returns true if each row has contains a qr code that is found. Rows and columns start at 1
    def isQrcodeOnEachRow(self,qrcodes):
        # create a checklist with element for each row
        checklist = [False] * self.nrOfLinesPerPage
        # loop through rows (rows start at 1)
        for i in range(1,len(checklist)+1):
            # loop through qr codes till a qr is found on the row under evaluation
            for j in range(0,len(qrcodes)): 
                # inspect if the row of the qr code is equal to row under evaluation
                if qrcodes[j].row == i:
                    # set checklist item for that row to true if qr is in that row
                    checklist[i-1]=True #(rows start at 1 but lists don't so -1)
                    # skip the remainder of the qr codes since requirement is satisfied
                    j = len(checklist)+1 
        # check if all rows contain a qr code.
        if all(element for element in checklist):
            return True
        # Return false if not every row has a qr code
        return False
    
    def has_quarter_spacing(self,page_nr,qrcodes,row_nr=None):
        # loop through rows
        if row_nr == None:
            start_row = 1
            end_row = self.nrOfLinesPerPage
        else:
            start_row = row_nr
            end_row = start_row+1
        
        # loop through rows
        for row in range(start_row,end_row):
            most_left_col_per_row,most_right_col_per_row = self.get_most_left_and_right(row,page_nr,qrcodes)
            # check if distance is at least quarter of nr of lines per box
            if self.check_hori_dist(most_left_col_per_row,most_right_col_per_row,self.dist_frac):
                return True
        return False
    
    # returns the most left and most right qr code column of found qr code in a row 
    def get_most_left_and_right(self,row_nr,page_nr,qrcodes):
        most_left_col_per_row = None
        most_right_col_per_row = None
        # loop through qr codes
        for i in range(0,len(qrcodes)):
            # if qr in row, 
            if qrcodes[i].row==row_nr and qrcodes[i].page_nr==page_nr:  
                # update most left qr position for that row
                most_left_col_per_row = self.get_most_left_column_per_row(most_left_col_per_row,qrcodes[i])
                # update most right qr position for that row
                most_right_col_per_row = self.get_most_right_column_per_row(most_right_col_per_row,qrcodes[i])
        return most_left_col_per_row,most_right_col_per_row
    
    # returns the most left column number of the qr code that is found in a row
    def get_most_left_column_per_row(self,most_left,qrcode):
        if most_left==None:
            return qrcode.column
        else:
            if most_left>qrcode.column:
                return qrcode.column
            else:
                return most_left
    
    # returns the most right column number of the qr code that is found in a row
    def get_most_right_column_per_row(self,most_right,qrcode):
        if most_right==None:
            return qrcode.column
        else:
            if most_right<qrcode.column:
                return qrcode.column
            else:
                return most_right
 
    # returns true if the distance between the codes on a line is larger than the given fraction of the nr of boxes of that line
    def check_hori_dist(self,left,right,fraction):
        if (not left==None) and (not right==None):
            # print(f'found={(right-left-1)/self.nrOfBoxesPerLine}')
            # print(f'margin={fraction}')
            if (right-left-1)/self.nrOfBoxesPerLine>=fraction:
                return True
        return False
 
    def avg_qr_width_per_row(self,qrcodes):
        widths = [False] * len(qrcodes)
        for i in range(0,len(qrcodes)):
            widths[i]=qrcodes[i].width
        return int(sum(widths) / len(widths))
 
    # returns list of indices of qr codes that are not detected in specific row
    def identify_unknown_qrcodes_in_row(self,row_nr,detected_qrcodes):
        print(f'row_nr={row_nr} and self.nrOfBoxesPerLine={self.nrOfBoxesPerLine}')
        start_index = (row_nr-1) * int(self.nrOfBoxesPerLine) + 1
        end_index = (row_nr-1)*(int(self.nrOfBoxesPerLine))+int(self.nrOfBoxesPerLine)
        missing_in_row = list(range(start_index,end_index+1))
        if len(detected_qrcodes)==0:
            return missing_in_row
        else:
            
            # compute found qr code symbol indices based on detected
            found_qrcode_index = list(map(lambda x: x.index,detected_qrcodes))
            print(f'found_qrcode_index={found_qrcode_index}')
            # compute found qr code page position index based on detected (as though on page 1)
            found_qrcode_index_on_page = list(map(lambda x: x.index-(x.page_nr-1)*self.nrOfBoxesPerLine*self.nrOfLinesPerPage,detected_qrcodes))
            
            # compute amount subtracted to normalize as though it was page 1
            subtracted = found_qrcode_index[0]-found_qrcode_index_on_page[0]
            print(f'found_qrcode_index_on_page={found_qrcode_index_on_page}')
            
            # compute which indices would be missing if as though it was on page 1
            missing_qrcode_index_on_page = self.remove_list_elements(missing_in_row,found_qrcode_index_on_page)
            print(f'missing_qrcode_index_on_page={missing_qrcode_index_on_page}')
            
            # compute absolute missing indices of symbols of qr codes that are not detected
            missing_qrcode_index = list(map(lambda x: x+subtracted,missing_qrcode_index_on_page))
            print(f'missing_qrcode_index={missing_qrcode_index}\n\n')
            return missing_qrcode_index
        
    def remove_list_elements(self,original,sublist):
        for x in sublist:
            if x in original:
                original.remove(x)
        return original
        
    # test unit tests
    def addThree(self,x):
        return x+3
        
        
        
        
        
        
        
        
        
        
        
    # extracts the remaining symbols that aren't found by combining the assumed geometry data (layout) with the measured geometry data(measurements)
    def geometric_inference(self,img_name,qrcodes):    
        for row_nr in range(1,self.nrOfLinesPerPage):
            missing_qrcodes_indices_in_row = self.identify_unknown_qrcodes_in_row(row_nr,qrcodes)
            if len(missing_qrcodes_indices_in_row)>0:
                page_nr_of_img = qrcodes[0].page_nr
                # TODO: Check if all qr codes have same page_nr, if not, throw exception
                if self.has_quarter_spacing(page_nr_of_img,qrcodes):# if page has quarter spacing
                    if self.has_quarter_spacing(page_nr_of_img,qrcodes,row_nr): # if row has quarter spacing
                        geometry_data = self.get_geometry_data(row_nr,page_nr_of_img,qrcodes)
                        self.extract_missing_symbols_in_line(img_name,row_nr,geometry_data,missing_qrcodes_indices_in_row,qrcodes)
                    else:
                        print('NEED THE GEOMETRIC UPDATE FROM THE OTHER ROW AS A REFERENCE TO DO GEOMETRIC INFERENCE. That has not yet been implemented though.')
                       #self.use_closest_row_with_distance_frac_for_extraction()
                else:
                    return print(f'NEED MORE IMAGES CANT DO GEOMETRIC INFERENCE.')
    
    # returns the average spacing between the qr codes in a specific row
    def get_avg_spacing_between_qrcodes(self,row_nr,qrcodes_in_row,page_nr_of_img,avg_width_in_row):
        most_left_col_per_row,most_right_col_per_row = self.get_most_left_and_right(row_nr,page_nr_of_img,qrcodes_in_row)
        nr_of_spaces_inbetween = most_right_col_per_row-most_left_col_per_row
        nr_of_qrcodes_inbetween = nr_of_spaces_inbetween-1 
        left_qrcode = self.get_qrcode_in_specific_column(qrcodes_in_row,most_left_col_per_row)
        right_qrcode = self.get_qrcode_in_specific_column(qrcodes_in_row,most_right_col_per_row)
        right_pos_of_left_qrcode = left_qrcode.right
        left_pos_of_right_qrcode = right_qrcode.left
        
        avg_spacing_between_qrcodes = ((left_pos_of_right_qrcode-right_pos_of_left_qrcode)-nr_of_qrcodes_inbetween*avg_width_in_row)/nr_of_spaces_inbetween
        return avg_spacing_between_qrcodes 
        
    # assumes qr code exists in list that is in that column and returns that qrcode
    def get_qrcode_in_specific_column(self,qrcodes,column_nr):
        for i in range(0,len(qrcodes)):
            if qrcodes[i].column == column_nr:
                return qrcodes[i]
        # TODO: throw exception should find qr code
        
    # returns the qr codes in a specific row, from a set of qr codes in a single page
    def get_found_qrcodes_in_row(self,row_nr,qrcodes):
        found_qrcodes_index = self.get_qrcode_indices_in_row(row_nr,qrcodes)
        print(f'found_qrcodes_index={found_qrcodes_index},with qrcodes={list(map(lambda x: x.index,qrcodes))}\n\n\n')
        found_qrcodes = []
        for i in range(0,len(found_qrcodes_index)):
            for j in range(0,len(qrcodes)):
                if qrcodes[j].index == found_qrcodes_index[i]:
                    found_qrcodes.append(qrcodes[j])
        return found_qrcodes

    # returns the absolute indicies of qr codes of a specific row selecting from all found qrcodes
    def get_qrcode_indices_in_row(self,row_nr,detected_qrcodes):
        start_index = (row_nr-1)*(self.nrOfBoxesPerLine)+1
        end_index = (row_nr-1)*(self.nrOfBoxesPerLine)+self.nrOfBoxesPerLine
        all_indices_of_row = list(range(start_index,end_index+1))
        print(f'all_indices_of_row={all_indices_of_row}')
        # compute found qr code symbol indices based on detected
        found_qrcode_index = list(map(lambda x: x.index,detected_qrcodes))
        
        # compute found qr code page position index based on detected (as though on page 1)
        found_qrcode_index_on_page = list(map(lambda x: x.index-(x.page_nr-1)*self.nrOfBoxesPerLine*self.nrOfLinesPerPage,detected_qrcodes))
        print(f'found_qrcode_index_on_page={found_qrcode_index_on_page}')
        # compute amount subtracted to normalize as though it was page 1
        subtracted = found_qrcode_index[0]-found_qrcode_index_on_page[0]
        
        # compute the qr codes that are found in row as though they were on page 1
        found_qr_codes_on_page = []
        for i in range(0,len(found_qrcode_index_on_page)):
            if found_qrcode_index_on_page[i] in all_indices_of_row:
                found_qr_codes_on_page.append(found_qrcode_index_on_page[i])        
       
        # compute absolute found indices of symbols of qr codes in a row that are found
        found_qr_codes_on_row = list(map(lambda x: x+subtracted,found_qr_codes_on_page))

        return found_qr_codes_on_row
            
            
    def update_geometry(self,original_row,closest_row):
        print(f'SHOULD DO UPDATING OF GEOMETRY WRT ACTUAL ROW!')
        
    # computes the geometry of qrcodes in a specific row on a specific page and returns it as list
    def get_geometry_data(self,row_nr,page_nr_of_img,qrcodes):
        qrcodes_in_row = self.get_found_qrcodes_in_row(row_nr,qrcodes) #0 
        print(f'row_nr={row_nr},page_nr={page_nr_of_img},qrcodes_in_row={qrcodes_in_row}')
        avg_width_in_row = round(mean(list(map(lambda x: x.width,qrcodes_in_row))),0) #1
        avg_height_in_row = round(mean(list(map(lambda x: x.height,qrcodes_in_row))),0) #2
        avg_top_in_row = round(mean(list(map(lambda x: x.top,qrcodes_in_row))),0) #3
        avg_bottom_in_row = round(mean(list(map(lambda x: x.bottom,qrcodes_in_row))),0) #4
        avg_qrcode_spacing_in_row = self.get_avg_spacing_between_qrcodes(row_nr,qrcodes_in_row,page_nr_of_img,avg_width_in_row) #5
        
        return [qrcodes_in_row,avg_width_in_row,avg_height_in_row,avg_top_in_row,avg_bottom_in_row,avg_qrcode_spacing_in_row]
    
    
    def extract_missing_symbols_in_line(self,img_name,row_nr,geometry_data,missing_qrcodes_indices_in_row,detected_qrcodes):
        found_qrcodes_in_row = geometry_data[0]
        missing_qrcodes = []
        for i in range(0,len(missing_qrcodes_indices_in_row)):
            missing_qrcode_filler = QrCode(missing_qrcodes_indices_in_row[i],1,2,3,4,self.nrOfSymbols,self.nrOfBoxesPerLine,self.nrOfLinesPerPage)
            nearest_qrcode, nr_of_qrcodes_inbetween = self.find_nearest_found_qrcode(missing_qrcode_filler,found_qrcodes_in_row)
            missing_qrcode_filler.top = geometry_data[3]
            missing_qrcode_filler.bottom = geometry_data[4]
            missing_qrcode_filler.width = geometry_data[1]
            missing_qrcode_filler.height = geometry_data[2]
            missing_qrcode_filler.left = self.get_left_pos_of_missing_qrcode(missing_qrcode_filler.column,nearest_qrcode,geometry_data)
            missing_qrcode = QrCode(missing_qrcodes_indices_in_row[i],missing_qrcode_filler.top,missing_qrcode_filler.left,missing_qrcode_filler.width,missing_qrcode_filler.height,self.nrOfSymbols,self.nrOfBoxesPerLine,self.nrOfLinesPerPage)
            missing_qrcodes.append(missing_qrcode)
            
            # reload full image
            full_img=Image.open(img_name)
            
            # export symbol to font directory
            self.get_symbol(full_img,missing_qrcode.index,missing_qrcode.top,missing_qrcode.left,missing_qrcode.bottom,missing_qrcode.right,missing_qrcode.width,missing_qrcode.height)
        
    
    # returns the left position of the missing qr code
    def get_left_pos_of_missing_qrcode(self,column_missing_qrcode,nearest_qrcode,geometry_data):
        avg_qrcode_width = geometry_data[1]
        avg_qrcode_spacing = geometry_data[5]
        nr_of_qrcodes_inbetween = abs(nearest_qrcode.column-column_missing_qrcode)-1
        missing_is_left_of_nearest = (nearest_qrcode.column>column_missing_qrcode)
        if missing_is_left_of_nearest:
            print(f'nearest_qrcode.column={nearest_qrcode.column} and LEFT of that = {nearest_qrcode.left} and right of that = {nearest_qrcode.right}')
            print(f'nr_of_qrcodes_inbetween={nr_of_qrcodes_inbetween}')
            print(f'avg_qrcode_width={avg_qrcode_width}')
            print(f'avg_qrcode_spacing={avg_qrcode_spacing}')
            left = nearest_qrcode.left-((nr_of_qrcodes_inbetween+1)*avg_qrcode_width+(nr_of_qrcodes_inbetween+1+1)*avg_qrcode_spacing) # +1 cause the symbol itself and spacing  must also be crossed to arrive at the symbol left
        else:
            
            left = nearest_qrcode.right+(nr_of_qrcodes_inbetween*avg_qrcode_width+(nr_of_qrcodes_inbetween+1)*avg_qrcode_spacing)
        print(f'LEFT={left}\n\n\n')
        
        return left
        
        
        
    
    # returns the closest qr code
    def find_nearest_found_qrcode(self,missing_qrcode_in_row,found_qrcodes_in_row):
        print(f'missing_qrcode_in_row={missing_qrcode_in_row.index}')
        closest_right_qrcode = self.find_closest_right(missing_qrcode_in_row,found_qrcodes_in_row)
        closest_left_qrcode = self.find_closest_left(missing_qrcode_in_row,found_qrcodes_in_row)
        #print(f'closest_right_qrcode={closest_right_qrcode.index}')
        #print(f'closest_left_qrcode={closest_left_qrcode.index}')
        if not closest_right_qrcode is None:
            distance_right = abs(missing_qrcode_in_row.column-closest_right_qrcode.column)
        else:
            distance_left = abs(missing_qrcode_in_row.column-closest_left_qrcode.column)
            return closest_left_qrcode,distance_left
        if not closest_left_qrcode is None:
            distance_left = abs(missing_qrcode_in_row.column-closest_left_qrcode.column)
            if distance_left<distance_right:
                return closest_left_qrcode,distance_left
            else:
                return closest_right_qrcode,distance_right
        print("ERROR COULD NOT FIND CLOSEST")
        # TODO: throw exception if no closest is found in row
        
    def find_closest_right(self,missing_qrcode_in_row,found_qrcodes_in_row):
        closest_right_column = 10000000
        closest_right_qr = None
        for i in range(0,len(found_qrcodes_in_row)):
            print(f'found_qrcodes_in_row[i].column={found_qrcodes_in_row[i].column} and missing_qrcode_in_row.column={missing_qrcode_in_row.column}')
            if found_qrcodes_in_row[i].column>missing_qrcode_in_row.column:
                if closest_right_column>found_qrcodes_in_row[i].column:
                    closest_right_column=found_qrcodes_in_row[i].column
                    closest_right_qr = found_qrcodes_in_row[i]
        if closest_right_column==10000000:
            return None
        else:
            return closest_right_qr
        
    def find_closest_left(self,missing_qrcode_in_row,found_qrcodes_in_row):
        closest_left_column = 0
        closest_left_qr = None
        for i in range(0,len(found_qrcodes_in_row)):
            if found_qrcodes_in_row[i].column<missing_qrcode_in_row.column:
                if closest_left_column<found_qrcodes_in_row[i].column:
                    closest_left_column=found_qrcodes_in_row[i].column
                    closest_left_qr = found_qrcodes_in_row[i]
        if closest_left_column==0:
            return None
        else:
            return closest_left_qr    
        
    def get_symbol_coords_of_qr_code(self,row_nr, column_index,geometry_data):
        return top,left,right,bottom
        
    # def use_closest_row_with_distance_frac_for_extraction():
        # closest_row = find closest line that has 25% fraction
        # geometry_data = self.get_geometry_data(self,closest_row)
        # updated_geometry_data = update_geometry()
        # self.extract_missing_symbols_in_line(img_name,row_nr,updated_geometry_data,missing_qrcodes)
        
        
        
        
        
        
        
        
        
        
        
        
        
    
# executes this main code
if __name__ == '__main__':
    main = ReadTemplate()
    main.perform_runs()