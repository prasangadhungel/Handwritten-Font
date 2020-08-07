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
import cv2
import numpy as np
import re
import sys
from io import StringIO ## for Python 3
from pdf2image import convert_from_path


class ReadTemplate:
    
    # intializes object
    def __init__(self):
        # Declare parameters
        self.image = None
        self.original = None
        self.gray = None
        self.blur  = None
        self.thresh = None
        self.cnts = None
        self.nrOfSymbols = None
        self.boxWidth = None
        self.boxHeight = None
        self.nrOfBoxesPerLine = None
        self.nrOfBoxesPerLineMinOne = None
        self.nrOfLinesPerPage = None
        self.nrOfLinesPerPageMinOne = None
        
        # hardcode relative qr specification file location
        self.spec_filename = 'symbol_spec.txt'
        self.spec_loc = f'../template_creating/{self.spec_filename}'
        self.read_image_specs()
        self.output_dir = "out"
        
        # execute file reading steps
        self.create_tempqr()
        self.clear_output_folder()
        self.convert_pdf_to_img()
        self.image, self.original,self.thresh = self.load_image()
        self.close,self.kernel = self.morph_image()
        self.cnts = self.find_contours()
        self.ROIs,self.ROIs_symbol = self.loop_through_contours()
        self.read_qr_imgs()
        # self.()
        
    def clear_output_folder(self):
        import os, re, os.path
        
        for root, dirs, files in os.walk(self.output_dir):
            for file in files:
                os.remove(os.path.join(root, file))
        
    def convert_pdf_to_img(self):
        # first convert the pdf into separate images        
        pages = convert_from_path('filled.pdf', 500)
        count=0
        for page in pages:
            page.save(f'out{count}.jpg', 'JPEG')
            count = count+ 1

    # TODO: Scan the directory for a single pdf.
    # TODO: Loop per page
    # TODO: change the dimensions of the searched qr code based on the outputted specs in create template from symbol_spec.txt
    # TODO: Loop through all qr codes
    # Load image, grayscale, Gaussian blur, Otsu's threshold
    def load_image(self):
        image = cv2.imread('out3.jpg')
        original = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9,9), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        return image,original,thresh
        
    # Morph close
    def morph_image(self):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        close = cv2.morphologyEx(self.thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        return close,kernel

    # Find contours and filter for QR code
    def find_contours(self):
        cnts = cv2.findContours(self.close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        return cnts

    # loop through all contours and find their respective positions
    def loop_through_contours(self):
        ROIs = []
        ROIs_symbol = []
        for c in self.cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True) # number of sides of polygon?
            x,y,w,h = cv2.boundingRect(approx)
            area = cv2.contourArea(c)
            ar = w / float(h)
            ROIs,ROIs_symbol = self.select_qr_contours(approx,ar,area,h,w,x,y,ROIs,ROIs_symbol)
        return ROIs,ROIs_symbol
        
    # TODO: change len(approx) to a standardized value from qr code sizes
    # TODO: change area size to output of box size in symbol_spec.txt
    # TODO: change ar size to output of box size in symbol_spec.txt    
    # Selects the contours based on their geometrics vs symbol spec file outputed by template creation
    def select_qr_contours(self,approx,ar,area,h,w,x,y,ROIs,ROIs_symbol):
        # if len(approx) == 4 and area > 1000 and (ar > .85 and ar < 1.3):
        #if len(approx) == 3 and area > 1000:
        if area > 10000 and (ar > .3 and ar < 0.38):
            cv2.rectangle(self.image, (x, y), (x + w, y + h), (36,255,12), 3)
            ROI = self.original[y:y+h, x:x+w]
            ROI_symbol = self.original[y-h:y, x:x+w]
            
            # convert images to grayscale
            ROI_gray = self.convert_img_to_grayscale(ROI)
            ROI_symbol_gray = self.convert_img_to_grayscale(ROI_symbol)
            
            # export images
            cv2.imwrite(f'{self.output_dir}/ROI_{len(ROIs)}_ar_{ar}.png', ROI_gray)
            cv2.imwrite(f'{self.output_dir}/ROI_symbol_{len(ROIs)}_ar_{ar}.png', ROI_symbol_gray)
            ROIs.append(ROI_gray)
            ROIs_symbol.append(ROI_symbol_gray)
        return ROIs,ROIs_symbol
            
    # Shows relevant countours which are believed to contain qr code.
    def show_qr_images(ROI,ROI_symbol):
        cv2.imshow('thresh', thresh)
        cv2.imshow('close', close)
        cv2.imshow('image', self.image)
        cv2.imshow('ROI', ROI)
        cv2.imshow('ROI_symbol', ROI_symbol)
        cv2.waitKey()
    
    # read image specs
    def read_image_specs(self):
        print(self.spec_loc)
        #with open(self.spec_loc) as f:
         #   content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        #content = [x.strip() for x in content] 
        # Using readlines() 
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
    
    def convert_img_to_grayscale(self,color_img):
        #print(f'ROI[0]={color_img.shape}')
        gray_img = np.zeros((color_img.shape[0],color_img.shape[1]),dtype=int)
        gray_img = color_img.mean(axis=2)
        
        return gray_img
    
    def read_qr_imgs(self):
        for i in range(0,len(self.ROIs)):
            # get qr code dimensions
            img = self.ROIs[i]
            
            # override from horn qr code
            # load and show an image with Pillow
            from PIL import Image
            img = Image.open('elbow.png')
            img = np.array(img)
            print(f'img={img}')
            print(f'img={img.shape}')
            
            qr_arr = self.binary_string(img,True)
            self.ndarray_to_txt(qr_arr)
            
            height = img.shape[0]
            width = img.shape[1]
            
            qr = img[height-width:height].astype(int)
            self.binary_string(qr)
            
            #cv2.imwrite(f'{self.output_dir}/qr_{i}.png', qr)
            #self.ndarray_to_txt(qr)
            
            # Source: https://www.thepythoncode.com/article/generate-read-qr-code-python
            
            
            # Source: https://stackoverflow.com/questions/27233351/how-to-decode-a-qr-code-image-in-preferably-pure-python
            # Decode qr cod
            # requires zbar
            # import qrtools
            # from qrtools import qrtools 
            # qrtool = qrtools.QR()
            # qrtool.decode(qr)
            # print(qrtool.data)
            
            
            # Source: https://github.com/svartalf/python-quirc/blob/master/examples/decode_opencv.py
            # Source: https://python-quirc.readthedocs.io/en/latest/usage/high-level.html
            # pip install quirc
            # does not register import cv in python 3
            
            
    def ndarray_to_txt(self,arr):
        str_arr = ""
        text_file = open("foo.txt", "w")
        np.set_printoptions(threshold=sys.maxsize)
        text_file.write(str(arr))
        text_file.close()
    
    def binary_string(self,arr, isbinary=False):
        output = StringIO()
        for row in range(0,arr.shape[0]):
            for column in range(0,arr.shape[1]):
                if (isbinary and arr[row][column]):
                    output.write('X')
                elif arr[row][column]>245:
                    output.write('X')
                else:
                    output.write('_')
            output.write('\n')
            content = output.getvalue()
        output.close()    
        
        return content
    
    def create_tempqr(self):
        # importing the module
        import qrcode 

        # information
        First_qrcode=qrcode.make(r'Includehelp is one of the best sites to learn any programming language from the basics. To visit the site click on the link:  https://www.includehelp.com .') 

        # to see the QR code on the computer screen
        print(f'First_qrcode={First_qrcode}')
        
if __name__ == '__main__':
    main = ReadTemplate()