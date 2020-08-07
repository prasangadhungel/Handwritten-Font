# 0. If it says: " import cv2" ... "ImportError: DLL load failed: The specified module could not be found." then:
# 0.1 Instal Visual Studio Code from https://code.visualstudio.com/Download
# 0.2 Click the Extensions view icon on the Sidebar (orCtrl+Shift+X keyboard combination).
# 0.3 Search of C++. And install (the most popular/top extension named:"C/C++"
# SOLUTION:
# 0.6 https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads download and install visual studio 2015, 2017 and 2019
# DONT? 0.2 Install opencv for windows from  https://opencv.org/releases/
# 0.1 open anaconda and type:
# DONT pip install opencv-python
# DONT pip install opencv-contrib-python
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


# import libraries
from pyzbar.pyzbar import decode
from PIL import Image

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
        self.img_path = 'testfiles/out3.jpg'
        
        # execute file reading steps
        self.clear_output_folder()
        self.convert_pdf_to_img()
        self.image, self.original,self.thresh = self.load_image(self.img_path)
        self.read_qr_imgs()
        
        
    def clear_output_folder(self):
        import os, re, os.path
        
        for root, dirs, files in os.walk(self.output_dir):
            for file in files:
                os.remove(os.path.join(root, file))
        
    def convert_pdf_to_img(self):
        # first convert the pdf into separate images        
        pages = convert_from_path('testfiles/filled.pdf', 500)
        count=0
        for page in pages:
            page.save(f'testfiles/out{count}.jpg', 'JPEG')
            count = count+ 1

    # TODO: Scan the directory for a single pdf.
    # TODO: Loop per page
    # TODO: change the dimensions of the searched qr code based on the outputted specs in create template from symbol_spec.txt
    # TODO: Loop through all qr codes
    # Load image, grayscale, Gaussian blur, Otsu's threshold
    def load_image(self,img_path):
        image = cv2.imread(img_path)
        original = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9,9), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        return image,original,thresh
        
    
    # read image specs
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
    
    def convert_img_to_grayscale(self,color_img):
        #print(f'ROI[0]={color_img.shape}')
        gray_img = np.zeros((color_img.shape[0],color_img.shape[1]),dtype=int)
        gray_img = color_img.mean(axis=2)
        return gray_img
    
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
    
    # read the qr code from an image
    def read_qr_imgs(self):
        #qrcode = decode(self.image)
        #print(f'Nr of qr codes = {len(qrcode)}')
        img = Image.open(self.img_path)
        #qrcode = decode(img)
        qrcode = self.preprocess_qrcode(cv2.imread(self.img_path))
        print(f'nr of qr codes = {len(qrcode)}')
        
        for i in range(0,len(qrcode)):
        #for i in range(0,1):
        
            # Get the rect/contour coordinates:
            left = qrcode[i].rect[0]
            top = qrcode[i].rect[1]
            width = qrcode[i].rect[2]
            height = qrcode[i].rect[3]
            print(f'left={left},top={top},width={width},height={height}\n\n and image height={img.height}\n\n and image width={img.width}')
            
            # get the rectangular contour corner coordinates
            # top_left = [top,left]
            # print(f'top_left={top_left}')
            # top_right = [top,left+width]
            # print(f'top_right={top_right}')
            # bottom_left = [top-height,left]
            # print(f'bottom_left={bottom_left}')
            # bottom_right = [top-height,left+width]
            # print(f'bottom_right={bottom_right}')
            
            self.show_qr(img,left,top,width,height,i)
            
    def show_qr(self,img,left,top,width,height,i):
        #im_crop = self.image.crop((tl[1], tl[0], tr[1], bl[0]))
        #self.image.show()
        #box=(left, upper, right, lower).
        print(f'left={left}\n upper={top}\n right={left+width}\n lower={top+height}')
        im_crop = img.crop((left,top,left+width,top+height))
        im_crop.save(f'out/{i}.png')
        #im_crop.show()
        #im_crop.save('data/dst/lena_pillow_crop.jpg', quality=95)
        
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
        
if __name__ == '__main__':
    main = ReadTemplate()