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
        
        # execute file reading steps
        self.create_tempqr()
        self.clear_output_folder()
        self.convert_pdf_to_img()
        self.image, self.original,self.thresh = self.load_image()
        self.close,self.kernel = self.morph_image()
        self.cnts = self.find_contours()
        self.ROIs,self.ROIs_symbol,self.ROIs_pos,self.ROIs_symbol_pos = self.loop_through_contours()
        self.read_qr_imgs()
        # self.()
        
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
    def load_image(self):
        image = cv2.imread('testfiles/out3.jpg')
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
        ROIs_pos = []
        ROIs_symbol_pos = []
        for c in self.cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True) # number of sides of polygon?
            x,y,w,h = cv2.boundingRect(approx)
            area = cv2.contourArea(c)
            ar = w / float(h)
            ROIs,ROIs_symbol,ROIs_pos,ROIs_symbol_pos = self.select_qr_contours(approx,ar,area,h,w,x,y,ROIs,ROIs_symbol,ROIs_pos,ROIs_symbol_pos)
        return ROIs,ROIs_symbol,ROIs_pos,ROIs_symbol_pos
        
    # TODO: change len(approx) to a standardized value from qr code sizes
    # TODO: change area size to output of box size in symbol_spec.txt
    # TODO: change ar size to output of box size in symbol_spec.txt    
    # Selects the contours based on their geometrics vs symbol spec file outputed by template creation
    def select_qr_contours(self,approx,ar,area,h,w,x,y,ROIs,ROIs_symbol,ROIs_pos,ROIs_symbol_pos):
        # if len(approx) == 4 and area > 1000 and (ar > .85 and ar < 1.3):
        #if len(approx) == 3 and area > 1000:
        if area > 10000 and (ar > .3 and ar < 0.38):
            cv2.rectangle(self.image, (x, y), (x + w, y + h), (36,255,12), 3)
            ROI = self.original[y:y+h, x:x+w]
            ROI_symbol = self.original[y-h:y, x:x+w]
            
            # left = qrcode[0].rect[0]
            # top = qrcode[0].rect[1]
            # width = qrcode[0].rect[2]
            # height = qrcode[0].rect[3]
            
            #ROI_pos =[left,top,width,height] measured from top to bottom and left edge to right
            ROI_pos =[x,y+h,w,h]
            ROI_symbol_pos = [x,y,w,h]
            
            # convert images to grayscale
            ROI_gray = self.convert_img_to_grayscale(ROI)
            ROI_symbol_gray = self.convert_img_to_grayscale(ROI_symbol)
            
            # export images
            #cv2.imwrite(f'{self.output_dir}/ROI_{len(ROIs)}_ar_{ar}.png', ROI_gray)
            #cv2.imwrite(f'{self.output_dir}/ROI_symbol_{len(ROIs)}_ar_{ar}.png', ROI_symbol_gray)
            cv2.imwrite(f'{self.output_dir}/ROI_{len(ROIs)}.png', ROI_gray)
            cv2.imwrite(f'{self.output_dir}/ROI_symbol_{len(ROIs)}.png', ROI_symbol_gray)
            ROIs.append(ROI_gray)
            ROIs_symbol.append(ROI_symbol_gray)
            ROIs_pos.append(ROI_pos)
            ROIs_symbol_pos.append(ROI_symbol_pos)
        return ROIs,ROIs_symbol,ROIs_pos,ROIs_symbol_pos
            
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
            contour = self.ROIs[i]
            img = cv2.imread(f'{self.output_dir}/ROI_{i}.png')
            qrcode = self.preprocess_qrcode(img)
            
            # Read qr from contour
            #qrcode = decode(contour)
            print(f'nr of qr codes = {len(qrcode)}')
            
            if len(qrcode)>0:
                # Get the qr coordinates relative to the contour:
                left_qr_cont = qrcode[0].rect[0]
                top_qr_cont = qrcode[0].rect[1]
                width_qr_cont = qrcode[0].rect[2]
                height_qr_cont = qrcode[0].rect[3]
                print(f'relative qr code positions = left={left_qr_cont},top={top_qr_cont},width={width_qr_cont},height={height_qr_cont}')
        
                
                left_ct = self.ROIs_pos[i][0]
                top_ct = self.ROIs_pos[i][1]
                width_ct = self.ROIs_pos[i][2]
                height_ct = self.ROIs_pos[i][3]
                bottom_ct = top_ct+height_ct
                right_ct = left_ct+width_ct        
                print(f'The original contour coordinates are= left={left_ct},top={top_ct},bottom={bottom_ct},right={right_ct}\n\n')
        
                
                # compute qr position
                top_qr = top_ct+top_qr_cont
                left_qr = left_ct+left_qr_cont
                bottom_qr =top_ct+top_qr_cont+height_qr_cont
                right_qr = left_ct+ left_qr_cont+width_qr_cont
   
                from matplotlib import pyplot as plt
                #full_img=Image.open(f'{self.output_dir}/ROI_{i}.png')
                #full_img = cv2.imread('testfiles/out3.jpg')
                full_img=Image.open('testfiles/out3.jpg')
                print(full_img.size)
                
                contour = full_img.crop((left_ct,top_ct,right_ct,bottom_ct))
                contour.save(f'out/contour{i}.png')
                #im_crop.show()
                
                print(f'left={left_qr}\n upper={top_qr}\n right={right_qr}\n lower={bottom_qr}')
                im_crop = full_img.crop((left_qr,top_qr,right_qr,bottom_qr))
                im_crop.save(f'out/crop{i}.png')
                #im_crop.show()
                
                # get the rectangular contour corner coordinates
                # top_left = [top,left]
                # print(f'top_left={top_left}')
                # top_right = [top,left+width]
                # print(f'top_right={top_right}')
                # bottom_left = [top-height,left]
                # print(f'bottom_left={bottom_left}')
                # bottom_right = [top-height,left+width]
                # print(f'bottom_right={bottom_right}')
                
                #self.show_qr(contour,left,top,width,height)
                
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
            
    def show_qr(self,img,left,top,width,height):
        #im_crop = self.image.crop((tl[1], tl[0], tr[1], bl[0]))
        #self.image.show()
        #box=(left, upper, right, lower).
        print(f'left={left}\n upper={top}\n right={left+width}\n lower={top+height}')
        im_crop = img.crop((left,top,left+width,top+height))
        im_crop.save('out/my.png')
        im_crop.show()
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
    
    def create_tempqr(self):
        # importing the module
        import qrcode 

        # information
        First_qrcode=qrcode.make(r'Includehelp is one of the best sites to learn any programming language from the basics. To visit the site click on the link:  https://www.includehelp.com .') 

        # to see the QR code on the computer screen
        print(f'First_qrcode={First_qrcode}')
        
if __name__ == '__main__':
    main = ReadTemplate()