# conda install potrace
import os, re, os.path
from os import listdir
from os.path import isfile, join
from potrace import Bitmap

class CreateFont:
    
    # intializes object
    def __init__(self):
    
        ## Declare parameters
        # scanned filled image that is read in for processing
        self.image = None 
        
        ## Specifications that are outputed by the template creating module
        self.nrOfSymbols = None
        self.boxWidth = None
        self.boxHeight = None
        self.nrOfBoxesPerLine = None
        self.nrOfBoxesPerLineMinOne = None
        self.nrOfLinesInTemplate = None
        self.nrOfLinesInTemplateMinOne = None
        self.maxNrOfLinesPerPage = None
        
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
        # Specify font input directory
        self.font_out = 'font_out'
        self.font_in = 'font_in'
        
        self.create_font()
        
    def create_font(self):
        

        # Initialize data, for example convert a PIL image to a numpy array
        # [...]
        

        bitmap = Bitmap(data)
        path = bitmap.trace()
        
    
# executes this main code
if __name__ == '__main__':
    main = CreateFont()
    main.perform_runs()