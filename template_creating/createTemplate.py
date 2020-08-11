import math

class CreateTemplate:
    
    # intializes object
    def __init__(self):
        
        self.encoding = "utf-8" # encoding used to handle symbol encoding
        # TODO: export encoding to import it with latex.
        # TODO: automatically select encoding based on user chosen language.
        
        # Specify inout and output file locations and names
        source_dir = "./"
        ext = ".txt"
        self.symbols_file_name = "symbols"
        self.page_setting_filename = 'symbol_spec'
        
        # specify page settings
        self.max_sheet_width = 210 # mm
        self.max_horizontal_margin = 5 # mm (2.5 on each side)
        self.max_sheet_height = 297 # mm
        self.box_width = 19 #mm
        self.box_spacing = 3 #mm
        
        # specify qr code line margins
        self.sheet_top_margin = 6 #measured in mm
        print(f'self.sheet_top_margin={self.sheet_top_margin}')
        self.sheet_bottom_margin = self.max_sheet_height*(844-768)/844 # measured in pixels based on showgeometry package lines
        self.spacing_between_two_lines = self.max_sheet_height*(64/2338) #measured in pixels for box width 20 mm
        
        
        
        # read the symbols from file
        self.symbol_lines = self.read_txt(source_dir,self.symbols_file_name,ext)
        print(f'lines={self.symbol_lines}')
        self.nrOfSymbols=self.symbol_lines.count('\n') +1 
        print(f'self.nrOfSymbols={self.nrOfSymbols}')
    
        # generate page distribution params
        [self.nr_of_boxes_per_line,self.nr_of_lines_in_template] = self.optimise_box_distribution()
        print(f'nr_of_boxes_per_line={self.nr_of_boxes_per_line}')
        print(f'nr_of_lines_in_template={self.nr_of_lines_in_template}')
    
        # infer maximum number of lines per page
        self.max_nr_of_lines_per_page = self.compute_max_nr_of_lines_per_page()
        print(f'max_nr_of_lines_per_page={self.max_nr_of_lines_per_page}')
    
        # generate page settings
        self.page_setting_lines = self.generate_page_settings()
        print(f'self.page_setting_lines={self.page_setting_lines}')
        
        # write page specifications to file
        self.write_txt(source_dir,self.page_setting_filename,ext,self.page_setting_lines)
        
        
    # checks if file exists
    def files_exists(self,str):
        my_file = Path(str)
        if my_file.is_file():
            # file exist
            return True
        else:
            return False
    
    
    # read the symbols from the file lines into string array
    def read_nr_of_symbols(self, lines):
        # create an array to store the symbols in
        
        # loop through the lines and store the symbols
        for line in lines:
            symbol[line] = lines(line)
                
        self.nrOfSymbols = len(symbols)
        return symbols


    # read txt file and return as string (newlines become spaces)
    def read_txt(self,source_dir,filename,ext):
        str = source_dir+filename+ext
        try:
            with open(str, 'r',encoding="utf-8") as myfile:
                data=myfile.read().replace(' ','')
            return data
        except:
            pass

    
    # read txt file and return as string (newlines become spaces)
    def write_txt(self,source_dir,filename,ext,lines):
        str = source_dir+filename+ext
        try:
            with open(str, 'w',encoding="utf-8") as myfile:
                print(f'writing lines = {lines}')
                for i in range(0,len(lines)):
                    myfile.write(f'{lines[i]}\n')
            myfile.close()
        except:
            pass
    
    
    # returns the nr of boxes per line and nr of lines per page
    def optimise_box_distribution(self):
        
        nr_of_boxes_per_line = math.floor((self.max_sheet_width-self.max_horizontal_margin+self.box_spacing)/ (self.box_width+self.box_spacing))
        print(f'{self.max_sheet_width-self.max_horizontal_margin+self.box_spacing}div{self.box_width+self.box_spacing} = {(self.max_sheet_width-self.max_horizontal_margin+self.box_spacing)/ (self.box_width+self.box_spacing)} [boxes/row]')
        
        
        
        #
        nr_of_lines_in_template = round(self.nrOfSymbols/nr_of_boxes_per_line)
        return [nr_of_boxes_per_line,nr_of_lines_in_template]
    
    # computes the maximum nr of lines that fit on a page
    def compute_max_nr_of_lines_per_page(self):
        
        available_height = self.max_sheet_height-self.sheet_top_margin-self.sheet_bottom_margin
        remainder_height = available_height
        height_per_line = self.box_width*3
        
        print(f' at start remainder_height={remainder_height}\n\n')
        # see how many lines you can fit in the available height
        max_nr_of_lines = 0
        for i in range(1,self.nrOfSymbols):
            subtracted = i*height_per_line + (i-1)*self.spacing_between_two_lines
            print(f'i={i},subtracted={subtracted }')
            remainder_height = available_height-subtracted 
            
            print(f'i={i} and remainder_height={remainder_height}, height_per_line = {height_per_line}, self.spacing_between_two_lines = {self.spacing_between_two_lines}')
            if remainder_height>0:
                max_nr_of_lines = i
            else:
                break 
        return max_nr_of_lines
        
    # Put the specificatins in a string array
    def generate_page_settings(self):
        lines = []
        lines.append(f'nrOfSymbols = {self.nrOfSymbols}')
        lines.append(f'boxWidth = {self.box_width}')
        lines.append(f'boxHeight = {self.box_width}')
        lines.append(f'nrOfBoxesPerLine = {self.nr_of_boxes_per_line}')
        lines.append(f'nrOfBoxesPerLineMinOne = {self.nr_of_boxes_per_line-1}')
        lines.append(f'nrOfLinesInTemplate = {self.nr_of_lines_in_template}')
        lines.append(f'nrOfLinesInTemplateMinOne = {self.nr_of_lines_in_template-1}')
        lines.append(f'maxNrOfLinesPerPage = {self.max_nr_of_lines_per_page}')
        return lines
        # nrOfSymbols = 30
        # boxWidth = 5.2
        # boxHeight = 7.2
        # nrOfBoxesPerLine = 10
        # nrOfBoxesPerLineMinOne = 9
        # nrOfLinesInTemplate = 6
        # nrOfLinesInTemplateMinOne = 5
    

    def set_box_distribution(self):
        self.nrOfBoxesPerLine = 10
        self.nrOfBoxesPerLineMinOne = 9
        self.nrOfLinesInTemplate = 6
        self.nrOfLinesInTemplateMinOne = 5

if __name__ == '__main__':
    main = CreateTemplate()