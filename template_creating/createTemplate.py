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
        self.max_sheet_height = 297 # mm
        self.box_width = 20 #mm
        self.box_spacing = 3 #mm
        
        # read the symbols from file
        self.symbol_lines = self.read_txt(source_dir,self.symbols_file_name,ext)
        print(f'lines={self.symbol_lines}')
        self.nrOfSymbols=self.symbol_lines.count('\n') +1 
        print(f'self.nrOfSymbols={self.nrOfSymbols}')
    
        # generate page distribution params
        [self.nr_of_boxes_per_line,self.nr_of_lines] = self.optimise_box_distribution()
        print(f'nr_of_boxes_per_line={self.nr_of_boxes_per_line}')
        print(f'nr_of_lines={self.nr_of_lines}')
    
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
        # TODO: Round down
        nr_of_boxes_per_line = round(self.max_sheet_width/ (self.box_width+self.box_spacing),0)
        # TODO: Round up (done by default)
        nr_of_lines = round(self.nrOfSymbols/nr_of_boxes_per_line)
        return [nr_of_boxes_per_line,nr_of_lines]
        
        
    # Put the specificatins in a string array
    def generate_page_settings(self):
        lines = []
        lines.append(f'nrOfSymbols = {self.nrOfSymbols}')
        lines.append(f'boxWidth = {self.box_width}')
        lines.append(f'boxHeight = {self.box_width}')
        lines.append(f'nrOfBoxesPerLine = {self.nr_of_boxes_per_line}')
        lines.append(f'nrOfBoxesPerLineMinOne = {self.nr_of_boxes_per_line-1}')
        lines.append(f'nrOfLinesPerPage = {self.nr_of_lines}')
        lines.append(f'nrOfLinesPerPageMinOne = {self.nr_of_lines-1}')
        return lines
        # nrOfSymbols = 30
        # boxWidth = 5.2
        # boxHeight = 7.2
        # nrOfBoxesPerLine = 10
        # nrOfBoxesPerLineMinOne = 9
        # nrOfLinesPerPage = 6
        # nrOfLinesPerPageMinOne = 5
    
    
    def set_box_dimensions(self):
        self.boxWidth = 5.2
        self.boxHeight = 7.2


    def set_box_distribution(self):
        self.nrOfBoxesPerLine = 10
        self.nrOfBoxesPerLineMinOne = 9
        self.nrOfLinesPerPage = 6
        self.nrOfLinesPerPageMinOne = 5

if __name__ == '__main__':
    main = CreateTemplate()