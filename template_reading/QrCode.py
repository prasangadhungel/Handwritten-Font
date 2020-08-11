import math
class QrCode:
    def __init__(self, index, top,left,width,height,nrOfSymbols,nrOfBoxesPerLine,nrOfLinesInTemplate,maxNrOfLinesPerPage):

        # initialize qr code properties
        self.top = top
        self.left = left
        self.width = width
        self.height = height
        self.index = int(index)
        self.nrOfSymbols = nrOfSymbols
        self.nrOfBoxesPerLine = nrOfBoxesPerLine
        self.nrOfLinesInTemplate = nrOfLinesInTemplate
        self.maxNrOfLinesPerPage = maxNrOfLinesPerPage
        
        # infer qr code properties
        self.nrOfLinesInPage = self.get_nr_of_lines_in_page()
        self.bottom = top + self.height
        self.right = left + self.width
        self.row = self.get_qr_row()
        self.column = self.get_qr_column()
        self.page_nr = self.get_page_nr()
        self.nrOfLinesInCurrentPage = self.get_nr_of_lines_in_page()
        self.hashcode = self.__hash__()
        
    def __hash__(self):
        return hash(('top', self.top,
                     'left', self.left,
                     'width', self.width,
                     'height', self.height,
                     'index', self.index
                     ))
                 
    # Returns the number of lines in the page on which the qrcode is positioned
    def get_nr_of_lines_in_page(self):
        nr_of_lines_in_page = self.maxNrOfLinesPerPage
        
        if self.is_last_page():
            print(f'self.get_nr_of_lines_in_last_page()={self.get_nr_of_lines_in_last_page()}')
            return self.get_nr_of_lines_in_last_page()
        print(f'returning self.maxNrOfLinesPerPage={self.maxNrOfLinesPerPage}')
        return self.maxNrOfLinesPerPage

    # Returns true if qrcode is on last page.
    def is_last_page(self):
        last_page_nr = int(math.ceil(self.nrOfLinesInTemplate/self.maxNrOfLinesPerPage))
        if last_page_nr ==self.get_page_nr():
            return True
        else:
            return False
     
    # computes the number of pages in the last page on which this qrcode is positioned.
    def get_nr_of_lines_in_last_page(self):
        if self.nrOfLinesInTemplate%self.maxNrOfLinesPerPage == 0:
            #print(f'self.nrOfLinesInTemplate={self.nrOfLinesInTemplate} and self.maxNrOfLinesPerPage={self.maxNrOfLinesPerPage}')
            # modulo without remainder implies the maxNrOfLinesPerPage lines are on the last page
            return self.maxNrOfLinesPerPage
        else:
            # modulo returns the remaining lines for the last page
            return self.nrOfLinesInTemplate%self.maxNrOfLinesPerPage 
                 
    # returns page nr starting at 1
    def get_page_nr(self):
        #print(f'self.index={self.index}, self.nrOfBoxesPerLine={self.nrOfBoxesPerLine}, self.maxNrOfLinesPerPage={self.maxNrOfLinesPerPage}')
        rounded_down = math.floor(self.index/(self.nrOfBoxesPerLine*self.maxNrOfLinesPerPage))
        remainder = self.index%(self.nrOfBoxesPerLine*self.maxNrOfLinesPerPage)
        
        # e.g. symbol index 8/(2*2)=2.0, and symbol index 8 is on page 2
        if self.index%(self.nrOfBoxesPerLine*self.maxNrOfLinesPerPage) == 0:
            return int(rounded_down)
        else: 
            # e.g. symbol index 9/(2*2)=2.25, =floored to 2.0 and symbol index 9 is on page 3
            return int(rounded_down+1)
        

    # returns row number of qr code starting at row 1
    def get_qr_row(self):
        # subtract previous pages if not the first page
        local_index = self.index-(self.get_page_nr()-1)*self.nrOfBoxesPerLine*self.maxNrOfLinesPerPage    
        row_nr = math.ceil(local_index/self.nrOfBoxesPerLine)
        
        return row_nr

    # returns the column of a box starting at 1 for most left column
    def get_qr_column(self):
        # subtract previous pages if not the first page
        local_index = self.index-(self.get_page_nr()-1)*self.nrOfBoxesPerLine*self.maxNrOfLinesPerPage    
        return local_index-(self.get_qr_row()-1)*self.nrOfBoxesPerLine