import math
class QrCode:
    def __init__(self, index, top,left,width,height,nrOfSymbols,nrOfBoxesPerLine,nrOfLinesPerPage):

        # initialize qr code properties
        self.top = top
        self.left = left
        self.width = width
        self.height = height
        self.index = int(index)
        self.nrOfSymbols = nrOfSymbols
        self.nrOfBoxesPerLine = nrOfBoxesPerLine
        self.nrOfLinesPerPage = nrOfLinesPerPage

        # infer qr code properties
        self.bottom = top + self.height
        self.right = left + self.width
        self.row = self.get_qr_row()
        self.column = self.get_qr_column()
        self.page_nr = self.get_page_nr()
        
    # returns page nr starting at 1
    def get_page_nr(self):
        return math.floor(self.index/(self.nrOfBoxesPerLine*self.nrOfLinesPerPage))+1 

    # returns row number of qr code starting at row 1
    def get_qr_row(self):
        # subtract previous pages if not the first page
        local_index = self.index-(self.get_page_nr()-1)*self.nrOfBoxesPerLine*self.nrOfLinesPerPage    
        return math.ceil(local_index/self.nrOfBoxesPerLine)

    # returns the column of a box starting at 1 for most left column
    def get_qr_column(self):
        # subtract previous pages if not the first page
        local_index = self.index-(self.get_page_nr()-1)*self.nrOfBoxesPerLine*self.nrOfLinesPerPage    
        return local_index-(self.get_qr_row()-1)*self.nrOfBoxesPerLine