import math
class QrCode:
    def __init__(self, index, top,left,width,height,nrOfSymbols,nrOfBoxesPerLine):

        # initialize qr code properties
        self.top = top
        self.left = left
        self.width = width
        self.height = height
        self.index = int(index)
        self.nrOfSymbols = nrOfSymbols
        self.nrOfBoxesPerLine = nrOfBoxesPerLine

        # infer qr code properties
        self.bottom = top + self.height
        self.right = left + self.width
        self.row = self.get_qr_row()
        self.column = self.get_qr_column()

    # returns row number of qr code starting at row 1
    def get_qr_row(self):
        return math.ceil(self.index/self.nrOfBoxesPerLine)

    # returns the column of a box starting at 1 for most left column
    def get_qr_column(self):
        return self.index-(self.get_qr_row()-1)*self.nrOfBoxesPerLine