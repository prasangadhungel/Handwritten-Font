import unittest
from read14 import ReadTemplate
from QrCode import QrCode
import math

class TestStringMethods(unittest.TestCase):

    # this is executed before all tests are ran
    @classmethod
    def setUpClass(self):
        # initialize class that is tested
        self.read = ReadTemplate()

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)
    
    # test the unit tests on the class ReadTemplate (referred to as self.read)
    def test_addThree(self):
        result = self.read.addThree(4)
        expected = 7
        self.assertEqual(expected,result)
        # self.assertEqual(expected,result+1)#fails as expected
        self.assertNotEqual(expected,result+1)
        
    # tests whether it correctly identifies if there exists(within system) a row without qr code
    def test_not_qr_in_each_row(self):
        self.read.nrOfBoxesPerLine = 9
        self.read.maxNrOfLinesPerPage = 3
        self.read.nrOfSymbols = 100
        self.read.nrOfLinesInTemplate = math.ceil(self.read.nrOfSymbols/(self.read.maxNrOfLinesPerPage))
        qr_one = QrCode(2,1,2,3,4,self.read.nrOfSymbols,self.read.nrOfBoxesPerLine,self.read.nrOfLinesInTemplate,self.read.maxNrOfLinesPerPage)
        qr_two = QrCode(18,1,2,3,4,self.read.nrOfSymbols,self.read.nrOfBoxesPerLine,self.read.nrOfLinesInTemplate,self.read.maxNrOfLinesPerPage)
        #qr_three = QrCode(23,1,2,3,4,self.read.nrOfSymbols,self.read.nrOfBoxesPerLine,self.read.nrOfLinesInTemplate,self.read.maxNrOfLinesPerPage)
        qrcodes = [qr_one,qr_two]
        self.assertFalse(self.read.isQrcodeOnEachRow(qrcodes))
        #self.assertEqual(1,1)
    
    # tests whether it correctly identifies if each row has a qr code
    def test_qr_in_each_row(self):
        self.read.nrOfBoxesPerLine = 9
        self.read.maxNrOfLinesPerPage = 3
        self.read.nrOfSymbols = 100
        self.read.nrOfLinesInTemplate = math.ceil(self.read.nrOfSymbols/(self.read.maxNrOfLinesPerPage))
        qr_one = QrCode(2,1,2,3,4,self.read.nrOfSymbols,self.read.nrOfBoxesPerLine,self.read.nrOfLinesInTemplate,self.read.maxNrOfLinesPerPage)
        qr_two = QrCode(18,1,2,3,4,self.read.nrOfSymbols,self.read.nrOfBoxesPerLine,self.read.nrOfLinesInTemplate,self.read.maxNrOfLinesPerPage)
        qr_three = QrCode(23,1,2,3,4,self.read.nrOfSymbols,self.read.nrOfBoxesPerLine,self.read.nrOfLinesInTemplate,self.read.maxNrOfLinesPerPage)
        qrcodes = [qr_one,qr_two,qr_three]
        self.assertTrue(self.read.isQrcodeOnEachRow(qrcodes))
    
    # tests whether it correctly identifies if each row has a qr code
    def test_qr_in_each_row_last_page(self):
        self.read.nrOfBoxesPerLine = 9
        self.read.maxNrOfLinesPerPage = 3
        self.read.nrOfSymbols = 53
        self.read.nrOfLinesInTemplate = math.ceil(self.read.nrOfSymbols/(self.read.maxNrOfLinesPerPage))
        qr_one = QrCode(28,1,2,3,4,self.read.nrOfSymbols,self.read.nrOfBoxesPerLine,self.read.nrOfLinesInTemplate,self.read.maxNrOfLinesPerPage)
        qr_two = QrCode(37,1,2,3,4,self.read.nrOfSymbols,self.read.nrOfBoxesPerLine,self.read.nrOfLinesInTemplate,self.read.maxNrOfLinesPerPage)
        qr_three = QrCode(52,1,2,3,4,self.read.nrOfSymbols,self.read.nrOfBoxesPerLine,self.read.nrOfLinesInTemplate,self.read.maxNrOfLinesPerPage)
        qrcodes = [qr_one,qr_two,qr_three]
        self.assertTrue(self.read.isQrcodeOnEachRow(qrcodes))

    # tests if the most left qr code is identified correclty
    def test_most_left_colum(self):
        self.read.nrOfBoxesPerLine = 9
        self.read.nrOfLinesPerPage = 3
        self.read.nrOfSymbols = 100
        qr_one = QrCode(3,1,2,3,4,self.read.nrOfSymbols,self.read.nrOfBoxesPerLine,self.read.nrOfLinesInTemplate,self.read.maxNrOfLinesPerPage)
        qr_two = QrCode(5,1,2,3,4,self.read.nrOfSymbols,self.read.nrOfBoxesPerLine,self.read.nrOfLinesInTemplate,self.read.maxNrOfLinesPerPage)
        qr_three = QrCode(8,1,2,3,4,self.read.nrOfSymbols,self.read.nrOfBoxesPerLine,self.read.nrOfLinesInTemplate,self.read.maxNrOfLinesPerPage)
        qrcodes = [qr_one,qr_two,qr_three]
        r1 = self.read.get_most_left_column_per_row(1,qr_one)
        r2 = self.read.get_most_left_column_per_row(3,qr_one)
        r3 = self.read.get_most_left_column_per_row(4,qr_one)
        r4 = self.read.get_most_left_column_per_row(99,qr_three)
        
        self.assertEqual(1,r1)
        self.assertEqual(3,r2)
        self.assertEqual(3,r3)
        self.assertEqual(8,r4)
    
    # tests if the most right qr code is identified correclty
    def test_most_right_colum(self):
        self.read.nrOfBoxesPerLine = 9
        self.read.maxNrOfLinesPerPage = 3
        self.read.nrOfSymbols = 100
        self.read.nrOfLinesInTemplate = math.ceil(self.read.nrOfSymbols/(self.read.maxNrOfLinesPerPage))
        qr_one = QrCode(3,1,2,3,4,self.read.nrOfSymbols,self.read.nrOfBoxesPerLine,self.read.nrOfLinesInTemplate,self.read.maxNrOfLinesPerPage)
        qr_two = QrCode(5,1,2,3,4,self.read.nrOfSymbols,self.read.nrOfBoxesPerLine,self.read.nrOfLinesInTemplate,self.read.maxNrOfLinesPerPage)
        qr_three = QrCode(8,1,2,3,4,self.read.nrOfSymbols,self.read.nrOfBoxesPerLine,self.read.nrOfLinesInTemplate,self.read.maxNrOfLinesPerPage)
        qrcodes = [qr_one,qr_two,qr_three]
        r1 = self.read.get_most_right_column_per_row(1,qr_one)
        r2 = self.read.get_most_right_column_per_row(3,qr_one)
        r3 = self.read.get_most_right_column_per_row(4,qr_one)
        r4 = self.read.get_most_right_column_per_row(99,qr_three)
        
        self.assertEqual(3,r1)
        self.assertEqual(3,r2)
        self.assertEqual(4,r3)
        self.assertEqual(99,r4)
    
    # tests if the horizontal distance between most left and most right qr code on a line is
    # detected correclty for when distance is larger than required
    def test_check_hori_dist(self):
        self.read.nrOfBoxesPerLine = 9
        self.read.nrOfLinesPerPage = 3
        left = 2
        right = 7
        result = self.read.check_hori_dist(left,right,0.25)
        expected = True # 7-2=5>= 0.25*9=2.25
        self.assertEqual(result,expected)

    
    def test_check_hori_dist_minimal(self):
        self.read.nrOfBoxesPerLine = 4
        self.read.nrOfLinesPerPage = 3
        left = 2
        right = 4
        result = self.read.check_hori_dist(left,right,0.25)
        expected = True # 7-2=5>= 0.25*9=2.25
        self.assertEqual(result,expected)
    
    def test_check_hori_dist_false(self):
        left = 3
        right = 4
        result = self.read.check_hori_dist(left,right,0.25)
        expected = False # 7-2=5>= 0.25*9=2.25
        self.assertEqual(result,expected)
    
    # tests if the horizontal distance between most left and most right qr code on a line is
    # detected correclty for when distance is smaller than required
    def test_check_hori_dist_smaller(self):
        left = 5
        right = 7
        result = self.read.check_hori_dist(left,right,0.25)
        expected = False # 7-2=5>= 0.25*9=2.25
        self.assertEqual(result,expected)
    
    
    # tests if the most right qr code is identified correclty
    def test_has_quarter_spacing_false(self):
        self.read.nrOfBoxesPerLine = 9
        self.read.maxNrOfLinesPerPage = 3
        self.read.nrOfSymbols = 100
        self.read.nrOfLinesInTemplate = math.ceil(self.read.nrOfSymbols/(self.read.maxNrOfLinesPerPage))
        qr_one = QrCode(3,1,2,3,4,self.read.nrOfSymbols,self.read.nrOfBoxesPerLine,self.read.nrOfLinesInTemplate,self.read.maxNrOfLinesPerPage)
        qr_two = QrCode(5,1,2,3,4,self.read.nrOfSymbols,self.read.nrOfBoxesPerLine,self.read.nrOfLinesInTemplate,self.read.maxNrOfLinesPerPage)
        qr_three = QrCode(18,1,2,3,4,self.read.nrOfSymbols,self.read.nrOfBoxesPerLine,self.read.nrOfLinesInTemplate,self.read.maxNrOfLinesPerPage)
        qrcodes = [qr_one,qr_two,qr_three]
        self.assertFalse(self.read.has_quarter_spacing(qrcodes))
    
        # tests if the most right qr code is identified correclty
    def test_has_quarter_spacing_false_but_close(self):
        self.read.nrOfBoxesPerLine = 9
        self.read.maxNrOfLinesPerPage = 3
        self.read.nrOfLinesInTemplate = math.ceil(self.read.nrOfSymbols/(self.read.maxNrOfLinesPerPage))
        self.read.nrOfSymbols = 100
        qr_one = QrCode(3,1,2,3,4,self.read.nrOfSymbols,self.read.nrOfBoxesPerLine,self.read.nrOfLinesInTemplate,self.read.maxNrOfLinesPerPage)
        qr_two = QrCode(5,1,2,3,4,self.read.nrOfSymbols,self.read.nrOfBoxesPerLine,self.read.nrOfLinesInTemplate,self.read.maxNrOfLinesPerPage)
        qr_three = QrCode(6,1,2,3,4,self.read.nrOfSymbols,self.read.nrOfBoxesPerLine,self.read.nrOfLinesInTemplate,self.read.maxNrOfLinesPerPage)
        qrcodes = [qr_one,qr_two,qr_three]
        self.assertFalse(self.read.has_quarter_spacing(qrcodes))    
    
    # tests if the most right qr code is identified correclty
    def test_has_quarter_spacing_true(self):
        self.read.nrOfBoxesPerLine = 9
        self.read.maxNrOfLinesPerPage = 3
        self.read.nrOfSymbols = 100
        self.read.nrOfLinesInTemplate = math.ceil(self.read.nrOfSymbols/(self.read.maxNrOfLinesPerPage))
        qr_one = QrCode(3,1,2,3,4,self.read.nrOfSymbols,self.read.nrOfBoxesPerLine,self.read.nrOfLinesInTemplate,self.read.maxNrOfLinesPerPage)
        qr_two = QrCode(5,1,2,3,4,self.read.nrOfSymbols,self.read.nrOfBoxesPerLine,self.read.nrOfLinesInTemplate,self.read.maxNrOfLinesPerPage)
        qr_three = QrCode(7,1,2,3,4,self.read.nrOfSymbols,self.read.nrOfBoxesPerLine,self.read.nrOfLinesInTemplate,self.read.maxNrOfLinesPerPage)
        qrcodes = [qr_one,qr_two,qr_three]
        self.assertTrue(self.read.has_quarter_spacing(qrcodes))    
    
    def test_avg_qr_width_per_row(self):
        self.read.nrOfBoxesPerLine = 9
        self.read.maxNrOfLinesPerPage = 3
        self.read.nrOfSymbols = 100
        self.read.nrOfLinesInTemplate = math.ceil(self.read.nrOfSymbols/(self.read.maxNrOfLinesPerPage))
        self.read.m = 100
        width_one = 5
        width_two = 10
        width_three = 20
        qr_one = QrCode(3,1,2,width_one,4,self.read.nrOfSymbols,self.read.nrOfBoxesPerLine,self.read.nrOfLinesInTemplate,self.read.maxNrOfLinesPerPage)
        qr_two = QrCode(5,1,2,width_two,4,self.read.nrOfSymbols,self.read.nrOfBoxesPerLine,self.read.nrOfLinesInTemplate,self.read.maxNrOfLinesPerPage)
        qr_three = QrCode(7,1,2,width_three,4,self.read.nrOfSymbols,self.read.nrOfBoxesPerLine,self.read.nrOfLinesInTemplate,self.read.maxNrOfLinesPerPage)
        qrcodes = [qr_one,qr_two,qr_three]
        self.assertEqual(int(35/3),self.read.avg_qr_width_per_row(qrcodes))
        self.assertNotEqual(int(32/3),self.read.avg_qr_width_per_row(qrcodes))
    
    def test_identify_unknown_qrcodes_in_row(self):
        self.read.nrOfBoxesPerLine = 9
        self.read.maxNrOfLinesPerPage = 3
        self.read.nrOfSymbols = 100
        self.read.nrOfLinesInTemplate = math.ceil(self.read.nrOfSymbols/(self.read.maxNrOfLinesPerPage))
        
        qr_one = QrCode(3,1,2,3,4,self.read.nrOfSymbols,self.read.nrOfBoxesPerLine,self.read.nrOfLinesInTemplate,self.read.maxNrOfLinesPerPage)
        qr_two = QrCode(5,1,2,3,4,self.read.nrOfSymbols,self.read.nrOfBoxesPerLine,self.read.nrOfLinesInTemplate,self.read.maxNrOfLinesPerPage)
        qr_three = QrCode(7,1,2,3,4,self.read.nrOfSymbols,self.read.nrOfBoxesPerLine,self.read.nrOfLinesInTemplate,self.read.maxNrOfLinesPerPage)
        qrcodes = [qr_one,qr_two,qr_three]
        result = self.read.identify_unknown_qrcodes_in_row(1,qrcodes)
        expected = [1,2,4,6,8,9]
        self.assertEqual(expected,result)
        
    def test_identify_unknown_qrcodes_in_row_next_page(self):
        self.read.nrOfBoxesPerLine = 9
        self.read.maxNrOfLinesPerPage = 3
        self.read.nrOfSymbols = 45
        self.read.nrOfLinesInTemplate = math.ceil(self.read.nrOfSymbols/(self.read.maxNrOfLinesPerPage))
        qr_one = QrCode(38,1,2,3,4,self.read.nrOfSymbols,self.read.nrOfBoxesPerLine,self.read.nrOfLinesInTemplate,self.read.maxNrOfLinesPerPage)
        qr_two = QrCode(43,1,2,3,4,self.read.nrOfSymbols,self.read.nrOfBoxesPerLine,self.read.nrOfLinesInTemplate,self.read.maxNrOfLinesPerPage)
        
        qrcodes = [qr_one,qr_two]
        result = self.read.identify_unknown_qrcodes_in_row(2,qrcodes)
        expected = [37,39,40,41,42,44,45]
        self.assertEqual(expected,result)
    
    def test_qrcode_get_nr_of_lines_in_last_page(self):
        self.read.nrOfBoxesPerLine = 2
        self.read.maxNrOfLinesPerPage = 2
        self.read.nrOfSymbols = 5
        self.read.nrOfLinesInTemplate = math.ceil(self.read.nrOfSymbols/(self.read.maxNrOfLinesPerPage))
        qr_one = QrCode(5,1,2,3,4,self.read.nrOfSymbols,self.read.nrOfBoxesPerLine,self.read.nrOfLinesInTemplate,self.read.maxNrOfLinesPerPage)
        result = qr_one.get_nr_of_lines_in_last_page()
        expected = 1
        self.assertEqual(expected,result)
    
    def test_qrcode_get_nr_of_lines_in_last_pageV2(self):
        self.read.nrOfBoxesPerLine = 2
        self.read.maxNrOfLinesPerPage = 2
        self.read.nrOfSymbols = 19
        self.read.nrOfLinesInTemplate = math.ceil(self.read.nrOfSymbols/(self.read.maxNrOfLinesPerPage))
        
        qr_one = QrCode(5,1,2,3,4,self.read.nrOfSymbols,self.read.nrOfBoxesPerLine,self.read.nrOfLinesInTemplate,self.read.maxNrOfLinesPerPage)
        result = qr_one.get_nr_of_lines_in_last_page()
        expected = 2
        self.assertEqual(expected,result)
        
    def test_qrcode_get_nr_of_lines_in_last_pageV3(self):
        self.read.nrOfBoxesPerLine = 2
        self.read.maxNrOfLinesPerPage = 2
        self.read.nrOfSymbols = 49
        self.read.nrOfLinesInTemplate = math.ceil(self.read.nrOfSymbols/(self.read.maxNrOfLinesPerPage))
        qr_one = QrCode(49,1,2,3,4,self.read.nrOfSymbols,self.read.nrOfBoxesPerLine,self.read.nrOfLinesInTemplate,self.read.maxNrOfLinesPerPage)
        result = qr_one.get_nr_of_lines_in_last_page()
        expected = 1
        self.assertEqual(expected,result)
        
    def test_qrcode_is_last_page(self):
        self.read.nrOfBoxesPerLine = 2
        self.read.maxNrOfLinesPerPage = 2
        self.read.nrOfSymbols = 5
        self.read.nrOfLinesInTemplate = math.ceil(self.read.nrOfSymbols/(self.read.maxNrOfLinesPerPage))
        qr_one = QrCode(5,1,2,3,4,self.read.nrOfSymbols,self.read.nrOfBoxesPerLine,self.read.nrOfLinesInTemplate,self.read.maxNrOfLinesPerPage)
        self.assertTrue(qr_one.is_last_page())
     
    def test_qrcode_is_last_pageV0(self):
        print(f'\n\n\n')
        self.read.nrOfBoxesPerLine = 2
        self.read.maxNrOfLinesPerPage = 2
        self.read.nrOfSymbols = 49
        self.read.nrOfLinesInTemplate = math.ceil(self.read.nrOfSymbols/(self.read.maxNrOfLinesPerPage))
        
        
        qr_one = QrCode(49,1,2,3,4,self.read.nrOfSymbols,self.read.nrOfBoxesPerLine,self.read.nrOfLinesInTemplate,self.read.maxNrOfLinesPerPage)
        self.assertTrue(qr_one.is_last_page())
    
if __name__ == '__main__':
    unittest.main()