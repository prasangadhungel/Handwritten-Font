import unittest
from read14 import ReadTemplate

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
        print(f'initialized={self.x}')
        result = self.read.addThree(4)
        expected = 7
        self.assertEqual(expected,result)
        #self.assertEqual(expected,result+1)#fails as expected
        self.assertNotEqual(expected,result+1)
        
    
if __name__ == '__main__':
    unittest.main()
    