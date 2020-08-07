# import libraries
from pyzbar.pyzbar import decode
from PIL import Image

# read the qr code from an image
qrcode = decode(Image.open('testfiles/out0.jpg'))
print(f'nr of qr codes = {len(qrcode)}')

# Get the rect/contour coordinates:
left = qrcode[0].rect[0]
top = qrcode[0].rect[1]
width = qrcode[0].rect[2]
height = qrcode[0].rect[3]
print(f'left={left},top={top},width={width},height={height}')

# get the rectangular contour corner coordinates
top_left = [top,left]
print(f'top_left={top_left}')
top_right = [top,left+width]
print(f'top_right={top_right}')
bottom_left = [top-height,left]
print(f'bottom_left={bottom_left}')
bottom_right = [top-height,left+width]
print(f'bottom_right={bottom_right}')