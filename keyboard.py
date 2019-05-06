from glfunctions import *
import cv2
import sys

def processNormalKeys(key, a, b):
        global xcount, ycount, sxspace, lxspace
        global lis, lissize
        global fontHs, fontHl, fontWl, fontWs
        space = lxspace
        val = int.from_bytes(key, 'big')
        if(val == 13):
                ycount += fontHl
                xcount = 20
        
        elif(int.from_bytes(key, 'big') == 27):
                sys.exit(0)
        
        elif(val == 32):
                xcount += fontWl

        elif(val == 8):
                xcount -= space
                i = lissize[-1]
                while (i>lissize[-2]):
                        lis.pop()
                        i-=1
                lissize.pop()

        elif(val == 9):
                xcount += 4*fontWl
                if(xcount >780):
                        xcount = 20

        elif(val > 1 and val < 128):
                img = cv2.imread('images/'+str(val)+'.png',0)
                if(val > 94):
                        fh = fontHs
                        fw = fontWs
                else:
                        fh = fontHl
                        fw = fontWl
                img = cv2.resize(img, (fw, fh))
                for i in range(len(img)):
	                for j in range(len(img[0])):
		                if(img[i][j] < 150):
			                img[i][j] = 1
		                else:
			                img[i][j] = 0

                for j in range(fh):
                        for k in range(fw):
                	        if(img[j][k] != 0):
        	        	        lis.append((k+xcount,j+ycount))
                
                lissize.append(len(lis))        
                if val > 94:
                        space = sxspace
                        xcount += sxspace
                else:
                        space = lxspace
                        xcount += lxspace
                if(xcount > 780):
                        xcount = 20
                        ycount += (fontHl)
