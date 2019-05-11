from glfunctions import *
import cv2
import sys

def processNormalKeys(key, a, b):
        global xcount, ycount, sxspace, lxspace
        global lis, lissize
        global fontHs, fontHl, fontWl, fontWs
        
        for j in range(fontHl):
                lis.pop()
                
        space = lxspace
        val = int.from_bytes(key, 'big')
        if(val == 13):
                ycount += (fontHl * 3)//2
                xcount = 20


        elif(int.from_bytes(key, 'big') == 27):
                sys.exit(0)
        
        elif(val == 32):
                xcount += fontWl
                if(xcount > 780):
                        xcount = 20
                        ycount -= (fontHl * 3)//2

        elif(val == 8):
                xcount -= lxspace
                if(xcount < 20):
                        xcount = 780
                        ycount -= (fontHl * 3)//2
                i = lissize[-1]
                while (i>lissize[-2]):
                        lis.pop()
                        i-=1
                lissize.pop()

        elif(val == 9):
                xcount += 4*fontWl
                if(xcount >780):
                        xcount = 20
                        ycount += (fontHl * 3)//2

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
                if(val == 103 or val == 121 or val == 112 or val == 113 or val == 59 or val == 106 or val==44):
                        for j in range(fh):
                                for k in range(fw):
                        	        if(img[j][k] != 0):
        	                	        lis.append((k+xcount,j+ycount + fontHs//3 + fontHl - fontHs))
                elif(val > 94):
                        for j in range(fh):
                                for k in range(fw):
                                        if(img[j][k] != 0):
                                                lis.append((k+xcount,j+ycount + fontHl - fontHs))


                else:
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
                        ycount += (fontHl * 3)//2

        for j in range(fontHl):
                lis.append((xcount, ycount + j))
        
        
