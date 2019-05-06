import numpy as np
import cv2
from OpenGL.GL import *
import sys
from OpenGL.GLUT import *
from OpenGL.GLU import *

width, height = 810, 700
window = 0                                             # glut window number
xcount = 20
ycount = 20
fontHs = 14
fontHl = 17
fontWs = 14
fontWl = 17
lxspace = 15
sxspace = 11
prevlis = []
lis = []
lissize = [0,0]
pressed = 0

def putpixel(x, y):
    glBegin(GL_POINTS)                                  # start drawing a rectangle
    glVertex2f(x, height - y)                                   # bottom left point
    glEnd()    
    
def refresh2d(width, height):
    glViewport(0, 0, width, height)
    glClearColor(1.0,1.0,1.0,0.0)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0.0, width, 0.0, height, 0.0, 1.0)
    glMatrixMode (GL_MODELVIEW)
    glLoadIdentity()


        
def draw():                                            # ondraw is called all the time
    global lis
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT) # clear the screen
    glLoadIdentity()                                   # reset position
    refresh2d(width, height)                           # set mode to 2d
    glColor3f(0.0, 0.0, 0.0)
    for i in lis:
        putpixel(i[0],i[1])
    glutSwapBuffers()                                  # important for double buffering
    

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


                


if __name__ == '__main__':
	glutInit()                                             # initialize glut
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH)
	glutInitWindowSize(width, height)                      # set window size
	glutInitWindowPosition(100, 100)                       # set window position
	window = glutCreateWindow("noobtuts.com")              # create window with title
	glutDisplayFunc(draw)                                  # set draw function callback
	glutIdleFunc(draw)                                     # draw  continuously
	glutKeyboardFunc(processNormalKeys)
	glutMainLoop()                                         # start everything
