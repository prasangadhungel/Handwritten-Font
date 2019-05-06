from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

width, height = 810, 700                               # window size
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

def putpixel(x, y):
    glBegin(GL_POINTS)                                  		# start drawing a rectangle
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

def reshape(w, h):
	if(h == 0):
		h = 1
	ratio = w * 1.0/h
	glMatrixMode(GL_PROJECTION)
	glLoadIdentity()
	glViewport(0, 0, w, h)
	gluPerspective(45.0, ratio, 0.1, 100.0)
	glMatrixMode(GL_MODELVIEW)

def draw():                                            # ondraw is called all the time
    global lis
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT) # clear the screen
    glLoadIdentity()                                   # reset position
    refresh2d(width, height)                           # set mode to 2d
    glColor3f(0.0, 0.0, 0.0)
    for i in lis:
        putpixel(i[0],i[1])
    glutSwapBuffers()                                  # important for double buffering

