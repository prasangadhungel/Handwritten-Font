from glfunctions import *
from keyboard import *    
glutInit()                                             # initialize glut
glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH)
glutInitWindowSize(width, height)                      # set window size
glutInitWindowPosition(100, 100)                       # set window position
window = glutCreateWindow("noobtuts.com")              # create window with title
glutReshapeFunc(reshape)
glutDisplayFunc(draw)                                  # set draw function callback
glutIdleFunc(draw)                                     # draw  continuously
glutKeyboardFunc(processNormalKeys)
glutMainLoop()                                         # start everything
