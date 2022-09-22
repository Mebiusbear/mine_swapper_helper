from turtle import screensize
import pyscreenshot as ImageGrab

def get_screen(filename):
    base_x,base_y = (283.1074523925781, 475.88787841796875)
    s1_x = base_x - 15
    s1_y = base_y - 145
    s2_x = base_x + 30*30 + 15
    s2_y = base_y + 16*30 + 13
    im = ImageGrab.grab(bbox=(s1_x,s1_y,s2_x,s2_y))
    im.save(filename)