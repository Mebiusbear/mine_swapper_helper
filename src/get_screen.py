import pyscreenshot as ImageGrab
import src.setting_file as setting_file

BASE_X,BASE_Y = setting_file.BASE_POINT
BOX_SIZE = setting_file.BOX_SIZE
EASY_POINTS = setting_file.EASY_POINTS
DIFFICULT_POINTS = setting_file.DIFFICULT_POINTS

def get_screen(level,filename):
    if level == "difficult":
        ROW,COL = DIFFICULT_POINTS
    elif level == "easy":
        ROW,COL = EASY_POINTS
    s1_x = BASE_X - 15
    s1_y = BASE_Y - 145
    s2_x = BASE_X + COL*BOX_SIZE + 15
    s2_y = BASE_Y + ROW*BOX_SIZE + 13
    im = ImageGrab.grab(bbox=(s1_x,s1_y,s2_x,s2_y))
    im.save(filename)