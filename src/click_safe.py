import pymouse as pm
import src.setting_file as setting_file

BASE_X,BASE_Y = setting_file.BASE_POINT
BOX_SIZE = setting_file.BOX_SIZE

def box_click(row,col):

    m = pm.PyMouse()
    base_x, base_y = BASE_X+BOX_SIZE//2, BASE_Y+BOX_SIZE//2
    click_x = col*BOX_SIZE + base_x
    click_y = row*BOX_SIZE + base_y
    m.move(click_x, click_y)  #鼠标移动到（1410,5）
    m.click(click_x, click_y)
    m.click(click_x, click_y)

def jiaozhun():
    m = pm.PyMouse()
    a = m.position()
    print (a)
    return a

if __name__ == "__main__":
    jiaozhun()