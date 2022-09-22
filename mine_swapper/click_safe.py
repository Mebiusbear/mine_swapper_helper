import pymouse as pm



def box_click(row,col):
    m = pm.PyMouse()

    base_x,base_y = (283.1074523925781+1, 475.88787841796875)
    base_x,base_y = base_x+15,base_y+15
    click_x = col*30 + base_x
    click_y = row*30 + base_y
    m.move(click_x, click_y)  #鼠标移动到（1410,5）
    m.click(click_x, click_y)
    m.click(click_x, click_y)


