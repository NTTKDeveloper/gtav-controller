import pydirectinput
import time
import pygetwindow as gw
import pyautogui

class GyroController():
    def __init__(self, name):
        self.name = name

    #Lấy giá trị giữa màn hình 
    def get_centerxy(self):
        game_window = gw.getWindowsWithTitle(self.name)[0]
        # Lấy kích thước và vị trí của cửa sổ game
        return game_window.left + game_window.width // 2,  game_window.top + game_window.height // 2 
        
    #Đưa chuột về giữa màn hình game
    def move_in_center(center_x, center_y):
        # Di chuyển chuột đến trung tâm cửa sổ game
        pydirectinput.moveTo(center_x, center_y)

    #Di chuyễn tới vị trí xy trả về 1 nếu xy hợp lệ, ngược lại trả về 0
    def move_xy(step_pixel_x, step_pixel_y, center_x, center_y, width, height):
        current_x, current_y = pyautogui.position()
        limit_x = width / 4 
        limit_y = height / 2 

        # Kiểm tra nếu di chuyển vẫn nằm trong hình chữ nhật
        new_x = current_x + step_pixel_x
        new_y = current_y + step_pixel_y

        if ((center_x - limit_x) <= new_x <= (center_x + limit_x) and
            (center_y - limit_y) <= new_y <= (center_y + limit_y)):
            return 1  # Di chuyển hợp lệ
        return 0  # Di chuyển vượt giới hạn
    
    #Giả lập bàn phím và nhớ tắt unikey
    def press_key(seft, key, duration):
        pydirectinput.keyDown(key) # Nhấn phím
        time.sleep(duration)
        pydirectinput.keyUp(key) # Nhả phím

# # Thời gian chờ trước khi bắt đầu
# print("Bắt đầu điều khiển chuột sau 3 giây...")
# time.sleep(3)

# gyroController = GyroController("Emulation")
# gyroController.press_key('w', 5)