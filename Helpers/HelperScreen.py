import win32gui as pyw
import win32ui as pywui
import win32api as pywapi
import win32process as pywproc
import win32con as pywcon
from PIL import Image
import numpy as np
import numpy.typing as npt
import cv2
import matplotlib.pyplot as plt
from multiprocessing.sharedctypes import Synchronized

WINDOW_BAR_HEIGHT = 30 # px

# Class philosophy is once a ScreenCapture object is created it never needs to change
# no data passing is ever done in this class, it is mainly a wrapper for image processing
# that makes memory management and calls to the screen more straightforward
class ScreenCapture:
    def __init__(self, hwnd:int, filename:str=''):
        self.wDC = None
        self.dcObj = None
        self.cDC = None
        self.dataBitMap = None
        self.oldObj = None

        self.height = -1
        self.width = -1
        self.hwnd = hwnd
        last_idx = -1
        for i in range(len(filename)):
            if (filename[i] == '.'):
                last_idx = i
        if (last_idx > -1):
            self.filename = filename[:last_idx]
        else:
            self.filename = filename
        self.bmp_filename = self.filename+'.bmp'
        self.game_filename = self.filename+'GameWnd'+'.bmp'
        
        # Potion Motion template files
        self.top_left_template = openImage('PotionMotion/topLeft.png')
        self.bot_right_template = openImage('PotionMotion/botRight.png')

    def getHwnd(self) -> int:
        return self.hwnd
    def getFilename(self) -> str:
        return self.filename

    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        self.cleanUp()
    
    def cleanUp(self):
        if (self.cDC is not None and self.oldObj is not None):
            try:
                self.cDC.SelectObject(self.oldObj)
            except Exception:
                pass
            finally:
                self.oldObj = None

        if (self.dataBitMap is not None):
            try:
                pyw.DeleteObject(self.dataBitMap.GetHandle())
            except Exception:
                pass
            finally:
                self.dataBitMap = None

        if (self.cDC is not None):
            try:
                self.cDC.DeleteDC()
            except Exception:
                pass
            finally:
                self.cDC = None

        if (self.dcObj is not None):
            try:
                self.dcObj.DeleteDC()
            except Exception:
                pass
            finally:
                self.dcObj = None

        if (self.wDC is not None):
            try:
                pyw.ReleaseDC(self.hwnd, self.wDC)
            except Exception:
                pass
            finally:
                self.wDC = None
        
        self.height = -1
        self.width = -1
    
    def recreateBitmap(self, height:int, width:int):
        # 1) Unselect old bitmap
        if (self.cDC is not None and self.oldObj is not None):
            try:
                self.cDC.SelectObject(self.oldObj)
            except Exception:
                pass
            finally:
                self.oldObj = None

        # 2) Delete old bitmap handle
        if (self.dataBitMap is not None):
            try:
                pyw.DeleteObject(self.dataBitMap.GetHandle())
            except Exception:
                pass
            finally:
                self.dataBitMap = None

        self.dataBitMap = pywui.CreateBitmap()
        self.dataBitMap.CreateCompatibleBitmap(self.dcObj, width, height)
        self.oldObj = self.cDC.SelectObject(self.dataBitMap)

        self.height = height
        self.width = width
        
    def saveBitmap(self, height:int=-1, width:int=-1, src_x_offset:int=0, src_y_offset:int=0, dest_x_offset:int=0, dest_y_offset:int=0,
                   set_focus:bool=True, fn:str=None, save_to_file=False) -> str|Image.Image|None:
        try:
            if (set_focus):
                setFocus(self.hwnd)
            if (width <= 0 and height <= 0):
                _,_,width,height = pyw.GetClientRect(self.hwnd)
            
            if (width <= 0 or height <= 0):
                return None

            if (self.wDC is None):
                self.wDC = pyw.GetWindowDC(self.hwnd)
            if (self.dcObj is None):
                self.dcObj=pywui.CreateDCFromHandle(self.wDC)
            if (self.cDC is None):
                self.cDC=self.dcObj.CreateCompatibleDC()

            if (self.dataBitMap is None):
                self.dataBitMap = pywui.CreateBitmap()
            if (width != self.width or height != self.height or self.oldObj is None):
                self.recreateBitmap(height, width)

            #if (self.oldObj is None):
            #    self.oldObj = self.cDC.SelectObject(self.dataBitMap)

            #self.cDC.BitBlt((src_x_offset,src_y_offset), (width, height) , self.dcObj, (dest_x_offset,dest_y_offset), pywcon.SRCCOPY)
            self.cDC.BitBlt((dest_x_offset, dest_y_offset), (width, height) , self.dcObj, (src_x_offset, src_y_offset), pywcon.SRCCOPY)
            if (save_to_file):
                save_name = self.bmp_filename
                if (fn is not None):
                    save_name = fn
                self.dataBitMap.SaveBitmapFile(self.cDC, save_name)
                return save_name
            else:
                buf = self.dataBitMap.GetBitmapBits(True)
                pil_img = Image.frombuffer('RGBA',
                                           (width, height),
                                           buf,
                                           'raw',
                                           'BGRA',
                                           0,
                                           1).copy()
                return pil_img
        except Exception:
            self.cleanUp()
            print('Something went wrong in the screen capture function')
            return None

    def saveGameWindow(self, height:int, width:int, x_offset:int, y_offset:int, safe:Synchronized, set_focus=True, save_to_file=False) -> str|Image.Image|None:
        while True:
            with safe.get_lock():
                if (safe.value):
                    break
        bitmap_return = self.saveBitmap(height, width, src_x_offset=x_offset, src_y_offset=y_offset, set_focus=set_focus, fn=self.game_filename, save_to_file=save_to_file)
        return bitmap_return

    def setupGameWindowPM(self, save_to_file=False):
        setFocus(self.hwnd)
        y_screenOffset, x_screenOffset = getScreenOffset(self.hwnd)
        _,_,windowWidth, windowHeight = pyw.GetClientRect(self.hwnd)

        bitmap_return = None
        while (bitmap_return is None):
            bitmap_return = self.saveBitmap(height=windowHeight, width=windowWidth, save_to_file=save_to_file)
        src = None
        if (save_to_file and bitmap_return is str):
            src:Image.Image = openImage(bitmap_return)
            if (src is None):
                raise Exception('Something is wrong when opening saved file in setupGameWindowPM')
        else:
            src = bitmap_return
        if (src is None):
            #print('Something is wrong in: setupGameWindowPM')
            raise Exception('Something is wrong in: setupGameWindowPM')
        topLeft,_,_ = templateMatch(src, self.top_left_template, save_img=False)
        botRight,_,_ = templateMatch(src, self.bot_right_template, save_img=False)
        
        # Resolution: 1024x768
        widthOffset  = 19
        heightOffset = 19
        old_y,old_x = topLeft
        topLeft = (old_y+heightOffset, old_x+widthOffset)
        adjWidth  = botRight[1]-topLeft[1]
        adjHeight = botRight[0]-topLeft[0]
        y_gameOffset = y_screenOffset+topLeft[0]
        x_gameOffset = x_screenOffset+topLeft[1]
        return (adjHeight, adjWidth, (y_gameOffset, x_gameOffset), (y_screenOffset, x_screenOffset), topLeft)

def relevantWindow(windowText:str, wordToFind:str=''):
    windowText = windowText.lower()
    wordToFind = wordToFind.lower()
    return wordToFind in windowText

def addToWindowList(hwnd:int, arr:list[int]):
    arr.append(hwnd)

def wizardWindowCallback(hwnd:int, window_list:list[int], relevant_word:str='Wizard101'):
    window_text = pyw.GetWindowText(hwnd)
    if (relevantWindow(window_text, relevant_word)):
        addToWindowList(hwnd, window_list)

def getWizardWindow() -> int:
    windowList = []
    pyw.EnumWindows(wizardWindowCallback, windowList)
    #if (len(windowList) <= 0 or len(windowList) > 1):
    if (len(windowList) <= 0):
        raise Exception('Problem finding Wizard101 window')
    # Assume only 1 window open for now
    wizardHwnd:int = windowList[0]
    return wizardHwnd

def setFocus(hwnd:int) -> None:
    attached = False
    current_thread_id = pywapi.GetCurrentThreadId()
    hwnd_tid,_ = pywproc.GetWindowThreadProcessId(hwnd)
    
    try:
        if (current_thread_id != hwnd_tid):
            pywproc.AttachThreadInput(current_thread_id, hwnd_tid, True)
            attached = True
        pyw.SetForegroundWindow(hwnd)
    except Exception:
        _=0
    finally:
        if (attached):
            try:
                pywproc.AttachThreadInput(current_thread_id, hwnd_tid, False)
            except:
                pass

#y,x
def plotX(src:Image.Image, point:tuple[int,int]):
    # y,x
    r,c = point

    plt.figure(figsize=(6, 6))

    shape = np.shape(src)
    # Display image
    if len(shape) == 2:
        plt.imshow(src, cmap='gray')
    else:
        plt.imshow(src)

    # Overlay red X
    plt.scatter(
        c, r,            # x = col, y = row
        marker='x',
        s=65,
        c='red',
        linewidths=1
    )

    plt.title("Template Match Location")
    plt.axis('off')
    plt.show()

# Assumes hwnd is windowed
def getScreenOffset(hwnd:int):
    y_screenOffset, x_screenOffset = getWindowPos(hwnd)
    y_screenOffset -= WINDOW_BAR_HEIGHT
    return (y_screenOffset, x_screenOffset)

def getWindowPos(hwnd:int):
    setFocus(hwnd)
    """
    _,_,_,_,location = pyw.GetWindowPlacement(hwnd)
    x_start,y_start,_,_ = location
    x_start = max(0,x_start)
    y_start = max(0,y_start)
    """
    x_start,y_start = pyw.ClientToScreen(hwnd, (0, 0))
    return (y_start,x_start)

def openImage(img_name:str, ptr=False) -> Image.Image|None:
    image = None
    try:
        image = Image.open(img_name)
    except Exception:
        print('Issue opening image file')
        return None
    
    if (image is None):
        return None
    if (ptr):
        return image
    image_return = image.copy()
    image.close()
    return image_return

def grayscale(arr:npt.ArrayLike) -> npt.ArrayLike:
    grayscale_array = np.mean(arr, axis=-1)
    return grayscale_array

# Returns top left pixel of template in src
def templateMatch(src:Image.Image, template:Image.Image, img_name='templateMatch', confidence_interval=.9, save_img=False):
    src_gray:npt.ArrayLike = grayscale(src)
    template_gray:npt.ArrayLike = grayscale(template)

    #src_f = src_gray.astype(np.float32)
    #tpl_f = template_gray.astype(np.float32)
    src_f = np.astype(src_gray, np.float32)
    tpl_f = np.astype(template_gray, np.float32)

    out = None
    try:
        out = cv2.matchTemplate(src_f, tpl_f, cv2.TM_CCOEFF_NORMED)
        topLeft = np.unravel_index(np.argmax(out), out.shape)
        confidence = out[topLeft]
        confident_match = True
        if (confidence < confidence_interval):
            confident_match = False
        
        if save_img:
            vis = out.copy()
            vis -= vis.min()
            if vis.max() > 0:
                vis /= vis.max()
            vis = (vis * 255).astype(np.uint8)
            Image.fromarray(vis, mode="L").save(img_name + ".png")
            plotX(src, topLeft)
    except Exception:
        print('Issue in the matchTemplate function')
        return ((0,0), 0, False)
    return (topLeft, confidence, confident_match)
