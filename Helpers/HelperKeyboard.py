import win32api as pywapi

def key_to_VK(key_char:str) -> str:
    return pywapi.VkKeyScan(key_char)

def key_pressed(key:str):
    key_char = key_to_VK(key)
    #if (pywapi.GetAsyncKeyState(key_char) < 0):
    #    return True
    if (pywapi.GetAsyncKeyState(key_char)):
        while (pywapi.GetAsyncKeyState(key_char) != 0):
            _=0
        return True
    return False
