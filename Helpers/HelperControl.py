from __future__ import annotations
import inspect as _inspect
import pickle
from enum import Enum
from typing import Iterable, Any, Callable

import multiprocessing
from multiprocessing.sharedctypes import Synchronized, SynchronizedArray
import ctypes

import Helpers.HelperKeyboard as helper_keyboard

GLOBAL_ID = 0

# Use to interact with the sync_variable since ControlStruct can't be passed between processes
class ControlStruct:
    STOPPED:int = 0
    RUNNING:int = 1
    PAUSED:int = 2
    def __init__(self, parent:ControlStruct|None=None):
        self.started:Synchronized[bool] = multiprocessing.Value(ctypes.c_bool, False)
        #self.paused:Synchronized[bool] = multiprocessing.Value(ctypes.c_bool, False)
        
        self.children:list[ControlStruct] = []
        #self.children:SynchronizedArray[int] = multiprocessing.Array(ctypes.c_int, size_or_initializer=20)
        self.parent = None
        #self.parent:Synchronized[int] = multiprocessing.Value(ctypes.c_int, -1)
        if (parent is not None):
            self.setParent(parent)

        # Mutable objects that both the parent and the child need to know about
        
        # an item is only ever put into the queue if there is an update
        self.sync_variable:Synchronized[int] = multiprocessing.Value(ctypes.c_int, ControlStruct.STOPPED)
        global GLOBAL_ID
        GLOBAL_ID += 1

    def __enter__(self):
        self.start()
        return self
    def __exit__(self, exc_type, exc, tb):
        self.sync_variable.value = ControlStruct.STOPPED
        self.stop()

    def getParent(self) -> ControlStruct|None:
        return self.parent   
    def setParent(self, new_parent:ControlStruct) -> None:
        if (self.parent is not None):
            self.parent.children.remove(self)
        new_parent.children.append(self)
        self.parent = new_parent

    def start(self):
        with (self.started.get_lock(),
              self.sync_variable.get_lock()):
            if (not self.started.value and self.sync_variable.value != ControlStruct.PAUSED):
                self.started.value = True
                self.sync_variable.value = ControlStruct.RUNNING
    def stop(self):
        with (self.started.get_lock()):
            if (self.started.value):
                # when the main process dies the children should too, to prevent zombies
                for child in self.children:
                    child.stop()
                self.started.value = False
                with self.sync_variable.get_lock():
                    self.sync_variable.value = ControlStruct.STOPPED
    def stopChildren(self):
        for child in self.children:
            # like pause, when a ControlStruct is stopped, all offspring should die too
            # easy way of memory leak prevention by removing likelihood of zombies
            child.stopChildren()
            child.stop()

    def pause(self):
        with (self.sync_variable.get_lock()):
            self.sync_variable.value = ControlStruct.PAUSED
            for child in self.children:
                child.pause()
    def pauseChildren(self):
        for child in self.children:
            child.pause()
    def unpause(self, unpause_all_offspring=False):
        with self.sync_variable.get_lock():
            if (unpause_all_offspring):
                for child in self.children:
                    child.unpause(unpause_all_offspring=unpause_all_offspring)
            self.sync_variable.value = ControlStruct.RUNNING

def placeInQueue(queue:multiprocessing.Queue, item, queued_moves = dict[int,bool], drop_item=True, unique=False, timeout=0) -> None:
    item_hash = hash(item)
    item_exists = queued_moves.get(item_hash)
    if (item_exists is None):
        item_exists = False
    if (drop_item):
        try:
            if (unique):
                if (not item_exists):
                    queue.put(item, block=True, timeout=timeout)
                    queued_moves[item_hash] = True
            else:
                queue.put(item, block=True, timeout=timeout)
        except Exception:
            return
    else:
        if (unique):
            if (not item_exists):
                queue.put(item, block=True)
                queued_moves[item_hash] = True
        else:
            queue.put(item, block=True)

def ControlStructWrapper(control_struct:ControlStruct, func:Callable[..., Any], args:Iterable[Any]):
    while (control_struct.sync_variable.value != ControlStruct.STOPPED):
        # func MUST do a piece of work and exit
        # so we can pause without re-creating this wrapper
        func(*args)

        # Spin while paused
        while (control_struct.sync_variable.value == ControlStruct.PAUSED):
            _=0
            
def keyb_control_struct_manager(control_struct:ControlStruct, paused_children:Synchronized):
    if (helper_keyboard.key_pressed('q')):
        print('quit')
        control_struct.stop()
        return

    if (helper_keyboard.key_pressed('p')):
        # have this be a function that blindly wakes up offspring
        # since our current control_struct isn't paused but the children are
        if (paused_children.value):
            with paused_children.get_lock():
                paused_children.value = False
                print('unpaused')
                control_struct.unpause(unpause_all_offspring=True)
        else:
            with paused_children.get_lock():
                paused_children.value = True
                print('paused')
                control_struct.pauseChildren()
        return
