import time
import sys
import os
import threading

"""
Display an animated statusbar, with progress and percentage
( items-completed/items-total )
displayed below the statusbar. Seperate thread is used to
display the spinning "icon." In order to stop the statusbar thread
early, calling thread can use join() 

example output created by StatusBar thread:

[===============\--------------]
30/60  50%

Halt the status bar by catching KeyboardInterrupt (CTRL+C):

    progress = pyLTR.StatusBar(0, nSteps)
    progress.start()
    try:
        # long running task
        progress.increment()
    except KeyboardInterrupt:
        progress.stop()
        progress.join()

Written by chi (aka chi42) from 42gems.com-- 11 Sept, 2009
http://42gems.com/?p=617

Modified by Peter Schmitt -- December 2010 (added stop() routine to
halt the status bar)

Copyright (C) 2009 chi (from 42gems.com)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

Please see http://www.gnu.org/licenses/ for a copy of the license.
"""

class StatusBar(threading.Thread):

    # class variables:
    # max:  number of total items to be completed
    # pos:  number of completed items
    # inc:  amount of items to increment completed 'pos' by
    #           (shared resource)
    # comp: amount of '=' to display in the progress bar
    # running: whether or not the statusbar is running
    def __init__(self, pos=0, max=100):
        threading.Thread.__init__(self)
        self.pos = pos
        self.max = max
        self.busy_char = '|'
        self._running = 0
        self.inc =  0
        self.__getsize()
        self.comp = int(float(self.pos) / self.max * self.columns)
        self.inc_lock = threading.Lock() 

        self._stopper = threading.Event()

    def stop(self):
        """
        Stop the status bar early.  Call StatusBar::join to deal with
        extra threads.
        
        This may be useful when catching KeyboardInterrupt (CTRL+C):
        
            progress = pyLTR.StatusBar(0, nSteps)
            progress.start()
            try:
                # long running task
                progress.increment()
            except KeyboardInterrupt:
                progress.stop()
                progress.join()        
        """
        self._running = 0
        self._stopper.set()

    def __stopped(self):
        return self._stopper.isSet()

    # find number of columns in terminal
    def __getsize(self):
        rows, columns = os.popen('stty size', 'r').read().split()
        if int(columns) > 80:
            self.columns = 80 - 2
        else:
            self.columns = int(columns) - 2
        return

    # redraw progress bar and all numerial values
    def __print(self):
        self.__getsize()

        sys.stdout.write('\x1b[1G')
        sys.stdout.write('[' + '=' * self.comp + self.busy_char + \
            '-'*(self.columns - self.comp - 1) + ']'   )
        sys.stdout.write('\n\x1b[0K' + str(self.pos) + \
            '/' + str(self.max) + '\t' + \
            str( round(float(self.pos) / self.max * 100, 2)) + '%')
        sys.stdout.write('\x1b[1A\x1b[' + \
            str(self.comp + 2) + 'G')
        sys.stdout.flush()
        return

    # run the thread
    def run(self):
        global busy_chars, inteval
        busy_chars = ['|','/','-','\\']
        interval = 0.05

        self._running = 1 

        self.__print()
        while not self.__stopped():
            # loop and display the busy spinning icon
            for c in busy_chars:
                self.busy_char = c
                sys.stdout.write(c + '\x1b[1D')
                sys.stdout.flush()
                time.sleep(interval)

                self.inc_lock.acquire()
                if self.inc:
                    if (self.pos + self.inc) >= self.max:
                        self.inc_lock.release()
                        self.pos = self.max
                        self.comp = self.columns
                        self.busy_char = ''
                        self.__print()
                        sys.stdout.write('\n\n')
                        self._running = 0
                        return 0
                    else:
                        self.pos += self.inc
                        self.inc = 0
                        self.inc_lock.release()
                        self.comp = int(float(self.pos) / self.max \
                            * self.columns)
                        self.__print()
                else:
                    self.inc_lock.release()

        sys.stdout.write('\n\n')
        sys.stdout.flush()        

        return 1

    # increment number of completed items used by calling thread
    def increment(self):
        if self._running:
            self.inc_lock.acquire()
            self.inc += 1
            self.inc_lock.release()
            return 0
        else:
            return 1
