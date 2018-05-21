import cv2, os
import numpy as np
import tkinter.filedialog
from tkinter import *

class RoiGui:
    def __init__(self, master):
        self.master = master
        master.wm_title("MMSE Pentagon ROI")

        mmse_path = tkinter.filedialog.askopenfilename()
        #print(mmse_path)
        self.select_pent_roi(mmse_path)

    def select_pent_roi(self, fname):
        file_list = self.make_file_list(fname)
        # Read image
        dir_name = os.path.dirname(fname)
        save_file = os.path.join(dir_name, 'roi_file.txt')
        processed = []
        if os.path.exists(save_file):
           with open(save_file, 'r') as text_file:
               for line in text_file:
                   procf =  line.split(":")[0]
                   processed.append(procf)                   
        for proc in processed:
            file_list.remove(proc)
        for file in file_list:
            path = os.path.join(dir_name,file)
            im = cv2.imread(path)
             
            # Select ROI
            fromCenter = False
            r = cv2.selectROI(im, fromCenter)
             
            # Crop image
            imCrop = im[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
            #print(int(r[1]),int(r[1]+r[3]), int(r[0]),int(r[0]+r[2]))
            # Display cropped image
            cv2.imshow("Pentagon", imCrop)
            #cv2.waitKey(0)
            
            basename = file
            print(basename)
            #savef = os.path.join(save_folder, 'EXTRACTED_PENT-' + basename)
            k = cv2.waitKey()
            #print(k)
            if k==32 or k ==13: # SPACE TO SAVE
                text = (basename + ":ROI:" + str(int(r[1])) + "," + str(int(r[1]+r[3]))+  "," + str(int(r[0]))+  "," +  str(int(r[0]+r[2])) + "\n")
                if os.path.exists(save_file):
                    append_write = 'a' # append if already exists
                else:
                    append_write = 'w' # make a new file if not
                with open(save_file, append_write) as text_file:
                    text_file.write(text)
                #print(int(r[1]),int(r[1]+r[3]), int(r[0]),int(r[0]+r[2]))
                
                print("saved")
                cv2.destroyAllWindows()
            else:
                cv2.destroyAllWindows()

    def make_file_list(self,fname):
        dir_name = os.path.dirname(fname)
        #print(dir_name)
        f = []
        for (dirpath, dirnames, filenames) in os.walk(dir_name):
            f.append(filenames)
            break
        #print(f[0])
        return f[0]
    
root = Tk()
my_gui = RoiGui(root)
root.mainloop()



