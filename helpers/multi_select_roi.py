#!/usr/bin/env python3

import cv2, os
import numpy as np
import tkinter.filedialog
from tkinter import *
from tkinter import messagebox
import argparse


class RoiGui:
    def __init__(self, master, pos_neg_arg):
        self.pos_neg_arg = pos_neg_arg
        self.master = master
        self.master.wm_title("MMSE Pentagon ROI")


        mmse_path = tkinter.filedialog.askopenfilename()
        # print(mmse_path)
        self.select_pent_roi(mmse_path)


    def select_pent_roi(self, fname):
        count = 0
        file_list = self.make_file_list(fname)
        # Read image
        dir_name = os.path.dirname(fname)
        # Save dirs
        positive_save_dir_name = os.path.join(dir_name, 'positive')
        negative_save_dir_name = os.path.join(dir_name, 'negative')
        other_save_dir_name = os.path.join(dir_name, 'other')
        if not os.path.exists(positive_save_dir_name):
            os.makedirs(positive_save_dir_name)
        if not os.path.exists(negative_save_dir_name):
            os.makedirs(negative_save_dir_name)
        if not os.path.exists(other_save_dir_name):
            os.makedirs(other_save_dir_name)
        
        save_dir = other_save_dir_name
        if self.pos_neg_arg == 1:
            save_dir = positive_save_dir_name
        elif self.pos_neg_arg == 0:
            save_dir = negative_save_dir_name

        messagebox.showinfo("Info", "Use mouse to bound boxes: Press \n [SPACE] or [ENTER ]\n [N] to save\n [Re-click and drag] to clear\n[ESC] to exit")

        for file in file_list:
            path = os.path.join(dir_name, file)
            im = cv2.imread(path)
            basename = os.path.basename(path)
            finish = False

            boxes = []
            roi_box_count = 0
            # Select ROI
            while not finish:
                fromCenter = False

                r = cv2.selectROI(im,  fromCenter)
                k = cv2.waitKey()
                imCrop = im[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

                if (k == 32 or k == 13) and imCrop.size != 0:
                    savef = os.path.join(save_dir, f'{basename}_selected_ROI_' + str(roi_box_count) + '.jpg')
                    cv2.imwrite(savef, imCrop)
                    cv2.rectangle(im, (int(r[0]), int(r[1])), (int(r[0]+ r[2]), int(r[1] + r[3])),  (0,0,255), 2)
                    #clear the rectangle      
                    roi_box_count += 1
                    count += 1
                elif k == 27: # ESC to exit
                    finish = True
                    cv2.destroyAllWindows()

        messagebox.showinfo("Info", f"Finished! Saved {count} images to {save_dir}")
        root.destroy()

    def make_file_list(self, fname):
        dir_name = os.path.dirname(fname)
        # print(dir_name)
        f = []
        for (dirpath, dirnames, filenames) in os.walk(dir_name):
            f.append(filenames)
            break
        # print(f[0])
        return f[0]

parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('pos_neg_arg', type=int, choices=[0, 1], #
                help='A required integer argument for pos[1] or neg[0]')
args = parser.parse_args()
root = Tk()
my_gui = RoiGui(root, args.pos_neg_arg)
root.mainloop()



