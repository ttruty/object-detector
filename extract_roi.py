import corner_dets_methods
import os

dir_name = r'F:\pent_python\pent_bins\all_pos_trial'
save_file = os.path.join(dir_name, 'roi_file.txt')

for root,  dirs, files in os.walk(dir_name):
    for file in files:
        print(file)
        path = os.path.join(dir_name, file)
        detections, found_dets = corner_dets_methods.find_contours(path, 20, .05, 15, 0.0)
        print(found_dets)
        # print(found_dets)
        if found_dets != []:
            x,y,_, w,h = found_dets[0]
        text = (file + ":ROI:" + str(y) + "," + str(int(y+h)) + "," + str(x) + "," + str(int(x+w)) + "\n")
        if os.path.exists(save_file):
            append_write = 'a' # append if already exists
        else:
            append_write = 'w' # make a new file if not
        with open(save_file, append_write) as text_file:
            text_file.write(text)