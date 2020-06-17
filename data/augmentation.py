'''Augmentation methods used are Padding, Horizontal shearing in clockwise direction,
Vertical shearing  in clockwise direction, Horizontal shearing in anti-clockwise direction, 
Vertical shearing in anti-clockwise direction'''

# Importing libraries
import cv2
import numpy as np
import pandas as pd
from augmentation_methods import pad, shear

# Reading csv file for license plate bounding box coordinates
df = pd.read_csv('dataset/indian_license_plates.csv')

def augmentAndSave(df_row, aug_type):
    ''' Calls augmentation function on images and write augmented images back to disk.
        Also returns a dictionary with updated details of the augmented image''' 

    img = cv2.imread('dataset/Indian Number Plates/'+df_row[0]+'.jpeg')
    bbox = np.array((df_row[3]*df_row[1], df_row[4]*df_row[2], df_row[5]*df_row[1], df_row[6]*df_row[2]))
    
    if aug_type == 'pad':
        img_new, new_bbox = pad(img, bbox.copy(), 16)  
    elif aug_type == 'shearYNeg':
        img_new, new_bbox = shear(img, bbox.copy(), -0.1, 1)
    elif aug_type == 'shearYPos':
        img_new, new_bbox = shear(img, bbox.copy(), 0.1, 1)
    elif aug_type == 'shearXNeg':
        img_new, new_bbox = shear(img, bbox.copy(), -0.1, 0)
    elif aug_type == 'shearXPos':
        img_new, new_bbox = shear(img, bbox.copy(), 0.1, 0)
    cv2.imwrite('dataset/Indian Number Plates/'+aug_type+'_'+df.iloc[i,0]+'.jpeg', img_new)
    
    dict_for_df = {'image_name':aug_type+'_'+df_row[0], 'image_width':img_new.shape[1], 
                   'image_height':img_new.shape[0], 'top_x':new_bbox[0]/img_new.shape[1], 
                   'top_y':new_bbox[1]/img_new.shape[0], 'bottom_x':new_bbox[2]/img_new.shape[1],
                   'bottom_y':new_bbox[3]/img_new.shape[0]}
    
    return dict_for_df

No_of_images = df.shape[0]

# Padding images
for i in range(0, No_of_images):
    if i%10 == 0: # Images skipped for vaildation set
        continue
    else:
        df = df.append(augmentAndSave(df.iloc[i, :], 'pad'), ignore_index=True)

No_of_images = 2*(No_of_images - (int(No_of_images/10) + 1)) + (int(No_of_images/10) + 1)

# Shearing images
for i in range(0, No_of_images):
    if i%10 == 0 and i<236: # Images skipped for vaildation set
        continue
    else:
        df = df.append(augmentAndSave(df.iloc[i, :], 'shearYNeg'), ignore_index=True)
        df = df.append(augmentAndSave(df.iloc[i, :], 'shearYPos'), ignore_index=True)
        df = df.append(augmentAndSave(df.iloc[i, :], 'shearXNeg'), ignore_index=True)
        df = df.append(augmentAndSave(df.iloc[i, :], 'shearXPos'), ignore_index=True)

# Saving updated dataframe back to disk
df = df.sample(frac=1).reset_index(drop=True)
df.to_csv('dataset/indian_license_plates.csv', index = None)