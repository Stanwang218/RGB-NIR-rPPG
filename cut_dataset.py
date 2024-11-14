import os
import numpy as np
import cv2 as cv
import pandas as pd
import ast


map_list = ['RGB', 'YUV', 'CHROM', 'POS'] # input the preprocessed map list !!!!!!! important

def str_to_list( s):
    return ast.literal_eval(s)

# csv_file = pd.read_csv('/data/PreprocessedData/raw/hr.csv')
csv_file = pd.read_csv('/data/PreprocessedData/UBFC_new/data_raw.csv', index_col = 0)

chrom_path = '/data/PreprocessedData/UBFC_FULL_STMap/CHROM'
pos_path = '/data/PreprocessedData/UBFC_FULL_STMap/POS'
yuv_path = '/data/PreprocessedData/UBFC_FULL_STMap/YUV'
nir_path = '/data/PreprocessedData/UBFC_FULL_STMap/NIR'
rgb_path = '/data/PreprocessedData/UBFC_FULL_STMap/RGB'
save_path = '/data/PreprocessedData/UBFC_new'
video_len = 900
img_size = 224
df_dict = {
    'project_name': [],
    'label': []
}
d = dict()
bvp_cut = True
for i, row in csv_file.iterrows():
    label_name = os.path.basename(row['label_path'])
    bvp = np.load(row['label_path'])
    hr = str_to_list(row['heart_rate'])
    data_name = label_name.replace('label', 'input').replace('npy','png')
    project_name = data_name.split('input')[0]
    
    # chrom_data = os.path.join(chrom_path, data_name)
    # pos_data = os.path.join(pos_path, data_name)
    # chrom_pic, pos_pic = cv.imread(chrom_data), cv.imread(pos_data)
    
    for _map in map_list:
        if _map == 'CHROM':
            map_path = os.path.join(chrom_path, data_name)
            # map_save_path = os.path.join(os.path.join(save_path, 'CHROM'),  f"{new_name}.png")
        elif _map == 'POS':
            map_path = os.path.join(pos_path, data_name)
            # map_save_path = os.path.join(os.path.join(save_path, 'POS'),  f"{new_name}.png")
        elif _map == 'YUV':
            map_path = os.path.join(yuv_path, data_name)
        elif _map == 'NIR':
            map_path = os.path.join(nir_path, data_name)
        elif _map == 'RGB':
            map_path = os.path.join(rgb_path, data_name)
            
            # map_save_path = os.path.join(os.path.join(save_path, 'YUV'),  f"{new_name}.png")
        
        new_pic = cv.imread(map_path)
        print(new_pic.shape)
        print(new_pic.shape[1] // 224, len(hr))
        for i in range(min(new_pic.shape[1] // 224, len(hr))):
            new_name = f"{project_name}input{d.get(project_name, 0)}"
            map_save_path = os.path.join(os.path.join(save_path, _map),  f"{new_name}.png")
            if bvp_cut : 
                save_path_bvp = os.path.join(os.path.join(save_path, 'bvp'), f"{new_name}.npy")
                new_bvp = bvp[i*img_size: (i + 1) * img_size]
                np.save(save_path_bvp, new_bvp)
            
            
            new_pic_temp = new_pic[:, i*img_size: (i + 1) * img_size]    
            cv.imwrite(map_save_path, new_pic_temp)    
            # new_chrom_pic, new_pos_pic = chrom_pic[:, i*img_size: (i + 1) * img_size], pos_pic[:, i*img_size: (i + 1) * img_size]
            # save_path_chrom = os.path.join(os.path.join(save_path, 'CHROM'),  f"{new_name}.png")
            # save_path_pos = os.path.join(os.path.join(save_path, 'POS'), f"{new_name}.png")
            # cv.imwrite(save_path_chrom, new_chrom_pic)
            # cv.imwrite(save_path_pos, new_pos_pic)
            
            print(map_save_path)
            d[project_name] = d.get(project_name, 0) + 1
            df_dict['project_name'].append(new_name)
            df_dict['label'].append(hr[i])
        d[project_name] = 0
    #break

new_df = pd.DataFrame(df_dict)
new_df.to_csv(f"{save_path}/data.csv")
print(new_df)