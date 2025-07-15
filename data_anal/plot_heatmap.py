import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

def diff_matrix(true_matrix, predict_matrix, row_idx=None,col_idx=None):
    diff = np.where(
                (true_matrix[:,:]==0) & (predict_matrix[:,:]==0), 0,
            np.where(
                (true_matrix[:,:]!=0) & (predict_matrix[:,:]==0), 1,
            np.where(
                (true_matrix[:,:]==0) & (predict_matrix[:,:]!=0), 2,
            np.where(
                (true_matrix[:,:]!=0) & (predict_matrix[:,:]==true_matrix), 3, 4))))
    if row_idx:
        diff = diff[row_idx,:]
    if col_idx:
        diff = diff[:,col_idx]
    return diff

' load test labels & TRN closure results '
test_idx = [37,38,39,40,41,42,43,44,45,46,47,48, 49,50,51,52,53,54, 55,56,57, 28,29,30,58,59,60,61]
label_set = pd.read_csv('dataset/label_set_iml.csv', index_col=0)
Y_test = np.load('dataset/ncbi-sra/Y_label.npy')[test_idx][:,list(label_set['matrix_idx'])]
#Y_test = np.load('dataset/precise1k/Y_label.npy')[:,list(label_set['precise1k_idx'])]
Y_deduction = np.load('data_anal/abduction_results/Yd_ABL0_regulator.npy')
Y_pseudo = np.load('data_anal/abduction_results/Yp_ABL0_regulator.npy')

label_mask = [3, 6, 12, 14, 15, 16, 17, 18, 19, 24, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 47, 48, 51, 59, 73, 76, 79, 89, 90, 94, 99, 102, 103, 106, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 142, 143, 144, 145, 147, 148, 149, 150, 154, 158, 165, 166, 167, 168, 175, 177, 178, 180, 181, 182, 184, 185, 186, 188, 189, 190, 191, 192, 193, 194, 197, 198, 199, 200, 201, 202, 203, 219, 222, 223, 224, 225, 226, 227, 228, 229, 230, 232, 235, 236, 238, 245, 247, 250, 251, 254, 255, 256, 257, 258, 259, 260, 270, 271, 275, 276, 277, 288, 290, 291, 292, 293, 295, 296, 298, 300, 304, 306, 307, 308, 313, 314, 315, 318, 319, 320, 321, 322, 323, 324, 325, 331, 333, 334, 335, 340, 341, 351, 356, 357, 358, 359, 360, 372, 373, 376, 377, 378, 379, 380, 381, 382, 386, 387, 388, 390, 391, 392, 393, 401, 407, 409, 410, 417, 423, 442, 450, 457, 459, 460, 461, 464, 472, 473, 474, 475, 477, 478, 479, 480, 481, 482, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 500, 508, 509, 511, 512, 513, 514, 515, 521, 529, 530, 533, 543, 544, 545, 546, 553, 554, 555, 556, 560, 563, 564, 565, 566, 567, 571, 573, 575, 576, 580, 590, 591, 592, 593, 594, 595, 598, 599, 602, 604, 605, 606, 607, 608, 618, 619, 620, 623, 625, 626, 627, 629, 630, 635, 637, 638, 639, 645, 646, 649, 650, 656, 659, 664, 666, 668, 669, 670, 671, 673, 674, 675, 677, 678, 679, 680, 682, 688, 694, 695, 696, 697, 698, 699, 703, 709, 710, 715, 728, 729, 730, 731, 732, 733, 736, 740, 741, 755, 756, 757, 758, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 780, 781, 782, 797, 798, 801, 803, 804, 805, 808, 809, 811, 818, 820, 821, 822, 824, 827, 832, 837, 845, 848, 849, 851, 855, 856, 858, 861, 862, 863, 864, 865, 866, 867, 868, 869, 870, 871, 872, 874, 875, 878, 890, 891, 902, 903, 909, 910, 911, 916, 917, 924, 927, 928, 931, 935, 936, 937, 938, 939, 940, 941, 942, 943, 944, 946, 948, 956, 959, 960, 961, 962, 964, 968, 969, 970, 971, 975, 977, 979, 980, 981, 982, 984, 985, 986, 987, 988, 989, 990, 991, 992, 994, 995, 1000, 1005, 1006, 1007, 1008, 1009, 1010, 1014, 1015, 1016, 1023, 1026, 1028, 1033, 1041, 1044, 1045, 1048, 1053, 1057, 1059, 1060, 1061, 1068, 1078, 1079, 1081, 1083, 1084, 1087, 1089, 1096, 1097, 1098, 1099, 1101, 1102, 1104, 1107, 1108, 1109, 1111, 1112, 1114, 1115, 1118, 1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1134, 1135, 1138, 1143, 1144, 1146, 1147, 1148, 1149, 1150, 1151, 1152, 1154, 1155, 1156, 1157, 1158, 1159, 1163, 1164, 1166, 1167, 1171, 1172, 1186, 1187, 1188, 1189, 1190, 1191, 1193, 1194, 1195, 1196, 1197, 1198, 1199, 1200, 1202, 1206, 1207, 1218, 1220, 1221, 1222, 1223, 1224, 1225, 1227, 1228, 1229, 1230, 1231, 1232, 1233, 1236, 1237, 1238, 1239, 1240, 1241, 1242, 1243, 1245, 1246, 1247, 1248, 1249, 1250, 1253, 1254, 1255, 1256, 1274, 1275, 1276, 1280, 1281, 1282, 1291, 1297, 1302, 1304, 1305, 1306, 1308, 1309, 1316, 1317, 1318, 1319, 1323, 1324, 1325, 1326, 1327, 1328, 1329, 1330, 1335, 1336, 1337, 1338, 1340, 1343, 1344, 1350, 1351, 1353, 1355, 1357, 1358, 1359, 1361, 1362, 1363, 1364, 1370, 1371, 1372, 1373, 1374, 1375, 1376, 1377, 1378, 1379, 1380, 1384, 1385, 1387, 1394, 1398, 1400, 1409, 1410, 1411, 1412, 1413, 1414, 1415, 1416, 1417, 1418, 1419, 1420, 1421, 1422, 1423, 1424, 1425, 1427, 1428, 1431, 1432, 1434, 1435, 1436, 1437, 1440, 1447, 1449, 1453, 1454, 1455, 1456, 1457, 1458, 1459, 1461, 1463, 1464, 1472, 1477, 1478, 1479, 1481, 1483, 1484, 1485, 1486, 1487, 1493, 1494, 1495, 1499, 1502, 1504, 1505, 1506, 1509, 1510, 1511, 1512, 1513]

diff_deduction = diff_matrix(Y_test, Y_deduction, col_idx=label_mask)
diff_pseudo = diff_matrix(Y_test, Y_pseudo, col_idx=label_mask)


' init color mapping '
#cmap_colors = [
#    'white',    # -1
#    'lightgray',      # 0
#    'yellowgreen',      # 1
#    'lightblue',      # 2
#    'teal',      # 3
#    'cyan', # 4
#    'red',      # 6
#    'violet',      # 5
#    'maroon'       # 7
#]
cmap_colors = [
    'lightgreen',      # 0
    'violet',      # 1
    'red',      # 2
    'teal',      # 3
    'maroon', # 4
]
cmap = mcolors.ListedColormap(cmap_colors)
print(cmap.N)
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

legend_labels = {
    0: 'true==0 & pred==true',
    1: 'true!=0 & pred==0',
    2: 'true==0 & pred!=0',
    3: 'true!=0 & pred==true',
    4: 'true!=0 & pred!=true',
}


test_genes = ['arcZ']*12 + ['gcvB']*6 + ['micA']*3 + ['ryhB']*7

' heatmap 1 '
#for s, matrix in [('KB_deduction',diff_deduction), ('neural_pred',diff_pseudo)]:#, ('pred',diff_pred)]:
#    plt.figure(figsize=(12, 8))
#    heatmap = sns.heatmap(matrix, 
#                         cmap=cmap, 
#                         norm=norm,
#                         annot=False, 
#                         #linecolor='white',
#                         #linewidths=.2,
#                         fmt="d",
#                         cbar=False)
#    heatmap.set_yticklabels([v for i,v in enumerate(test_genes)], rotation=0)
#    
#    # Create custom legend
#    #legend_labels = {
#    #    -1: 'Not a Regulator',
#    #    0: 'Not regulated',
#    #    1: 'Consistent with KB',
#    #    2: 'Dual Regulation (0 in data)',
#    #    3: 'Dual Regulation (1 in Data)',
#    #    4: 'Dual Regulation (-1 in Data)',
#    #    5: 'Inconsist (Missing in KB)',
#    #    6: 'Inconsist (Missing in Data)',
#    #    7: 'Inconsist (Reversed)'
#    #}
#    patches = [Patch(color=cmap_colors[i], label=legend_labels[i]) for i in range(0, 5)]
#    plt.legend(handles=patches, 
#               bbox_to_anchor=(1.05, 1),
#               loc='upper left', 
#               title='Value Meanings')
#    
#    
#    #sns.heatmap(jiff_matrix, annot=False, cmap='viridis')
#    plt.title('Inconsistency with Regulation')
#    plt.xlabel('Genes')
#    plt.ylabel('Overexpression')
#    plt.tight_layout()
#    
#    
#    plt.savefig(f'data_anal/heatmap_{s}.png', dpi=600)
#    plt.show()

from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

plt.figure(figsize=(20, 16))

# 创建上三角元素
patches_upper = []
values_upper = []
for i in range(diff_deduction.shape[0]):
    for j in range(diff_deduction.shape[1]):
        triangle = Polygon([[j, i], [j+1, i], [j+1, i+1]], closed=True)
        patches_upper.append(triangle)
        values_upper.append(diff_deduction[i,j])

# 创建下三角元素
patches_lower = []
values_lower = []
for i in range(diff_deduction.shape[0]):
    for j in range(diff_deduction.shape[1]):
        triangle = Polygon([[j, i], [j, i+1], [j+1, i+1]], closed=True)
        patches_lower.append(triangle)
        values_lower.append(diff_pseudo[i,j])

# 添加上三角
pc_upper = PatchCollection(patches_upper, cmap=cmap, norm=norm, alpha=0.8)
pc_upper.set_array(np.array(values_upper))
plt.gca().add_collection(pc_upper)

# 添加下三角
pc_lower = PatchCollection(patches_lower, cmap=cmap, norm=norm, alpha=0.8)
pc_lower.set_array(np.array(values_lower))
plt.gca().add_collection(pc_lower)

plt.xlim(0, diff_deduction.shape[1])
plt.ylim(0, diff_deduction.shape[0])
plt.gca().invert_yaxis()
#plt.colorbar(pc_upper, label='Upper Triangle Values')
#plt.colorbar(pc_lower, label='Lower Triangle Values')
plt.title("Triangle Patch Heatmap")

patches = [Patch(color=cmap_colors[i], label=legend_labels[i]) for i in range(0, 5)]
plt.legend(handles=patches, 
           bbox_to_anchor=(1.05, 1),
           loc='upper left', 
           title='Value Meanings')


#plt.savefig(f'data_anal/heatmap_sra_deduce_vs_pseudo.png', dpi=600)
plt.show()
