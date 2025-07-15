import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np
from torch.distributions import Bernoulli

from egoal.reasoner import RegualtoryKB

class ReflectNN(nn.Module):
    """ Network Structure of Base Learner with Reflect Output (RL) """

    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Args:
            input_dim:
            hidden_dim:
            output_dim:
        """
        super(ReflectNN, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        #TODO use MLP temp, GNN embd?
        self.relu = nn.ReLU()

        ' Head 1: Classification (y) '
        self.y_head = nn.Linear(hidden_dim, output_dim*3)
        self.softmax = nn.Softmax(dim=-1)

        ' Head 2: REINFORCE (r): Logits for binary actions '
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.r_head = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        emb = self.embedding(x)

        ' clf head '
        output_y = self.y_head(emb)
        output_y = self.softmax(output_y.view(output_y.shape[0], -1, 3))

        ' action head '
        output_r = self.r_head(self.relu(self.fc(emb)))
        output_r = self.sigmoid(output_r)

        return output_y, output_r

    def predict(self,x):
        output_y, _ = self.forward(x)
        return torch.argmax(output_y, dim=-1) -1

    def reflection(self, x):
        _, output_r = self.forward(x)
        return torch.round(output_r)

class ReflectLearner():
    def __init__(self,
        input_dim,
        output_dim,
        hidden_dim = 64,
        use_gpu = False,
        log_path = '',
    ) -> None:
        '''
        Args:
            KB:
            log_path (optional):
        '''

        ' 4639 genes of whole genome, 241 output genes '
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        ' weight of classes for CE loss '
        self.clf_weight = torch.Tensor([.4,.2,.4])

        self.model = ReflectNN(self.input_dim, self.hidden_dim,  self.output_dim)
        self.train_loader = None
        self.test_loader = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'cuda availability: {torch.cuda.is_available()}')
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.model = self.model.to(self.device)
            self.clf_weight = self.clf_weight.to(self.device)

        self.log_path = log_path


    def consistency_reward(self,
                           KB: RegualtoryKB,
                           x: torch.Tensor,
                           y_probs: torch.Tensor,
                           r_binary: torch.Tensor):
        '''
        KB:
        x:
        y_probs:
        r_binary:
        Reward = count_nonzero(y_binary & r_actions - x_binary)
        '''

        # TODO
        y = torch.abs(torch.argmax(y_probs, dim=-1) -1)
        r_binary = r_binary.bool()
        
        #labels = [6, 7, 9, 43, 44, 61, 68, 72, 82, 84, 85, 86, 87, 91, 98, 105, 124, 141, 142, 143, 144, 150, 194, 220, 221, 232, 260, 268, 287, 288, 289, 290, 291, 292, 294, 320, 322, 345, 353, 355, 387, 397, 406, 407, 409, 411, 419, 455, 483, 485, 487, 488, 489, 490, 498, 501, 510, 525, 528, 535, 536, 545, 547, 548]
        #labels = set([3, 6, 12, 14, 15, 16, 17, 18, 19, 24, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 47, 48, 51, 59, 73, 76, 79, 89, 90, 94, 99, 102, 103, 106, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 142, 143, 144, 145, 147, 148, 149, 150, 154, 158, 165, 166, 167, 168, 175, 177, 178, 180, 181, 182, 184, 185, 186, 188, 189, 190, 191, 192, 193, 194, 197, 198, 199, 200, 201, 202, 203, 219, 222, 223, 224, 225, 226, 227, 228, 229, 230, 232, 235, 236, 238, 245, 247, 250, 251, 254, 255, 256, 257, 258, 259, 260, 270, 271, 275, 276, 277, 288, 290, 291, 292, 293, 295, 296, 298, 300, 304, 306, 307, 308, 313, 314, 315, 318, 319, 320, 321, 322, 323, 324, 325, 331, 333, 334, 335, 340, 341, 351, 356, 357, 358, 359, 360, 372, 373, 376, 377, 378, 379, 380, 381, 382, 386, 387, 388, 390, 391, 392, 393, 401, 407, 409, 410, 417, 423, 442, 450, 457, 459, 460, 461, 464, 472, 473, 474, 475, 477, 478, 479, 480, 481, 482, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 500, 508, 509, 511, 512, 513, 514, 515, 521, 529, 530, 533, 543, 544, 545, 546, 553, 554, 555, 556, 560, 563, 564, 565, 566, 567, 571, 573, 575, 576, 580, 590, 591, 592, 593, 594, 595, 598, 599, 602, 604, 605, 606, 607, 608, 618, 619, 620, 623, 625, 626, 627, 629, 630, 635, 637, 638, 639, 645, 646, 649, 650, 656, 659, 664, 666, 668, 669, 670, 671, 673, 674, 675, 677, 678, 679, 680, 682, 688, 694, 695, 696, 697, 698, 699, 703, 709, 710, 715, 728, 729, 730, 731, 732, 733, 736, 740, 741, 755, 756, 757, 758, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 780, 781, 782, 797, 798, 801, 803, 804, 805, 808, 809, 811, 818, 820, 821, 822, 824, 827, 832, 837, 845, 848, 849, 851, 855, 856, 858, 861, 862, 863, 864, 865, 866, 867, 868, 869, 870, 871, 872, 874, 875, 878, 890, 891, 902, 903, 909, 910, 911, 916, 917, 924, 927, 928, 931, 935, 936, 937, 938, 939, 940, 941, 942, 943, 944, 946, 948, 956, 959, 960, 961, 962, 964, 968, 969, 970, 971, 975, 977, 979, 980, 981, 982, 984, 985, 986, 987, 988, 989, 990, 991, 992, 994, 995, 1000, 1005, 1006, 1007, 1008, 1009, 1010, 1014, 1015, 1016, 1023, 1026, 1028, 1033, 1041, 1044, 1045, 1048, 1053, 1057, 1059, 1060, 1061, 1068, 1078, 1079, 1081, 1083, 1084, 1087, 1089, 1096, 1097, 1098, 1099, 1101, 1102, 1104, 1107, 1108, 1109, 1111, 1112, 1114, 1115, 1118, 1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1134, 1135, 1138, 1143, 1144, 1146, 1147, 1148, 1149, 1150, 1151, 1152, 1154, 1155, 1156, 1157, 1158, 1159, 1163, 1164, 1166, 1167, 1171, 1172, 1186, 1187, 1188, 1189, 1190, 1191, 1193, 1194, 1195, 1196, 1197, 1198, 1199, 1200, 1202, 1206, 1207, 1218, 1220, 1221, 1222, 1223, 1224, 1225, 1227, 1228, 1229, 1230, 1231, 1232, 1233, 1236, 1237, 1238, 1239, 1240, 1241, 1242, 1243, 1245, 1246, 1247, 1248, 1249, 1250, 1253, 1254, 1255, 1256, 1274, 1275, 1276, 1280, 1281, 1282, 1291, 1297, 1302, 1304, 1305, 1306, 1308, 1309, 1316, 1317, 1318, 1319, 1323, 1324, 1325, 1326, 1327, 1328, 1329, 1330, 1335, 1336, 1337, 1338, 1340, 1343, 1344, 1350, 1351, 1353, 1355, 1357, 1358, 1359, 1361, 1362, 1363, 1364, 1370, 1371, 1372, 1373, 1374, 1375, 1376, 1377, 1378, 1379, 1380, 1384, 1385, 1387, 1394, 1398, 1400, 1409, 1410, 1411, 1412, 1413, 1414, 1415, 1416, 1417, 1418, 1419, 1420, 1421, 1422, 1423, 1424, 1425, 1427, 1428, 1431, 1432, 1434, 1435, 1436, 1437, 1440, 1447, 1449, 1453, 1454, 1455, 1456, 1457, 1458, 1459, 1461, 1463, 1464, 1472, 1477, 1478, 1479, 1481, 1483, 1484, 1485, 1486, 1487, 1493, 1494, 1495, 1499, 1502, 1504, 1505, 1506, 1509, 1510, 1511, 1512, 1513])
        labels = [1, 2, 23, 82, 83, 84, 97, 107, 159, 160, 161, 162, 163, 243, 246, 263, 264, 265, 266, 267, 268, 269, 272, 284, 286, 301, 302, 303, 328, 337, 338, 342, 343, 348, 353, 362, 363, 418, 432, 436, 443, 444, 448, 458, 471, 501, 549, 550, 558, 561, 562, 570, 579, 584, 585, 586, 587, 588, 589, 621, 622, 632, 633, 634, 641, 661, 683, 705, 719, 720, 734, 748, 752, 753, 762, 763, 764, 765, 766, 799, 833, 904, 925, 934, 945, 950, 951, 952, 953, 954, 955, 965, 997, 998, 999, 1003, 1011, 1012, 1013, 1029, 1030, 1031, 1032, 1043, 1052, 1066, 1071, 1091, 1094, 1095, 1132, 1133, 1136, 1140, 1141, 1142, 1175, 1178, 1183, 1204, 1209, 1213, 1215, 1251, 1252, 1262, 1320, 1321, 1322, 1339, 1342, 1347, 1348, 1349, 1366, 1381, 1382, 1388, 1389, 1391, 1392, 1402, 1403, 1429, 1430, 1438, 1439, 1442, 1443, 1452, 1462, 1474, 1475, 1497, 1498, 1507]
        #labels = set([6, 18, 19, 47, 109, 114, 146, 149, 166, 172, 177, 197, 296, 305, 312, 313, 323, 335, 389, 401, 409, 422, 440, 448, 472, 475, 480, 509, 527, 557, 569, 578, 647, 654, 659, 690, 700, 707, 765, 768, 770, 776, 791, 797, 828, 846, 871, 907, 977, 987, 1048, 1076, 1095, 1125, 1138, 1227, 1230, 1242, 1342, 1357, 1382, 1456, 1473, 1486, 1490, 1499])
        #labels = set([66, 70, 160, 162, 163, 169, 213, 220, 262, 264, 265, 385, 414, 421, 426, 444, 501, 541, 561, 578, 616, 633, 684, 689, 702, 707, 752, 873, 887, 904, 920, 966, 997, 999, 1024, 1029, 1030, 1031, 1032, 1036, 1038, 1062, 1066, 1132, 1211, 1273, 1277, 1285, 1399, 1444])
        label_restriction = torch.count_nonzero(r_binary[:,[i for i in range(r_binary.shape[1]) if i not in labels]])
        #violated = - KB.violated(Y=y, X=x, mask=~r_binary)
        violated = - KB.violated(Y=y, X=x, mask=~r_binary) - label_restriction#torch.count_nonzero(r_binary) - label_restriction
        #violated /= torch.count_nonzero(~r_binary)
        return violated
    

    def load_data(self,
                  X_train: torch.Tensor,
                  Y_train: torch.Tensor,
                  X_test: torch.Tensor,
                  Y_test: torch.Tensor,
                  batch_size=64):
        ''' define train & test data loader '''

        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(Y_train) > 0
        assert len(Y_test) > 0
        train_dataset = TensorDataset(X_train, Y_train)
        test_dataset = TensorDataset(X_test, Y_test)
        
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        ' reset classification loss weight with new Y_train '
        flat_y = Y_train.flatten()
        weights = [1/(torch.sum(flat_y==-1).item() + 1e-6),
                   1/(torch.sum(flat_y==0).item() + 1e-6),
                   1/(torch.sum(flat_y==1).item() + 1e-6)]
        #self.clf_weight = torch.Tensor(weights) / sum(weights)
        #if self.use_gpu:
        #    self.clf_weight = self.clf_weight.to(self.device)


    def train(
        self, 
        KB: RegualtoryKB,
        epochs= 10, 
        C= 100,
        lr= 1e-3, 
        gamma= 0.95
    ):
        '''
        Train the Clf + Refl Model
        Args:
            epochs: 
            lr:
            gamma: discount factor for RL baseline reward
        '''

        ''' Training loop '''
        criterion = nn.CrossEntropyLoss(weight=self.clf_weight)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()

        baseline = 0.

        for epoch in range(epochs):

            ' get the global reward '
            r_actions_batch = []
            violated, r_nonzero = 0., 0.
            for X_batch, _ in self.train_loader:

                output_y, output_r = self.model(X_batch)

                dist = Bernoulli(output_r)
                r_actions = dist.sample()  # Shape: (batch_size, output_dim)
                r_actions_batch.append(r_actions)

                violated += self.consistency_reward(KB, X_batch, output_y, r_actions).detach().item()
                r_nonzero += torch.count_nonzero(1-r_actions).detach().item()

            #reward = violated / (r_nonzero + 1e-6)
            reward = violated

            ' opt local loss_y & global loss_r '
            loss_y, loss_r = 0.,0.
            for (X_batch, Y_batch), r_actions in zip(self.train_loader, r_actions_batch):
                output_y, output_r = self.model(X_batch)
                Y_batch = Y_batch.to(int)+1

                ' CE loss '
                loss_y += criterion(output_y.view(-1,3), Y_batch.view(-1))

                ' RL for discrete action opt '
                ' sample from Ber distribution, '
                dist = Bernoulli(output_r)

                ' Update baseline (exponential moving average) '
                baseline = gamma * baseline + (1 - gamma) * reward

                # REINFORCE loss
                log_probs = dist.log_prob(r_actions).sum(dim=1)
                loss_r += -torch.mean((reward - baseline) * log_probs)
    
    
            ' backprop '
            if epoch % 1000 == 0:
                C1, C2 = 1, 100
            else:
                C1, C2 = 0, 100

            total_loss = C1 * loss_y + C2 * loss_r  # Scale REINFORCE loss to balance
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if (epoch+1)%100 == 0:
                print(f"Epoch {epoch+1}, Total loss: {total_loss.item():.4f}, CE loss: {loss_y.item():.4f}, RL loss: {loss_r.item():.4f}, Reward: {reward:.4f}")


    #########################################################################

    def eval(self):
        assert self.test_loader != None

        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            f1_micro2 = 0
            f1_macro2 = 0

            Y_test, Y_pred, Y_prob = [],[],[]

            for X_batch, Y_batch in self.test_loader:
                outputs = self.model.predict(X_batch)

                Y_test.append(Y_batch)
                Y_pred.append(outputs)
            #    Y_prob.append(self.predict_prob(X_batch).max(dim=-1).values)
                total += Y_batch.size(0)
                correct += (outputs == Y_batch).sum(dim=0)

            Y_test = torch.concat(Y_test, dim=0)
            Y_pred = torch.concat(Y_pred, dim=0)

            if self.use_gpu:
                Y_test = Y_test.cpu()
                Y_pred = Y_pred.cpu()
                correct = correct.cpu()
            #Y_prob = torch.concat(Y_prob, dim=0)

            ''' compute total confusion matrix '''
            flat_y_t = Y_test.flatten()
            flat_y_p = Y_pred.flatten()
            confusion = confusion_matrix(flat_y_t, flat_y_p, labels=[-1, 0,1])
            confusion = confusion / confusion.sum().sum()
            f1_macro = f1_score(flat_y_t, flat_y_p, average='macro') # micro on labels, macro on classes
            f1_micro = f1_score(flat_y_t, flat_y_p, average='micro') # micro on labels, micro on classes

            ' compute weighted f1 by ground truth proportion '
            weights = [1/(torch.sum(flat_y_t==-1).item() + 1e-6),
                       1/(torch.sum(flat_y_t==0).item() + 1e-6),
                       1/(torch.sum(flat_y_t==1).item() + 1e-6)]
            weights = torch.Tensor(weights) / sum(weights)
            f1_class = f1_score(flat_y_t, flat_y_p, average=None)
            f1_weighted = sum([f1*w for f1,w in zip(f1_class,weights)])
            #f1_weighted = f1_class[0]*weights[0] + f1_class[2]*weights[2]

            for label_idx in range(Y_test.shape[1]):
                f1_macro2 += f1_score(Y_test[:,label_idx], Y_pred[:,label_idx], average='macro') # macro on labels, macro on classes
                f1_micro2 += f1_score(Y_test[:,label_idx], Y_pred[:,label_idx], average='micro') # macro on labels, macro on classes
                    
            f1_macro2 /= Y_test.shape[1]
            f1_micro2 /= Y_test.shape[1]

            ''' compute acc & confusion matrix on each gene '''
            per_label_accuracy = correct / total
            #f1 /= Y_test.shape[1]
            if self.log_path != '':
                with open(self.log_path,'a') as f:
                    f.write('label ')
                    for i in range(len(per_label_accuracy)):
                        f.write(f'{i:8}\t')
                    f.write('\n   acc ')
                    for acc in per_label_accuracy:
                        f.write(f'{acc * 100:7.2f}%\t')
                    f.write('\n    f1 ')
                    for label_idx in range(Y_test.shape[1]):
                        f.write(f"{f1_score(Y_test[:,label_idx], Y_pred[:,label_idx], average='macro'):8.4f}\t")

                    #for data_idx in range(Y_test.shape[0]):
                    #    f.write(f'\npred{data_idx:2} ')
                    #    for y_pred in Y_pred[data_idx]:
                    #        f.write(f'{y_pred:8}\t')
                    #    f.write(f'\nprob{data_idx:2} ')
                    #    for y_prob in Y_prob[data_idx]:
                    #        f.write(f'{y_prob:8.2f}\t')
                    #    f.write(f'\ntest{data_idx:2} ')
                    #    for y_test in Y_test[data_idx]:
                    #        f.write(f'{y_test:8}\t')

                    f.write(f'\n------\nconfusion matrix:\n{confusion}\n')
                    f.write(f'macro f1: {f1_macro}\n')
                    f.write(f'micro f1: {f1_micro}\n')
                    f.write(f'weighted f1: {f1_weighted}\n')
                    f.write(f'class -1 f1: {f1_class[0]}\n')
                    f.write(f'class  0 f1: {f1_class[1]}\n')
                    f.write(f'class  1 f1: {f1_class[2]}\n')
                    f.write(f'average label-wise acc: {np.mean(np.array(per_label_accuracy))*100:.2f}%\n')
            else:
                print(f'Average Per-label Acc: {np.mean(np.array(per_label_accuracy))*100:.2f}%\n')

            return f1_macro

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def predict(self, x: torch.Tensor):
        return self.model.predict(x)

    def predict_prob(self, x: torch.Tensor):
        outputs, _ = self.model(x)
        return outputs



if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)

    X_train = torch.tensor(np.load('dataset/precise1k/X_label.npy'), dtype=torch.float32)
    Y_train = torch.tensor(np.load('dataset/precise1k/Y_train.npy'), dtype=int)
    X_test = torch.tensor(np.load('dataset/ncbi-sra/X_label.npy'), dtype=torch.float32)
    Y_test = torch.tensor(np.load('dataset/ncbi-sra/Y_train.npy'), dtype=int)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, Y_train = X_train.to(device), Y_train.to(device)
    X_test, Y_test = X_test.to(device), Y_test.to(device)


    from scipy.sparse import load_npz
    KB = torch.tensor(load_npz('rules/regu_pos_clo.npz').toarray()).float()

    import pandas as pd
    label_set = pd.read_csv('dataset/ncbi-sra/label_set.csv')
    idx_list = list(label_set['matrix_idx'])
    KB = KB[:,idx_list]

    input_dim = X_train.shape[1]
    output_dim = Y_train.shape[1]
    hidden_dim = 128
    batch_size = 64
    

    # Initialize model
    #data_loader = DataLoader(TensorDataset(X_train,Y_train), batch_size=batch_size, shuffle=True)
    #learner.train_loader = data_loader

    # Train
    learner = ReflectLearner(KB, use_gpu=True, log_path='log.txt')
    learner.load_data(X_train, Y_train, X_test, Y_test, batch_size=batch_size)
    print(learner.eval())
    learner.train(epochs=10000, lr=1e-4)
    print(learner.eval())
