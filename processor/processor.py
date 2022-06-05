import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
import numpy as np
import copy
from datasets.bases import ImageDataset
import torchvision.transforms as T
from timm.data.random_erasing import RandomErasing
from torch.utils.data import DataLoader
from datasets.sampler import RandomIdentitySampler
from datasets.make_dataloader import train_collate_fn
from model.make_model import weights_init_classifier
from sklearn.cluster import DBSCAN
import collections
from loss.triplet_loss import TripletLoss
import torch.nn.functional as F

class IterLoader:
    def __init__(self, loader):
        self.loader = loader
        self.iter = iter(self.loader)

    def next_one(self):
        try:
            return next(self.iter)
        except:
            self.iter = iter(self.loader)
            return next(self.iter)

def kl_loss(a,b):
    loss = F.kl_div(a.softmax(dim=-1).log(), b.softmax(dim=-1), reduction='mean')
    return loss

def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat.cpu().numpy()

def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat

def mmm(per_cam_features):
    per_cam_features_list = []
    for i in range(len(per_cam_features)):
        per_cam_features_list.append(torch.cat(per_cam_features[i], dim=0))


    return per_cam_features_list

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def dist(per_cam_features_list):
    dists = {}
    num = len(per_cam_features_list)
    for i in range(num):
        for j in range(num):
            key = str(i)+str(j)
            if i!=j and i<j:
                dists[key]=normalization(euclidean_distance(per_cam_features_list[i], per_cam_features_list[j]))
    return dists

def a_dist(dists,key,thres):
    dist = dists[key]
    zero = np.zeros_like(dist)
    one = np.ones_like(dist)
    a = np.where(dist > thres, zero, dist)
    b = np.where(a > 0, one, a)
    index = np.argwhere(b==1)
    return index

def thres_index(dist,thres):
    zero = np.zeros_like(dist)
    one = np.ones_like(dist)
    a = np.where(dist > thres, zero, dist)
    b = np.where(a > 0, one, a)
    index = np.argwhere(b==1)
    return index

# def dd(indexs):
#     # H,W = indexs.shape[0],indexs.shape[1]
#     for i in range(64):
#         index = np.argwhere(indexs[0] == 0)





def sim_label(dists,camera_num,thres):
    # lenth = len(6)
    img_dict = {}
    for i in range(camera_num):
        for j in range(camera_num):
            if i != j and i<j:
                key = str(i) + str(j)
                if key not in img_dict.keys():
                    img_dict[key] = {}
                index_tensor = a_dist(dists,key,thres)

                num = index_tensor.shape[0]
                for n in range(num):
                    index = index_tensor[n]
                    a,b =index[0],index[1]
                    if a not in img_dict[key].keys():
                        img_dict[key][a] = []
                    img_dict[key][a].append(b)

    return  img_dict

def merge(img_dict,per_c_pid_dict,camera_num):
    # new_c_class_dict = copy.deepcopy(img_dict)
    new_c_class_dict = {}
    for i in range(camera_num):
        for j in range(camera_num):
            if i != j and i < j:
                key = str(i) + str(j)
                if key not in new_c_class_dict.keys():
                    new_c_class_dict[key] = {}
    for i in range(camera_num):
        for j in range(camera_num):
            if i != j and i < j:
                key = str(i) + str(j)
                for per_key in per_c_pid_dict[i].keys():
                    class_index = per_c_pid_dict[i][per_key]
                    for index in class_index:
                        if index in img_dict[key].keys():
                            if per_key not in new_c_class_dict[key].keys():
                                new_c_class_dict[key][per_key] = []
                            new_c_class_dict[key][per_key]=new_c_class_dict[key][per_key]+img_dict[key][index]
                            new_c_class_dict[key][per_key] = list(set(new_c_class_dict[key][per_key]))
    return new_c_class_dict

def merge2(img_dict,per_c_pid_dict,camera_num):
    # new_c_class_dict = copy.deepcopy(img_dict)
    new_c_class_dict = {}
    for i in range(camera_num):
        for j in range(camera_num):
            if i != j and i < j:
                key = str(i) + str(j)
                if key not in new_c_class_dict.keys():
                    new_c_class_dict[key] = {}
    for i in range(camera_num):
        for j in range(camera_num):
            if i != j and i < j:
                key = str(i) + str(j)
                for child_key in img_dict[key].keys():
                    img_lists = img_dict[key][child_key]
                    id_list = []
                    for img_index in img_lists:
                        id = per_c_pid_dict[j][img_index]
                        id_list.append(id)
                    id_list=list(set(id_list))
                    new_c_class_dict[key][child_key]=id_list
    return new_c_class_dict

def find_repeat(source,elmt): # The source may be a list or string.
    elmt_index=[]
    s_index = 0;e_index = len(source)
    while(s_index < e_index):
        try:
            temp = source.index(elmt,s_index,e_index)
            elmt_index.append(temp)
            s_index = temp + 1
        except ValueError:
            break
    return elmt_index

def nnn(per_cam_pid):
    len_per_cam_pid = len(per_cam_pid)
    per_c_pid_dict = {}
    for i in range(len_per_cam_pid):
        camera_pids = per_cam_pid[i]
        camera_pid_set = set(camera_pids)
        per_c_pid_dict[i] = {}
        for pid in camera_pid_set:
            indexx = find_repeat(camera_pids, pid)
            if pid not in per_c_pid_dict[i].keys():
                per_c_pid_dict[i][pid] = indexx
    return per_c_pid_dict

def intersec(a,b):
    return  list(set(a).intersection(set(b)))
def different(a,b):
    return (list(set(b).difference(set(a)))) #b have a not

def hard_sample_inference(dict,camera_num):
    # keys = []
    dict2 = copy.deepcopy(dict)
    for i in range(camera_num):
        for j in range(camera_num):
            # and i < j
            if i != j and i < j :
                key1 = str(i) + str(j)

                for key_1 in dict[key1].keys():
                    id_list = dict[key1][key_1]
                    if len(id_list)!=1:
                        for key_2 in dict[key1].keys():
                            if len(dict[key1][key_2])==1:
                                result = intersec(dict[key1][key_1],dict[key1][key_2])
                                # if len(result) ==1:
                                dict2[key1][key_1]=different(result,dict2[key1][key_1])
                                if len(dict2[key1][key_1])==1:
                                    break

                for key_4 in list(dict2[key1]):
                    try:
                        value = dict2[key1][key_4]
                    except:
                        continue
                    if len(value)!=1:
                        dict2[key1].pop(key_4)
                    else:
                        for key_5 in list(dict2[key1]):
                            try:
                                if key_4!=key_5 and dict2[key1][key_4][0]==dict2[key1][key_5][0]:
                                    del dict2[key1][key_4]
                                    del dict2[key1][key_5]
                            except:
                                continue


    return dict2

def make_id(dict_id,per_cam_fname,per_cam_pid,per_c_pid_dict,camera_num):
    camera_pid ={}
    camera_pid_index={}
    for i in range(camera_num):
        camera_pid[i]={}
        camera_pid_index[i]={}
        for j in range(camera_num):
            if i != j and i < j:
                key = str(i) + str(j)
                for k in dict_id[key].keys():
                    if k not in camera_pid[i].keys():
                        camera_pid[i][k]=[]
                        camera_pid_index[i][k]=[]
                    if i+1==j:
                        img_index1 = per_c_pid_dict[i][k]
                        for index1 in img_index1:
                            img_path1 = per_cam_fname[i][index1]
                            sc_id = str(i) + '_' + str(k)
                            camera_pid[i][k].append([img_path1,sc_id])
                            camera_pid_index[i][k].append(sc_id)
                    j_id = dict_id[key][k][0] #查询k在目标相机下匹配上的身份id
                    img_index = per_c_pid_dict[j][j_id] #查询目标相机身份ID j_id 对应的图片索引
                    for index in img_index:
                        img_path = per_cam_fname[j][index]
                        c_id = str(j)+'_'+str(j_id)
                        camera_pid[i][k].append([img_path,c_id])
                        camera_pid_index[i][k].append(c_id)#遍历j_id 对应的图片索引 ，将对应的图片【图片、路径】存放list
        # camera_pid[i]=list(set(camera_pid[i]))
    return camera_pid,camera_pid_index

def inter(a,b):
    return list(set(a)&set(b))

def bbb(camera_pid,camera_pid_index,k,k2,v2,repeat_list,class_list):
    for key in camera_pid_index.keys():
        if key!=k:
            for key2 in camera_pid_index[key].keys():
                input_key = str(k)+'_'+str(k2)
                if input_key not in repeat_list:
                    if input_key not in class_list.keys():
                        class_list[input_key]=[]
                        class_list[input_key].append(camera_pid[k][k2])
                    value = camera_pid_index[key][key2]
                    inter_list = inter(v2,value)
                    if len(inter_list)>0:
                        repeat_list.append(str(key)+'_'+str(key2))
                        class_list[input_key].append(camera_pid[key][key2])
    return class_list,repeat_list



def make_id2(camera_pid,camera_pid_index,camera_num):
    repeat_list=[]
    class_list = {}
    for k,v in camera_pid_index.items():
        for k2,v2 in v.items():
            class_list,repeat_list = bbb(camera_pid,camera_pid_index,k,k2,v2,repeat_list,class_list)
    return class_list

def make_dataset(class_list):
    i = 0
    dataset=[]
    for k in list(class_list.keys()):
        num=0
        v = class_list[k]
        for j in range(len(v)):
            num = num + len(v[j])
        if num<2:
            del class_list[k]

    for key,v in class_list.items():
        for j in range(len(v)):
            for k in range(len(v[j])):
                value = v[j][k]
                img_path = value[0]
                c_id = value[1].split('_')[0]
                pid = i
                dataset.append((img_path,pid,int(c_id),1))
        i=i+1
    return dataset

def make_train_transforms(cfg):
    train_transforms = T.Compose([
                T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
                # T.RandomGrayscale(p=0.5),
                T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
                T.Pad(cfg.INPUT.PADDING),
                T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
                T.ToTensor(),
                T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),

                RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
                # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
            ])
    return train_transforms

def make_test_transforms(cfg):
    test_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])
    return test_transforms

def compute_jaccard_dist(target_features, k1=20, k2=6, print_flag=True,
                         lambda_value=0, source_features=None, use_gpu=False):
    end = time.time()
    N = target_features.size(0)
    if (use_gpu):
        # accelerate matrix distance computing
        target_features = target_features.cuda()
        if (source_features is not None):
            source_features = source_features.cuda()

    if ((lambda_value>0) and (source_features is not None)):
        M = source_features.size(0)
        sour_tar_dist = torch.pow(target_features, 2).sum(dim=1, keepdim=True).expand(N, M) + \
                        torch.pow(source_features, 2).sum(dim=1, keepdim=True).expand(M, N).t()
        sour_tar_dist.addmm_(1, -2, target_features, source_features.t())
        sour_tar_dist = 1-torch.exp(-sour_tar_dist)
        sour_tar_dist = sour_tar_dist.cpu()
        source_dist_vec = sour_tar_dist.min(1)[0]
        del sour_tar_dist
        source_dist_vec /= source_dist_vec.max()
        source_dist = torch.zeros(N, N)
        for i in range(N):
            source_dist[i, :] = source_dist_vec + source_dist_vec[i]
        del source_dist_vec


    if print_flag:
        print('Computing original distance...')

    original_dist = torch.pow(target_features, 2).sum(dim=1, keepdim=True) * 2
    original_dist = original_dist.expand(N, N) - 2 * torch.mm(target_features, target_features.t())
    original_dist /= original_dist.max(0)[0]
    original_dist = original_dist.t()
    initial_rank = torch.argsort(original_dist, dim=-1)

    original_dist = original_dist.cpu()
    initial_rank = initial_rank.cpu()
    all_num = gallery_num = original_dist.size(0)

    del target_features
    if (source_features is not None):
        del source_features

    if print_flag:
        print('Computing Jaccard distance...')

    nn_k1 = []
    nn_k1_half = []
    for i in range(all_num):
        nn_k1.append(k_reciprocal_neigh(initial_rank, i, k1))
        nn_k1_half.append(k_reciprocal_neigh(initial_rank, i, int(np.around(k1/2))))

    V = torch.zeros(all_num, all_num)
    for i in range(all_num):
        k_reciprocal_index = nn_k1[i]
        k_reciprocal_expansion_index = k_reciprocal_index
        for candidate in k_reciprocal_index:
            candidate_k_reciprocal_index = nn_k1_half[candidate]
            if (len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index)) > 2/3*len(candidate_k_reciprocal_index)):
                k_reciprocal_expansion_index = torch.cat((k_reciprocal_expansion_index,candidate_k_reciprocal_index))

        k_reciprocal_expansion_index = torch.unique(k_reciprocal_expansion_index)  ## element-wise unique
        weight = torch.exp(-original_dist[i,k_reciprocal_expansion_index])
        V[i,k_reciprocal_expansion_index] = weight/torch.sum(weight)

    if k2 != 1:
        k2_rank = initial_rank[:,:k2].clone().view(-1)
        V_qe = V[k2_rank]
        V_qe = V_qe.view(initial_rank.size(0),k2,-1).sum(1)
        V_qe /= k2
        V = V_qe
        del V_qe
    del initial_rank

    invIndex = []
    for i in range(gallery_num):
        invIndex.append(torch.nonzero(V[:,i])[:,0])  #len(invIndex)=all_num

    jaccard_dist = torch.zeros_like(original_dist)
    for i in range(all_num):
        temp_min = torch.zeros(1,gallery_num)
        indNonZero = torch.nonzero(V[i,:])[:,0]
        # indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+ torch.min(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
        jaccard_dist[i] = 1-temp_min/(2-temp_min)
    del invIndex

    del V

    pos_bool = (jaccard_dist < 0)
    jaccard_dist[pos_bool] = 0.0
    if print_flag:
        print ("Time cost: {}".format(time.time()-end))

    if (lambda_value>0):
        return jaccard_dist*(1-lambda_value) + source_dist*lambda_value
    else:
        return jaccard_dist

def k_reciprocal_neigh(initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i,:k1+1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
    fi = torch.nonzero(backward_k_neigh_index==i)[:,0]
    return forward_k_neigh_index[fi]

def get_cluster(rerank_dist,n):
    tri_mat = np.triu(rerank_dist, 1)
    tri_mat = tri_mat[np.nonzero(tri_mat)] # tri_mat.dim=1
    tri_mat = np.sort(tri_mat,axis=None)
    rho = 0.1e-3
    rho = rho*n
    top_num = np.round(rho*tri_mat.size).astype(int)
    if top_num==0:
        top_num = np.round(rho*16*tri_mat.size).astype(int)
    eps = tri_mat[:top_num].mean()
    print('eps for cluster: {:.3f}'.format(eps))
    # eps = 0.1
    if n<=1:
        cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)
    else:
        cluster = DBSCAN(eps=eps, min_samples=2, metric='precomputed', n_jobs=-1)
    return cluster

def intra_camera_leanring(epoch,cluster,model,t_train_loader,device,camera_num,train_transforms,test_transforms,cfg):


    per_cam_features, per_cam_fname, per_cam_pid = extract_features_per_cam(model, t_train_loader, device,is_c_intra=True)

    rerank_dists = []
    num_ids = []
    c_labels = []
    camera_intra_train_loader = []
    cluster_centers = []
    t_loader = []
    for c_i in range(len(per_cam_features)):
        per_cam_features[c_i]=torch.stack(per_cam_features[c_i])
        rerank_dist = compute_jaccard_dist(per_cam_features[c_i], use_gpu=True).numpy()
        rerank_dists.append(rerank_dist)
        if epoch%10== 1 and epoch<=160:
            cluster = get_cluster(rerank_dist,(epoch//10)+1)

        print('Clustering and labeling...')
        labels = cluster.fit_predict(rerank_dist)
        num_id = len(set(labels)) - (1 if -1 in labels else 0)
        num_ids.append(num_id)
        c_labels.append(labels)
        dataset_c_intra=[]
        t_dataset = []
        t_dataset_test = []
        cluster_center = collections.defaultdict(list)
        NO_i = 0
        for img_i in range(len(per_cam_fname[c_i])):
            fname = per_cam_fname[c_i][img_i]
            label = labels[img_i]
            t_dataset_test.append((fname,label,c_i,1))
            if label==-1:
                NO_i = NO_i + 1
                labelc=num_id+NO_i
                t_dataset.append((fname,labelc,c_i,1))
            else:
                t_dataset.append((fname,label,c_i,1))
            if label==-1:continue
            dataset_c_intra.append((fname,label,c_i,1))
            cluster_center[label].append(per_cam_features[c_i][img_i])

        camera_intra_train_set = ImageDataset(dataset_c_intra, train_transforms)
        if cfg.DATASETS.IS_NEW:
            camera_intra_train_loader.append(DataLoader(
                camera_intra_train_set, batch_size=cfg.SOLVER.CITRA_IMS_PER_BATCH,
                sampler=RandomIdentitySampler(camera_intra_train_set.dataset, cfg.SOLVER.CITRA_IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=8, collate_fn=train_collate_fn,drop_last=True
            ))
        else:
            camera_intra_train_loader.append(DataLoader(
                camera_intra_train_set, batch_size=cfg.SOLVER.CITRA_IMS_PER_BATCH,
                sampler=RandomIdentitySampler(camera_intra_train_set.dataset, cfg.SOLVER.CITRA_IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=8, collate_fn=train_collate_fn
            ))

        t_datat_train_set = ImageDataset(t_dataset, test_transforms)
        t_loader.append(DataLoader(
            t_datat_train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            num_workers=8, collate_fn=train_collate_fn
        ))
    return camera_intra_train_loader,num_ids,cluster_centers,cluster,t_loader

def LRI_data(model,t_train_loader,device,camera_num,train_transforms,test_transforms,cfg,epoch):
    per_cam_features, per_cam_fname, per_cam_pid = extract_features_per_cam(model, t_train_loader, device)
    per_cam_features_list = mmm(per_cam_features)
    dists = dist(per_cam_features_list)
    img_dict = sim_label(dists, camera_num, cfg.SOLVER.THRES)
    per_c_pid_dict = nnn(per_cam_pid)
    new_c_class_dict = merge(img_dict, per_c_pid_dict,camera_num)
    new_c_class_dict2 = merge2(new_c_class_dict, per_cam_pid,camera_num)
    # Similar-class Fusion
    dict_id = hard_sample_inference(new_c_class_dict2, camera_num)
    camera_pid, camera_pid_index = make_id(dict_id, per_cam_fname, per_cam_pid, per_c_pid_dict, camera_num)
    class_list = make_id2(camera_pid, camera_pid_index, camera_num)
    dataset = make_dataset(class_list)
    dataset = list(dict.fromkeys(dataset))
    try:
        class_num = dataset[-1][1]
    except:
        print('')

    camera_inter_test_set = ImageDataset(dataset, test_transforms)
    camera_inter_test_loader = DataLoader(
        camera_inter_test_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
        num_workers=8, collate_fn=train_collate_fn
    )
    # Similar-class Fusion
    repeatclass_ids = get_repeatclass_id(model,camera_inter_test_loader,cfg)
    dataset,class_num  = merge_class_dataset(dataset,repeatclass_ids,class_num)

    camera_inter_train_set = ImageDataset(dataset, train_transforms)
    camera_inter_train_loader = DataLoader(
        camera_inter_train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
        sampler=RandomIdentitySampler(dataset, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
        num_workers=8, collate_fn=train_collate_fn
    )
    print('data_loader make done!')


    return camera_inter_train_loader,class_num,camera_inter_test_loader

def merge_class_dataset(dataset,repeatclass_ids,class_num):
    new_dataset = []
    for data in dataset:
        img_path,pid,c_id, = data[0],data[1],data[2]
        for i in range(len(repeatclass_ids)):
            ids = repeatclass_ids[i]
            if pid in ids:
                pid = class_num + i + 1
        new_dataset.append((img_path,pid,c_id,1))

    new_dataset,class_nums = _relabels(new_dataset,1)


    return new_dataset,class_nums

def _relabels(samples, label_index):
    '''
    reorder labels
    map labels [1, 3, 5, 7] to [0,1,2,3]
    '''
    ids = []
    for sample in samples:
        ids.append(sample[label_index])
    ids = list(set(ids))
    ids.sort()
    for i in range(len(samples)):
        samples[i] = (samples[i][0],ids.index(samples[i][label_index]),samples[i][2],1)

    return samples,len(ids)

def get_repeatclass_id(model,camera_inter_test_loader,cfg):
    model.eval()
    print("Start extract features center")
    cluster_center = collections.defaultdict(list)
    for n_iter, (img, vid, camid, target_view,fnames) in enumerate(camera_inter_test_loader):
        img = img.cuda()
        with torch.no_grad():
            outputs = model(img)
            outputs = outputs.cpu()
        if n_iter==0:
            features = outputs
            idxs = vid
        else:
            idxs = torch.cat((idxs,vid),dim=0)
            features = torch.cat((features,outputs),dim=0)

    for i in range(len(features)):
        cluster_center[int(idxs[i])].append(features[i])

    cluster_center = [torch.stack(cluster_center[idx]).mean(0)
                      for idx in sorted(cluster_center.keys())]
    cluster_center = torch.stack(cluster_center)
    distance_dict = normalization(euclidean_distance(cluster_center, cluster_center))
    index_tensor = thres_index(distance_dict,cfg.SOLVER.THRES)
    index_tensor2 = copy.deepcopy(index_tensor)

    id_list = []
    for v in index_tensor2:
        if v[0]!=v[1]:
            if [v[1],v[0]] not in id_list:
                id_list.append([v[0],v[1]])


    id_s = merge_list(id_list)

    print("Extract features center done!")
    model.train()
    return id_s

def merge_list(a):
    b = len(a)
    for i in range(b):
        for j in range(b):
            x = list(set(a[i]+a[j]))
            y = len(a[j])+len(a[i])
            if i == j or a[i] == 0 or a[j] == 0:
                break
            elif len(x) < y:
                a[i] = x
                a[j] = [0]

    a = [i for i in a if i != [0]]
    return a

def do_comparative(cfg,
             model,
             t_train_loader,
             val_loader,
             optimizer,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    best_rank1_map =  0
    best_cmc = 0
    best_rank1 = 0
    best_epoch = 0
    device = "cuda"
    # epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    train_transforms=make_train_transforms(cfg)
    test_transforms = make_test_transforms(cfg)
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    # train
    cluster = 0
    camera_number = cfg.DATALOADER.CAMERA_NUM
    for epoch in range(1, 201):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        if epoch<=10:
            if epoch%1==0 or epoch==1:
                camera_intra_train_loader,class_num,cluster_centers,cluster,t_loader = \
                    intra_camera_leanring(epoch,cluster,model, t_train_loader, device, camera_number, train_transforms, test_transforms,cfg)

            for i in range(camera_number):
                camera_num=i
                train_proess(camera_intra_train_loader[i], optimizer, device, model, loss_fn, scaler, cfg, epoch, loss_meter, acc_meter,
                             log_period, logger, scheduler, start_time,camera_num,a=1,b=cfg.SOLVER.b,)
        else:
            if epoch%1==0 or epoch==1:
                camera_intra_train_loader,class_num,cluster_centers,cluster,t_loader = \
                    intra_camera_leanring(epoch,cluster,model, t_train_loader, device, camera_number, train_transforms, test_transforms,cfg)
                camera_inter_train_loader,class_num2,camera_inter_test_loader = \
                    LRI_data(model, t_loader, device, camera_number, train_transforms,test_transforms, cfg,epoch)

            train_proess(camera_inter_train_loader, optimizer, device, model, loss_fn, scaler, cfg, epoch, loss_meter,
                         acc_meter, log_period, logger, scheduler, start_time,iter_num=cfg.SOLVER.ITERS,a=cfg.SOLVER.a,b=cfg.SOLVER.b,is_source=True)

        if epoch % checkpoint_period == 0:
            torch.save(model.state_dict(),
                       os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch > cfg.SOLVER.STATR_EPOCH and epoch % eval_period == 0 :
            model.eval()
            for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                with torch.no_grad():
                    img = img.to(device)
                    camids = camids.to(device)
                    target_view = target_view.to(device)
                    feat = model(img, cam_label=camids, view_label=target_view)
                    evaluator.update((feat, vid, camid))
            cmc, mAP, _, _, _, _, _ = evaluator.compute()
            logger.info("Validation Results - Epoch: {}".format(epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            torch.cuda.empty_cache()

            if cmc[0]>best_rank1:
                torch.save(model.state_dict(),os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_best.pth'))
                best_rank1 = cmc[0]
                best_rank1_map = mAP
                best_cmc = cmc
                best_epoch = epoch
            elif cmc[0]==best_rank1 and mAP>best_rank1_map:
                torch.save(model.state_dict(),os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_best.pth'))
                best_rank1 = cmc[0]
                best_rank1_map = mAP
                best_cmc = cmc
                best_epoch = epoch
            logger.info("########################")
            logger.info("Best Results - Epoch: {}".format(best_epoch))
            logger.info("mAP: {:.1%}".format(best_rank1_map))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, best_cmc[r - 1]))




def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        for n_iter, (img, vid, target_cam, target_view,_) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            with amp.autocast(enabled=True):
                score, feat = model(img, target, cam_label=target_cam, view_label=target_view )
                loss = loss_fn(score, feat, target, target_cam)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                        .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        feat = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()



def extract_features_per_cam(model, train_loader,device,is_c_intra=False):
    model.eval()
    per_cam_features = {}
    per_cam_fname = {}
    per_cam_pid = {}
    print("Start extract features per camera")
    for camera_loader in train_loader:
        for n_iter, (img, vid, camid, target_view,fnames) in enumerate(camera_loader):
            # if n_iter<3:
            img = img.to(device)
            camid = list(camid)
            for cam in camid:
                cam = cam.item()
                if cam not in per_cam_features.keys():
                    per_cam_features[cam] = []
                    per_cam_fname[cam] = []
                    per_cam_pid[cam]=[]

            with torch.no_grad():
                outputs = model(img)
                outputs = outputs.cpu()

            for fname,pid, output, cam in zip(fnames,vid, outputs, camid):
                cam = cam.item()
                if not is_c_intra:
                    output = output.unsqueeze(dim=0)
                per_cam_features[cam].append(output)
                per_cam_fname[cam].append(fname)
                per_cam_pid[cam].append(pid.numpy().tolist())
    print("Extract features per camera done!")
    model.train()
    return per_cam_features, per_cam_fname,per_cam_pid

def train_proess(train_loader, optimizer, device, model, loss_fn, scaler, cfg, epoch, loss_meter, acc_meter,
                     log_period, logger, scheduler, start_time,camera_num=0,iter_num=20,a=1,b=1,is_source=False):
    model.train()
    n_iter = 0
    loader_lenth = len(train_loader)
    train_iter = IterLoader(train_loader)
    if cfg.DATASETS.IS_NEW:
        loader_lenth +=1
    if loader_lenth > 1:
        for n_iter in range(iter_num):
            img, vid, target_cam, target_view,img_path = train_iter.next_one()
            optimizer.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            with amp.autocast(enabled=True):
                feat,feat2 = model(img,is_source,camera_num,target, cam_label=target_cam, view_label=target_view)
                loss_1 = loss_fn( feat, target)
                loss_2 = loss_fn( feat2, target)
                loss_3 = kl_loss(feat,feat2)
                loss = a*loss_1 + b*(loss_2 + loss_3)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), img.shape[0])
            # acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, , Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg,  scheduler._get_lr(epoch)[0]))
    logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f},  Base Lr: {:.2e}, a: {:.1f}, b: {:.1f}"
                .format(epoch, (n_iter + 1), len(train_loader),loss_meter.avg,  scheduler._get_lr(epoch)[0],a,b))
    end_time = time.time()
    time_per_batch = (end_time - start_time) / (n_iter + 1)
    if cfg.MODEL.DIST_TRAIN:
        pass
    else:
        logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]

