import cv2
from PIL import Image


import os
import sys
import torch
import argparse
import numpy as np
from modules import utils
from train import train
from data import VideoDataset
from torchvision import transforms
import data.transforms as vtf
from models.retinanet import build_retinanet
from gen_dets import gen_dets, eval_framewise_dets
from tubes import build_eval_tubes
from val import val
import torch.utils.data as data_utils
from data import custum_collate

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def set_video(inp, video_name):
    cap = cv2.VideoCapture(inp)
    ret, frame = cap.read()
    frheight, frwidth, ch = frame.shape
    print(frheight,'.......',frwidth)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    video_width = 1381
    video_height = 777
    size = (video_width, video_height)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video = cv2.VideoWriter(video_name, fourcc, fps, size)
    return cap, video, fps

def set_out_video(video_name):
    fps = 12
    video_width = 1280
    video_height = 960
    size = (video_width, video_height)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video = cv2.VideoWriter(video_name, fourcc, fps, size)
    return video


def main():
    parser = argparse.ArgumentParser(description='Training single stage FPN with OHEM, resnet as backbone')
    parser.add_argument('DATA_ROOT', help='Location to root directory for dataset reading') # /mnt/mars-fast/datasets/
    parser.add_argument('SAVE_ROOT', help='Location to root directory for saving checkpoint models') # /mnt/mars-alpha/
    parser.add_argument('MODEL_PATH',help='Location to root directory where kinetics pretrained models are stored')
    
    parser.add_argument('--MODE', default='train',
                        help='MODE can be train, gen_dets, eval_frames, eval_tubes define SUBSETS accordingly, build tubes')
    # Name of backbone network, e.g. resnet18, resnet34, resnet50, resnet101 resnet152 are supported
    parser.add_argument('--ARCH', default='resnet50', 
                        type=str, help=' base arch')
    parser.add_argument('--MODEL_TYPE', default='I3D',
                        type=str, help=' base model')
    parser.add_argument('--ANCHOR_TYPE', default='RETINA',
                        type=str, help='type of anchors to be used in model')
    
    parser.add_argument('--SEQ_LEN', default=8,
                        type=int, help='NUmber of input frames')
    parser.add_argument('--TEST_SEQ_LEN', default=8,
                        type=int, help='NUmber of input frames')
    parser.add_argument('--MIN_SEQ_STEP', default=1,
                        type=int, help='DIFFERENCE of gap between the frames of sequence')
    parser.add_argument('--MAX_SEQ_STEP', default=1,
                        type=int, help='DIFFERENCE of gap between the frames of sequence')
    # if output heads are have shared features or not: 0 is no-shareing else sharining enabled
    # parser.add_argument('--MULIT_SCALE', default=False, type=str2bool,help='perfrom multiscale training')
    parser.add_argument('--HEAD_LAYERS', default=3, 
                        type=int,help='0 mean no shareding more than 0 means shareing')
    parser.add_argument('--NUM_FEATURE_MAPS', default=5, 
                        type=int,help='0 mean no shareding more than 0 means shareing')
    parser.add_argument('--CLS_HEAD_TIME_SIZE', default=3, 
                        type=int, help='Temporal kernel size of classification head')
    parser.add_argument('--REG_HEAD_TIME_SIZE', default=3,
                    type=int, help='Temporal kernel size of regression head')
    
    #  Name of the dataset only voc or coco are supported
    parser.add_argument('--DATASET', default='road', 
                        type=str,help='dataset being used')
    parser.add_argument('--TRAIN_SUBSETS', default='train_3,', 
                        type=str,help='Training SUBSETS seprated by ,')
    parser.add_argument('--VAL_SUBSETS', default='', 
                        type=str,help='Validation SUBSETS seprated by ,')
    parser.add_argument('--TEST_SUBSETS', default='', 
                        type=str,help='Testing SUBSETS seprated by ,')
    # Input size of image only 600 is supprted at the moment 
    parser.add_argument('--MIN_SIZE', default=512, 
                        type=int, help='Input Size for FPN')
    
    #  data loading argumnets
    parser.add_argument('-b','--BATCH_SIZE', default=4, 
                        type=int, help='Batch size for training')
    parser.add_argument('--TEST_BATCH_SIZE', default=1, 
                        type=int, help='Batch size for testing')
    # Number of worker to load data in parllel
    parser.add_argument('--NUM_WORKERS', '-j', default=8, 
                        type=int, help='Number of workers used in dataloading')
    # optimiser hyperparameters
    parser.add_argument('--OPTIM', default='SGD', 
                        type=str, help='Optimiser type')
    parser.add_argument('--RESUME', default=0, 
                        type=int, help='Resume from given epoch')
    parser.add_argument('--MAX_EPOCHS', default=30, 
                        type=int, help='Number of training epoc')
    parser.add_argument('-l','--LR', '--learning-rate', 
                        default=0.004225, type=float, help='initial learning rate')
    parser.add_argument('--MOMENTUM', default=0.9, 
                        type=float, help='momentum')
    parser.add_argument('--MILESTONES', default='20,25', 
                        type=str, help='Chnage the lr @')
    parser.add_argument('--GAMMA', default=0.1, 
                        type=float, help='Gamma update for SGD')
    parser.add_argument('--WEIGHT_DECAY', default=1e-4, 
                        type=float, help='Weight decay for SGD')
    
    # Freeze layers or not 
    parser.add_argument('--FBN','--FREEZE_BN', default=True, 
                        type=str2bool, help='freeze bn layers if true or else keep updating bn layers')
    parser.add_argument('--FREEZE_UPTO', default=1, 
                        type=int, help='layer group number in ResNet up to which needs to be frozen')
    
    # Loss function matching threshold
    parser.add_argument('--POSTIVE_THRESHOLD', default=0.5, 
                        type=float, help='Min threshold for Jaccard index for matching')
    parser.add_argument('--NEGTIVE_THRESHOLD', default=0.4,
                        type=float, help='Max threshold Jaccard index for matching')
    # Evaluation hyperparameters
    parser.add_argument('--EVAL_EPOCHS', default='30', 
                        type=str, help='eval epochs to test network on these epoch checkpoints usually the last epoch is used')
    parser.add_argument('--VAL_STEP', default=2, 
                        type=int, help='Number of training epoch before evaluation')
    parser.add_argument('--IOU_THRESH', default=0.5, 
                        type=float, help='Evaluation threshold for validation and for frame-wise mAP')
    parser.add_argument('--CONF_THRESH', default=0.025, 
                        type=float, help='Confidence threshold for to remove detection below given number')
    parser.add_argument('--NMS_THRESH', default=0.5, 
                        type=float, help='NMS threshold to apply nms at the time of validation')
    parser.add_argument('--TOPK', default=10, 
                        type=int, help='topk detection to keep for evaluation')
    parser.add_argument('--GEN_CONF_THRESH', default=0.025, 
                        type=float, help='Confidence threshold at the time of generation and dumping')
    parser.add_argument('--GEN_TOPK', default=100, 
                        type=int, help='topk at the time of generation')
    parser.add_argument('--GEN_NMS', default=0.5, 
                        type=float, help='NMS at the time of generation')
    parser.add_argument('--CLASSWISE_NMS', default=False, 
                        type=str2bool, help='apply classwise NMS/no tested properly')
    parser.add_argument('--JOINT_4M_MARGINALS', default=False, 
                        type=str2bool, help='generate score of joints i.e. duplexes or triplet by marginals like agents and actions scores')
    
    ## paths hyper parameters
    parser.add_argument('--COMPUTE_PATHS', default=False, 
                        type=str2bool, help=' COMPUTE_PATHS if set true then it overwrite existing ones')
    parser.add_argument('--PATHS_IOUTH', default=0.5,
                        type=float, help='Iou threshold for building paths to limit neighborhood search')
    parser.add_argument('--PATHS_COST_TYPE', default='score',
                        type=str, help='cost function type to use for matching, other options are scoreiou, iou')
    parser.add_argument('--PATHS_JUMP_GAP', default=4,
                        type=int, help='GAP allowed for a tube to be kept alive after no matching detection found')
    parser.add_argument('--PATHS_MIN_LEN', default=6,
                        type=int, help='minimum length of generated path')
    parser.add_argument('--PATHS_MINSCORE', default=0.1,
                        type=float, help='minimum score a path should have over its length')
    
    ## paths hyper parameters
    parser.add_argument('--COMPUTE_TUBES', default=False, type=str2bool, help='if set true then it overwrite existing tubes')
    parser.add_argument('--TUBES_ALPHA', default=0,
                        type=float, help='alpha cost for changeing the label')
    parser.add_argument('--TRIM_METHOD', default='none',
                        type=str, help='other one is indiv which works for UCF24')
    parser.add_argument('--TUBES_TOPK', default=10,
                        type=int, help='Number of labels to assign for a tube')
    parser.add_argument('--TUBES_MINLEN', default=5,
                        type=int, help='minimum length of a tube')
    parser.add_argument('--TUBES_EVAL_THRESHS', default='0.2,0.5',
                        type=str, help='evaluation threshold for checking tube overlap at evaluation time, one can provide as many as one wants')
    # parser.add_argument('--TRAIL_ID', default=0,
    #                     type=int, help='eval TUBES_Thtrshold at evaluation time')
    
    ###
    parser.add_argument('--LOG_START', default=10, 
                        type=int, help='start loging after k steps for text/tensorboard') 
    parser.add_argument('--LOG_STEP', default=10, 
                        type=int, help='Log every k steps for text/tensorboard')
    parser.add_argument('--TENSORBOARD', default=1,
                        type=str2bool, help='Use tensorboard for loss/evalaution visualization')

    # Program arguments
    parser.add_argument('--MAN_SEED', default=123, 
                        type=int, help='manualseed for reproduction')
    parser.add_argument('--MULTI_GPUS', default=True, type=str2bool, help='If  more than 0 then use all visible GPUs by default only one GPU used ') 

    # Use CUDA_VISIBLE_DEVICES=0,1,4,6 to select GPUs to use


    ## Parse arguments
    args = parser.parse_args()
    
    args = utils.set_args(args) # set directories and SUBSETS fo datasets
    args.MULTI_GPUS = False if args.BATCH_SIZE == 1 else args.MULTI_GPUS
    ## set random seeds and global settings
    np.random.seed(args.MAN_SEED)
    torch.manual_seed(args.MAN_SEED)
    # torch.cuda.manual_seed_all(args.MAN_SEED)
    torch.set_default_tensor_type('torch.FloatTensor')

    args = utils.create_exp_name(args)

    utils.setup_logger(args)
    logger = utils.get_logger(__name__)
    logger.info(sys.version)

    assert args.MODE in ['train','val','test','gen_dets','eval_frames', 'eval_tubes'], 'MODE must be from ' + ','.join(['train','test','tubes'])

    if args.MODE == 'train':
        args.TEST_SEQ_LEN = args.SEQ_LEN
    else:
        args.SEQ_LEN = args.TEST_SEQ_LEN

    ttransform = transforms.Compose([
                        vtf.ResizeClip(args.MIN_SIZE, args.MAX_SIZE),
                        vtf.ToTensorStack(),
                        vtf.Normalize(mean=args.MEANS, std=args.STDS)])
    if args.MODE in ['train','val']:
        # args.CONF_THRESH = 0.05
        args.SUBSETS = args.TRAIN_SUBSETS
        train_transform = transforms.Compose([
                            vtf.ResizeClip(args.MIN_SIZE, args.MAX_SIZE),
                            vtf.ToTensorStack(),
                            vtf.Normalize(mean=args.MEANS, std=args.STDS)])
        
        # train_skip_step = args.SEQ_LEN
        # if args.SEQ_LEN>4 and args.SEQ_LEN<=10:
        #     train_skip_step = args.SEQ_LEN-2
        if args.SEQ_LEN>10:
            train_skip_step = args.SEQ_LEN + (args.MAX_SEQ_STEP - 1) * 2 - 2
        else:
            train_skip_step = args.SEQ_LEN 

        train_dataset = VideoDataset(args, train=True, skip_step=train_skip_step, transform=train_transform)
        logger.info('Done Loading Dataset Train Dataset')
        ## For validation set
        full_test = False
        args.SUBSETS = args.VAL_SUBSETS
        skip_step = args.SEQ_LEN*8
    else:
        args.SEQ_LEN = args.TEST_SEQ_LEN
        args.MAX_SEQ_STEP = 1
        args.SUBSETS = args.TEST_SUBSETS
        full_test = True #args.MODE != 'train'
        args.skip_beggning = 0
        args.skip_ending = 0
        if args.MODEL_TYPE == 'I3D':
            args.skip_beggning = 2
            args.skip_ending = 2
        elif args.MODEL_TYPE != 'C2D':
            args.skip_beggning = 2

        skip_step = args.SEQ_LEN - args.skip_beggning

    

    val_transform = transforms.Compose([ 
                        vtf.ResizeClip(args.MIN_SIZE, args.MAX_SIZE),
                        vtf.ToTensorStack(),
                        vtf.Normalize(mean=args.MEANS,std=args.STDS)])
    

    val_dataset = VideoDataset(args, train=False, transform=val_transform, skip_step=skip_step, full_test=full_test)
    logger.info('Done Loading Dataset Validation Dataset')


    args.num_classes =  val_dataset.num_classes
    # one for objectness
    args.label_types = val_dataset.label_types
    args.num_label_types = val_dataset.num_label_types
    args.all_classes =  val_dataset.all_classes
    args.num_classes_list = val_dataset.num_classes_list
    args.num_ego_classes = val_dataset.num_ego_classes
    args.ego_classes = val_dataset.ego_classes
    args.head_size = 256
    # olympia_classes = val_dataset.olympia_classes

    if args.MODE in ['train', 'val','test','gen_dets']:
        net = build_retinanet(args).cuda()
        logger.info('\nLets do dataparallel\n')
        net = torch.nn.DataParallel(net)


    net.eval()
    args.MODEL_PATH = args.SAVE_ROOT + 'model_{:06d}.pth'.format(args.EVAL_EPOCHS[0])
    logger.info('Loaded model from :: '+args.MODEL_PATH)
    net.load_state_dict(torch.load(args.MODEL_PATH))

    val_data_loader = data_utils.DataLoader(val_dataset, args.BATCH_SIZE, num_workers=args.NUM_WORKERS,
                                            shuffle=False, pin_memory=True, collate_fn=custum_collate)

    with torch.no_grad():
        for val_itr, (images, gt_boxes, gt_targets, ego_labels, batch_counts, img_indexs, wh) in enumerate(val_data_loader):
            



    # test_vid_path = '../olympia/test/videos/C0003_fb.MP4'
    # test_vid_out = 'C0003_fb_splt_th_0_1.MP4'

    # cap,video,fps = set_video(test_vid_path,test_vid_out)
    # activation = torch.nn.Sigmoid().cuda()
    # f_n=1
    # with torch.no_grad():
    #     while(cap):
    #         images = []
    #         images_org = []
            
    #         for i in range(args.SEQ_LEN):
    #             ret, frame = cap.read()
    #             f_n += 1
    #             print(f_n)
    #             if ret==False:
    #                 break
    #             # You may need to convert the color.
    #             img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #             im_pil = Image.fromarray(img)
    #             images.append(im_pil)
    #             images_org.append(frame)
    #         clip = ttransform(images)
    #         height, width = clip.shape[-2:]
    #         n_imgs = clip.shape[1]
    #         if n_imgs !=8:
    #             break
    #         clip = torch.unsqueeze(clip,0)
    #         # print(clip.shape)
            
    #         clip = clip.cuda(0, non_blocking=True)
    #         decoded_boxes, confidence, ego_preds = net(clip)
    #         confidence = activation(confidence)
    #         # print(decoded_boxes.shape)
    #         # print(confidence.shape)
    #         # print(ego_preds.shape)
    #         det_boxes = []
    #         for nlt in range(args.num_label_types):
    #             numc = args.num_classes_list[nlt]
    #             det_boxes.append([[] for _ in range(numc)])
    #         for s in range(args.SEQ_LEN):
    #             image = images_org[s]
    #             image = cv2.resize(image,(width,height))
    #             decoded_boxes_frame = decoded_boxes[0, s].clone()
    #             cc = 0 
    #             for nlt in range(1,args.num_label_types):
    #                 num_c = args.num_classes_list[nlt]
    #                 # tgt_labels = gt_labels_batch[:,cc:cc+num_c]
    #                 # # print(gt_boxes_batch.shape, tgt_labels.shape)
    #                 # frame_gt = get_individual_labels(gt_boxes_batch, tgt_labels)
    #                 # gt_boxes_all[nlt].append(frame_gt)
    #                 for cl_ind in range(num_c):
    #                     scores = confidence[0, s, :, cc].clone().squeeze()
    #                     cc += 1
    #                     cls_dets = utils.filter_detections(args, scores, decoded_boxes_frame)
    #                     det_boxes[nlt][cl_ind].append(cls_dets)
    #                     # print(cls_dets)                   
    #                     classname = olympia_classes[cl_ind]
    #                     # print(classname)
    #                     # scores = conf_scores[:, cl_ind].squeeze()
    #                     # #print(scores.shape)
    #                     # c_mask = scores.gt(args.conf_thresh)  # greater than minmum threshold
    #                     # scores = scores[c_mask].squeeze()
    #                     # #print(scores)
    #                     # # print('scores size',scores.size())
    #                     # if scores.dim() == 0:
    #                     #     # print(len(''), ' dim ==0 ')
    #                     #     det_boxes[cl_ind - 1].append(np.asarray([]))
    #                     #     continue
    #                     # boxes = decoded_boxes.clone()
                    
    #                     # l_mask = c_mask.unsqueeze(1).expand_as(boxes)
                    
    #                     # boxes = boxes[l_mask].view(-1, 4)
    #                     # #print(boxes.shape)

    #                     # # idx of highest scoring and non-overlapping boxes per class
    #                     # ids, counts = nms(boxes, scores, args.nms_thresh, args.topk)  # idsn - ids after nms
    #                     # #print(counts)
    #                     # scores = scores[ids[:counts]].numpy()
    #                     # #print(boxes)
    #                     # boxes = boxes[ids[:counts]].numpy()
    #                     # #print('boxes shape',boxes)
    #                     # boxes[:, 0] *= width
    #                     # boxes[:, 2] *= width
    #                     # boxes[:, 1] *= height
    #                     # boxes[:, 3] *= height
    #                     #print(boxes)
    #                     boxes = cls_dets
    #                     if boxes.shape !=(0,4):
    #                         #print('boxes sahpe',boxes[0][0])
    #                         for bb in range(boxes.shape[0]):
    #                             boxes[bb, 0] = max(0, boxes[bb, 0])
    #                             boxes[bb, 2] = min(width, boxes[bb, 2])
    #                             boxes[bb, 1] = max(0, boxes[bb, 1])
    #                             boxes[bb, 3] = min(height, boxes[bb, 3])
    #                             # print(int(boxes[bb][0]), int(boxes[bb][1]),int(boxes[bb][2]), int(boxes[bb][3]))
    #                             cv2.rectangle(image, (int(boxes[bb][0]), int(boxes[bb][1])), (int(boxes[bb][2]), int(boxes[bb][3])), (0, 255, 0), 2)
    #                             cv2.putText(image, classname, (int(boxes[bb][0]), int(boxes[bb][1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 1)
    #                         #cv2.rectangle(image2, (boxes[1][0], boxes[1][1]), (boxes[1][2], boxes[1][3]), (0, 255, 0), 2)
    #                         #cv2.putText(image2, classname, (int(boxes[1][0]), int(boxes[1][1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 1)
    #                 # cv2.imwrite('out.png',image)
    #                 video.write(image)
    #         # break
    # cap.release()
    # video.release()
        
                        





    # for arg in sorted(vars(args)):
    #     logger.info(str(arg)+': '+str(getattr(args, arg)))
    
    # if args.MODE == 'train':
    #     if args.FBN:
    #         if args.MULTI_GPUS:
    #             net.module.backbone.apply(utils.set_bn_eval)
    #         else:
    #             net.backbone.apply(utils.set_bn_eval)
    #     train(args, net, train_dataset, val_dataset)
    # args.EVAL_EPOCHS = args.MAX_EPOCHS
    # args.MODE = 'val'
    # val(args, net, val_dataset)
    # args.MODE = 'test'
    # val(args, net, test_dataset)


    # # elif args.MODE == 'gen_dets':
    # #     gen_dets(args, net, val_dataset)
    # #     eval_framewise_dets(args, val_dataset)
    # #     build_eval_tubes(args, val_dataset)
    # # elif args.MODE == 'eval_frames':
    # #     eval_framewise_dets(args, val_dataset)
    # # elif args.MODE == 'eval_tubes':
    # #     build_eval_tubes(args, val_dataset)
    

if __name__ == "__main__":
    main()
