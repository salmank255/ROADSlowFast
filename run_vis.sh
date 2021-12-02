CUDA_VISIBLE_DEVICES=1 python inference_vis.py ../../ ../../ ../../kinetics-pt/ --MODE=gen_dets --ARCH=resnet50 --MODEL_TYPE=I3D --DATASET=road --TRAIN_SUBSETS=train_3 --TEST_SUBSETS=test --SEQ_LEN=8 --TEST_SEQ_LEN=08 --BATCH_SIZE=4 --LR=0.0041 --GEN_CONF_THRESH=0.6

CUDA_VISIBLE_DEVICES=2 python inference_vis.py ../../ ../../ ../../kinetics-pt/ --MODE=gen_dets --ARCH=resnet50 --MODEL_TYPE=I3D --DATASET=road --TRAIN_SUBSETS=train_3 --TEST_SUBSETS=test --SEQ_LEN=8 --TEST_SEQ_LEN=08 --BATCH_SIZE=4 --LR=0.0041 --GEN_CONF_THRESH=0.5

CUDA_VISIBLE_DEVICES=5 python inference_vis.py ../../ ../../ ../../kinetics-pt/ --MODE=gen_dets --ARCH=resnet50 --MODEL_TYPE=I3D --DATASET=road --TRAIN_SUBSETS=train_3 --TEST_SUBSETS=test --SEQ_LEN=8 --TEST_SEQ_LEN=08 --BATCH_SIZE=4 --LR=0.0041 --GEN_CONF_THRESH=0.4

CUDA_VISIBLE_DEVICES=4 python inference_vis.py ../../ ../../ ../../kinetics-pt/ --MODE=gen_dets --ARCH=resnet50 --MODEL_TYPE=I3D --DATASET=road --TRAIN_SUBSETS=train_3 --TEST_SUBSETS=test --SEQ_LEN=8 --TEST_SEQ_LEN=08 --BATCH_SIZE=4 --LR=0.0041 --GEN_CONF_THRESH=0.3

CUDA_VISIBLE_DEVICES=6 python inference_vis.py ../../ ../../ ../../kinetics-pt/ --MODE=gen_dets --ARCH=resnet50 --MODEL_TYPE=I3D --DATASET=road --TRAIN_SUBSETS=train_3 --TEST_SUBSETS=test --SEQ_LEN=8 --TEST_SEQ_LEN=08 --BATCH_SIZE=4 --LR=0.0041 --GEN_CONF_THRESH=0.2
