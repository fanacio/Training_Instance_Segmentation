## 模型操作流程

    大致流程如下所示：
    1. 先使用labelme进行数据标注，标注后的目录结构如下所示：
    ├─Annotations
    ├─JPEGImages
    └─Segmentation
    其中Annotations中存放标注出的json文件，JPEGImages中存放了所标注的images，images与json文件数量一一对应。
    
    2. 将JPEGImages和Annotations文件夹下的文件全部放在一个命名为data的文件夹下，并将其移动到../dataset目录下，然后在../dataset目录下有一个readme.md文档详细介绍了数据增强工作和labelme格式转换为coco格式的过程。总之，通过此步骤，能够得到符合模型训练要求的数据集。
    
    3. 训练前准备工作
     3.1 需准备预训练权重，并将其保存至weights文件夹下：
        .
        |--base_54_800000.onnx
        |-- base_54_800000.pth
        |-- darknet53_54_800000.pth
        |--im700_54_800000.pth
        `-- resnet50_54_800000.pth
     3.2 修改config.py文件，下面列出主要需要修改的参数：
        "COCO_CLASSES"参数：修改为和标注时的序号一致，也要和步骤2中的labelme2coco.py中的id对应；
        "COCO_LABEL_MAP"参数：与COCO_CLASSES也是一一对应即可；
        "dataset_base"中，设置训练集和验证集的路径及json文件的路径；
        "coco_base_config"中，修改"num_classes"为类别数+1、"max_iter"训练迭代数，这里包含了很多训练过程中需要设置的参数，根据自己所需进行修改；
        "base_config"中的"num_classes"和"max_iter"进行修改，max_iter控制了最终的循环迭代次数；
    
    4. 开始训练
    训练时需要设置一些参数，主要参数如下列出(可以在根目录执行 python train.py -h 命令获取):
    --batch_size       训练时batch设置，建议根据显卡数量设置，一般设置为8*显卡数量；
    --num_workers      线程数，一般都是需要设置为"0",默认为4；
    --lr               学习率，默认为none；
    --save_folder      训练所得权重保存的路径，默认为weights/；
    --config           配置文件选择，默认为none；
    --save_interval    每10000次迭代保存模型权重，这里调小一点可以多保存中间过程；
    --validation_epoch 每*个epoch进行一次验证计算，默认为2；
    --dataset          数据集路径；
    
    一般只需要执行的命令为：
    python train.py --config=base_config --batch_size=16
    即可，如果报错了就把batch_size调小一点。训练所得权重保存在save_folder目录下，其格式为<config>_<epoch>_<iter>.pth，如果是中断训练，则命名为<config>_<epoch>_<iter>_interrupt.pth
    
    注意，如果使用上次训练的权重进行再训练，则如下：
    #使用吗，默认基础配_base_config训练，指定权重文件
    python train.py --config=base_config --resume=weights/base_10_32100.pth --start_iter=-1
    
    5. 训练过程如下：
    本人在A10卡上训练的，其显卡占用情况如下所示: data/nvidia-smi.png
    
    6. 训练后模型测试
    测试时可指定一些参数，如下：
    --trained_model     训练所得模型权重；
    --score_threshold   置信度阈值，如果设置为0.5，则表示剔除低于0.5置信度的目标；
    --top_k=15          保存置信度最高的前15个目标；
    --config            配置文件选择，默认为none；
    --images            检测的图片的目录:检测结果存放的目录；
    --image             检测单张图片；
    --video             检测视频；
    
    一般只需要执行的命令为：
    python eval.py --trained_model=weights/base_374_1500.pth --score_threshold=0.75 --top_k=15 --images=fanyichao_test/images:fanyichao_test/images_out/


