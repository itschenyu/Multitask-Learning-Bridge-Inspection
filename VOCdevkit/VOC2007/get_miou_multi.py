import os

from PIL import Image
from tqdm import tqdm

from hrnet_multi import HRnet_Segmentation
from utils.utils_metrics_multi import compute_mIoU, show_results

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "03"   #(xxxx is your specific GPU ID)

'''
进行指标评估需要注意以下几点：
1、该文件生成的图为灰度图，因为值比较小，按照PNG形式的图看是没有显示效果的，所以看到近似全黑的图是正常的。
2、该文件计算的是验证集的miou，当前该库将测试集当作验证集使用，不单独划分测试集
'''
if __name__ == "__main__":
    #---------------------------------------------------------------------------#
    #   miou_mode用于指定该文件运行时计算的内容
    #   miou_mode为0代表整个miou计算流程，包括获得预测结果、计算miou。
    #   miou_mode为1代表仅仅获得预测结果。
    #   miou_mode为2代表仅仅计算miou。
    #---------------------------------------------------------------------------#
    miou_mode       = 0
    #------------------------------#
    #   分类个数+1、如2+1
    #------------------------------#
    num_classes     = [7, 2]
    #--------------------------------------------#
    #   区分的种类，和json_to_dataset里面的一样
    #--------------------------------------------#
    # name_classes    = ["background","aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    # name_classes    = ["_background_","cat","dog"]
    name_classes    = [["_background_","bearing","bracing","deck","floor_beam","girder","substructure"], ["_background_","Corrosion"]]
    # name_classes    = [["_background_","Corrosion"], ["_background_","Corrosion"]]
    
    #-------------------------------------------------------#
    #   指向VOC数据集所在的文件夹
    #   默认指向根目录下的VOC数据集
    #-------------------------------------------------------#
    VOCdevkit_path  = 'VOCdevkit'

    image_ids       = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),'r').read().splitlines() 
    gt_dir_e          = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass/element/")
    gt_dir_d          = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass/defect/")
    miou_out_path_e   = "miou_out/element"
    miou_out_path_d   = "miou_out/defect"
    pred_dir_e        = os.path.join(miou_out_path_e, 'detection-results/')
    pred_dir_d        = os.path.join(miou_out_path_d, 'detection-results/')

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir_e):
            os.makedirs(pred_dir_e)
        if not os.path.exists(pred_dir_d):
            os.makedirs(pred_dir_d) 

        print("Load model.")
        hrnet = HRnet_Segmentation()
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
            image       = Image.open(image_path)
            image       = hrnet.get_miou_png(image)
            image[0].save(os.path.join(pred_dir_e, image_id + ".png"))
            image[1].save(os.path.join(pred_dir_d, image_id + ".png"))
        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        hist_e, IoUs_e, PA_Recall_e, Precision_e, hist_d, IoUs_d, PA_Recall_d, Precision_d = compute_mIoU(gt_dir_e, pred_dir_e, gt_dir_d, pred_dir_d, image_ids, num_classes, name_classes)  # 执行计算mIoU的函数
        print("Get miou done.")
        show_results(miou_out_path_e, hist_e, IoUs_e, PA_Recall_e, Precision_e, name_classes[0])
        show_results(miou_out_path_d, hist_d, IoUs_d, PA_Recall_d, Precision_d, name_classes[1])

