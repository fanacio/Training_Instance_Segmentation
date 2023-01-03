import os
import cv2
import numpy as np
import torch
# 默认的颜色跟类别的绑定列表
colors = [
        [56, 0, 255],
        [226, 255, 0],
        [0, 94, 255],
        [0, 37, 255],
        [0, 255, 94],
        [255, 226, 0],
        [0, 18, 255],
        [255, 151, 0],
        [170, 0, 255],
        [0, 255, 56],
        [255, 0, 75],
        [0, 75, 255],
        [0, 255, 169],
        [255, 0, 207],
        [75, 255, 0],
        [207, 0, 255],
        [37, 0, 255],
        [0, 207, 255],
        [94, 0, 255],
        [0, 255, 113],
        [255, 18, 0],
        [255, 0, 56],
        [18, 0, 255],
        [0, 255, 226],
        [170, 255, 0],
        [255, 0, 245],
        [151, 255, 0],
        [132, 255, 0],
        [75, 0, 255],
        [151, 0, 255],
        [0, 151, 255],
        [132, 0, 255],
        [0, 255, 245],
        [255, 132, 0],
        [226, 0, 255],
        [255, 37, 0],
        [207, 255, 0],
        [0, 255, 207],
        [94, 255, 0],
        [0, 226, 255],
        [56, 255, 0],
        [255, 94, 0],
        [255, 113, 0],
        [0, 132, 255],
        [255, 0, 132],
        [255, 170, 0],
        [255, 0, 188],
        [113, 255, 0],
        [245, 0, 255],
        [113, 0, 255],
        [255, 188, 0],
        [0, 113, 255],
        [255, 0, 0],
        [0, 56, 255],
        [255, 0, 113],
        [0, 255, 188],
        [255, 0, 94],
        [255, 0, 18],
        [18, 255, 0],
        [0, 255, 132],
        [0, 188, 255],
        [0, 245, 255],
        [0, 169, 255],
        [37, 255, 0],
        [255, 0, 151],
        [188, 0, 255],
        [0, 255, 37],
        [0, 255, 0],
        [255, 0, 170],
        [255, 0, 37],
        [255, 75, 0],
        [0, 0, 255],
        [255, 207, 0],
        [255, 0, 226],
        [255, 245, 0],
        [188, 255, 0],
        [0, 255, 18],
        [0, 255, 75],
        [0, 255, 151],
        [255, 56, 0],
        [245, 255, 0],
    ]

# 使用cv2.dnn加载onnx进行推理的接口
class yolact_onnx():
    def __init__(self, onnx_path:str, confThreshold=0.5, nmsThreshold=0.5, keep_top_k=200, COCO_CLASSES=('background', 'T','S','Z','L','J','O','I')):
        r"""初始化yolact分割接口

        Args:
            onnx_path (str): 传入的onnx模型路径
            confThreshold (float): 后处理的判别置信度,越大则去除的目标个数越多,默认是0.5, 取值范围是[0.,1.].
            nmsThreshold (float): iou极大值抑制的阈值,越大则相邻重叠框越多,默认是0.5,取值范围是[0.,1.].
            keep_top_k (int): nms最大保留的目标个数，默认是200.
            COCO_CLASSES (tuple): 表示对应的类别名称，默认是('background', 'T','S','Z','L','J','O','I')，长度是1+类别数.
        """
        self.target_size = 550
        self.MEANS = np.array([103.94, 116.78, 123.68], dtype=np.float32).reshape(1, 1, 3)
        self.STD = np.array([57.38, 57.12, 58.40], dtype=np.float32).reshape(1, 1, 3)
        self.net = cv2.dnn.readNet(onnx_path)
        self.confidence_threshold = confThreshold
        self.nms_threshold = nmsThreshold
        self.keep_top_k = keep_top_k
        self.conv_ws = [69, 35, 18, 9, 5]
        self.conv_hs = [69, 35, 18, 9, 5]
        self.aspect_ratios = [1, 0.5, 2]
        self.scales = [24, 48, 96, 192, 384]
        self.variances = [0.1, 0.2]
        self.last_img_size = None
        self.priors = self.make_priors()
        self.COCO_CLASSES = COCO_CLASSES

    # 创建锚点矩阵
    def make_priors(self):
        """ Note that priors are [x,y,width,height] where (x,y) is the center of the box. """
        if self.last_img_size != (self.target_size, self.target_size):
            prior_data = []

            for conv_w, conv_h, scale in zip(self.conv_ws, self.conv_hs, self.scales):
                for i in range(conv_h):
                    for j in range(conv_w):
                        # +0.5 because priors are in center-size notation
                        cx = (j + 0.5) / conv_w
                        cy = (i + 0.5) / conv_h

                        for ar in self.aspect_ratios:
                            ar = np.sqrt(ar)

                            w = scale * ar / self.target_size
                            h = scale / ar / self.target_size

                            # 因为目前的锚点都是正方形的，所以目前暂且也是正方形的，需要配合训练时的cfg配置锚点信息
                            h = w

                            prior_data += [cx, cy, w, h]

            self.priors = np.array(prior_data).reshape(-1, 4)
            self.last_img_size = (self.target_size, self.target_size)
        return self.priors

    def decode(self, loc, priors, img_w, img_h):
        boxes = np.concatenate(
            (
                priors[:, :2] + loc[:, :2] * self.variances[0] * priors[:, 2:],
                priors[:, 2:] * np.exp(loc[:, 2:] * self.variances[1]),
            ),
            1,
        )
        boxes[:, :2] -= boxes[:, 2:] / 2
        # boxes[:, 2:] += boxes[:, :2]

        # crop
        np.where(boxes[:, 0] < 0, 0, boxes[:, 0])
        np.where(boxes[:, 1] < 0, 0, boxes[:, 1])
        np.where(boxes[:, 2] > 1, 1, boxes[:, 2])
        np.where(boxes[:, 3] > 1, 1, boxes[:, 3])

        # decode to img size
        boxes[:, 0] *= img_w
        boxes[:, 1] *= img_h
        boxes[:, 2] = boxes[:, 2] * img_w + 1
        boxes[:, 3] = boxes[:, 3] * img_h + 1
        return boxes

    def detect(self, srcimg):
        r"""进行分割推理的接口

        Args:
            srcimg (numpy.ndarray): 通过opencv读取出来的图片信息，要求是[H,W,C=BGR]的图像数据.
        """
        # 1. 前处理流程：resize->norm->bgr2rgb->[h,w,c]2[c,h,w]
        img_h, img_w = srcimg.shape[:2]
        img = cv2.resize(srcimg, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        img = (img - self.MEANS) / self.STD
        # 2. 正式放入dnn中进行推理
        blob = cv2.dnn.blobFromImage(img, swapRB=True)
        self.net.setInput(blob)
        loc_data, conf_preds, mask_data, proto_data = self.net.forward(self.net.getUnconnectedOutLayersNames())
        # 3. 后处理流程
        # 3.1. 获取对应的最大类别
        cur_scores = conf_preds[:, 1:]
        num_class = cur_scores.shape[1]
        # 为了容错，这里根据num_class来添加类别
        if len(self.COCO_CLASSES) < num_class + 1:
            coconum = len(self.COCO_CLASSES)
            for cls_name in range(num_class + 1 - coconum):
                self.COCO_CLASSES = self.COCO_CLASSES + (str(cls_name + coconum), )
        classid = np.argmax(cur_scores, axis=1)
        # conf_scores = np.max(cur_scores, axis=1)
        conf_scores = cur_scores[range(cur_scores.shape[0]), classid]

        # 3.2. 根据置信度过滤得分低的框，同时解码框
        keep = conf_scores > self.confidence_threshold
        conf_scores = conf_scores[keep]
        classid = classid[keep]
        loc_data = loc_data[keep, :]
        prior_data = self.priors[keep, :]
        masks = mask_data[keep, :]
        boxes = self.decode(loc_data, prior_data, img_w, img_h)
        # 3.3. 调用nms进行框的极大值抑制
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), conf_scores.tolist(), self.confidence_threshold, self.nms_threshold , top_k=self.keep_top_k)
        for i in indices:
            #idx = i[0]
            idx =i
            left, top, width, height = boxes[idx, :].astype(np.int32).tolist()
            cv2.rectangle(srcimg, (left, top), (left+width, top+height), (0, 0, 255), thickness=1)
            cv2.putText(srcimg, self.COCO_CLASSES[classid[idx]+1]+':'+str(round(conf_scores[idx], 2)), (left, top-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)

            # 3.4. 生成分割掩模，proto_data是一个三维的张量，[138,138,32]
            mask = proto_data @ masks[idx, :].reshape(-1,1)
            mask = 1 / (1 + np.exp(-mask))  ###sigmoid

            # 重新scale回输入图像的尺寸
            mask = cv2.resize(mask.squeeze(), (img_w, img_h), interpolation=cv2.INTER_LINEAR)
            mask = mask > 0.5
            srcimg[mask] = srcimg[mask] * 0.5 + np.array(colors[classid[idx]+1]) * 0.5
        return srcimg


def eval_using_onnx(onnx_path:str, imgpath:str, respath:str, confThreshold=0.5, nmsThreshold=0.5, keep_top_k=200, COCO_CLASSES=('background', 'T','S','Z','L','J','O','I')):
    """ 用于测试生成的onnx是否正常的接口 """
    myyolact = yolact_onnx(onnx_path=onnx_path, confThreshold=confThreshold, nmsThreshold=nmsThreshold, keep_top_k=keep_top_k, COCO_CLASSES=COCO_CLASSES)
    srcimg = cv2.imread(imgpath)
    srcimg = myyolact.detect(srcimg)
    cv2.imwrite(respath, srcimg)


def torch2onnx(torchmodel_path:str, onnx_path:str):
    r"""用于torch模型转换onnx部署模型的接口

    Args:
        torchmodel_path (str): 传入的内存中的pytorch模型路径.
        onnx_path (str): onnx模型保持的路径.
    """
    from yolact import Yolact
    torchmodel = Yolact()
    assert os.path.exists(torchmodel_path)
    torchmodel.load_weights(torchmodel_path)
    torchmodel.eval()
    device = 'cpu'
    torchmodel.eval()
    torchmodel.to(device)
    input = torch.randn(1, 3, 550, 550).to(device)
    print('convert',onnx_path,'begin')
    torch.onnx.export(torchmodel, input, onnx_path, verbose=False, opset_version=12, input_names=['image'],
                      output_names=['loc', 'conf', 'mask', 'proto'])
    print('convert', onnx_path, 'to onnx finish!!!')


def may_make_dir(path):
    path = os.path.dirname(os.path.abspath(path))
    if path in [None, '']:
        return
    if not os.path.exists(path):
        os.makedirs(path)


if __name__=='__main__':
    """ 测试代码，直接运行本脚本将会运行如下命令 """
    this_path = os.path.dirname(os.path.realpath(__file__))
    pth_path = os.path.join(this_path, "pth_model/yolact_base_505_100000.pth")
    onnx_path = os.path.join(this_path, "onnx_model/yolact_base_505_100000.onnx")
    input_images_path = os.path.join(this_path, "images/Color_20220411_130208.bmp")
    output_images_path = os.path.join(this_path, "images_out/Color_20220411_130208.bmp")
    # 保证文件存在性
    assert os.path.exists(pth_path) and os.path.exists(input_images_path)
    may_make_dir(onnx_path)
    may_make_dir(output_images_path)
    torch2onnx(pth_path, onnx_path)
    eval_using_onnx(onnx_path,
                    input_images_path,
                    output_images_path)