# 【飞桨学习赛：钢铁缺陷检测】第7名方案

## 项目描述
> 飞桨钢铁检测参赛项目

## 项目结构
> 一目了然的项目结构能帮助更多人了解，目录树以及设计思想都很重要~
```
-|data
-|work
-README.MD
-main.ipynb
```
## 训练模型

以下训练模型以ppyoloe-plus-s为例

优化器选择的是momentum
初始学习率是0,01
学习率衰减在前5eopch选择LinearWarmup，后面的是CosineDecay
训练300epochs
加载预训练权重https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_s_80e_coco.pdparams
数据增强使用了:RandomCrop,RandomFlip,BatchRandomResize等
有兴趣可以看一下项目里的具体配置

下面列出ppyoloe_plus_crn_s_80e_coco.yml：

```
_BASE_: [
  '../datasets/voc.yml',
  '../runtime.yml',
  './_base_/optimizer_80e.yml',
  './_base_/ppyoloe_plus_crn.yml',
  './_base_/ppyoloe_plus_reader.yml',
]

log_iter: 100
snapshot_epoch: 5
weights: weights/best_model

depth_mult: 0.33
width_mult: 0.50


pretrain_weights: https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_l_80e_coco.pdparams

```
voc.yml：voc数据集的说明
optimizer_300e:关于优化器的配置
ppyoloe_plus_crn：关于网络结构的配置
ppyoloe_plus_reader：关于数据管道的配置

按照自己的需求修改好配置文件之后就开始训练了：
```
python tools/train.py -c configs/ppyoloe/ppyoloe_plus_crn_s_80e_coco.yml
```
如果需要边训练边评估：
```
python tools/train.py -c configs/ppyoloe/ppyoloe_plus_crn_s_80e_coco.yml --eval
```

## 预测保存结果
训练好模型之后，就可以进行模型推理了。可以在ppyoloe_plus_reader配置文件下修改模型推理的相应的数据管道，比如修改推理图片的大小。当然在这里也需要将我们的检测结果保存到csv文件下，因此对infer的代码添加了一些代码(因为懒，当时报名的时候只剩下10月最后两三天了，代码可能写的有点粗糙，然后后面一直关于优化训练过程，忘记了这里)，并不是像大佬们一样写了一个json转换到csv文件的程序.

对trainer.py的Trainer.predict的修改如下(代码太长，所以只亮出了修改的部分)：
```
 if visualize:
            import csv
            if os.path.exists('submission.csv'):
                os.remove('submission.csv')

            f = open('submission.csv', 'w', encoding='UTF8', newline='')
            writer = csv.writer(f)
            writer.writerow(['image_id', 'bbox', 'category_id', 'confidence'])
            for outs in results:
                batch_res = get_infer_results(outs, clsid2catid,writer)
                bbox_num = outs['bbox_num']
```

对coco_utils.py的get_infer_results函数修改如下
```
def get_infer_results(outs, catid, writer,bias=0):
    
    if outs is None or len(outs) == 0:
        raise ValueError(
            'The number of valid detection result if zero. Please use reasonable model and check input data.'
        )

    im_id = outs['im_id']

    infer_res = {}
    if 'bbox' in outs:
        if len(outs['bbox']) > 0 and len(outs['bbox'][0]) > 6:
            infer_res['bbox'] = get_det_poly_res(
                outs['bbox'], outs['bbox_num'], im_id, catid, bias=bias)
        else:
            infer_res['bbox'] = get_det_res(
                outs['bbox'], outs['bbox_num'], im_id, catid, writer,bias=bias)
```

然后对json_result.py的get_det_res函数修改如下：
```
def get_det_res(bboxes, bbox_nums, image_id, label_to_cat_id_map, writer,bias=0):
    det_res = []
    k = 0
    for i in range(len(bbox_nums)):
        cur_image_id = int(image_id[i][0])
        det_nums = bbox_nums[i]
        for j in range(det_nums):
            dt = bboxes[k]
            k = k + 1
            num_id, score, xmin, ymin, xmax, ymax = dt.tolist()
            if xmin<0:
                xmin=0
            if ymin<0:
                ymin=0

            w_bbox='[{},{},{},{}]'.format(xmin,ymin,xmax,ymax)
            if int(num_id) < 0:
                continue
            category_id = label_to_cat_id_map[int(num_id)]
            info = [str(cur_image_id), w_bbox, str(category_id),str(score)]
            writer.writerow(info)
            w = xmax - xmin + bias
            h = ymax - ymin + bias
            bbox = [xmin, ymin, w, h]
            dt_res = {
                'image_id': cur_image_id,
                'category_id': category_id,
                'bbox': bbox,
                'score': score
            }
            det_res.append(dt_res)
    return det_res
```

然后还需要记得把配置文件ppyoloe_plus_crn_s_80e_coco.yml的weights改成自己的模型。

这样我们就可以预测了：
```
python tools/infer.py -c configs/ppyoloe/ppyoloe_plus_crn_s_80e_coco.yml
```
然后提交我们的csv文件即可。
