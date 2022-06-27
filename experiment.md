# 病理图预测TMB/MSI等基因指标

## 结果

|方法|结果|{batch_size}\_{lr}\_{Epoch}|{ACC}|{Recall}|{F1}|{AUC}|
|--------|--------|--------|--------|--------|--------|--------|
|tile|
|resnet18|不好|config={}||


## 1. 预处理步骤

>  * `执行命令`: ```python utils/slide_core.py```
>  * `参数设置`：
>       * `-m`/`--magnification`: 选择病理图的放大倍数
>       * `--tile_size`: 选择切割出来的tile的大小，常用512
>       * `--document_root`: 
>       * `--data_root`: 存放数据的根目录

  

## 2. data preparing

### 2.1. csv合并

- [x] 合并tile的信息
- [x] 合并slides和tmb的信息
- [x] 分train-test


### 2.2. 建立dataset
 **对于test的dataset应另外设置**
- [x] 基于tile建立dataset
- [ ] 基于mil建立dataset

## 3. training

- [ ] 基于tile的trainer
- [ ] 基于tile的test
  
- [ ] 基于mil的trainer
- [ ] 基于mil的test

#### 4. 可视化
- [ ] 训练集合tile的分布情况
- [ ] 

### 文件结构

#### 1. 训练代码结构

```
    
```

#### 2. 结果文件结构

```
./results/
├── documents
│   └── all_models.csv
└── trained_models
    └── model_1
        ├── checkpoints
        ├── tb
        └── train_mode.logs
```

#### 3. 数据文件结构

```
.
├── slides
│   ├── TCGA-A6-3810-01Z-00-DX1.2940ca70-013a-4bc3-ad6a-cf4d9ffa77ce.svs
│   ├── TCGA-AA-3517-01Z-00-DX1.dac0f9a3-fa10-42e7-acaf-e86fff0829d2.svs
│   ├── ...
│   ├── TCGA-EI-6882-01Z-00-DX1.b4b4638d-be05-453a-9028-de170db5b51e.svs
│   └── TCGA-F5-6811-01Z-00-DX1.8959e075-c641-472e-88a4-0b6081f58ad9.svs
├── thumbnails
│   ├── TCGA-A6-3810-01Z-00-DX1.png
│   ├── TCGA-AA-3517-01Z-00-DX1.png
│   ├── ...
│   ├── TCGA-EI-6882-01Z-00-DX1.png
│   └── TCGA-F5-6811-01Z-00-DX1.png
├── tiles
│   └── 20_224
│       ├── data
│       │   ├── TCGA-A6-3810-01Z-00-DX1
│       │   │   ├── TCGA-A6-3810-01Z-00-DX1_2_[1120_1120].png
│       │   │   ├── TCGA-A6-3810-01Z-00-DX1_2_[1120_224].png
│       │   │   ├── ...
│       │   │   ├── TCGA-A6-3810-01Z-00-DX1_2_[896_672].png
│       │   │   └── TCGA-A6-3810-01Z-00-DX1_2_[896_896].png
│       │   ├── ...
│       │   ├── TCGA-EI-6882-01Z-00-DX1
│       │   │   ├── TCGA-EI-6882-01Z-00-DX1_2_[2240_448].png
│       │   │   ├── TCGA-EI-6882-01Z-00-DX1_2_[2240_672].png
│       │   │   ├── ...
│       │   │   ├── TCGA-EI-6882-01Z-00-DX1_2_[896_448].png
│       │   │   └── TCGA-EI-6882-01Z-00-DX1_2_[896_672].png
│       │   └── TCGA-F5-6811-01Z-00-DX1
│       │       ├── TCGA-F5-6811-01Z-00-DX1_2_[1120_1120].png
│       │       ├── TCGA-F5-6811-01Z-00-DX1_2_[1120_1344].png
│       │       ├── ...
│       │       ├── TCGA-F5-6811-01Z-00-DX1_2_[896_672].png
│       │       └── TCGA-F5-6811-01Z-00-DX1_2_[896_896].png
│       ├── documents
│       │   ├── all_slides_info.csv
│       │   └── slides_tiles_csv
│       │       ├── TCGA-A6-3810-01Z-00-DX1.csv
│       │       ├── TCGA-AA-3517-01Z-00-DX1.csv
│       │       ├── ...
│       │       ├── TCGA-EI-6882-01Z-00-DX1.csv
│       │       └── TCGA-F5-6811-01Z-00-DX1.csv
│       ├── tiles_generator.logs
│       └── vectors
│               ├── TCGA-A6-3810-01Z-00-DX1.h5
│               ├── TCGA-AA-3517-01Z-00-DX1.h5
│               ├── ...
│               ├── TCGA-EI-6882-01Z-00-DX1.h5
│               └── TCGA-F5-6811-01Z-00-DX1.h5
```

#### 4. test result文件结构
```
test_result/
├── model_name                         # 对应模型的名字
│   ├── figure
│   │   ├── slide_0_heatmap.png        # slide_0的热图
│   │   └── slide_0_thumbnail.png      # slide_0的缩略图
│   ├── test.log                       # 测试的日志
│   ├── test_results.csv                # 存储所有slide相应的预测指标、预测label、真实label
│   └── test_tiles_results
│       └── slide_0.csv                # 存储slide_0的tile预测信息
└── models_results.csv                 # 存储所有模型的测试指标
```







