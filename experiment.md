# 病理图预测TMB/MSI等基因指标

## To-Do

- [ ] 要不要打包成docker
- [ ] 把预处理流程打包在一起

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

- [x] 基于tile的trainer
- [x] 基于tile的test
  
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
tumor
├── slides
│   ├── TCGA-A6-3810-01Z-00-DX1.2940ca70-013a-4bc3-ad6a-cf4d9ffa77ce.svs
│   ├── TCGA-AA-3517-01Z-00-DX1.dac0f9a3-fa10-42e7-acaf-e86fff0829d2.svs
│   └── ...
├── thumbnails
│   ├── TCGA-A6-3810-01Z-00-DX1.png
│   └── ...
└── tiles
    └── 20_224
        ├── data 
        │   ├── TCGA-A6-3810-01Z-00-DX1 
        │   │   ├── TCGA-A6-3810-01Z-00-DX1_2_[1120_1120].png
        │   │   ├── TCGA-A6-3810-01Z-00-DX1_2_[1120_224].png
        │   │   └── ...
        │   └── ...
        ├── documents
        │   ├── all_slides_info.csv
        │   └── slides_tiles_csv
        │       ├── TCGA-A6-3810-01Z-00-DX1.csv
        │       ├── TCGA-AA-3517-01Z-00-DX1.csv
        │       └── ...
        ├── tiles_generator.logs
        └── vectors
                ├── TCGA-A6-3810-01Z-00-DX1.pkl
                ├── TCGA-AA-3517-01Z-00-DX1.pkl
                └── ...
```








